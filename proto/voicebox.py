"""The on-device half of Kaabil, simulated: one Ear (STT) + one Voice (TTS)
running as two worker processes that share ONE memory budget, the way they
would on a 2 GB phone.

How the cap works.  Android kills an app when the memory it actually *uses*
(resident pages) is too high; it does not care how much address space a
library reserves.  Windows commit accounting does count reservations: Whisper
reserves 2.7 GB but uses ~360 MB.  So the cap enforced here is on RESIDENT
memory (sum of both workers' working sets, sampled every 100 ms), like the
Android low-memory killer.  A Job Object commit limit (4x the cap) stays on as
a backstop.  Both numbers are reported.

    box = VoiceBox(stt="shruti", tts="piper", lang="hi", limit_mb=2048)
    box.start()                         # loads both; raises CapExceeded if they don't fit
    r = box.transcribe("clip.wav")      # {"text", "secs", ...}
    r = box.speak("नमस्ते", "out.flac")  # {"secs", "first_audio_s", "duration", ...}
    box.stats()                         # peak committed MB for the pair
    box.close()
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "proto"))
sys.path.insert(0, str(ROOT / "rig"))

from engines import RIG_HOME  # noqa: E402
from memcap import CapJob  # noqa: E402

MAIN_PY = Path(sys.executable)
# Engines that need PyTorch run under their own interpreter.
ENGINE_PY = {
    "indic-tts": RIG_HOME / "venv_indic" / "Scripts" / "python.exe",
    "indic-tts-full": RIG_HOME / "venv_indic" / "Scripts" / "python.exe",
    "parler": RIG_HOME / "venv_parler" / "Scripts" / "python.exe",
}
LOGS = RIG_HOME / "worker_logs"


class CapExceeded(RuntimeError):
    pass


class WorkerDied(RuntimeError):
    pass


class Worker:
    def __init__(self, kind: str, engine: str, job: CapJob, threads: int, watch: list | None = None):
        py = ENGINE_PY.get(engine, MAIN_PY) if kind == "tts" else MAIN_PY
        if not Path(py).exists():
            raise RuntimeError(f"{engine} needs {py}; run setup_models.py")
        LOGS.mkdir(parents=True, exist_ok=True)
        self.log = LOGS / f"{kind}_{engine}.log"
        env = dict(os.environ, VOICE_RIG_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
                   PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HOME=str(RIG_HOME / "hf_cache"))
        self.p = subprocess.Popen([str(py), "-X", "utf8", str(ROOT / "proto" / "worker.py"),
                                   "--kind", kind, "--engine", engine],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(self.log, "w", encoding="utf-8"),
                                  text=True, encoding="utf-8", env=env,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        job.add(self.p)
        if watch is not None:
            watch.append(self)
        self.kind, self.engine = kind, engine
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.ready = self._get(180)

    def _pump(self):
        for line in self.p.stdout:
            if line.strip().startswith("{"):
                self.q.put(json.loads(line))
        self.q.put(None)

    def _get(self, timeout):
        try:
            r = self.q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"{self.kind} worker '{self.engine}' did not answer in {timeout}s")
        if r is None:
            tail = self.log.read_text(encoding="utf-8", errors="replace")[-600:]
            raise WorkerDied(f"{self.kind} worker '{self.engine}' exited (code {self.p.wait()}). {tail}")
        return r

    def call(self, timeout=600, **req):
        if self.p.poll() is not None:
            raise WorkerDied(f"{self.kind} worker '{self.engine}' is not running")
        self.p.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        return self._get(timeout)

    def close(self):
        try:
            if self.p.poll() is None:
                self.p.stdin.write('{"op":"quit"}\n'); self.p.stdin.flush()
                self.p.wait(5)
        except Exception:
            self.p.kill()


class Monitor(threading.Thread):
    """Samples the workers' resident memory; kills them if the sum exceeds the cap."""

    def __init__(self, box: "VoiceBox"):
        super().__init__(daemon=True)
        self.box, self.peak, self.peak_commit, self.killed, self.stop = box, 0, 0, None, False

    def run(self):
        import psutil
        while not self.stop:
            rss = commit = 0
            for w in list(self.box.workers.values()) + self.box.starting:
                try:   # venv python.exe is a launcher; the interpreter is its child
                    root = psutil.Process(w.p.pid)
                    for pr in [root] + root.children(recursive=True):
                        mi = pr.memory_info()
                        rss += mi.rss; commit += getattr(mi, "private", 0)
                except Exception:
                    pass
            self.peak = max(self.peak, rss >> 20)
            self.peak_commit = max(self.peak_commit, commit >> 20)
            if (rss >> 20) > self.box.cfg["limit_mb"] and not self.killed:
                self.killed = f"resident memory reached {rss >> 20} MB, over the {self.box.cfg['limit_mb']} MB cap"
                for w in list(self.box.workers.values()):
                    try:
                        w.p.kill()
                    except Exception:
                        pass
            time.sleep(0.1)


class VoiceBox:
    def __init__(self, stt: str | None, tts: str | None, lang: str, limit_mb: int = 2048,
                 threads: int = 4, cpu_percent: float | None = None, voice: str | None = None):
        self.cfg = dict(stt=stt, tts=tts, lang=lang, limit_mb=limit_mb, threads=threads,
                        cpu_percent=cpu_percent, voice=voice)
        self.job = None
        self.workers: dict[str, Worker] = {}
        self.load = {}
        self.lock = threading.Lock()
        self.mon = None
        self.starting: list = []   # worker being spawned, watched before it is ready

    def start(self) -> dict:
        c = self.cfg
        self.job = CapJob(max(c["limit_mb"] * 4, 8192), c["cpu_percent"])   # backstop only
        self.mon = Monitor(self)
        self.mon.start()
        try:
            for kind in ("stt", "tts"):           # Ear first: it must stay resident while the Voice loads
                if not c[kind]:
                    continue
                w = Worker(kind, c[kind], self.job, c["threads"], self.starting)
                self.workers[kind] = w
                self.starting.clear()
                r = w.call(op="load", lang=c["lang"], voice=c["voice"], timeout=900)
                if not r.get("ok"):
                    raise (CapExceeded if r.get("oom") else RuntimeError)(f"{kind} {c[kind]}: {r.get('error')}")
                self.load[kind] = {"secs": r["secs"], "commit_mb": r.get("commit_mb")}
        except (WorkerDied, CapExceeded, TimeoutError) as ex:
            killed = self.mon.killed
            backstop = max(c["limit_mb"] * 4, 8192)
            if not killed and isinstance(ex, WorkerDied) and self.job and self.job.peak_mb() >= backstop * 0.95:
                killed = (f"committed memory hit the {backstop} MB safety backstop while loading "
                          f"(resident was {self.mon.peak} MB and still rising); the process crashed")
            self.close()
            if killed or isinstance(ex, CapExceeded):
                raise CapExceeded(f"does not fit in {c['limit_mb']} MB: {killed or ex}") from None
            raise
        return self.stats()

    def _call(self, kind: str, **req) -> dict:
        with self.lock:
            try:
                return self.workers[kind].call(**req)
            except WorkerDied:
                if self.mon and self.mon.killed:
                    raise CapExceeded(self.mon.killed) from None
                raise

    def transcribe(self, wav: str) -> dict:
        return self._call("stt", op="stt", wav=str(wav), lang=self.cfg["lang"])

    def speak(self, text: str, out: str, speed: float = 1.0) -> dict:
        return self._call("tts", op="tts", text=text, lang=self.cfg["lang"],
                          voice=self.cfg["voice"], speed=speed, out=str(out))

    def stats(self) -> dict:
        m = getattr(self, "mon", None)
        return {**self.cfg, "peak_mb": m.peak if m else None, "peak_commit_mb": m.peak_commit if m else None,
                "killed": m.killed if m else None, "load": self.load}

    def alive(self) -> bool:
        return bool(self.workers) and all(w.p.poll() is None for w in self.workers.values())

    def close(self):
        if getattr(self, "mon", None):
            self.mon.stop = True
        for w in self.workers.values():
            w.close()
        self.workers = {}
        if self.job:
            self.job.close()
            self.job = None
