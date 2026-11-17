"""One on-device model in its own process, driven by JSON lines on stdin.

    python proto/worker.py --kind stt --engine shruti
    python proto/worker.py --kind tts --engine piper

Requests  {"op": "load", "lang": "hi"}
          {"op": "stt", "wav": path, "lang": "hi"}
          {"op": "tts", "text": "...", "lang": "hi", "voice": null, "speed": 1.0, "out": path}
Replies   one JSON line each: {"ok": true, ...} or {"ok": false, "error": "..."}

Workers are started by voicebox.py inside a capped Job Object.  The TTS
worker may run under a different Python (the PyTorch venvs), so this file
imports nothing beyond numpy/soundfile at the top.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rig"))
sys.path.insert(0, str(ROOT / "proto"))

SENT = re.compile(r"(?<=[.!?।॥])\s+")


def mem() -> dict:
    try:
        import psutil
        m = psutil.Process().memory_info()
        return {"rss_mb": round(m.rss / 1048576), "commit_mb": round(getattr(m, "private", m.rss) / 1048576),
                "peak_commit_mb": round(getattr(m, "peak_pagefile", 0) / 1048576)}
    except ImportError:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["stt", "tts"], required=True)
    ap.add_argument("--engine", required=True)
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf

    if a.kind == "stt":
        from stt import get_stt, load_16k
        model = get_stt(a.engine)
    else:
        from engines import get_engine
        model = get_engine(a.engine)

    def reply(**kw):
        sys.stdout.write(json.dumps(kw, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    reply(ok=True, ready=True, **mem())
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        t0 = time.perf_counter()
        try:
            op = req["op"]
            if op == "quit":
                reply(ok=True); break
            if op == "mem":
                reply(ok=True, **mem()); continue
            if op == "load":
                if a.kind == "stt":
                    model.transcribe(np.zeros(16000, dtype=np.float32), req["lang"])
                else:
                    warm = {"hi": "नमस्ते।", "mr": "नमस्कार.", "en": "Hello."}[req["lang"]]
                    model.synth(warm, req["lang"], req.get("voice"))
                reply(ok=True, secs=round(time.perf_counter() - t0, 3), **mem())
            elif op == "stt":
                audio = load_16k(req["wav"])
                text = model.transcribe(audio, req["lang"])
                secs = time.perf_counter() - t0
                reply(ok=True, text=text, secs=round(secs, 3), audio_s=round(len(audio) / 16000, 2), **mem())
            elif op == "tts":
                from core import normalise
                parts = [s for s in SENT.split(req["text"].strip()) if s.strip()] or [req["text"]]
                chunks, sr, first = [], None, None
                for i, s in enumerate(parts):
                    audio, sr = model.synth(s, req["lang"], req.get("voice"), float(req.get("speed", 1.0)))
                    if first is None:
                        first = time.perf_counter() - t0          # sentence streaming: playback starts here
                    chunks.append(audio)
                    chunks.append(np.zeros(int(0.18 * sr), dtype=np.float32))
                audio = normalise(np.concatenate(chunks[:-1]), sr)
                Path(req["out"]).parent.mkdir(parents=True, exist_ok=True)
                sf.write(req["out"], audio, sr, format="FLAC", subtype="PCM_16")
                secs = time.perf_counter() - t0
                reply(ok=True, secs=round(secs, 3), first_audio_s=round(first, 3), sentences=len(parts),
                      duration=round(len(audio) / sr, 2), sample_rate=sr, **mem())
            else:
                reply(ok=False, error=f"unknown op {op}")
        except MemoryError:
            reply(ok=False, error="MemoryError: the memory cap was hit", oom=True, **mem())
        except Exception as ex:
            oom = "bad_alloc" in str(ex) or "Failed to allocate" in str(ex) or "not enough memory" in str(ex).lower()
            reply(ok=False, error=f"{type(ex).__name__}: {str(ex)[:400]}", oom=oom,
                  trace=traceback.format_exc()[-800:], **mem())


if __name__ == "__main__":
    main()
