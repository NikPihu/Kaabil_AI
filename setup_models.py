"""Download every model and package the rig needs, and report what is missing.

    python setup_models.py            # fetch everything that is missing
    python setup_models.py --check    # only print the checklist
    python setup_models.py --only shruti,vosk

Safe to re-run: anything already present is skipped.  Everything goes to
VOICE_RIG_HOME (default C:/Users/<you>/voice_rig), outside OneDrive.
Sources are the model makers' own repositories only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HOME = Path(os.environ.get("VOICE_RIG_HOME", Path.home() / "voice_rig"))
MODELS = HOME / "models"
VENV = HOME / "venv"
PY = VENV / "Scripts" / "python.exe"
TORCH_VENVS = {"indic": HOME / "venv_indic", "parler": HOME / "venv_parler"}
PROJECT = Path(__file__).resolve().parent


def _env() -> dict:
    env = dict(os.environ)
    p = PROJECT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    env.setdefault(k.strip(), v.strip())
    env["HF_HOME"] = str(HOME / "hf_cache")
    return env


def hf(repo: str, dest: Path, patterns: list[str] | None = None) -> None:
    code = ("import sys; from huggingface_hub import snapshot_download as s; "
            "s(sys.argv[1], local_dir=sys.argv[2], allow_patterns=(sys.argv[3].split('|') if sys.argv[3] else None))")
    subprocess.run([str(PY), "-c", code, repo, str(dest), "|".join(patterns or [])],
                   check=True, env=_env())


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"    downloading {url}")
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length", 0)); done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk); done += len(chunk)
            if total and done % (100 << 20) < (1 << 20):
                print(f"      {done >> 20} / {total >> 20} MB", flush=True)
    part.rename(dest)


def unzip(zf: Path, dest: Path) -> None:
    with zipfile.ZipFile(zf) as z:
        for m in z.namelist():                     # refuse paths that escape dest
            if (dest / m).resolve().is_relative_to(dest.resolve()) is False:
                raise RuntimeError(f"unsafe path in {zf}: {m}")
        z.extractall(dest)


def pip(python: Path, *args: str) -> None:
    subprocess.run([str(python), "-m", "pip", "install", "-q", *args], check=True)


def venv(path: Path) -> Path:
    py = path / "Scripts" / "python.exe"
    if not py.exists():
        subprocess.run(["py", "-3.12", "-m", "venv", str(path)], check=True)
        pip(py, "--upgrade", "pip")
    return py


def has_pkg(python: Path, *mods: str) -> bool:
    """Fast check (find_spec, no import), one interpreter start for all modules."""
    code = "import importlib.util as u, sys; sys.exit(0 if all(u.find_spec(m) for m in sys.argv[1:]) else 1)"
    return python.exists() and subprocess.run([str(python), "-c", code, *mods],
                                              capture_output=True).returncode == 0


def size_mb(p: Path) -> int:
    if p.is_file():
        return round(p.stat().st_size / 1e6)
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6) if p.exists() else 0


# --------------------------------------------------------------------------
# Every item: role, where it lives, how to tell it is there, how to get it.
# --------------------------------------------------------------------------
def _shruti(lang):
    return {"id": f"shruti-{lang}", "role": f"Ear (STT) {lang}", "path": MODELS / "shruti" / lang,
            "ok": lambda p: (p / "sherpa-onnx" / "encode.onnx").exists() and (p / "onnx-int8" / "encoder.onnx").exists(),
            "get": lambda p: hf("udgamlabs/Shruti-ONNX", MODELS / "shruti", [f"{lang}/sherpa-onnx/*", f"{lang}/onnx-int8/*", "LICENSE", "NOTICE"]),
            "manual": f"https://huggingface.co/udgamlabs/Shruti-ONNX/tree/main/{lang}"}


def _indic(lang):
    p = MODELS / "indic-tts" / lang
    def get(p, lang=lang):
        z = MODELS / "indic-tts" / f"{lang}.zip"
        if not z.exists():
            fetch(f"https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/{lang}.zip", z)
        unzip(z, MODELS / "indic-tts")
        z.unlink()
        # inference-only copy (drops optimizer state): the as-released files do not fit in 2 GB
        subprocess.run([str(TORCH_VENVS["indic"] / "Scripts" / "python.exe"), str(PROJECT / "proto" / "slim_indic.py"), lang], check=True)
    return {"id": f"indic-tts-{lang}", "role": f"Voice {lang} (deck's choice)", "path": p,
            "ok": lambda p: (p / "slim" / "hifigan.pth").exists(),
            "get": get, "size_hint": "1.5 GB download",
            "manual": f"https://github.com/AI4Bharat/Indic-TTS/releases/tag/v1-checkpoints-release  ({lang}.zip, unzip into {p.parent})"}


def _vosk_get(p):
    z = MODELS / "vosk-small-hi.zip"
    fetch("https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip", z)
    unzip(z, MODELS)
    (MODELS / "vosk-model-small-hi-0.22").rename(p)
    z.unlink()


ITEMS = [
    # ---- packages
    {"id": "pkg-core", "role": "Rig runtime", "path": VENV,
     "ok": lambda p: has_pkg(PY, "sherpa_onnx", "faster_whisper", "piper", "kokoro_onnx", "supertonic", "psutil"),
     "get": lambda p: pip(PY, "-r", str(PROJECT / "requirements.txt"))},
    {"id": "pkg-vosk", "role": "Ear (STT) runtime", "path": VENV,
     "ok": lambda p: has_pkg(PY, "vosk"), "get": lambda p: pip(PY, "vosk")},
    {"id": "pkg-indic", "role": "Indic-TTS runtime (PyTorch + Coqui TTS)", "path": TORCH_VENVS["indic"],
     "ok": lambda p: has_pkg(p / "Scripts" / "python.exe", "TTS"),
     "get": lambda p: (pip(venv(p), "torch", "--index-url", "https://download.pytorch.org/whl/cpu"),
                       pip(venv(p), "coqui-tts", "soundfile", "psutil", "numpy"))},
    {"id": "pkg-parler", "role": "Parler-TTS runtime (PyTorch)", "path": TORCH_VENVS["parler"],
     "ok": lambda p: has_pkg(p / "Scripts" / "python.exe", "parler_tts"),
     "get": lambda p: (pip(venv(p), "torch", "--index-url", "https://download.pytorch.org/whl/cpu"),
                       pip(venv(p), "git+https://github.com/huggingface/parler-tts.git", "soundfile", "psutil"))},
    # ---- ears
    _shruti("hi"), _shruti("mr"),
    {"id": "whisper-small", "role": "Ear (STT) hi/mr/en", "path": MODELS / "whisper-small",
     "ok": lambda p: (p / "model.bin").exists(),
     "get": lambda p: hf("Systran/faster-whisper-small", p), "manual": "https://huggingface.co/Systran/faster-whisper-small"},
    {"id": "vosk-small-hi", "role": "Ear (STT) hi, team's v1 choice", "path": MODELS / "vosk-small-hi",
     "ok": lambda p: (p / "am").exists(), "get": _vosk_get,
     "manual": "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip"},
    # ---- voices
    _indic("hi"), _indic("mr"),
    {"id": "parler-mini", "role": "English voice (deck's choice)", "path": MODELS / "parler-tts-mini-v1",
     "ok": lambda p: (p / "model.safetensors").exists(),
     "get": lambda p: hf("parler-tts/parler-tts-mini-v1", p), "size_hint": "3.5 GB download",
     "manual": "https://huggingface.co/parler-tts/parler-tts-mini-v1"},
    # ---- from the first session (checked, re-fetched only if missing)
    {"id": "piper", "role": "Voice hi/mr/en", "path": MODELS / "piper", "ok": lambda p: (p / "mr").exists(),
     "get": lambda p: hf("rhasspy/piper-voices", p, ["hi/hi_IN/*/medium/*.onnx*", "mr/mr_IN/google/medium/*.onnx*",
                                                    "en/en_US/amy/medium/*.onnx*", "en/en_US/ryan/medium/*.onnx*"])},
    {"id": "supertonic-int8", "role": "Voice hi/en (phone build)", "path": MODELS / "sherpa-onnx-supertonic-3-tts-int8-2026-05-11",
     "ok": lambda p: (p / "vocoder.int8.onnx").exists(), "get": None,
     "manual": "https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models"},
    {"id": "kokoro", "role": "Voice hi/en", "path": MODELS / "kokoro", "ok": lambda p: (p / "kokoro-v1.0.int8.onnx").exists(),
     "get": None, "manual": "https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0"},
    {"id": "whisper-judge", "role": "Judge (scores clips, runs outside the cap)", "path": HOME / "hf_cache" / "hub" / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo",
     "ok": lambda p: p.exists(), "get": None, "manual": "downloaded automatically by rig/evaluate.py"},
]

KEYS = [
    ("OPENROUTER_API_KEY", True, "The brain: profile filling and follow-up questions"),
    ("HF_TOKEN", False, "Faster downloads; gated Indic Parler-TTS"),
    ("SARVAM_API_KEY", False, "Sarvam Bulbul comparison voice"),
    ("ELEVENLABS_API_KEY", False, "ElevenLabs reference voice"),
]


def status() -> dict:
    env = _env()
    return {
        "home": str(HOME),
        "keys": [{"name": k, "required": req, "unlocks": what, "set": bool(env.get(k))} for k, req, what in KEYS],
        "items": [{"id": it["id"], "role": it["role"], "path": str(it["path"]), "present": bool(it["ok"](it["path"])),
                   "size_mb": None if it["id"].startswith("pkg-") else size_mb(it["path"]),
                   "manual": it.get("manual", ""), "size_hint": it.get("size_hint", "")} for it in ITEMS],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    MODELS.mkdir(parents=True, exist_ok=True)
    if not a.check:
        for it in ITEMS:
            if only and it["id"] not in only:
                continue
            if it["ok"](it["path"]):
                continue
            if it["get"] is None:
                print(f"[{it['id']}] missing and has no automatic source: {it.get('manual')}")
                continue
            print(f"[{it['id']}] fetching {it.get('size_hint', '')}", flush=True)
            try:
                it["get"](it["path"])
            except Exception as ex:
                print(f"[{it['id']}] FAILED: {type(ex).__name__}: {ex}\n    get it by hand: {it.get('manual', '')}")
    s = status()
    if a.json:
        print(json.dumps(s, indent=1)); return
    print("\nKeys (.env)")
    for k in s["keys"]:
        print(f"  {'OK ' if k['set'] else ('-- ' if not k['required'] else 'XX ')} {k['name']:20s} {'required' if k['required'] else 'optional'}  {k['unlocks']}")
    print("\nModels and packages")
    for i in s["items"]:
        print(f"  {'OK ' if i['present'] else 'XX '} {i['id']:16s} {i['role']:45s} {str(i['size_mb'] or '') + ' MB' if i['size_mb'] else ''}")
    free = shutil.disk_usage(HOME).free / 1e9
    print(f"\nFree disk: {free:.0f} GB")


if __name__ == "__main__":
    main()
