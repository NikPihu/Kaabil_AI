"""Kaabil voice prototype server.

    python rig/server.py            ->  http://127.0.0.1:8765

Serves the UI, the Conversation and Judges brief APIs (proto/convo_api.py) and
the Live bench: synthesis of any text with any voice, and the dialogue mixer.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from core import OUT, WARMUP, analyse, render_dialogue, render_single, save  # noqa: E402
from engines import PROJECT, engines, get_engine  # noqa: E402

WEB = PROJECT / "web"
LIVE = OUT / "live"
_lock = threading.Lock()
_warm: set[tuple] = set()

app = FastAPI(title="Kaabil voice prototype")
OUT.mkdir(exist_ok=True)


def _json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/state")
def state():
    return {
        "engines": [e.info() for e in engines().values()],
        "questions": _json(PROJECT / "bench" / "questions.json", {}),
    }


class SynthReq(BaseModel):
    engine: str
    lang: str
    text: str
    voice: str | None = None
    speed: float = 1.0
    dialogue: bool = False
    overlap: float = 0.8


@app.post("/api/synth")
def synth(r: SynthReq):
    if r.engine not in engines():
        raise HTTPException(404, "unknown engine")
    eng = get_engine(r.engine)
    ok, why = eng.available()
    if not ok:
        raise HTTPException(400, why)
    if r.lang not in eng.voices:
        raise HTTPException(400, f"{eng.name} has no {r.lang} voice")
    text = r.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    if len(text) > 3000:
        raise HTTPException(400, "text too long (3000 characters max)")
    speed = min(2.0, max(0.5, r.speed))
    extra = {}
    try:
        # First use of a voice loads its model; do that outside the timed run.
        warm = [v["id"] for v in eng.voices[r.lang][:3]] if r.dialogue else [r.voice or eng.voices[r.lang][0]["id"]]
        for v in warm:
            if (r.engine, r.lang, v) not in _warm:
                eng.synth(WARMUP[r.lang], r.lang, v)
                _warm.add((r.engine, r.lang, v))
        if r.dialogue:
            turns = []
            for line in text.splitlines():
                m = re.match(r"\s*([A-Ea-e])\s*[:：]\s*(.+)", line)
                if m:
                    turns.append({"who": m.group(1).upper(), "text": m.group(2).strip(),
                                  "overlap": r.overlap if turns else 0.0})
            if len(turns) < 2:
                raise HTTPException(400, "Dialogue needs at least two lines like 'A: hello' / 'B: hi'")
            audio, sr, gen, extra = render_dialogue(r.engine, turns, r.lang, speed)
        else:
            audio, sr, gen = render_single(r.engine, text, r.lang, r.voice, speed)
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(500, f"{type(ex).__name__}: {str(ex)[:300]}")
    name = f"{int(time.time() * 1000)}_{r.engine}.flac"
    save(audio, sr, LIVE / name)
    m = analyse(audio, sr)
    m.update(extra)
    m.update({"url": f"/audio/live/{name}", "file": f"live/{name}", "engine": r.engine,
              "lang": r.lang, "sample_rate": sr, "gen_time": round(gen, 2),
              "rtf": round(gen / max(m["duration"], 0.01), 3)})
    return m


class AsrReq(BaseModel):
    file: str
    lang: str
    reference: str = ""


@app.post("/api/transcribe")
def transcribe(r: AsrReq):
    """Whisper round-trip for a clip (loads a 1.6 GB model on first use)."""
    path = (OUT / r.file).resolve()
    if OUT.resolve() not in path.parents or not path.exists():
        raise HTTPException(404, "no such clip")
    import evaluate as ev
    with _lock:
        hyp = ev.transcribe(str(path), r.lang)
    return {"asr_text": hyp, "cer": ev.cer(r.reference, hyp) if r.reference else None}




sys.path.insert(0, str(PROJECT / "proto"))
from convo_api import router as convo_router  # noqa: E402
app.include_router(convo_router)

app.mount("/audio", StaticFiles(directory=OUT), name="audio")
app.mount("/web", StaticFiles(directory=WEB), name="web")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8765")), log_level="warning")
