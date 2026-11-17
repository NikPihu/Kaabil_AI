"""Conversation endpoints: the prototype loop, served next to the rig.

POST /api/convo/start  {lang, stt, tts, voice, limit_mb, threads, cpu_percent}
POST /api/convo/turn   {sid, text?, audio_b64? | example?, echo?}
GET  /api/setup        keys + models checklist
Only one VoiceBox (Ear + Voice under the cap) is alive at a time.
"""
from __future__ import annotations

import base64
import json
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "proto"))
sys.path.insert(0, str(ROOT))

import llm  # noqa: E402
from agent import ProfileAgent  # noqa: E402
from core import OUT  # noqa: E402
from engines import engines  # noqa: E402
from stt import stts  # noqa: E402
from voicebox import CapExceeded, VoiceBox  # noqa: E402

router = APIRouter()
CONVOS = OUT / "convos"
_box: VoiceBox | None = None
_box_lock = threading.Lock()
_sessions: dict[str, dict] = {}


@router.get("/api/setup")
def setup():
    import setup_models
    engines_ = [e.info() for e in engines().values() if e.kind == "local" or e.id in ("sarvam", "elevenlabs", "edge")]
    return {**setup_models.status(), "llm": {"available": llm.available()[0], "model": llm.model_name()},
            "stt": [s.info() for s in stts().values()], "tts": engines_,
            "examples": json.loads((ROOT / "proto" / "examples.json").read_text(encoding="utf-8")),
            "schema": json.loads((ROOT / "proto" / "schema.json").read_text(encoding="utf-8"))["fields"]}


class StartReq(BaseModel):
    lang: str = "hi"
    stt: str = "shruti"
    tts: str = "piper"
    voice: str | None = None
    limit_mb: int = 2048
    threads: int = 4
    cpu_percent: float | None = None


def _box_for(r: StartReq) -> tuple[VoiceBox, dict]:
    global _box
    want = dict(stt=r.stt, tts=r.tts, lang=r.lang, limit_mb=r.limit_mb, threads=r.threads,
                cpu_percent=r.cpu_percent, voice=r.voice)
    with _box_lock:
        if _box and _box.cfg == want and _box.alive():
            return _box, _box.stats() | {"reused": True}
        if _box:
            _box.close()
            _box = None
        box = VoiceBox(**want)
        t = time.perf_counter()
        stats = box.start()                   # may raise CapExceeded
        _box = box
        return box, stats | {"start_s": round(time.perf_counter() - t, 2), "reused": False}


@router.post("/api/convo/start")
def start(r: StartReq):
    try:
        box, stats = _box_for(r)
    except CapExceeded as ex:
        return {"fits": False, "error": str(ex)}
    except Exception as ex:
        raise HTTPException(500, f"{type(ex).__name__}: {str(ex)[:500]}")
    sid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    agent = ProfileAgent(r.lang)
    say = agent.start()
    out = CONVOS / sid / "k00.flac"
    sp = box.speak(say, str(out))
    _sessions[sid] = {"agent": agent, "cfg": r.model_dump(), "turns": [], "box": box}
    first = {"role": "kaabil", "text": say, "audio": f"/audio/convos/{sid}/k00.flac" if sp.get("ok") else None,
             "tts": sp, "fixed_prompt": True}
    _sessions[sid]["turns"].append(first)
    _save(sid)
    return {"fits": True, "sid": sid, "box": stats, "turn": first, "state": agent.state(),
            "llm": {"available": llm.available()[0], "model": llm.model_name()}}


class TurnReq(BaseModel):
    sid: str
    text: str | None = None
    audio_b64: str | None = None      # 16 kHz mono WAV from the browser mic
    example: str | None = None        # "hi/h3" or "hi/h3_phone": pre-voiced example clip
    echo: bool = False                # skip the LLM: Kaabil repeats the transcript (voice-only test)


@router.post("/api/convo/turn")
def turn(r: TurnReq):
    s = _sessions.get(r.sid)
    if not s:
        raise HTTPException(404, "session expired; press Start again")
    box: VoiceBox = s["box"]
    if not box.alive():
        raise HTTPException(409, "the voice workers stopped (memory cap?); press Start again")
    agent: ProfileAgent = s["agent"]
    n = len(s["turns"])
    t_all = time.perf_counter()
    timing, student = {}, {"role": "student"}

    # 1. Ear
    wav = None
    if r.audio_b64:
        wav = CONVOS / r.sid / f"s{n:02d}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(base64.b64decode(r.audio_b64))
        student["audio"] = f"/audio/convos/{r.sid}/{wav.name}"
    elif r.example:
        wav = OUT / "examples" / f"{r.example}.flac"
        if not wav.exists():
            raise HTTPException(404, "example clip missing; run proto/make_example_audio.py")
        student["audio"] = f"/audio/examples/{r.example}.flac"
    if wav is not None:
        h = box.transcribe(str(wav))
        if not h.get("ok"):
            raise HTTPException(500, f"speech-to-text failed: {h.get('error')}")
        text = h["text"]
        timing["stt_s"] = h["secs"]
        student["audio_s"] = h.get("audio_s")
        if r.example:
            lang, eid = r.example.split("/")
            ex = json.loads((ROOT / "proto" / "examples.json").read_text(encoding="utf-8"))["examples"][lang]
            student["reference"] = next((e["text"] for e in ex if e["id"] == eid.replace("_phone", "")), "")
    else:
        text = (r.text or "").strip()
        timing["stt_s"] = 0.0
    if not text:
        raise HTTPException(400, "nothing heard; speak a little longer or type")
    student["text"] = text
    student["turn"] = n

    # 2. Brain
    if r.echo:
        from numwords import speakable
        say, result = text, {"llm": None, "state": agent.state()}
        spoken = speakable(text, agent.lang)
    else:
        try:
            result = agent.turn(text)
        except llm.LLMError as ex:
            raise HTTPException(502, f"LLM: {ex}")
        say = result["say"] or "…"
        spoken = result.get("say_spoken") or say
        timing["llm_ttft_s"] = result["llm"]["ttft_s"]
        timing["llm_s"] = result["llm"]["total_s"]

    # 3. Voice
    out = CONVOS / r.sid / f"k{n + 1:02d}.flac"
    sp = box.speak(spoken, str(out))
    if sp.get("ok"):
        timing["tts_first_s"] = sp["first_audio_s"]
        timing["tts_s"] = sp["secs"]
    timing["total_s"] = round(time.perf_counter() - t_all, 2)
    # what the student waits before hearing Kaabil, with sentence streaming of the reply
    timing["wait_s"] = round(timing["stt_s"] + timing.get("llm_s", 0) + timing.get("tts_first_s", 0), 2)
    kaabil = {"role": "kaabil", "text": say, "spoken": spoken if spoken != say else None, "audio": f"/audio/convos/{r.sid}/{out.name}" if sp.get("ok") else None,
              "tts": sp, "timing": timing, "llm": result.get("llm"), "asks_about": result.get("asks_about", []),
              "changes": result.get("changes", []), "phase": result.get("state", {}).get("phase")}
    s["turns"] += [student, kaabil]
    _save(r.sid)
    return {"student": student, "kaabil": kaabil, "state": result["state"], "box": box.stats()}


def _save(sid: str) -> None:
    s = _sessions[sid]
    p = CONVOS / sid / "session.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"cfg": s["cfg"], "turns": s["turns"], "state": s["agent"].state(),
                             "card": s["agent"].card()}, ensure_ascii=False, indent=1), encoding="utf-8")


@router.post("/api/convo/stop")
def stop():
    global _box
    with _box_lock:
        if _box:
            _box.close()
            _box = None
    return {"ok": True}


@router.get("/api/brief")
def brief():
    p = ROOT / "JUDGES_BRIEF.md"
    return {"markdown": p.read_text(encoding="utf-8") if p.exists() else
            "# No brief yet\n\nRun `python bench/make_brief.py` after the benchmarks."}
