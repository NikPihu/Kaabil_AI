"""Career recommendation endpoints.

GET  /api/career/profiles   saved conversations with a usable profile + the demo students
POST /api/career/suggest    {source: "session"|"persona", id, exclude?, feedback?, refresh?}
POST /api/career/feedback   {key, career_id, kind: "interested"|"not_fit", reason?}
GET  /api/career/review     staged government-PDF extractions awaiting human review
POST /api/career/approve    {id, reviewer}  publish a reviewed extraction into careers.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "proto"), str(ROOT / "rig")]

import career_agent  # noqa: E402
import ingest  # noqa: E402
import llm  # noqa: E402

router = APIRouter()
CONVOS = ROOT / "outputs" / "convos"
CACHE = ROOT / "outputs" / "careers"
PERSONAS = ROOT / "bench" / "personas.json"
MIN_FIELDS = 6


def _load_session(sid: str) -> dict:
    p = CONVOS / sid / "session.json"
    if not p.exists():
        raise HTTPException(404, "conversation not found")
    s = json.loads(p.read_text(encoding="utf-8"))
    fields = s["state"]["fields"]
    prof = {f["id"]: f["value"] for f in fields if f.get("value") and f.get("status") != "skipped"}
    ev = {f["id"]: f["evidence"] for f in fields if f.get("evidence") and f.get("value")}
    return {"profile": prof, "evidence": ev, "lang": s["cfg"]["lang"], "name": prof.get("name", ""),
            "phase": s["state"].get("phase"), "summary": s["state"].get("summary", "")}


def _load_persona(pid: str) -> dict:
    ps = {p["id"]: p for p in json.loads(PERSONAS.read_text(encoding="utf-8"))["personas"]}
    if pid not in ps:
        raise HTTPException(404, "demo student not found")
    p = ps[pid]
    return {"profile": dict(p["truth"]), "evidence": {}, "lang": p["lang"], "name": p["truth"].get("name", ""),
            "phase": "demo", "summary": p["who"]}


@router.get("/api/career/profiles")
def profiles():
    out = []
    if CONVOS.exists():
        for d in sorted(CONVOS.iterdir(), reverse=True):
            f = d / "session.json"
            if not f.exists():
                continue
            try:
                s = _load_session(d.name)
            except Exception:
                continue
            if len(s["profile"]) >= MIN_FIELDS:
                out.append({"source": "session", "id": d.name, "name": s["name"] or "Unnamed student",
                            "lang": s["lang"], "phase": s["phase"], "fields": len(s["profile"])})
    for p in json.loads(PERSONAS.read_text(encoding="utf-8"))["personas"]:
        out.append({"source": "persona", "id": p["id"], "name": p["truth"]["name"], "lang": p["lang"],
                    "phase": "demo", "fields": len(p["truth"])})
    return {"profiles": out, "llm": {"available": llm.available()[0], "model": llm.model_name()}}


class SuggestReq(BaseModel):
    source: str
    id: str
    exclude: list[str] = []
    feedback: str = ""
    refresh: bool = False


def _key(r: SuggestReq) -> str:
    return f"{r.source}-{r.id}"


@router.post("/api/career/suggest")
def suggest(r: SuggestReq):
    who = _load_session(r.id) if r.source == "session" else _load_persona(r.id)
    CACHE.mkdir(parents=True, exist_ok=True)
    key = _key(r)
    cache = CACHE / f"{key}.json"
    sig = hashlib.sha1(json.dumps([who["profile"], sorted(r.exclude), r.feedback,
                                   career_agent.kb().get("version")], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    store = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {"feedback": []}
    if not r.refresh and store.get("sig") == sig:
        return store["result"] | {"key": key, "cached": True, "feedback": store["feedback"], "student": _public(who)}
    try:
        res = career_agent.suggest(who["profile"], who["evidence"], who["lang"], r.exclude, r.feedback, who["name"])
    except llm.LLMError as ex:
        raise HTTPException(502, f"LLM: {ex}")
    store.update(sig=sig, result=res, at=time.strftime("%Y-%m-%d %H:%M"))
    cache.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    return res | {"key": key, "cached": False, "feedback": store["feedback"], "student": _public(who)}


def _public(who: dict) -> dict:
    return {"name": who["name"], "lang": who["lang"], "phase": who["phase"], "profile": who["profile"],
            "evidence": who["evidence"]}


class FeedbackReq(BaseModel):
    key: str
    career_id: str
    kind: str
    reason: str = ""


@router.post("/api/career/feedback")
def feedback(r: FeedbackReq):
    if r.kind not in ("interested", "not_fit"):
        raise HTTPException(400, "kind must be interested or not_fit")
    cache = CACHE / f"{r.key}.json"
    store = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {"feedback": []}
    store["feedback"] = [f for f in store["feedback"] if f["career_id"] != r.career_id] + \
                        [{"career_id": r.career_id, "kind": r.kind, "reason": r.reason, "at": time.strftime("%Y-%m-%d %H:%M")}]
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"ok": True, "feedback": store["feedback"]}


@router.get("/api/career/review")
def review():
    items = []
    for p in sorted(ingest.STAGED.glob("*.json")) if ingest.STAGED.exists() else []:
        items.append(json.loads(p.read_text(encoding="utf-8")))
    titles = {c["id"]: c["title"]["en"] for c in career_agent.kb()["careers"]}
    return {"items": [i | {"title": titles.get(i["id"], i["id"])} for i in items], "kb_version": career_agent.kb().get("version")}


class ApproveReq(BaseModel):
    id: str
    reviewer: str = "reviewer in the Kaabil app"


@router.post("/api/career/approve")
def approve(r: ApproveReq):
    try:
        return ingest.approve(r.id, r.reviewer)
    except FileNotFoundError:
        raise HTTPException(404, "nothing staged for that career")
