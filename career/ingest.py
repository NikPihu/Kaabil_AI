"""Curated ingestion of government curriculum PDFs into the career knowledge base
(pattern B of the team's Data Sources guide: fetch -> diff -> extract -> human review -> publish).

    python career/ingest.py                 # fetch + extract every career with an "ingest" URL
    python career/ingest.py --approve iti_electrician   # publish a reviewed record into careers.json

1. Fetch   the PDF; keep the raw copy and its SHA-256 under VOICE_RIG_HOME/career_raw/
2. Diff    unchanged hash -> skip
3. Extract text with pypdf; the LLM fills a small schema ONLY from that text and must
           quote the line each item came from; items whose quote is not in the text are dropped
4. Stage   career/staged/<id>.json with status "extracted, awaiting human review"
5. Publish --approve (or the Review panel in the Career tab) copies the modules into
           careers.json with learn_source="official" and bumps the version
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "proto"), str(ROOT / "rig")]

import llm  # noqa: E402
from engines import RIG_HOME  # noqa: E402

KB = ROOT / "career" / "careers.json"
STAGED = ROOT / "career" / "staged"
RAW = RIG_HOME / "career_raw"

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["modules", "progression", "entry", "duration"],
    "properties": {
        "modules": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                 "required": ["name", "quote"],
                                                 "properties": {"name": {"type": "string"}, "quote": {"type": "string"}}}},
        "progression": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                     "required": ["role", "quote"],
                                                     "properties": {"role": {"type": "string"}, "quote": {"type": "string"}}}},
        "entry": {"type": "object", "additionalProperties": False, "required": ["text", "quote"],
                  "properties": {"text": {"type": "string"}, "quote": {"type": "string"}}},
        "duration": {"type": "object", "additionalProperties": False, "required": ["text", "quote"],
                     "properties": {"text": {"type": "string"}, "quote": {"type": "string"}}},
    },
}

PROMPT = """You read an official Indian government curriculum / qualification document and pull out facts for a career guide for rural students.
Use ONLY the document text below. For every item give `quote`: an exact short phrase (5-15 words) copied from the text that supports it.
- modules: 6-10 main topics a student learns, in plain simple English (e.g. "House wiring and earthing").
- progression: job roles this leads to, as named in the document (employment avenues, progression, next level).
- entry: the entry qualification and minimum age if stated.
- duration: course duration / notional hours if stated.
If something is not in the text, return an empty list or text "" with quote "".

DOCUMENT ({name}):
{text}"""


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Kaabil career ingestion)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def pdf_text(blob: bytes, max_chars: int = 14000) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(blob))
    pages = [(p.extract_text() or "") for p in reader.pages]
    # the course-information pages come first; add any page that mentions progression/employment
    keep = pages[:8] + [p for p in pages[8:] if re.search(r"employment|progression|career|job role|avenue", p, re.I)]
    text = re.sub(r"[ \t]+", " ", "\n".join(keep))
    return text[:max_chars]


def _supported(quote: str, text: str) -> bool:
    q = re.sub(r"\s+", " ", quote).strip().lower()
    t = re.sub(r"\s+", " ", text).lower()
    return bool(q) and (q in t or (len(q) > 20 and q[:20] in t))


def extract(career: dict, url: str) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    blob = fetch(url)
    digest = hashlib.sha256(blob).hexdigest()
    raw_path = RAW / f"{career['id']}_{digest[:12]}.pdf"
    staged_path = STAGED / f"{career['id']}.json"
    if staged_path.exists() and json.loads(staged_path.read_text(encoding="utf-8")).get("sha256") == digest:
        return {"id": career["id"], "skipped": "unchanged since last fetch"}
    raw_path.write_bytes(blob)
    text = pdf_text(blob)
    r = llm.chat([{"role": "user", "content": PROMPT.format(name=career["title"]["en"], text=text)}],
                 schema=SCHEMA, temperature=0, reasoning="off", max_tokens=1500)
    j = r["json"]
    # keep only items whose quote really appears in the document
    mods = [m for m in j.get("modules", []) if _supported(m.get("quote", ""), text)]
    prog = [p for p in j.get("progression", []) if _supported(p.get("quote", ""), text)]
    record = {
        "id": career["id"], "source_url": url, "fetched_on": time.strftime("%Y-%m-%d"), "sha256": digest,
        "status": "extracted, awaiting human review", "text_chars": len(text),
        "modules": mods, "progression": prog,
        "entry": j.get("entry") if _supported((j.get("entry") or {}).get("quote", ""), text) else None,
        "duration": j.get("duration") if _supported((j.get("duration") or {}).get("quote", ""), text) else None,
        "dropped_unsupported": (len(j.get("modules", [])) - len(mods)) + (len(j.get("progression", [])) - len(prog)),
    }
    STAGED.mkdir(parents=True, exist_ok=True)
    staged_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def approve(cid: str, reviewer: str = "human reviewer") -> dict:
    rec = json.loads((STAGED / f"{cid}.json").read_text(encoding="utf-8"))
    kb = json.loads(KB.read_text(encoding="utf-8"))
    c = next(x for x in kb["careers"] if x["id"] == cid)
    if rec["modules"]:
        c["learn"] = [m["name"] for m in rec["modules"]]
        c["learn_source"] = "official"
    c.setdefault("sources", []).append({"name": "Curriculum extracted and reviewed", "url": rec["source_url"],
                                        "checked": f"official PDF, fetched {rec['fetched_on']}, reviewed by {reviewer}"})
    kb["version"] = kb.get("version", 1) + 1
    kb["updated"] = time.strftime("%Y-%m-%d")
    KB.write_text(json.dumps(kb, ensure_ascii=False, indent=2), encoding="utf-8")
    rec["status"] = f"published in careers.json v{kb['version']}"
    (STAGED / f"{cid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approve", default="")
    a = ap.parse_args()
    if a.approve:
        print(json.dumps(approve(a.approve), ensure_ascii=False, indent=1)); return
    kb = json.loads(KB.read_text(encoding="utf-8"))
    for c in kb["careers"]:
        for url in c.get("ingest", []):
            try:
                rec = extract(c, url)
                print(c["id"], "->", rec.get("skipped") or f"{len(rec['modules'])} modules, {len(rec['progression'])} roles, "
                      f"{rec['dropped_unsupported']} unsupported items dropped")
            except Exception as ex:
                print(c["id"], "FAILED:", type(ex).__name__, str(ex)[:200])


if __name__ == "__main__":
    main()
