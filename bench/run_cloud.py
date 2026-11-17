"""Benchmark 2b - cloud voices for the 'fixed prompts / server' role.
Speaks the same follow-up questions as run_pairs.py with Sarvam Bulbul,
ElevenLabs and the Microsoft neural voice, then the same Whisper judge scores them.

    python bench/run_cloud.py
"""
from __future__ import annotations

import json
import statistics as stx
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "proto"), str(ROOT / "rig")]

from core import OUT, normalise, save  # noqa: E402
from engines import get_engine  # noqa: E402
from numwords import speakable  # noqa: E402

RES = ROOT / "bench" / "results"
CLOUD = ["sarvam", "elevenlabs", "edge"]


def main() -> None:
    from evaluate import cer, transcribe
    qs = json.loads((ROOT / "bench" / "questions.json").read_text(encoding="utf-8"))
    out_file = RES / "cloud.json"
    res = json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() else {}
    for eid in CLOUD:
        e = get_engine(eid)
        if not e.available()[0]:
            print(eid, "skipped:", e.available()[1]); continue
        for lang in ("hi", "mr", "en"):
            key = f"{lang}/{eid}"
            if key in res and res[key].get("judge_cer") is not None:
                continue
            row = {"lang": lang, "tts": eid, "items": []}
            for i, q in enumerate(qs[lang]):
                p = OUT / "bench_cloud" / lang / eid / f"q{i}.flac"
                try:
                    t = time.perf_counter()
                    a, sr = e.synth(speakable(q, lang), lang, None, 1.0)
                    secs = time.perf_counter() - t
                    save(normalise(a, sr), sr, p)
                    hyp = transcribe(str(p), lang)
                    row["items"].append({"q": q, "file": str(p.relative_to(OUT)), "secs": round(secs, 2),
                                         "duration": round(len(a) / sr, 2), "asr": hyp, "cer": cer(q, hyp)})
                except Exception as ex:
                    row["items"].append({"q": q, "error": f"{type(ex).__name__}: {str(ex)[:200]}"})
            ok = [i for i in row["items"] if "cer" in i]
            row["judge_cer"] = round(stx.mean(i["cer"] for i in ok), 3) if ok else None
            row["secs_median"] = round(stx.median(i["secs"] for i in ok), 2) if ok else None
            row["errors"] = len(row["items"]) - len(ok)
            res[key] = row
            RES.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{key:16s} judge error {row['judge_cer']}  {row['secs_median']}s per sentence  errors {row['errors']}", flush=True)


if __name__ == "__main__":
    main()
