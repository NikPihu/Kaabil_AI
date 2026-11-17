"""Benchmark 1 - the Ear.  Every speech-to-text model, inside the memory cap,
on the spoken example answers (clean and noisy 8 kHz phone copies) and on any
real recordings in bench/recordings/<lang>/ (name.wav + name.txt with what was said).

    python bench/run_ear.py                 # all languages, 2048 MB
    python bench/run_ear.py --langs mr --limit 1024

Writes bench/results/ear.json.  Error = character error rate after
normalisation, numbers ignored (see rig/evaluate.py).  Lower is better.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "proto"), str(ROOT / "rig")]

from core import OUT  # noqa: E402
from evaluate import cer, norm  # noqa: E402
from stt import stts  # noqa: E402
from voicebox import CapExceeded, VoiceBox  # noqa: E402

RES = ROOT / "bench" / "results"


def clips(lang: str) -> list[dict]:
    ex = json.loads((ROOT / "proto" / "examples.json").read_text(encoding="utf-8"))["examples"][lang]
    out = []
    for e in ex:
        for cond, suffix in (("clean", ""), ("phone", "_phone")):
            p = OUT / "examples" / lang / f"{e['id']}{suffix}.flac"
            if p.exists():
                out.append({"id": e["id"], "kind": e["kind"], "cond": cond, "path": str(p), "ref": e["text"]})
    rec = ROOT / "bench" / "recordings" / lang
    for w in sorted(rec.glob("*.wav")) if rec.exists() else []:
        t = w.with_suffix(".txt")
        if t.exists():
            out.append({"id": w.stem, "kind": "real", "cond": "real", "path": str(w),
                        "ref": t.read_text(encoding="utf-8").strip()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="hi,mr,en")
    ap.add_argument("--limit", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    out_file = RES / "ear.json"
    results = json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() else {}
    for lang in a.langs.split(","):
        cs = clips(lang)
        for sid, s in stts().items():
            if lang not in s.langs or not s.available()[0]:
                continue
            key = f"{lang}/{sid}/{a.limit}"
            box = VoiceBox(sid, None, lang, a.limit, a.threads)
            row = {"lang": lang, "stt": sid, "limit_mb": a.limit, "threads": a.threads, "items": []}
            try:
                st = box.start()
                row["load_s"] = st["load"]["stt"]["secs"]
                for c in cs:
                    r = box.transcribe(c["path"])
                    row["items"].append({**{k: c[k] for k in ("id", "kind", "cond", "ref")},
                                         "hyp": r.get("text", ""), "secs": r.get("secs"), "audio_s": r.get("audio_s"),
                                         # digits clip: Whisper-style spelling differences make CER meaningless
                                         "cer": None if c["id"] == "h10" else (cer(c["ref"], r.get("text", "")) if r.get("ok") else 1.0),
                                         # script-blind check for skipped speech: share of the reference's
                                         # length that came back (Latin words are ~same length as Devanagari)
                                         "coverage": round(len(norm(r.get("text", "")).replace(" ", "")) /
                                                           max(1, len(norm(c["ref"]).replace(" ", ""))), 2),
                                         "error": r.get("error")})
                row["fits"] = True
            except CapExceeded as ex:
                row["fits"], row["error"] = False, str(ex)
            finally:
                x = box.stats(); box.close()
            row["peak_mb"], row["peak_commit_mb"] = x["peak_mb"], x["peak_commit_mb"]
            it = row["items"]
            for cond in ("clean", "phone", "real"):
                v = [i["cer"] for i in it if i["cond"] == cond and i["cer"] is not None]
                row[f"dropped_{cond}"] = sum(1 for i in it if i["cond"] == cond and i["coverage"] < 0.75)
                row[f"cer_{cond}"] = round(sum(v) / len(v), 3) if v else None
            secs = sum(i["secs"] or 0 for i in it); aud = sum(i["audio_s"] or 0 for i in it)
            row["x_realtime"] = round(aud / secs, 1) if secs else None
            results[key] = row
            print(f"{key:28s} fits={row['fits']} peak {row['peak_mb']} MB  clean {row['cer_clean']}  phone {row['cer_phone']}"
                  f"  real {row['cer_real']}  dropped {row['dropped_clean']}+{row['dropped_phone']}  {row['x_realtime']}x real time", flush=True)
            out_file.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
