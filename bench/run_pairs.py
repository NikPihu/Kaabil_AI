"""Benchmark 2 - the cap matrix.  Every Ear x Voice pair per language, loaded
together under one memory cap, speaking real follow-up questions.

    python bench/run_pairs.py                     # 2048 and 1024 MB, all languages
    python bench/run_pairs.py --langs hi --caps 2048
    python bench/run_pairs.py --judge-only        # just (re)score the saved audio

Per pair: fits?, resident peak (what Android counts), committed peak, load
times, Voice time to first sound and real-time factor.  Then the Whisper
judge (outside the cap) transcribes every spoken question; error = how much
of the question a listener-model could not recover.
"""
from __future__ import annotations

import argparse
import json
import statistics as stx
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "proto"), str(ROOT / "rig")]

from core import OUT  # noqa: E402
from engines import engines  # noqa: E402
from stt import stts  # noqa: E402
from voicebox import CapExceeded, VoiceBox  # noqa: E402

RES = ROOT / "bench" / "results"
AUDIO = OUT / "bench_pairs"
VOICES = ["piper", "kokoro", "kokoro-int8", "supertonic-int8", "mms", "kitten", "indic-tts", "indic-tts-full", "parler"]
SPEAKER = {"indic-tts": "female", "indic-tts-full": "female"}


def questions(lang: str, n: int) -> list[str]:
    q = json.loads((ROOT / "bench" / "questions.json").read_text(encoding="utf-8"))[lang]
    return q[:n]


def run(langs, caps, nq) -> dict:
    out_file = RES / "pairs.json"
    res = json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() else {}
    E = engines()
    for lang in langs:
        ears = [s for s in stts().values() if lang in s.langs and s.available()[0]]
        voices = [E[v] for v in VOICES if v in E and lang in E[v].voices and E[v].available()[0]]
        for cap in caps:
            for v in voices:
                for ear in ears:
                    key = f"{lang}/{ear.id}/{v.id}/{cap}"
                    if key in res and res[key].get("done"):
                        continue
                    # at the smaller cap, skip pairs already known not to fit at the larger one
                    big = res.get(f"{lang}/{ear.id}/{v.id}/{max(caps)}")
                    if cap < max(caps) and big and not big.get("fits"):
                        res[key] = {"lang": lang, "stt": ear.id, "tts": v.id, "limit_mb": cap, "fits": False,
                                    "error": f"did not fit in {max(caps)} MB", "done": True}
                        continue
                    box = VoiceBox(ear.id, v.id, lang, cap, voice=SPEAKER.get(v.id))
                    row = {"lang": lang, "stt": ear.id, "tts": v.id, "limit_mb": cap, "items": []}
                    try:
                        st = box.start()
                        row["load"] = st["load"]
                        # full question set once per voice (with the first Ear); 2 questions otherwise
                        qs = questions(lang, nq if ear is ears[0] and cap == max(caps) else 2)
                        for i, q in enumerate(qs):
                            p = AUDIO / lang / v.id / f"q{i}.flac"
                            r = box.speak(q, str(p))
                            if not r.get("ok"):
                                raise CapExceeded(r.get("error")) if r.get("oom") else RuntimeError(r.get("error"))
                            row["items"].append({"q": q, "file": str(p.relative_to(OUT)), "first_s": r["first_audio_s"],
                                                 "secs": r["secs"], "duration": r["duration"]})
                        # after speaking, confirm the Ear still works with the Voice loaded
                        h = box.transcribe(str(OUT / "examples" / lang / "n1.flac"))
                        row["ear_ok"] = bool(h.get("ok"))
                        row["fits"] = True
                    except CapExceeded as ex:
                        row["fits"], row["error"] = False, str(ex)[:300]
                    except Exception as ex:
                        row["fits"], row["error"] = None, f"{type(ex).__name__}: {str(ex)[:300]}"
                    finally:
                        x = box.stats(); box.close()
                    row.update(peak_mb=x["peak_mb"], peak_commit_mb=x["peak_commit_mb"], done=True)
                    it = row["items"]
                    if it:
                        row["first_s_median"] = round(stx.median(i["first_s"] for i in it), 2)
                        row["x_realtime"] = round(sum(i["duration"] for i in it) / sum(i["secs"] for i in it), 1)
                    res[key] = row
                    print(f"{key:44s} fits={row['fits']} resident {row['peak_mb']} MB (commit {row['peak_commit_mb']})"
                          f" first sound {row.get('first_s_median')}s  {row.get('x_realtime')}x  {row.get('error', '')[:90]}",
                          flush=True)
                    RES.mkdir(parents=True, exist_ok=True)
                    out_file.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def judge() -> None:
    """Whisper large-v3-turbo transcribes each spoken question (outside the cap)."""
    from evaluate import cer, transcribe
    out_file = RES / "pairs.json"
    res = json.loads(out_file.read_text(encoding="utf-8"))
    done = {}
    for key, row in res.items():
        for it in row.get("items", []):
            if it["file"] in done:
                it["asr"], it["cer"] = done[it["file"]]
                continue
            if "cer" not in it:
                hyp = transcribe(str(OUT / it["file"]), row["lang"])
                it["asr"], it["cer"] = hyp, cer(it["q"], hyp)
            done[it["file"]] = (it["asr"], it["cer"])
        v = [i["cer"] for i in row.get("items", []) if i.get("cer") is not None]
        row["judge_cer"] = round(sum(v) / len(v), 3) if v else None
        out_file.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    # per-voice summary (full question set only)
    print("\nvoice quality (Whisper judge error on the spoken questions, lower is better)")
    seen = set()
    for key, row in res.items():
        k = (row["lang"], row["tts"])
        if k in seen or len(row.get("items", [])) < 3:
            continue
        seen.add(k)
        print(f"  {row['lang']} {row['tts']:16s} {row['judge_cer']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="hi,mr,en")
    ap.add_argument("--caps", default="2048,1024")
    ap.add_argument("--questions", type=int, default=8)
    ap.add_argument("--judge-only", action="store_true")
    a = ap.parse_args()
    if not a.judge_only:
        run(a.langs.split(","), sorted(map(int, a.caps.split(",")), reverse=True), a.questions)
    judge()


if __name__ == "__main__":
    main()
