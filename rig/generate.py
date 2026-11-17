"""Batch-generate the demo matrix: every scenario x language x engine.

    python rig/generate.py                 # everything that is missing
    python rig/generate.py --force         # regenerate all
    python rig/generate.py --engines piper,kokoro --langs hi --scenarios hurry

Each engine runs in its own process so load time and memory are measured
cleanly.  Results land in outputs/<scenario>/<lang>/<engine>.flac and
outputs/manifest.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core import OUT, analyse, load_scenarios, render_dialogue, render_single, save, scenario_text  # noqa: E402
from engines import engines, get_engine  # noqa: E402

MANIFEST = OUT / "manifest.json"
WARMUP = {"en": "This is a short warm up sentence.", "hi": "यह एक छोटा वाक्य है।", "mr": "हे एक छोटं वाक्य आहे."}


def read_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"clips": {}, "engines": {}}


def write_manifest(m: dict) -> None:
    OUT.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")


def run_engine(eid: str, langs: list[str], scen_ids: list[str] | None, force: bool) -> dict:
    """Runs inside the per-engine subprocess; returns {clips, engine_stats}."""
    import psutil
    eng = get_engine(eid)
    proc = psutil.Process()
    existing = read_manifest()["clips"]
    clips, stats = {}, {"cold_start_s": {}, "rtf": []}
    for lang in langs:
        if lang not in eng.voices:
            continue
        t = time.perf_counter()
        eng.synth(WARMUP[lang], lang)                     # includes model load on first call
        stats["cold_start_s"][lang] = round(time.perf_counter() - t, 2)
        if "ram_one_voice_mb" not in stats:               # one language, one voice loaded
            stats["ram_one_voice_mb"] = round(proc.memory_info().rss / 1e6)
        for v in eng.voices[lang][1:3]:                   # preload the dialogue voices so
            eng.synth(WARMUP[lang], lang, v["id"])        # clip timings exclude model loading
        for sc in load_scenarios():
            if scen_ids and sc["id"] not in scen_ids:
                continue
            if lang not in sc.get("turns", sc.get("text", {})):
                continue
            key = f'{sc["id"]}/{lang}/{eid}'
            path = OUT / sc["id"] / lang / f"{eid}.flac"
            if key in existing and path.exists() and not force:
                continue
            speed = float(sc.get("speed", 1.0))
            try:
                if "turns" in sc:
                    audio, sr, gen, extra = render_dialogue(eid, sc["turns"][lang], lang, speed)
                    voice = ", ".join(sorted(set(extra["voice_map"].values())))
                else:
                    voice = eng.voices[lang][0]["id"]
                    audio, sr, gen = render_single(eid, sc["text"][lang], lang, voice, speed)
                    extra = {}
            except Exception as ex:  # keep going: one failure must not sink the batch
                print(f"  FAIL {key}: {type(ex).__name__}: {ex}", flush=True)
                clips[key] = {"scenario": sc["id"], "lang": lang, "engine": eid,
                              "error": f"{type(ex).__name__}: {str(ex)[:200]}"}
                continue
            save(audio, sr, path)
            m = analyse(audio, sr)
            text = scenario_text(sc, lang)
            m.update(extra)
            m.update({
                "scenario": sc["id"], "lang": lang, "engine": eid, "voice": voice, "speed": speed,
                "file": f'{sc["id"]}/{lang}/{eid}.flac', "sample_rate": sr,
                "gen_time": round(gen, 2), "rtf": round(gen / m["duration"], 3),
                "chars_per_s": round(len(text.replace(" ", "")) / max(m.get("speech_time", m["duration"]), 0.1), 1),
                "experimental": lang in eng.experimental,
            })
            clips[key] = m
            stats["rtf"].append(m["rtf"])
            print(f'  {key:42s} {m["duration"]:6.1f}s  gen {gen:5.1f}s  rtf {m["rtf"]:.2f}', flush=True)
    mem = proc.memory_info()
    stats["peak_ram_mb"] = round(getattr(mem, "peak_wset", mem.rss) / 1e6)
    stats["rtf_median"] = round(sorted(stats["rtf"])[len(stats["rtf"]) // 2], 3) if stats["rtf"] else None
    del stats["rtf"]
    return {"clips": clips, "stats": stats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="")
    ap.add_argument("--langs", default="en,hi,mr")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--child", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    langs = a.langs.split(",")
    scen = a.scenarios.split(",") if a.scenarios else None

    if a.child:
        res = run_engine(a.child, langs, scen, a.force)
        print("@@RESULT@@" + json.dumps(res, ensure_ascii=False), flush=True)
        return

    todo = a.engines.split(",") if a.engines else [
        eid for eid, e in engines().items() if e.in_matrix and e.available()[0]]
    for eid in todo:
        ok, why = get_engine(eid).available()
        if not ok:
            print(f"[{eid}] skipped: {why}")
            continue
        print(f"[{eid}]", flush=True)
        cmd = [sys.executable, "-X", "utf8", __file__, "--child", eid, "--langs", a.langs]
        if a.scenarios:
            cmd += ["--scenarios", a.scenarios]
        if a.force:
            cmd.append("--force")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, encoding="utf-8")
        result = None
        for line in p.stdout:
            if line.startswith("@@RESULT@@"):
                result = json.loads(line[len("@@RESULT@@"):])
            else:
                print(line, end="", flush=True)
        p.wait()
        if result is None:
            print(f"[{eid}] crashed (exit {p.returncode})")
            continue
        m = read_manifest()
        m["clips"].update(result["clips"])
        full_run = not a.scenarios and set(langs) >= {"en", "hi", "mr"}
        if result["stats"]["rtf_median"] is not None and (full_run or eid not in m["engines"]):
            old = m["engines"].get(eid, {})
            old.update(result["stats"])
            m["engines"][eid] = old
        write_manifest(m)
    print("done ->", MANIFEST)


if __name__ == "__main__":
    main()
