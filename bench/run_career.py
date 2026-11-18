"""Benchmark 4 - the Career Agent on the six demo students.

    python bench/run_career.py

Checks done by code (no LLM judge):
  grounding   no money or percentage in the agent's own text (facts must come from careers.json)
  rules       every card's route passes the entry rules for that student
  balance     at least 2 non-STEM paths and at least 2 sectors
  quotes      share of "why" points that quote the student, and whether each quote is really
              in the profile (for demo students the quote must be a profile value)
  speed/cost  seconds per suggestion
Writes bench/results/career.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "career"), str(ROOT / "proto"), str(ROOT / "rig")]

import career_agent  # noqa: E402
import rules  # noqa: E402

MONEY = re.compile(r"₹|\brs\.?\s?\d|\b\d[\d,.]*\s?(%|lakh|thousand|rupees)", re.I)


def main() -> None:
    ps = json.loads((ROOT / "bench" / "personas.json").read_text(encoding="utf-8"))["personas"]
    kb = {c["id"]: c for c in career_agent.kb()["careers"]}
    out = {}
    for p in ps:
        prof = p["truth"]
        try:
            r = career_agent.suggest(prof, {}, p["lang"], name=prof["name"])
        except Exception as ex:
            out[p["id"]] = {"student": p["id"], "error": f"{type(ex).__name__}: {str(ex)[:200]}"}
            print(f"{p['id']:10s} FAILED: {out[p['id']]['error']}", flush=True)
            continue
        sit = rules.situation(prof)
        texts = [w["point"] for c in r["cards"] for w in c.get("why", [])] + \
                [c.get("watch_out", "") for c in r["cards"]] + [s for c in r["cards"] for s in c.get("first_steps", [])] + \
                [r["summary_local"]]
        money_hits = [t for t in texts if MONEY.search(t)]
        route_ok = all(rules.check(next(x for x in kb[c["id"]]["routes"] if x["id"] == c["route"]["id"]), sit)["status"] != "blocked"
                       for c in r["cards"])
        whys = [w for c in r["cards"] for w in c.get("why", [])]
        profile_text = " ".join(prof.values()).lower()
        quoted = [w for w in whys if w.get("quote")]
        grounded_quotes = [w for w in quoted if w["quote"].lower()[:25] in profile_text]
        row = {
            "student": p["id"], "lang": p["lang"], "cards": [c["id"] for c in r["cards"]],
            "non_stem": sum(1 for c in r["cards"] if not c["stem"]), "sectors": len({c["sector"] for c in r["cards"]}),
            "government": sum(1 for c in r["cards"] if "government" in c["work_type"]),
            "own_business": sum(1 for c in r["cards"] if "self-employment" in c["work_type"]),
            "money_in_llm_text": len(money_hits), "rules_ok": route_ok,
            "why_points": len(whys), "why_quoted": len(quoted), "quotes_found_in_profile": len(grounded_quotes),
            "ruled_out": len(r["blocked"]), "seconds": r["seconds"], "cost": r["llm"]["cost"],
            "profile_read": r["profile_read"],
        }
        out[p["id"]] = row
        print(f"{p['id']:10s} {len(r['cards'])} paths ({row['non_stem']} non-STEM, {row['sectors']} sectors) "
              f"money-in-text {row['money_in_llm_text']}  rules_ok {route_ok}  quotes {len(grounded_quotes)}/{len(quoted)} found  "
              f"ruled out {row['ruled_out']}  {row['seconds']}s", flush=True)
        print("            " + ", ".join(row["cards"]))
    (ROOT / "bench" / "results").mkdir(exist_ok=True)
    (ROOT / "bench" / "results" / "career.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
