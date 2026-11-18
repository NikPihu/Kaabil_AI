"""Hard limits for career routes ("AI decides, rules guard"). No LLM here.

situation(profile)  -> the student's situation in machine terms: completed level, stream,
                       years until the next level, age, marks, travel limit, money, interests text
check(route, sit)   -> status + plain-English reasons:
    now            the student already meets the entry requirement
    after_current  met once the current class/course is finished
    with_step      reachable, but needs a choice first (e.g. take Science in Class 11)
    blocked        cannot be met (stream already chosen, age limit passed) -> never shown
plus watch-outs (city move vs travel limit, cost vs income, marks below the minimum).
"""
from __future__ import annotations

import re

LEVELS = {"class_8": 8, "class_10": 10, "class_12": 12, "graduate": 15, "any": 0}
STREAM_REQ = {"class_12_science": {"pcb", "pcm", "science"}, "class_12_pcb": {"pcb"},
              "class_12_pcm_or_pcb": {"pcb", "pcm", "science"}, "class_12_commerce": {"commerce"}}


def _num(s: str | None) -> float | None:
    if not s:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", s.replace(",", ""))
    return float(m.group(1)) if m else None


def _income(s: str | None) -> float | None:
    """'about 1.5 lakh rupees a year' / '150000' / 'two to three lakh' -> rupees per year."""
    if not s:
        return None
    t = s.lower().replace(",", "")
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "half": 0.5, "one and a half": 1.5}
    n = _num(t)
    if n is None:
        for w, v in sorted(words.items(), key=lambda x: -len(x[0])):
            if w in t:
                n = v
                break
    if n is None:
        return None
    if "lakh" in t or "lac" in t:
        return n * 100000
    if "thousand" in t or "हजार" in t or "हज़ार" in t:
        return n * 1000
    return n if n > 1000 else None


def situation(p: dict) -> dict:
    """p = {field_id: value} in English (Profile Agent output)."""
    cls = (p.get("class_or_year") or "").lower()
    course = (p.get("stream_or_course") or "").lower()
    both = f"{cls} {course}"
    sit = {"completed": 0, "current": None, "years_to_next": 0, "stream": None, "in_college": False}
    # a degree mention wins over numbers like "2nd year"
    degree = re.search(r"b\.?\s?a\b|b\.?\s?com|b\.?\s?sc|\bbca\b|\bbba\b|b\.?\s?tech|\bb\.?\s?e\b|degree|graduat|college", cls)
    m = None if degree else re.search(r"class\s*(\d{1,2})|\b(\d{1,2})\s*(?:th|st|nd|rd)\b|\b(\d{1,2})\b", cls)
    if re.search(r"\biti\b", both):
        sit.update(completed=10, current="iti", years_to_next=1 if "1st" in cls or "first" in cls else 0)
    elif re.search(r"diploma|polytechnic", both):
        sit.update(completed=10, current="diploma", years_to_next=2)
    elif degree:
        yr = _num(cls) or 1
        sit.update(completed=12, current="graduate", in_college=True, years_to_next=max(0, 3 - int(yr) + 1))
    elif m:
        n = int(m.group(1) or m.group(2) or m.group(3))
        if n <= 10:
            sit.update(completed=n - 1, current=f"class_{n}", years_to_next=10 - n + 1)
        else:
            sit.update(completed=10, current=f"class_{n}", years_to_next=12 - n + 1)
    # stream (for Class 11-12 or the degree subject)
    if re.search(r"\bpcb\b|biology|bio\b", both):
        sit["stream"] = "pcb"
    elif re.search(r"\bpcm\b|math", both) and re.search(r"science|pcm|b\.?\s?sc", both):
        sit["stream"] = "pcm"
    elif re.search(r"science|b\.?\s?sc", both):
        sit["stream"] = "science"
    elif re.search(r"commerce|b\.?\s?com|accounts", both):
        sit["stream"] = "commerce"
    elif re.search(r"arts|humanities|b\.?\s?a\b", both):
        sit["stream"] = "arts"
    cons = (p.get("constraints") or "").lower()
    km = re.search(r"(\d+)\s*(km|kilomet)", cons)
    sit["travel_km"] = int(km.group(1)) if km else None
    sit["local_only"] = bool(re.search(r"cannot leave|can't leave|cannot travel|not.*far|near home|mother is alone|"
                                       r"village only|stay in|beyond|not go to|प्रवास|दूर", cons)) or (sit["travel_km"] is not None and sit["travel_km"] <= 40)
    sit["age"] = _num(p.get("age"))
    sit["marks"] = _num(p.get("last_marks"))
    sit["income"] = _income(p.get("family_income"))
    sit["low_income"] = sit["income"] is not None and sit["income"] <= 250000
    sit["text"] = " ".join(str(v) for v in p.values()).lower()
    return sit


def check(route: dict, sit: dict) -> dict:
    need = route["from"]
    base = "class_12" if need.startswith("class_12") else need
    lvl = LEVELS[base]
    reasons, watch = [], []
    status = "now"
    years_wait = 0
    if lvl > sit["completed"]:
        cur = sit["current"] or ""
        # finishing the current class/course reaches the level?
        reach = {"class_10": 10, "class_12": 12, "graduate": 15}.get(base, lvl)
        finish_gives = 10 if cur in {f"class_{n}" for n in range(6, 11)} else 12 if cur in ("class_11", "class_12") else \
            15 if cur == "graduate" else 10
        if reach <= finish_gives:
            status, years_wait = "after_current", sit["years_to_next"]
            target = {10: "Class 10", 12: "Class 12", 15: "your degree"}.get(finish_gives, "your current course")
            if cur in ("iti", "diploma"):
                target = "your " + cur.upper() if cur == "iti" else "your diploma"
            reasons.append(f"after you finish {target}")
        else:
            status = "with_step"
            years_wait = sit["years_to_next"] + (2 if base == "class_12" else
                                                 3 if sit["current"] in ("class_11", "class_12", "iti") else 5)
            reasons.append({"class_12": "after Class 12", "graduate": "after a degree (3 years after Class 12)",
                            "class_10": "after Class 10"}.get(base, "after more study"))
    # stream requirement
    if need in STREAM_REQ:
        ok = STREAM_REQ[need]
        if sit["completed"] >= 12 or sit["current"] in ("class_11", "class_12"):
            if sit["stream"] and sit["stream"] not in ok and not (sit["stream"] == "science" and ok & {"pcb", "pcm"}):
                return {"status": "blocked", "reasons": [f"needs {need.replace('class_12_', '').upper()} in Class 12; your stream is {sit['stream']}"], "watch": [], "years_wait": None}
            if not sit["stream"]:
                watch.append("check your Class 11–12 subjects: this route needs " + need.replace("class_12_", "").replace("_", " ").upper())
        else:
            status = "with_step" if status != "now" else status
            reasons.append("choose " + {"class_12_pcb": "Science with Biology", "class_12_science": "Science",
                                         "class_12_pcm_or_pcb": "Science (Biology or Maths)",
                                         "class_12_commerce": "Commerce"}[need] + " in Class 11")
    # age limits at the time the student becomes eligible
    if sit["age"] is not None:
        at = sit["age"] + (years_wait or 0)
        if route.get("max_age") and at > route["max_age"]:
            return {"status": "blocked", "reasons": [f"age limit is {route['max_age']}; you would be about {int(at)}"], "watch": [], "years_wait": None}
        if route.get("min_age") and at < route["min_age"]:
            reasons.append(f"minimum age {route['min_age']:g}; you can start at {route['min_age']:g}")
        if route.get("max_age"):
            watch.append(f"age limit {route.get('min_age', 0):g}–{route['max_age']:g} at the time of applying")
    # marks
    if route.get("min_marks") and sit["marks"] is not None and sit["marks"] < route["min_marks"]:
        watch.append(f"needs {route['min_marks']}% in Class 12; your last result was {sit['marks']:g}%")
    # place and money
    if route["where"] == "city" and (sit["local_only"]):
        watch.append("this route usually means moving to a city, which clashes with what you said about travel")
    if sit["low_income"] and route["cost"]["max"] >= 150000:
        watch.append("can be costly: pick a government college and check scholarships")
    return {"status": status, "reasons": reasons, "watch": watch, "years_wait": years_wait}


def evaluate(career: dict, sit: dict) -> dict:
    """Best route for this student (now > after_current > with_step), or None if all blocked."""
    order = {"now": 0, "after_current": 1, "with_step": 2}
    best = None
    for r in career["routes"]:
        c = check(r, sit)
        if c["status"] == "blocked":
            continue
        local_penalty = 1 if (sit["local_only"] and r["where"] == "city") else 0
        key = (order[c["status"]] + local_penalty, c["years_wait"] or 0, r["cost"]["max"])
        if best is None or key < best[0]:
            best = (key, r, c)
    if not best:
        blocked = [check(r, sit)["reasons"][0] for r in career["routes"]]
        return {"eligible": False, "why_not": blocked[0] if blocked else ""}
    return {"eligible": True, "route": best[1], "check": best[2],
            "options": [r["id"] for r in career["routes"] if check(r, sit)["status"] != "blocked"]}
