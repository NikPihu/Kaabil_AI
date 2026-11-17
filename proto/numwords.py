"""Digits -> spoken words in Hindi, Marathi and English, before text reaches a voice.

The LLM is told to write numbers as digits; this code says them.  Measured
reason: Nemotron wrote 72 as "सत्तर दो" (seventy-two, word for word) instead of
"बहत्तर", and Hindi/Marathi numbers below 100 are irregular, so they need a
table, not composition.  Also handles ₹, %, decimals and Indian grouping
(1,50,000 = one lakh fifty thousand).  Long digit strings (OTP, phone numbers)
are read digit by digit.
"""
from __future__ import annotations

import re

HI = ("शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस "
      "इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस चालीस "
      "इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ "
      "इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी अस्सी "
      "इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानबे बानबे तिरानबे चौरानबे पचानबे छियानबे सत्तानबे अट्ठानबे निन्यानबे").split()
MR = ("शून्य एक दोन तीन चार पाच सहा सात आठ नऊ दहा अकरा बारा तेरा चौदा पंधरा सोळा सतरा अठरा एकोणीस वीस "
      "एकवीस बावीस तेवीस चोवीस पंचवीस सव्वीस सत्तावीस अठ्ठावीस एकोणतीस तीस एकतीस बत्तीस तेहेतीस चौतीस पस्तीस छत्तीस सदतीस अडतीस एकोणचाळीस चाळीस "
      "एक्केचाळीस बेचाळीस त्रेचाळीस चव्वेचाळीस पंचेचाळीस सेहेचाळीस सत्तेचाळीस अठ्ठेचाळीस एकोणपन्नास पन्नास एक्कावन्न बावन्न त्रेपन्न चोपन्न पंचावन्न छप्पन्न सत्तावन्न अठ्ठावन्न एकोणसाठ साठ "
      "एकसष्ट बासष्ट त्रेसष्ट चौसष्ट पासष्ट सहासष्ट सदुसष्ट अडुसष्ट एकोणसत्तर सत्तर एक्काहत्तर बाहत्तर त्र्याहत्तर चौऱ्याहत्तर पंचाहत्तर शहात्तर सत्त्याहत्तर अठ्ठ्याहत्तर एकोणऐंशी ऐंशी "
      "एक्याऐंशी ब्याऐंशी त्र्याऐंशी चौऱ्याऐंशी पंच्याऐंशी शहाऐंशी सत्त्याऐंशी अठ्ठ्याऐंशी एकोणनव्वद नव्वद एक्क्याण्णव ब्याण्णव त्र्याण्णव चौऱ्याण्णव पंच्याण्णव शहाण्णव सत्त्याण्णव अठ्ठ्याण्णव नव्व्याण्णव").split()
EN_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
EN_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()

assert len(HI) == 100 and len(MR) == 100

WORDS = {
    "hi": {"hundred": "सौ", "thousand": "हज़ार", "lakh": "लाख", "crore": "करोड़", "point": "दशमलव",
           "rupees": "रुपये", "percent": "प्रतिशत"},
    "mr": {"hundred": "शे", "thousand": "हजार", "lakh": "लाख", "crore": "कोटी", "point": "दशांश",
           "rupees": "रुपये", "percent": "टक्के"},
    "en": {"hundred": "hundred", "thousand": "thousand", "lakh": "lakh", "crore": "crore", "point": "point",
           "rupees": "rupees", "percent": "percent"},
}


def _small(n: int, lang: str) -> str:
    if lang == "hi":
        return HI[n]
    if lang == "mr":
        return MR[n]
    return EN_ONES[n] if n < 20 else EN_TENS[n // 10] + ("" if n % 10 == 0 else " " + EN_ONES[n % 10])


def _hundreds(n: int, lang: str) -> str:
    h, r = divmod(n, 100)
    out = []
    if h:
        if lang == "mr":   # Marathi: शंभर, दोनशे, तीनशे ... (one word)
            out.append("शंभर" if h == 1 else MR[h] + "शे")
        else:
            out.append(f"{_small(h, lang)} {WORDS[lang]['hundred']}")
    if r:
        out.append(_small(r, lang))
    return " ".join(out)


def say_int(n: int, lang: str) -> str:
    if n < 100:
        return _small(n, lang)
    w = WORDS[lang]
    parts = []
    for unit, name in ((10 ** 7, "crore"), (10 ** 5, "lakh"), (1000, "thousand")):
        q, n = divmod(n, unit)
        if q:
            parts.append(f"{say_int(q, lang)} {w[name]}")
    if n:
        parts.append(_hundreds(n, lang))
    return " ".join(parts)


def _digits(s: str, lang: str) -> str:
    return " ".join(_small(int(d), lang) for d in s)


CODE_WORDS = re.compile(r"(otp|ओटीपी|pin|पिन|code|कोड|number|नंबर|नम्बर|क्रमांक|phone|फ़ोन|फोन|mobile|मोबाइल|मोबाईल)\W{0,12}$", re.I)


def say_number(tok: str, lang: str, before: str = "") -> str:
    plain = tok.replace(",", "")
    if "." in plain:
        a, b = plain.split(".", 1)
        return f"{say_int(int(a or 0), lang)} {WORDS[lang]['point']} {_digits(b, lang)}"
    # codes and phone numbers are read digit by digit; amounts are not
    if "," not in tok and (len(plain) >= 10 or (len(plain) >= 4 and CODE_WORDS.search(before))):
        return _digits(plain, lang)
    return say_int(int(plain), lang)


NUM = re.compile(r"(₹\s?)?(\d[\d,]*(?:\.\d+)?)(\s?%)?")


def speakable(text: str, lang: str) -> str:
    def rep(m):
        words = say_number(m.group(2).rstrip(","), lang, m.string[max(0, m.start() - 20):m.start()])
        if m.group(3):
            words += " " + WORDS[lang]["percent"]
        if m.group(1):
            words += " " + WORDS[lang]["rupees"]
        return words + ("," if m.group(2).endswith(",") else "")
    # Devanagari digits -> ASCII first
    text = text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    return NUM.sub(rep, text)


if __name__ == "__main__":
    for l in ("hi", "mr", "en"):
        print(l, "|", speakable("दसवीं में 72% और 72.4, आमदनी ₹1,50,000 या 150000, OTP 482916, नंबर 9876543210, 100, 250, 1999", l))
