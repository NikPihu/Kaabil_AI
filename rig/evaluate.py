"""The Whisper judge: transcribe a clip and compare it with what should have
been said. Used by the benchmarks in bench/.

CER (character error rate) is a proxy for intelligibility, not naturalness.
Whisper itself makes mistakes, especially in Marathi, so read CER relative to
the cloud reference voice on the same sentence, and always check by ear.
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from engines import RIG_HOME  # noqa: E402

os.environ.setdefault("HF_HOME", str(RIG_HOME / "hf_cache"))


_MODEL = None


def whisper():
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel
        _MODEL = WhisperModel(os.environ.get("VOICE_RIG_ASR", "large-v3-turbo"),
                              device="cpu", compute_type="int8", cpu_threads=8)
    return _MODEL


def load_16k(path: str):
    """Read a clip as 16 kHz mono (band-limited FFT resample, no extra deps)."""
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr == 16000:
        return x
    n = int(round(len(x) * 16000 / sr))
    spec = np.fft.rfft(x)
    keep = n // 2 + 1
    spec = spec[:keep] if keep <= len(spec) else np.pad(spec, (0, keep - len(spec)))
    return (np.fft.irfft(spec, n) * (n / len(x))).astype("float32")


def _decode(audio, lang: str, temperature=0.0) -> str:
    segs, _ = whisper().transcribe(audio, language=lang, beam_size=2, temperature=temperature,
                                   condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segs).strip()


def _pieces(x, max_s: float = 12.0):
    """Split at the quietest point near the middle until every piece is short."""
    import numpy as np
    if len(x) <= max_s * 16000:
        return [x]
    lo, hi = int(len(x) * 0.35), int(len(x) * 0.65)
    n = 320
    env = np.abs(x[lo:hi][: (hi - lo) // n * n]).reshape(-1, n).mean(axis=1)
    cut = lo + int(np.argmin(np.convolve(env, np.ones(8) / 8, "same"))) * n
    return _pieces(x[:cut], max_s) + _pieces(x[cut:], max_s)


def transcribe(path: str, lang: str, expect_chars: int = 0) -> str:
    """Whisper sometimes stops early on a long clip.  If the transcript is much
    shorter than the script, transcribe again in short pieces and keep the
    longer result, so an ASR dropout is not blamed on the voice."""
    x = load_16k(path)
    hyp = _decode(x, lang)
    if expect_chars and len(norm(hyp)) < 0.75 * expect_chars and len(x) > 12 * 16000:
        alt = " ".join(_decode(p, lang) for p in _pieces(x))
        if len(norm(alt)) > len(norm(hyp)):
            hyp = alt
    if expect_chars and len(norm(hyp)) > 1.5 * expect_chars:      # stuck in a repetition loop
        hyp = _decode(x, lang, temperature=(0.0, 0.2, 0.4, 0.6))
    return hyp


def norm(text: str) -> str:
    """Strip everything that is spelling convention rather than sound."""
    t = unicodedata.normalize("NFC", text).lower()
    t = t.replace("़", "")                  # nukta: फ़ -> फ
    t = t.replace("ँ", "ं")            # chandrabindu -> anusvara
    t = t.replace("‌", "").replace("‍", "")   # zero-width joiners
    # keep letters, combining marks (matras, virama) and digits; drop the rest
    t = "".join(ch if unicodedata.category(ch)[0] in "LMN" else " " for ch in t)
    return re.sub(r"\s+", " ", t).strip()


# Whisper writes "fifteen hundred" as "1500" whenever it likes, which would be
# scored as an error.  Numbers are therefore removed from both sides before
# scoring; the dedicated number scenarios are judged from the transcript instead.
NUMBER_WORDS = set("""
zero one two three four five six seven eight nine ten twelve fifteen twenty thirty hundred thousand first second third
शून्य एक दो तीन चार पांच छह छः सात आठ नौ दस बारह पंद्रह पच्चीस छब्बीस तीस सौ हजार साढे
दोन पाच सहा नऊ दहा बारा पंधरा पंधराशे पाचशे पंचवीस सव्वीस शे साडे साडेपाच
""".split())


def strip_numbers(t: str) -> str:
    t = re.sub(r"[0-9०-९]+(st|nd|rd|th)?", " ", t)
    return " ".join(w for w in t.split() if w not in NUMBER_WORDS)


def cer(ref: str, hyp: str) -> float:
    import jiwer
    r, h = strip_numbers(norm(ref)), strip_numbers(norm(hyp))
    if not r:
        return 0.0
    return round(min(1.0, float(jiwer.cer(r, h))), 3)
