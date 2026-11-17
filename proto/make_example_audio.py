"""Voice every example answer so the Ear can be tested without a microphone.

    python proto/make_example_audio.py

Clips land in outputs/examples/<lang>/<id>.flac. They are read by the
Microsoft cloud voices (a stand-in student, clearly not a real one), plus a
degraded copy (<id>_phone.flac: 8 kHz phone line + background noise).
Synthetic speech flatters speech-to-text; real recordings by the team go in
bench/recordings/<lang>/ and matter more.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rig"))
from core import OUT, normalise, save  # noqa: E402
from engines import get_engine  # noqa: E402

VOICE = {"hi": "hi-IN-MadhurNeural", "mr": "mr-IN-AarohiNeural", "en": "en-IN-PrabhatNeural"}


def phone_line(x: np.ndarray, sr: int, seed: int) -> tuple[np.ndarray, int]:
    """8 kHz telephone band (300-3400 Hz) plus low background noise at ~15 dB SNR."""
    n = int(len(x) * 8000 / sr)
    spec = np.fft.rfft(x)
    spec = spec[: n // 2 + 1]
    f = np.fft.rfftfreq(n, 1 / 8000)
    spec[(f < 300) | (f > 3400)] = 0
    y = np.fft.irfft(spec, n) * (n / len(x))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(y)).astype(np.float32)
    y = y + noise * np.sqrt(np.mean(y ** 2)) * 10 ** (-15 / 20)
    return normalise(y.astype(np.float32), 8000), 8000


def main() -> None:
    ex = json.loads((ROOT / "proto" / "examples.json").read_text(encoding="utf-8"))["examples"]
    edge = get_engine("edge")
    for lang, items in ex.items():
        for i, it in enumerate(items):
            p = OUT / "examples" / lang / f"{it['id']}.flac"
            if p.exists():
                continue
            audio, sr = edge.synth(it["text"], lang, VOICE[lang], 1.0)
            audio = normalise(audio, sr)
            save(audio, sr, p)
            y, sr2 = phone_line(audio, sr, i)
            save(y, sr2, p.with_name(f"{it['id']}_phone.flac"))
            print("ok", lang, it["id"], flush=True)


if __name__ == "__main__":
    main()
