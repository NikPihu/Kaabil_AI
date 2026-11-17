"""Rendering, dialogue mixing and signal measurements shared by the batch
generator and the live server."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from engines import PROJECT, get_engine

OUT = PROJECT / "outputs"
SCENARIOS = PROJECT / "scenarios.json"
PAN = {"A": -0.55, "B": 0.55, "C": 0.0, "D": -0.25, "E": 0.25}


def load_scenarios() -> list[dict]:
    return json.loads(SCENARIOS.read_text(encoding="utf-8"))["scenarios"]


# ---------------------------------------------------------------- loudness
def normalise(audio: np.ndarray, sr: int, target: float = 0.09) -> np.ndarray:
    """Match loudness across models so a louder clip does not 'win' by volume.
    Scales the 90th-percentile 25 ms frame RMS to `target`, then caps the peak."""
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    n = max(1, int(0.025 * sr))
    frames = mono[: len(mono) // n * n].reshape(-1, n)
    if len(frames) == 0:
        return audio
    rms = np.sqrt((frames ** 2).mean(axis=1))
    ref = np.percentile(rms, 90)
    if ref < 1e-5:
        return audio
    out = audio * (target / ref)
    peak = np.abs(out).max()
    return (out * (0.97 / peak) if peak > 0.97 else out).astype(np.float32)


# ---------------------------------------------------------------- analysis
def _resample(x: np.ndarray, sr: int, to: int) -> np.ndarray:
    if sr == to:
        return x
    n = int(len(x) * to / sr)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def analyse(audio: np.ndarray, sr: int) -> dict:
    """Cheap, model-independent measurements. They do not replace listening;
    they flag what to listen for."""
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    dur = len(mono) / sr
    out: dict = {"duration": round(dur, 2)}
    if dur < 0.3:
        return out

    # pauses: 20 ms frames more than 35 dB below the loud frames
    n = int(0.02 * sr)
    fr = mono[: len(mono) // n * n].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(axis=1)) + 1e-9
    quiet = 20 * np.log10(rms / np.percentile(rms, 95)) < -35
    runs, run = [], 0
    for q in quiet[np.argmax(~quiet): len(quiet) - np.argmax(~quiet[::-1])]:  # ignore lead/tail
        if q:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    pauses = [r * 0.02 for r in runs if r * 0.02 >= 0.15]
    out["pause_count"] = len(pauses)
    out["pause_total"] = round(sum(pauses), 2)
    out["speech_time"] = round(float((~quiet).sum() * 0.02), 2)

    # bandwidth: highest frequency still within 50 dB of the spectral peak
    # (dull, telephone-like audio vs crisp audio)
    nfft = 2048
    segs = mono[: len(mono) // nfft * nfft].reshape(-1, nfft) * np.hanning(nfft)
    if len(segs):
        psd = (np.abs(np.fft.rfft(segs, axis=1)) ** 2).mean(axis=0)
        db = 10 * np.log10(psd / psd.max() + 1e-15)
        above = np.nonzero(db > -50)[0]
        out["bandwidth_hz"] = int(above[-1] * sr / nfft) if len(above) else 0

    # pitch: autocorrelation F0 on 16 kHz, 40 ms frames / 10 ms hop
    x = _resample(mono, sr, 16000)
    win, hop, lo, hi = 640, 160, 16000 // 400, 16000 // 70
    thr = np.percentile(np.abs(x), 95) * 0.1
    f0 = []
    for s in range(0, len(x) - win, hop):
        f = x[s:s + win]
        if np.abs(f).max() < thr:
            continue
        f = f - f.mean()
        ac = np.correlate(f, f, "full")[win - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.5:
            f0.append(16000 / k)
    if len(f0) > 20:
        st = 12 * np.log2(np.array(f0) / np.median(f0))
        st = st[np.abs(st) < 12]          # drop octave errors
        out["f0_median"] = int(np.median(f0))
        out["f0_std_st"] = round(float(st.std()), 2)      # pitch movement, semitones
        out["f0_range_st"] = round(float(np.percentile(st, 95) - np.percentile(st, 5)), 1)
    return out


# ---------------------------------------------------------------- rendering
def render_single(engine_id: str, text: str, lang: str, voice: str | None, speed: float):
    eng = get_engine(engine_id)
    t = time.perf_counter()
    audio, sr = eng.synth(text, lang, voice, speed)
    gen = time.perf_counter() - t
    return normalise(audio, sr), sr, gen


def render_dialogue(engine_id: str, turns: list[dict], lang: str, speed: float):
    """Synthesise each turn with a different voice and overlay them.
    turn = {"who": "A", "text": "...", "overlap": seconds before previous ends}"""
    eng = get_engine(engine_id)
    ids = [v["id"] for v in eng.voices[lang]]
    speakers = sorted({t["who"] for t in turns})
    voice_of = {w: ids[i % len(ids)] for i, w in enumerate(speakers)}
    clips, gen, sr = [], 0.0, None
    for t in turns:
        t0 = time.perf_counter()
        a, sr = eng.synth(t["text"], lang, voice_of[t["who"]], speed)
        gen += time.perf_counter() - t0
        clips.append(normalise(a, sr))
    starts, end = [], 0
    for t, a in zip(turns, clips):
        s = max(0, end - int(float(t.get("overlap", 0)) * sr))
        starts.append(s)
        end = max(end, s + len(a))
    mix = np.zeros((end, 2), dtype=np.float32)
    for t, a, s in zip(turns, clips, starts):
        ang = (PAN.get(t["who"], 0.0) + 1) * np.pi / 4      # constant-power pan
        mix[s:s + len(a), 0] += a * np.cos(ang)
        mix[s:s + len(a), 1] += a * np.sin(ang)
    peak = np.abs(mix).max()
    if peak > 0.97:
        mix *= 0.97 / peak
    overlap_s = sum(max(0, (starts[i - 1] + len(clips[i - 1]) - starts[i])) for i in range(1, len(clips))) / sr
    extra = {"distinct_voices": len(set(voice_of.values())), "speakers": len(speakers),
             "overlap_total": round(overlap_s, 2), "voice_map": voice_of}
    return mix, sr, gen, extra


def save(audio: np.ndarray, sr: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, format="FLAC", subtype="PCM_16")


def scenario_text(sc: dict, lang: str) -> str:
    if "turns" in sc:
        return "\n".join(f'{t["who"]}: {t["text"]}' for t in sc["turns"][lang])
    return sc["text"][lang]
