"""Speech-to-text adapters ("the Ear").  Same shape as the TTS adapters:

    text = get_stt("shruti").transcribe(audio_16k_float32, lang="hi")
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rig"))
from engines import MODELS, THREADS, dir_mb  # noqa: E402


def load_16k(path: str) -> np.ndarray:
    """Any WAV/FLAC -> 16 kHz mono float32 (band-limited FFT resample)."""
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


class STT:
    id = ""
    name = ""
    maker = ""
    params = ""
    license = ""
    note = ""
    langs: tuple = ()

    def __init__(self):
        self._lock = threading.Lock()
        self._loaded: set = set()

    def available(self) -> tuple[bool, str]:
        return True, ""

    def disk_mb(self) -> dict:
        return {}

    def _load(self, lang: str) -> None:
        pass

    def _run(self, audio: np.ndarray, lang: str) -> str:
        raise NotImplementedError

    def transcribe(self, audio: np.ndarray, lang: str) -> str:
        if lang not in self.langs:
            raise ValueError(f"{self.id} has no {lang} model")
        with self._lock:
            if lang not in self._loaded:
                self._load(lang)
                self._loaded.add(lang)
            return self._run(np.asarray(audio, dtype=np.float32), lang).strip()

    def info(self) -> dict:
        ok, why = self.available()
        return {"id": self.id, "name": self.name, "maker": self.maker, "params": self.params,
                "license": self.license, "note": self.note, "langs": list(self.langs),
                "available": ok, "unavailable_reason": why, "disk_mb": self.disk_mb() if ok else {}}


class Shruti(STT):
    id = "shruti"
    name = "Shruti (sherpa-onnx)"
    maker = "Udgam Labs"
    params = "55M per language"
    license = "Apache-2.0"
    note = ("One ~54 MB model per language, run through sherpa-onnx (the Android runtime). Hindi model "
            "keeps English words in English; Marathi is a separate model. Language is auto-detected.")
    langs = ("hi", "mr")

    def available(self):
        ok = (MODELS / "shruti" / "hi" / "sherpa-onnx" / "encode.onnx").exists()
        return ok, "" if ok else "Run setup_models.py (shruti-hi, shruti-mr)"

    def disk_mb(self):
        return {l: dir_mb(MODELS / "shruti" / l / "sherpa-onnx") for l in self.langs}

    def _load(self, lang):
        import sherpa_onnx as so
        d = MODELS / "shruti" / lang / "sherpa-onnx"
        self.__dict__.setdefault("_r", {})[lang] = so.OfflineRecognizer.from_moonshine(
            preprocessor=str(d / "preprocess.onnx"), encoder=str(d / "encode.onnx"),
            uncached_decoder=str(d / "uncached_decode.onnx"), cached_decoder=str(d / "cached_decode.onnx"),
            tokens=str(d / "tokens.txt"), num_threads=THREADS)

    def _run(self, audio, lang):
        r = self._r[lang]
        out = []
        # the model decodes a whole clip at once; long clips are cut at quiet points
        for piece in _pieces(audio, 25.0):
            s = r.create_stream()
            s.accept_waveform(16000, piece)
            r.decode_stream(s)
            out.append(s.result.text)
        return " ".join(out)


class WhisperSmall(STT):
    id = "whisper-small"
    name = "Whisper small (int8)"
    maker = "OpenAI / CTranslate2"
    params = "244M"
    license = "MIT"
    note = "One multilingual model for all three languages, 8-bit. Strong in English, weak in Marathi."
    langs = ("hi", "mr", "en")

    def available(self):
        ok = (MODELS / "whisper-small" / "model.bin").exists()
        return ok, "" if ok else "Run setup_models.py (whisper-small)"

    def disk_mb(self):
        mb = dir_mb(MODELS / "whisper-small")
        return {l: mb for l in self.langs}

    def _load(self, lang):
        if not hasattr(self, "_m"):
            from faster_whisper import WhisperModel
            self._m = WhisperModel(str(MODELS / "whisper-small"), device="cpu", compute_type="int8",
                                   cpu_threads=THREADS)

    def _run(self, audio, lang):
        segs, _ = self._m.transcribe(audio, language=lang, beam_size=1, temperature=0.0,
                                     condition_on_previous_text=False, vad_filter=False)
        return " ".join(s.text.strip() for s in segs)


class Vosk(STT):
    id = "vosk"
    name = "Vosk small Hindi"
    maker = "Alpha Cephei"
    params = "Kaldi, ~42 MB"
    license = "Apache-2.0"
    note = "The team's round-1 choice. Hindi only; fixed vocabulary, so unseen names and English words are lost."
    langs = ("hi",)

    def available(self):
        try:
            import vosk  # noqa: F401
        except ImportError:
            return False, "Run setup_models.py (pkg-vosk)"
        ok = (MODELS / "vosk-small-hi" / "am").exists()
        return ok, "" if ok else "Run setup_models.py (vosk-small-hi)"

    def disk_mb(self):
        return {"hi": dir_mb(MODELS / "vosk-small-hi")}

    def _load(self, lang):
        import vosk
        vosk.SetLogLevel(-1)
        self._m = vosk.Model(str(MODELS / "vosk-small-hi"))

    def _run(self, audio, lang):
        import json
        import vosk
        rec = vosk.KaldiRecognizer(self._m, 16000)
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        out = []
        for i in range(0, len(pcm), 32000):
            if rec.AcceptWaveform(pcm[i:i + 32000]):
                out.append(json.loads(rec.Result()).get("text", ""))
        out.append(json.loads(rec.FinalResult()).get("text", ""))
        return " ".join(t for t in out if t)


def _pieces(x: np.ndarray, max_s: float):
    if len(x) <= max_s * 16000:
        return [x]
    lo, hi = int(len(x) * 0.35), int(len(x) * 0.65)
    n = 320
    env = np.abs(x[lo:hi][: (hi - lo) // n * n]).reshape(-1, n).mean(axis=1)
    cut = lo + int(np.argmin(np.convolve(env, np.ones(8) / 8, "same"))) * n
    return _pieces(x[:cut], max_s) + _pieces(x[cut:], max_s)


# --------------------------------------------------------------------------
# Cloud Ears: need internet and a key; use no phone memory.
# --------------------------------------------------------------------------
def _multipart(url: str, headers: dict, fields: dict, audio: np.ndarray) -> dict:
    import io, json as _j, urllib.request, urllib.error, uuid
    import soundfile as sf
    buf = io.BytesIO(); sf.write(buf, audio, 16000, format="WAV", subtype="PCM_16")
    b, NL = uuid.uuid4().hex, chr(13) + chr(10)
    parts = [f'--{b}{NL}Content-Disposition: form-data; name="{k}"{NL}{NL}{v}{NL}'.encode() for k, v in fields.items()]
    parts.append(f'--{b}{NL}Content-Disposition: form-data; name="file"; filename="a.wav"{NL}Content-Type: audio/wav{NL}{NL}'.encode()
                 + buf.getvalue() + f'{NL}--{b}--{NL}'.encode())
    body = b"".join(parts)
    req = urllib.request.Request(url, data=body, headers={**headers, "Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return _j.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}") from None


class SarvamSTT(STT):
    id = "sarvam-stt"
    name = "Sarvam Saarika (cloud)"
    maker = "Sarvam AI"
    params = "n/a (server-side)"
    license = "Commercial API"
    note = "Indian-language speech-to-text in the cloud. Needs internet and SARVAM_API_KEY; audio leaves the phone."
    langs = ("hi", "mr", "en")
    CODE = {"hi": "hi-IN", "mr": "mr-IN", "en": "en-IN"}

    def available(self):
        ok = bool(os.environ.get("SARVAM_API_KEY"))
        return ok, "" if ok else "Add SARVAM_API_KEY to .env"

    def _run(self, audio, lang):
        r = _multipart("https://api.sarvam.ai/speech-to-text", {"api-subscription-key": os.environ["SARVAM_API_KEY"]},
                       {"model": os.environ.get("SARVAM_STT_MODEL", "saarika:v2.5"), "language_code": self.CODE[lang]}, audio)
        return r.get("transcript", "")


class ElevenScribe(STT):
    id = "elevenlabs-stt"
    name = "ElevenLabs Scribe (cloud)"
    maker = "ElevenLabs"
    params = "n/a (server-side)"
    license = "Commercial API"
    note = "Multilingual speech-to-text in the cloud. Needs internet and ELEVENLABS_API_KEY; audio leaves the phone."
    langs = ("hi", "mr", "en")
    CODE = {"hi": "hin", "mr": "mar", "en": "eng"}

    def available(self):
        ok = bool(os.environ.get("ELEVENLABS_API_KEY"))
        return ok, "" if ok else "Add ELEVENLABS_API_KEY to .env"

    def _run(self, audio, lang):
        r = _multipart("https://api.elevenlabs.io/v1/speech-to-text", {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
                       {"model_id": os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v1"), "language_code": self.CODE[lang]}, audio)
        return r.get("text", "")


_REG: dict[str, STT] = {}


def stts() -> dict[str, STT]:
    if not _REG:
        for s in (Shruti(), WhisperSmall(), Vosk(), SarvamSTT(), ElevenScribe()):
            _REG[s.id] = s
    return _REG


def get_stt(sid: str) -> STT:
    return stts()[sid]
