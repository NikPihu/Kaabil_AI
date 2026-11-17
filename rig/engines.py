"""Engine adapters: every TTS model sits behind the same synth() call.

    audio, sr = get_engine("piper").synth(text, lang="hi", voice="priyamvada", speed=1.0)

audio is mono float32 in [-1, 1].  Languages are "en", "hi", "mr".
Model weights live outside the project (default: ~/voice_rig/models) so the
OneDrive folder only holds code and the generated clips.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
RIG_HOME = Path(os.environ.get("VOICE_RIG_HOME", Path.home() / "voice_rig"))
MODELS = RIG_HOME / "models"
THREADS = int(os.environ.get("VOICE_RIG_THREADS", "4"))


def load_env() -> None:
    """Read KEY=value lines from the project's .env (API keys for cloud engines)."""
    env = PROJECT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


load_env()


def dir_mb(*paths: Path) -> float:
    total = 0
    for p in paths:
        if p.is_file():
            total += p.stat().st_size
        elif p.is_dir():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / 1e6, 1)


_ESPEAK = threading.Lock()   # espeak-ng is not thread-safe


def _trim(wav: np.ndarray, sr: int, keep: float = 0.03) -> np.ndarray:
    """Cut leading/trailing near-silence, keeping `keep` seconds of padding."""
    loud = np.nonzero(np.abs(wav) > 0.02 * np.abs(wav).max())[0]
    if len(loud) == 0:
        return wav
    pad = int(keep * sr)
    return wav[max(0, loud[0] - pad): loud[-1] + pad]


def V(vid: str, label: str, gender: str) -> dict:
    return {"id": vid, "label": label, "gender": gender}


class Engine:
    id = ""
    name = ""
    kind = "local"          # local | cloud
    maker = ""
    params = ""
    license = ""
    arch = ""
    note = ""
    speed_control = True
    voices: dict[str, list[dict]] = {}
    experimental: set[str] = set()   # languages the model was NOT trained for
    in_matrix = True                 # include in the pre-generated demo matrix

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False

    # -- overridables -------------------------------------------------------
    def available(self) -> tuple[bool, str]:
        return True, ""

    def disk_mb(self) -> dict[str, float]:
        """Download size per language (what a phone would have to store)."""
        return {}

    def _load(self) -> None:
        pass

    def _synth(self, text: str, lang: str, voice: str, speed: float):
        raise NotImplementedError

    # -- public -------------------------------------------------------------
    def synth(self, text: str, lang: str, voice: str | None = None, speed: float = 1.0):
        if lang not in self.voices:
            raise ValueError(f"{self.id} has no voice for language '{lang}'")
        ids = [v["id"] for v in self.voices[lang]]
        voice = voice if voice in ids else ids[0]
        with self._lock:
            if not self._loaded:
                self._load()
                self._loaded = True
            audio, sr = self._synth(text, lang, voice, float(speed))
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        return audio, int(sr)

    def info(self) -> dict:
        ok, why = self.available()
        return {
            "id": self.id, "name": self.name, "kind": self.kind, "maker": self.maker,
            "params": self.params, "license": self.license, "arch": self.arch,
            "note": self.note, "voices": self.voices,
            "experimental": sorted(self.experimental),
            "available": ok, "unavailable_reason": why,
            "disk_mb": self.disk_mb() if ok and self.kind == "local" else {},
            "in_matrix": self.in_matrix, "speed_control": self.speed_control,
        }


# --------------------------------------------------------------------------
# Piper  (VITS, one small ONNX file per voice)
# --------------------------------------------------------------------------
class Piper(Engine):
    id = "piper"
    name = "Piper (VITS medium)"
    maker = "Open Home Foundation / Rhasspy"
    params = "~20M per voice"
    license = "Code GPL-3.0 (piper1-gpl); voices vary (MIT / CC-BY)"
    arch = "VITS, espeak-ng phonemes, ONNX"
    note = "One 60-77 MB file per voice. The only tiny model here with native Hindi AND Marathi."
    FILES = {
        "en": {"amy": "en/en_US/amy/medium/en_US-amy-medium",
               "ryan": "en/en_US/ryan/medium/en_US-ryan-medium"},
        "hi": {"priyamvada": "hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium",
               "pratham": "hi/hi_IN/pratham/medium/hi_IN-pratham-medium",
               "rohan": "hi/hi_IN/rohan/medium/hi_IN-rohan-medium"},
        "mr": {f"google-{i}": "mr/mr_IN/google/medium/mr_IN-google-medium" for i in (0, 3, 6, 8)},
    }
    voices = {
        "en": [V("amy", "Amy (US)", "F"), V("ryan", "Ryan (US)", "M")],
        "hi": [V("priyamvada", "Priyamvada", "F"), V("pratham", "Pratham", "M"), V("rohan", "Rohan", "M")],
        "mr": [V(f"google-{i}", f"Speaker {i}", "F") for i in (0, 3, 6, 8)],
    }

    def available(self):
        ok = (MODELS / "piper").exists()
        return ok, "" if ok else "Piper voices not downloaded (see README)"

    def disk_mb(self):
        return {lang: dir_mb(MODELS / "piper" / (next(iter(f.values())) + ".onnx"))
                for lang, f in self.FILES.items()}

    def _load(self):
        self._v = {}

    def _synth(self, text, lang, voice, speed):
        from piper import PiperVoice, SynthesisConfig
        stem = self.FILES[lang][voice]
        if stem not in self._v:
            self._v[stem] = PiperVoice.load(str(MODELS / "piper" / (stem + ".onnx")))
        pv = self._v[stem]
        sid = int(voice.split("-")[1]) if lang == "mr" else None
        base = getattr(pv.config, "length_scale", 1.0) or 1.0
        cfg = SynthesisConfig(speaker_id=sid, length_scale=base / speed, normalize_audio=True)
        chunks = list(pv.synthesize(text, syn_config=cfg))
        audio = np.concatenate([c.audio_int16_array for c in chunks]).astype(np.float32) / 32768.0
        return audio, pv.config.sample_rate


# --------------------------------------------------------------------------
# Kokoro-82M  (StyleTTS2-derived, one multilingual ONNX + a voice pack)
# --------------------------------------------------------------------------
class Kokoro(Engine):
    maker = "hexgrad"
    params = "82M"
    license = "Apache-2.0"
    arch = "StyleTTS2 + ISTFTNet, espeak-ng phonemes, ONNX"
    LANG = {"en": "en-us", "hi": "hi", "mr": "mr"}
    experimental = {"mr"}
    voices = {
        "en": [V("af_heart", "Heart (US)", "F"), V("am_michael", "Michael (US)", "M"),
               V("af_bella", "Bella (US)", "F"), V("bm_george", "George (UK)", "M"),
               V("bf_emma", "Emma (UK)", "F"), V("am_fenrir", "Fenrir (US)", "M")],
        "hi": [V("hf_alpha", "Alpha", "F"), V("hm_omega", "Omega", "M"),
               V("hf_beta", "Beta", "F"), V("hm_psi", "Psi", "M")],
        "mr": [V("hf_alpha", "Alpha (Hindi voice)", "F"), V("hm_omega", "Omega (Hindi voice)", "M"),
               V("hf_beta", "Beta (Hindi voice)", "F")],
    }

    def __init__(self, variant: str):
        super().__init__()
        self.variant = variant
        if variant == "fp32":
            self.id, self.name, self.file = "kokoro", "Kokoro-82M v1.0", "kokoro-v1.0.onnx"
            self.note = ("Top-rated tiny open model for English. Has 4 Hindi voices; Marathi is "
                         "NOT supported - the 'mr' option reads Marathi through Hindi voices as an experiment.")
        else:
            self.id, self.name, self.file = "kokoro-int8", "Kokoro-82M v1.0 (int8)", "kokoro-v1.0.int8.onnx"
            self.note = "Same model quantised to 8-bit: 3.5x smaller download. Compare by ear against fp32."
            self.in_matrix = False

    def available(self):
        ok = (MODELS / "kokoro" / self.file).exists()
        return ok, "" if ok else "Kokoro model not downloaded"

    def disk_mb(self):
        mb = dir_mb(MODELS / "kokoro" / self.file, MODELS / "kokoro" / "voices-v1.0.bin")
        return {"en": mb, "hi": mb, "mr": mb}

    def _load(self):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro as K
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = THREADS
        sess = ort.InferenceSession(str(MODELS / "kokoro" / self.file), opts,
                                    providers=["CPUExecutionProvider"])
        self._k = K.from_session(sess, str(MODELS / "kokoro" / "voices-v1.0.bin"))

    def _synth(self, text, lang, voice, speed):
        code = "en-gb" if voice.startswith("b") else self.LANG[lang]
        if lang == "en":
            return self._k.create(text, voice=voice, speed=speed, lang=code)
        # Stock kokoro-onnx has two bugs on Hindi/Marathi text, both patched here:
        #  1. espeak marks English words as "(en)...(hi)" and the model reads the
        #     markers aloud ("in application hi") -> ask espeak to drop the flags.
        #  2. the danda is not in the vocabulary, so sentences run together -> use ".".
        import phonemizer
        with _ESPEAK:
            ph = phonemizer.phonemize(text.replace("।", "."), code, preserve_punctuation=True,
                                      with_stress=True, language_switch="remove-flags")
        ph = "".join(c for c in ph if c in self._k.tokenizer.vocab).strip()
        return self._k.create(ph, voice=voice, speed=speed, lang=code, is_phonemes=True)


# --------------------------------------------------------------------------
# Supertonic 3  (flow-matching, 31 languages incl. Hindi, raw-text input)
# --------------------------------------------------------------------------
class Supertonic(Engine):
    id = "supertonic"
    name = "Supertonic 3"
    maker = "Supertone"
    params = "99M"
    license = "Weights OpenRAIL-M, code MIT"
    arch = "Flow-matching latent TTS, character input (no phonemiser), 44.1 kHz, ONNX"
    note = ("31 languages incl. Hindi, 44.1 kHz output, expression tags (<laugh> <breath> <sigh>). "
            "No Marathi - the 'mr' option feeds Marathi text with the Hindi language tag as an experiment.")
    experimental = {"mr"}
    voices = {lang: [V(n, n, n[0]) for n in ("F1", "M1", "F2", "M2", "F3", "M3")]
              for lang in ("en", "hi", "mr")}

    def available(self):
        ok = (MODELS / "supertonic-3" / "onnx").exists()
        return ok, "" if ok else "Supertonic 3 not downloaded"

    def disk_mb(self):
        mb = dir_mb(MODELS / "supertonic-3" / "onnx")
        return {"en": mb, "hi": mb, "mr": mb}

    def _load(self):
        from supertonic import TTS
        self._t = TTS(model="supertonic-3", model_dir=str(MODELS / "supertonic-3"),
                      auto_download=False, intra_op_num_threads=THREADS)
        self._styles = {}

    def _synth(self, text, lang, voice, speed):
        if voice not in self._styles:
            self._styles[voice] = self._t.get_voice_style(voice)
        wav, _ = self._t.synthesize(text, voice_style=self._styles[voice], total_steps=8,
                                    speed=1.05 * speed, lang="hi" if lang == "mr" else lang)
        return wav, self._t.sample_rate


# --------------------------------------------------------------------------
# sherpa-onnx runtime: MMS (VITS), KittenTTS, Supertonic int8
# sherpa-onnx is also the runtime you would ship on Android, so numbers here
# transfer to the phone more directly than the Python-only packages.
# --------------------------------------------------------------------------
class MMS(Engine):
    id = "mms"
    name = "MMS-TTS (VITS)"
    maker = "Meta"
    params = "36M per language"
    license = "CC-BY-NC 4.0 (non-commercial)"
    arch = "VITS, character input, 16 kHz, ONNX via sherpa-onnx"
    note = ("1,100+ languages, one voice each, 16 kHz. Covers Marathi, but the licence is "
            "non-commercial - fine for a hackathon prototype, not for a paid product.")
    CODE = {"en": "eng", "hi": "hin", "mr": "mar"}
    voices = {lang: [V("default", "Default", "M")] for lang in ("en", "hi", "mr")}

    def available(self):
        ok = (MODELS / "mms" / "hin" / "model.onnx").exists()
        return ok, "" if ok else "MMS models not downloaded"

    def disk_mb(self):
        return {lang: dir_mb(MODELS / "mms" / c) for lang, c in self.CODE.items()}

    def _load(self):
        self._t = {}

    def _synth(self, text, lang, voice, speed):
        import sherpa_onnx as so
        if lang not in self._t:
            d = MODELS / "mms" / self.CODE[lang]
            cfg = so.OfflineTtsConfig(model=so.OfflineTtsModelConfig(
                vits=so.OfflineTtsVitsModelConfig(model=str(d / "model.onnx"), tokens=str(d / "tokens.txt")),
                num_threads=THREADS), max_num_sentences=1)
            self._t[lang] = so.OfflineTts(cfg)
        # MMS was trained on lower-cased, punctuation-light text.
        out = self._t[lang].generate(text.lower() if lang == "en" else text, sid=0, speed=speed)
        return np.asarray(out.samples, dtype=np.float32), out.sample_rate


class Kitten(Engine):
    id = "kitten"
    name = "KittenTTS nano 0.8"
    maker = "KittenML"
    params = "15M"
    license = "Apache-2.0"
    arch = "StyleTTS2-lite, espeak-ng phonemes, ONNX"
    note = "Smallest usable neural TTS (57 MB here, 25 MB as int8). English only."
    DIR = "kitten-nano-0.8"
    _NAMES = ("Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo")
    voices = {"en": [V(n, n, "FM"[i % 2]) for i, n in enumerate(_NAMES)]}

    def available(self):
        ok = ((MODELS / self.DIR / "kitten_tts_nano_v0_8.onnx").exists()
              and (MODELS / "kitten-nano-en-v0_8-fp32" / "tokens.txt").exists())
        return ok, "" if ok else "KittenTTS not downloaded"

    def disk_mb(self):
        d = MODELS / self.DIR
        return {"en": dir_mb(d / "kitten_tts_nano_v0_8.onnx", d / "voices.npz")}

    def _load(self):
        import espeakng_loader
        import onnxruntime as ort
        from phonemizer.backend import EspeakBackend
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        d = MODELS / self.DIR
        self._cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        self._styles = np.load(d / "voices.npz")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = THREADS
        self._sess = ort.InferenceSession(str(d / "kitten_tts_nano_v0_8.onnx"), opts,
                                          providers=["CPUExecutionProvider"])
        self._g2p = EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
        self._tok = {}
        tok_file = MODELS / "kitten-nano-en-v0_8-fp32" / "tokens.txt"
        for line in tok_file.read_text(encoding="utf-8").splitlines():
            if line:
                sym, idx = line.rsplit(" ", 1)
                self._tok[sym or " "] = int(idx)

    def _synth(self, text, lang, voice, speed):
        import re
        vid = self._cfg["voice_aliases"][voice]
        speed = speed * self._cfg.get("speed_priors", {}).get(vid, 1.0)
        out = []
        for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
            if not sent:
                continue
            with _ESPEAK:
                ph = self._g2p.phonemize([sent])[0]
            ph = " ".join(re.findall(r"\w+|[^\w\s]", ph))
            ids = [0] + [self._tok[c] for c in ph if c in self._tok] + [10, 0]
            style = self._styles[vid][min(len(sent), self._styles[vid].shape[0] - 1)][None, :]
            wav = self._sess.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                                        "style": style.astype(np.float32),
                                        "speed": np.array([speed], dtype=np.float32)})[0]
            out.append(_trim(wav, 24000))
            out.append(np.zeros(int(0.25 * 24000), dtype=np.float32))
        return np.concatenate(out), 24000


class SupertonicInt8(Engine):
    id = "supertonic-int8"
    name = "Supertonic 3 (int8, sherpa-onnx)"
    maker = "Supertone / k2-fsa"
    params = "99M"
    license = "Weights OpenRAIL-M"
    arch = "Same model, 8-bit quantised, run through sherpa-onnx (the Android runtime)"
    note = "Phone-sized build of Supertonic 3: about a third of the fp32 download."
    experimental = {"mr"}
    in_matrix = False
    DIR = "sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
    voices = {lang: [V(str(i), f"Voice {i}", "?") for i in range(4)] for lang in ("en", "hi", "mr")}

    def available(self):
        ok = (MODELS / self.DIR / "vocoder.int8.onnx").exists()
        return ok, "" if ok else "Supertonic int8 not downloaded"

    def disk_mb(self):
        mb = dir_mb(MODELS / self.DIR)
        return {"en": mb, "hi": mb, "mr": mb}

    def _load(self):
        import sherpa_onnx as so
        d = MODELS / self.DIR
        cfg = so.OfflineTtsConfig(model=so.OfflineTtsModelConfig(
            supertonic=so.OfflineTtsSupertonicModelConfig(
                duration_predictor=str(d / "duration_predictor.int8.onnx"),
                text_encoder=str(d / "text_encoder.int8.onnx"),
                vector_estimator=str(d / "vector_estimator.int8.onnx"),
                vocoder=str(d / "vocoder.int8.onnx"),
                tts_json=str(d / "tts.json"),
                unicode_indexer=str(d / "unicode_indexer.bin"),
                voice_style=str(d / "voice.bin")),
            num_threads=THREADS))
        self._t = so.OfflineTts(cfg)
        self._so = so

    def _synth(self, text, lang, voice, speed):
        g = self._so.GenerationConfig()
        g.sid, g.speed, g.num_steps = int(voice), 1.05 * speed, 8
        g.extra = {"lang": "hi" if lang == "mr" else lang}
        out = self._t.generate(text, g)
        return np.asarray(out.samples, dtype=np.float32), out.sample_rate


# --------------------------------------------------------------------------
# Cloud reference voices
# --------------------------------------------------------------------------
def _decode(blob: bytes):
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def _post_json(url: str, headers: dict, body: dict, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}") from None


class EdgeNeural(Engine):
    id = "edge"
    name = "Microsoft Edge neural voices (cloud)"
    kind = "cloud"
    maker = "Microsoft"
    params = "n/a (server-side)"
    license = "Proprietary; reached through the unofficial edge-tts client"
    arch = "Azure neural TTS"
    note = ("Cloud REFERENCE only: no key needed, but text is sent to Microsoft and this endpoint is "
            "unofficial - do not ship it. Use it as the 'what a commercial voice sounds like' bar.")
    voices = {
        "en": [V("en-IN-NeerjaNeural", "Neerja (Indian English)", "F"),
               V("en-IN-PrabhatNeural", "Prabhat (Indian English)", "M"),
               V("en-US-AvaMultilingualNeural", "Ava (US)", "F")],
        "hi": [V("hi-IN-SwaraNeural", "Swara", "F"), V("hi-IN-MadhurNeural", "Madhur", "M")],
        "mr": [V("mr-IN-AarohiNeural", "Aarohi", "F"), V("mr-IN-ManoharNeural", "Manohar", "M")],
    }

    def available(self):
        try:
            import edge_tts  # noqa: F401
            return True, ""
        except ImportError:
            return False, "pip install edge-tts"

    def _synth(self, text, lang, voice, speed):
        import edge_tts

        async def run() -> bytes:
            com = edge_tts.Communicate(text, voice, rate=f"{round((speed - 1) * 100):+d}%")
            buf = bytearray()
            async for chunk in com.stream():
                if chunk["type"] == "audio":
                    buf.extend(chunk["data"])
            return bytes(buf)

        return _decode(asyncio.run(run()))


class ElevenLabs(Engine):
    id = "elevenlabs"
    name = "ElevenLabs (cloud, needs key)"
    kind = "cloud"
    maker = "ElevenLabs"
    params = "n/a"
    license = "Commercial API"
    arch = "Proprietary"
    note = ("Set ELEVENLABS_API_KEY in .env. ELEVENLABS_MODEL defaults to eleven_v3; "
            "ELEVENLABS_VOICE_ID picks the voice (falls back to the premade George voice). Tested 17 Nov 2026.")
    in_matrix = False
    voices = {lang: [V("env", "Voice from .env", "?")] for lang in ("en", "hi", "mr")}

    def available(self):
        ok = bool(os.environ.get("ELEVENLABS_API_KEY"))
        return ok, "" if ok else "Add ELEVENLABS_API_KEY to .env"

    def _synth(self, text, lang, voice, speed):
        vid = os.environ.get("ELEVENLABS_VOICE_ID", "")
        if len(vid) != 20:          # voice IDs are 20 chars; fall back to the premade "George" voice
            vid = "JBFqnCBsd6RMkjVDRZzb"
        raw = _post_json(
            f"https://api.elevenlabs.io/v1/text-to-speech/{vid}?output_format=pcm_24000",
            {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
            {"text": text, "model_id": os.environ.get("ELEVENLABS_MODEL", "eleven_v3"),
             "voice_settings": {"speed": min(1.2, max(0.7, speed))}})
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, 24000


class Sarvam(Engine):
    id = "sarvam"
    name = "Sarvam Bulbul (cloud, needs key)"
    kind = "cloud"
    maker = "Sarvam AI"
    params = "n/a"
    license = "Commercial API"
    arch = "Proprietary, Indian-language first"
    note = ("Set SARVAM_API_KEY in .env. SARVAM_MODEL defaults to bulbul:v3, SARVAM_SPEAKER to shubh. "
            "Tested 17 Nov 2026 with Hindi and Marathi.")
    in_matrix = False
    CODE = {"en": "en-IN", "hi": "hi-IN", "mr": "mr-IN"}
    voices = {lang: [V("env", "Speaker from .env", "?")] for lang in ("en", "hi", "mr")}

    def available(self):
        ok = bool(os.environ.get("SARVAM_API_KEY"))
        return ok, "" if ok else "Add SARVAM_API_KEY to .env"

    def _synth(self, text, lang, voice, speed):
        raw = _post_json(
            "https://api.sarvam.ai/text-to-speech",
            {"api-subscription-key": os.environ["SARVAM_API_KEY"]},
            {"text": text, "language_code": self.CODE[lang],
             "model": os.environ.get("SARVAM_MODEL", "bulbul:v3"),
             "speaker": os.environ.get("SARVAM_SPEAKER", "shubh"),
             "pace": min(2.0, max(0.5, speed)), "speech_sample_rate": 24000})
        return _decode(base64.b64decode(json.loads(raw)["audios"][0]))


class IndicParler(Engine):
    id = "indic-parler"
    name = "Indic Parler-TTS (needs HF access + PyTorch)"
    maker = "AI4Bharat"
    params = "0.9B"
    license = "Apache-2.0 (gated download)"
    arch = "Parler-TTS: T5 text encoder + audio LM over DAC codes, style set by a text description"
    note = ("Best open Hindi/Marathi expressive model, but gated: accept the terms on its Hugging Face "
            "page, set HF_TOKEN in .env, then `pip install torch parler-tts`. Slow on CPU (many times "
            "slower than real time). Adapter written from the model card, not yet run.")
    in_matrix = False
    voices = {
        "en": [V("Mary", "Mary", "F"), V("Thoma", "Thoma", "M")],
        "hi": [V("Divya", "Divya", "F"), V("Rohit", "Rohit", "M")],
        "mr": [V("Sunita", "Sunita", "F"), V("Sanjay", "Sanjay", "M")],
    }

    def available(self):
        try:
            import parler_tts  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            return False, "pip install torch parler-tts, accept the model terms, set HF_TOKEN"
        if not (os.environ.get("HF_TOKEN") or (Path.home() / ".cache/huggingface/token").exists()):
            return False, "Set HF_TOKEN in .env after accepting the model terms"
        return True, ""

    def _load(self):
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        repo = "ai4bharat/indic-parler-tts"
        self._m = ParlerTTSForConditionalGeneration.from_pretrained(repo)
        self._ptok = AutoTokenizer.from_pretrained(repo)
        self._dtok = AutoTokenizer.from_pretrained(self._m.config.text_encoder._name_or_path)

    def _synth(self, text, lang, voice, speed):
        pace = "slow" if speed < 0.9 else "fast" if speed > 1.15 else "moderate"
        desc = (f"{voice} speaks at a {pace} pace with a clear, expressive tone in a close-sounding "
                f"recording with very high quality and no background noise.")
        d = self._dtok(desc, return_tensors="pt")
        p = self._ptok(text, return_tensors="pt")
        out = self._m.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                               prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
        return out.cpu().numpy().squeeze(), self._m.config.sampling_rate


# --------------------------------------------------------------------------
_REGISTRY: dict[str, Engine] = {}


def engines() -> dict[str, Engine]:
    if not _REGISTRY:
        for e in (Piper(), Kokoro("fp32"), Supertonic(), MMS(), Kitten(), EdgeNeural(),
                  Kokoro("int8"), SupertonicInt8(), ElevenLabs(), Sarvam(), IndicParler()):
            _REGISTRY[e.id] = e
    return _REGISTRY


def get_engine(eid: str) -> Engine:
    return engines()[eid]


# --------------------------------------------------------------------------
# PyTorch engines (run by proto/worker.py under their own venv)
# --------------------------------------------------------------------------
class IndicTTS(Engine):
    """AI4Bharat Indic-TTS: FastPitch + HiFi-GAN via Coqui TTS (the deck's choice).

    variant "full": the release checkpoints as shipped (training state included)
    variant "slim": same weights with optimizer state removed (proto/slim_indic.py)
    """
    maker = "AI4Bharat (IIT Madras)"
    params = "56M FastPitch + 14M HiFi-GAN per language"
    license = "MIT"
    arch = "FastPitch + HiFi-GAN, character input, 22 kHz, PyTorch (Coqui TTS)"
    voices = {"hi": [V("female", "Female", "F"), V("male", "Male", "M")],
              "mr": [V("female", "Female", "F"), V("male", "Male", "M")]}

    def __init__(self, variant: str = "slim"):
        super().__init__()
        self.variant = variant
        if variant == "slim":
            self.id, self.name = "indic-tts", "Indic-TTS (AI4Bharat)"
            self.note = ("The deck's choice for Hindi/Marathi. Two voices per language, trained on studio "
                         "recordings. Checkpoints slimmed (training state removed); runs on PyTorch.")
        else:
            self.id, self.name = "indic-tts-full", "Indic-TTS (as released)"
            self.note = "Same model loaded from the 1.5 GB release files, to show the cost of shipping them unmodified."
            self.in_matrix = False

    def _paths(self, lang):
        d = MODELS / "indic-tts" / lang
        if self.variant == "slim":
            return d / "slim" / "fastpitch.pth", d / "slim" / "fastpitch.json", d / "slim" / "hifigan.pth", d / "hifigan" / "config.json", d / "fastpitch" / "speakers.pth"
        return d / "fastpitch" / "best_model.pth", d / "slim" / "fastpitch.json", d / "hifigan" / "best_model.pth", d / "hifigan" / "config.json", d / "fastpitch" / "speakers.pth"

    def available(self):
        py = RIG_HOME / "venv_indic" / "Scripts" / "python.exe"
        missing = [l for l in ("hi", "mr") if not self._paths(l)[0].exists()]
        if not py.exists():
            return False, "Run setup_models.py (pkg-indic)"
        if len(missing) == 2:
            return False, "Run setup_models.py (indic-tts-hi / -mr, then slimming)"
        return True, ""

    def disk_mb(self):
        out = {}
        for l in ("hi", "mr"):
            fp, _, voc, _, _ = self._paths(l)
            if fp.exists():
                out[l] = dir_mb(fp, voc)
        return out

    def _load(self):
        self._s = {}

    def _synth(self, text, lang, voice, speed):
        import torch
        torch.set_num_threads(THREADS)
        if lang not in self._s:
            from TTS.utils.synthesizer import Synthesizer
            fp, fpc, voc, vocc, spk = self._paths(lang)
            self._s[lang] = Synthesizer(tts_checkpoint=str(fp), tts_config_path=str(fpc), tts_speakers_file=str(spk),
                                        vocoder_checkpoint=str(voc), vocoder_config=str(vocc), use_cuda=False)
        syn = self._s[lang]
        syn.tts_model.length_scale = 1.0 / max(0.5, speed)
        with torch.inference_mode():
            wav = syn.tts(text, speaker_name=voice, split_sentences=False)
        return np.asarray(wav, dtype=np.float32), syn.output_sample_rate


class ParlerMini(Engine):
    """Parler-TTS mini v1 (the deck's English choice). Voice set by a text description."""
    id = "parler"
    name = "Parler-TTS mini (English)"
    maker = "Hugging Face"
    params = "880M"
    license = "Apache-2.0"
    arch = "T5 text encoder + decoder over DAC audio codes, 44 kHz, PyTorch"
    note = "The deck's choice for English. Style is described in words. Large for a phone; measured to see if it fits."
    voices = {"en": [V("Jon", "Jon", "M"), V("Lea", "Lea", "F"), V("Gary", "Gary", "M"), V("Laura", "Laura", "F")]}

    def available(self):
        if not (RIG_HOME / "venv_parler" / "Scripts" / "python.exe").exists():
            return False, "Run setup_models.py (pkg-parler)"
        ok = (MODELS / "parler-tts-mini-v1" / "model.safetensors").exists()
        return ok, "" if ok else "Run setup_models.py (parler-mini)"

    def disk_mb(self):
        return {"en": dir_mb(MODELS / "parler-tts-mini-v1")}

    def _load(self):
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        torch.set_num_threads(THREADS)
        d = str(MODELS / "parler-tts-mini-v1")
        self._m = ParlerTTSForConditionalGeneration.from_pretrained(d, torch_dtype=torch.float32).eval()
        self._tok = AutoTokenizer.from_pretrained(d)

    def _synth(self, text, lang, voice, speed):
        import torch
        pace = "slowly" if speed < 0.9 else "quickly" if speed > 1.15 else "at a moderate pace"
        desc = f"{voice} speaks {pace} in a clear, friendly voice. Very clear audio with no background noise."
        d = self._tok(desc, return_tensors="pt")
        p = self._tok(text, return_tensors="pt")
        with torch.inference_mode():
            out = self._m.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                   prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask,
                                   do_sample=True, temperature=0.8)
        return out.cpu().numpy().squeeze(), self._m.config.sampling_rate


def _register_torch():
    for e in (IndicTTS("slim"), IndicTTS("full"), ParlerMini()):
        _REGISTRY.setdefault(e.id, e)


_orig_engines = engines


def engines() -> dict[str, Engine]:   # noqa: F811
    _orig_engines()
    _register_torch()
    return _REGISTRY
