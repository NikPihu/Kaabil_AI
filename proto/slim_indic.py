"""Make inference-only copies of the AI4Bharat Indic-TTS checkpoints.

The release .pth files carry optimizer state and the HiFi-GAN discriminator
(training-only). This keeps the model weights, drops the rest, and writes a
FastPitch config whose speakers_file points at the local file.
Run with the Indic venv:  venv_indic/Scripts/python.exe proto/slim_indic.py
"""
import json
import os
import sys
from pathlib import Path

import torch

MODELS = Path(os.environ.get("VOICE_RIG_HOME", Path.home() / "voice_rig")) / "models" / "indic-tts"

for lang in sys.argv[1:] or ["hi", "mr"]:
    d = MODELS / lang
    if not (d / "fastpitch" / "best_model.pth").exists():
        print(lang, "not downloaded"); continue
    out = d / "slim"
    out.mkdir(exist_ok=True)
    cfg = json.loads((d / "fastpitch" / "config.json").read_text(encoding="utf-8"))
    spk = str(d / "fastpitch" / "speakers.pth")
    cfg["speakers_file"] = spk
    cfg["model_args"]["speakers_file"] = spk
    (out / "fastpitch.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    for name, src in (("fastpitch", d / "fastpitch" / "best_model.pth"), ("hifigan", d / "hifigan" / "best_model.pth")):
        dst = out / f"{name}.pth"
        if dst.exists():
            continue
        ck = torch.load(src, map_location="cpu", weights_only=False)
        torch.save({"model": ck["model"], "config": ck.get("config")}, dst)
        print(lang, name, f"{src.stat().st_size >> 20} MB -> {dst.stat().st_size >> 20} MB")
