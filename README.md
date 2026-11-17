# Kaabil voice prototype and test rig

A working prototype of Kaabil's voice profile loop, plus the benchmarks behind the
model choices:

**student speaks or types → on-device Ear (speech-to-text) → Brain (Nemotron via
OpenRouter) fills the profile and asks the next question → on-device Voice speaks it**

The Ear and the Voice run inside a shared **2 GB memory cap**, like on a budget phone.

## 1. Set up (once)

1. Open [.env](.env) and paste your key after `OPENROUTER_API_KEY=`. Every other key there is optional, and each line says what it unlocks and where to get it.
2. Fetch every model (about 12 GB; safe to re-run, skips what is there, prints a checklist):

```bash
C:/Users/nikks/voice_rig/venv/Scripts/python.exe -X utf8 setup_models.py
```

3. Check without downloading:

```bash
C:/Users/nikks/voice_rig/venv/Scripts/python.exe -X utf8 setup_models.py --check
```

Everything heavy lives in `C:\Users\nikks\voice_rig\` (outside OneDrive): `venv` (main), `venv_indic` and `venv_parler` (PyTorch models), `models\`, `hf_cache\`.

## 2. Run the app

Double-click `run.bat`, or:

```bash
C:/Users/nikks/voice_rig/venv/Scripts/python.exe -X utf8 rig/server.py
```

Open http://127.0.0.1:8765.

| Tab | What it is for |
|---|---|
| **Conversation** | The prototype. Pick language, Ear, Voice, RAM cap (2 GB / 1 GB / 768 MB), threads and an optional CPU cap, press **Start**. Talk with the mic, type, or click example chips: the label sends the text, 🔊 sends a spoken clip through the Ear (tick "noisy phone audio" for an 8 kHz + noise copy). Dashed chips are the hard cases; hover to see what each one tests. The profile card on the right fills live: green filled, amber doubt, red missing. Each reply shows how long the student waited and where the time went. **Echo mode** skips the LLM to test only Ear + Voice. |
| **Judges brief** | Which model plays which role and why, with the measured numbers. Generated from the benchmark results. |
| **Live bench** | Type any text and hear it in several voices side by side, with timing. The preset buttons are Kaabil's own follow-up questions (`bench/questions.json`). **Dialogue mix** gives each `A:` / `B:` / `C:` line its own voice. |

## 3. Each model: role, setup, how it is tested

| Role | Model | Set up by | Runs in | Tested by |
|---|---|---|---|---|
| Ear hi, mr | **Shruti** (Udgam Labs, sherpa-onnx) | `setup_models.py` (shruti-hi, shruti-mr) | main venv | `bench/run_ear.py`, Conversation tab |
| Ear hi, mr, en | Whisper-small int8 | `setup_models.py` (whisper-small) | main venv | same |
| Ear hi | Vosk small (round-1 choice) | `setup_models.py` (vosk-small-hi) | main venv | same |
| Voice hi, mr, en | **Piper** | already present | main venv | `bench/run_pairs.py`, Conversation tab |
| Voice hi, mr | **Indic-TTS** (AI4Bharat, deck's choice) | `setup_models.py` (indic-tts-hi/mr); slimmed automatically by `proto/slim_indic.py` | `venv_indic` (PyTorch) | same |
| Voice hi, en | Kokoro-82M, Supertonic 3 int8 | already present | main venv | same |
| Voice en | Parler-TTS mini (deck's choice) | `setup_models.py` (parler-mini) | `venv_parler` (PyTorch) | same |
| Brain | Nemotron 3.5 Super VL | `OPENROUTER_API_KEY` in `.env` | OpenRouter (cloud) | `bench/run_dialogues.py`, Conversation tab |

**How the 2 GB cap works.** The Ear and the Voice each run as a worker process
(`proto/worker.py`). `proto/voicebox.py` samples the resident memory of both
every 100 ms and kills them if the sum passes the cap, like Android's
low-memory killer, so a model that does not fit fails visibly. Committed memory
is reported too; Windows counts reservations that Android would not (Whisper
reserves 2.9 GB but uses a fraction). A Windows Job Object (`proto/memcap.py`)
adds a hard commit backstop and the optional CPU cap.

## 4. Benchmarks

Run in this order; each writes to `bench/results/`. Then build the brief.

```bash
C:/Users/nikks/voice_rig/venv/Scripts/python.exe -X utf8 bench/run_ear.py
```

```bash
C:/Users/nikks/voice_rig/venv/Scripts/python.exe -X utf8 bench/run_pairs.py
```

```bash
C:/Users/nikks/voice_rig/venv/Scripts/python.exe -X utf8 bench/run_dialogues.py
```

```bash
C:/Users/nikks/voice_rig/venv/Scripts/python.exe -X utf8 bench/run_dialogues.py --voice --personas hi_rahul,mr_priya,en_anil
```

```bash
C:/Users/nikks/voice_rig/venv/Scripts/python.exe -X utf8 bench/make_brief.py
```

| Benchmark | Measures |
|---|---|
| `run_ear.py` | Each Ear inside the cap: error on clean and noisy-phone example clips and on your real recordings, dropped speech, speed, memory. |
| `run_pairs.py` | Every Ear × Voice pair per language at 2 GB and 1 GB: fits?, resident memory, time to first sound, speed; then Whisper large judges how intelligible each voice is. |
| `run_dialogues.py` | Whole conversations with 6 simulated students (`bench/personas.json`) with twists (corrections, contradictions, vague income, refusal, off-topic, prompt injection). Scores fields right, re-asks of known facts (context loss), injections obeyed, seconds per turn, cost. `--voice` runs the student's line through a voice and the Ear, so speech errors compound. |

**Most valuable thing to add:** real recordings. Put `name.wav` (any rate, mono)
and `name.txt` (exactly what was said) in `bench/recordings/hi/`, `mr/` or `en/`,
then re-run `run_ear.py`. Synthetic speech flatters every Ear.

## 5. Files

| Path | What |
|---|---|
| `proto/agent.py` | Profile Agent: the LLM proposes, code decides (doubts, contradictions, locked fields, read-back). |
| `proto/schema.json` | The profile fields. Edit to change what is asked. |
| `proto/examples.json` | Example answers and hard cases per language (the chips). |
| `proto/numwords.py` | Digits → correct Hindi / Marathi / English words before speaking. |
| `proto/stt.py`, `rig/engines.py` | Ear and Voice adapters. |
| `proto/voicebox.py`, `proto/worker.py`, `proto/memcap.py` | The memory-capped Ear + Voice. |
| `proto/llm.py` | OpenRouter client (streams; records first-token time, tokens, cost). |
| `proto/convo_api.py`, `web/talk.js` | Conversation tab. |
| `outputs/convos/<session>/` | Every conversation: audio of each turn and `session.json` with timings and the profile. |
