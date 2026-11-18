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
| **Conversation** | The prototype. Pick language, Ear, Voice, RAM cap (2 GB / 1 GB / 768 MB), threads and an optional CPU cap, press **Start**. Talk with the mic, type, or click example chips: the label sends the text, **audio** sends a spoken clip through the Ear (tick "send audio as a noisy phone line" for an 8 kHz + noise copy). Once 6+ fields are filled, **Plan my path** opens the Career tab for this student. Dashed chips are the hard cases; hover to see what each one tests. The profile card on the right fills live: green filled, amber doubt, red missing. Each reply shows how long the student waited and where the time went. **Echo mode** skips the LLM to test only Ear + Voice. |
| **Career recommendations** | The Career Agent. Pick a finished conversation (or press **Plan my path** on the Conversation tab's profile card) or one of the 6 demo students, press **Find paths**. Left: what Kaabil knows about the student and how 34 paths were narrowed down (which were ruled out and why). Right: 5–8 path cards with fit, why it fits (quoting the student), watch-outs, duration, cost, when earning starts and where you study. **Open full path** shows the route steps, personalised next steps, what you will study, what you can become, pay and sources. **This doesn't fit** re-ranks without it; **I'm interested** is saved. Filters: Non-STEM, STEM, Government, Own business, Earn soon. **Listen** reads the summary aloud in the student's language. A **Data review** panel approves government-PDF extractions. |
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

```bash
C:/Users/nikks/voice_rig/venv/Scripts/python.exe -X utf8 bench/run_career.py
```

`run_career.py` checks the Career Agent on the 6 demo students by code: no money or percentages in the agent's own text (facts come only from `career/careers.json`), every route passes the entry rules, at least 2 non-STEM paths and 2 sectors, quotes really come from the profile, seconds per suggestion.

## 4b. Career Agent: how it decides

1. **Knowledge base** — `career/careers.json`: 34 paths in 14 sectors (13 STEM, 21 non-STEM: trades, healthcare, agriculture, commerce, government jobs, teaching, creative, media, hospitality, sports, law, own business). Each path has routes by entry level (after Class 8, 10, 12 by stream, or a degree) with where you study (village, district, city, distance, online), duration, cost, steps, what you learn, what you can become, pay (marked *official* or *estimate*) and sources with how each was checked.
2. **Rules** — `career/rules.py` reads the profile into a situation (class, stream, age, marks, travel limit, family income) and checks each route: *now*, *after your current class*, *needs a step first*, or *ruled out* (wrong stream, past the age limit). It adds warnings: moving to a city when the student can't travel, costly routes when money is tight, marks below a minimum. No LLM here.
3. **Shortlist** — `career/career_agent.py` scores eligible paths against the student's own words (interests, skills from daily life, aspiration) and keeps non-STEM and STEM options in the mix.
4. **LLM** picks and orders 5–8, explains *why* quoting the student, adds watch-outs and three personal next steps, and a spoken summary. Code then removes any money it wrote, keeps the rules' route if the LLM picked a blocked one, and makes sure a path the student named and at least 2 non-STEM paths are present.
5. **Government data** — `career/ingest.py` downloads official curriculum PDFs (DGT/BharatSkills, NQR qualification files), extracts the syllabus with the LLM, keeps only items whose quote is really in the PDF, and stages them for a person to approve in the Data review panel. Approved modules replace the typical syllabus (`learn_source: official`).

```bash
C:/Users/nikks/voice_rig/venv/Scripts/python.exe -X utf8 career/ingest.py
```

To add a career: add an entry to `career/careers.json` (copy one of the same kind), with official source URLs; add an `ingest` URL if a government curriculum PDF exists.

**Most valuable thing to add:** real recordings. Put `name.wav` (any rate, mono)
and `name.txt` (exactly what was said) in `bench/recordings/hi/`, `mr/` or `en/`,
then re-run `run_ear.py`. Synthetic speech flatters every Ear.

## 5. Files

| Path | What |
|---|---|
| `proto/agent.py` | Profile Agent: the LLM proposes, code decides (doubts, contradictions, locked fields, read-back). |
| `career/` | Career Agent: `careers.json` (knowledge base), `rules.py` (entry rules), `career_agent.py` (shortlist + LLM explanation), `ingest.py` (government PDFs), `career_api.py` (endpoints), `staged/` (extractions awaiting review). |
| `web/career.js`, `web/career.css` | Career recommendations tab. |
| `proto/schema.json` | The profile fields. Edit to change what is asked. |
| `proto/examples.json` | Example answers and hard cases per language (the chips). |
| `proto/numwords.py` | Digits → correct Hindi / Marathi / English words before speaking. |
| `proto/stt.py`, `rig/engines.py` | Ear and Voice adapters. |
| `proto/voicebox.py`, `proto/worker.py`, `proto/memcap.py` | The memory-capped Ear + Voice. |
| `proto/llm.py` | OpenRouter client (streams; records first-token time, tokens, cost). |
| `proto/convo_api.py`, `web/talk.js` | Conversation tab. |
| `outputs/convos/<session>/` | Every conversation: audio of each turn and `session.json` with timings and the profile. |
