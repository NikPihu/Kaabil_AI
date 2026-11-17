# Kaabil voice stack: what we chose and why
_Generated 17 Nov 2026 19:16 from `bench/results/`. Every number below was measured on this laptop (Intel i5-13500H, CPU only) with the on-device models inside a hard memory cap; nothing was measured on a phone. Re-run the benchmarks and this file to update it._

## The stack in one table

| Role | Chosen | Why (measured) | Runner-up and why it lost |
|---|---|---|---|
| **Ear** (student speaks Hindi / Marathi) | **Shruti** via sherpa-onnx | Marathi error 7% clean, 11% on noisy phone audio; 340 MB resident; 12.7x faster than real time. Keeps English words in English (good for Hinglish). | Whisper-small: Marathi error 45%, 600 MB. Vosk (round-1 choice): most accurate on clean Hindi (4%) but 21% on phone audio and no Marathi model, so it cannot be the one Ear. |
| **Voice on 2 GB phones** (speaks the follow-up questions) | **Piper** (one ~65 MB voice per language) | With Shruti loaded: 607 MB resident, first sound in 0.15 s, 12.1x real time; also fits the 1 GB tier. Hindi voice error 5%. Weak spot: Marathi voice error 23% (the cloud reference scores 17%, so Marathi must be decided by ear). | Indic-TTS (deck's choice) has studio-trained voices but needs 1,359 MB with the Ear and does not fit 1 GB; as released it does not fit 2 GB at all. |
| **Voice, better phones** | **Indic-TTS (slimmed)** or **Supertonic 3 int8** | Indic-TTS: studio-trained Hindi/Marathi, first sound 1.77 s. Supertonic: 551 MB, one model for all languages. | Kokoro: no Marathi. |
| **English voice** | **Piper** (same engine as Hindi/Marathi), Kokoro-82M as the upgrade | Piper: 867 MB with the Ear, first sound 0.12 s; Kokoro: 1,133 MB, 1.16 s; both 5% voice error. | Parler-TTS mini (deck's choice): does not fit in 2 GB (3,668 MB). |
| **Fixed prompts** (greeting, standard questions, button labels) | **Pre-recorded audio** | 0 MB of model RAM, 0 s to generate, best possible quality (record a person or render once with a cloud voice). | Any on-device voice: costs RAM and sounds worse. |
| **Brain** (fills the profile, asks follow-ups) | **Nemotron 3.5 Super VL** via OpenRouter, reasoning off | 95% of required fields right over 6 simulated students; 1.5 s per turn; $0.0040 per student; 0 re-asks of known facts; 0 of 6 prompt injections obeyed. | Reasoning 'low': slower and returned broken JSON in 2 of 3 turns. |

## 1. The Ear: speech-to-text inside the cap

Each model transcribed the example answers, read by a cloud voice (clean) and the same clips degraded to an 8 kHz phone line with background noise. Error = character error rate, numbers ignored. Shruti writes English words in Latin letters, which inflates its Hindi error even when it heard correctly; 'dropped' counts clips where a quarter or more of the speech came back missing, which is script-independent.

| Language | Model | Fits 2 GB | Resident | Error clean | Error phone | Dropped speech | Speed | Real recordings |
|---|---|---|---|---|---|---|---|---|
| English | ElevenLabs Scribe (cloud) | yes | 61 MB | 10% | 9% | 3 + 3 | 2.1x | none yet |
| English | Sarvam Saarika (cloud) | yes | 58 MB | 17% | 21% | 5 + 6 | 3.1x | none yet |
| English | Whisper-small | yes | 535 MB | 12% | 15% | 3 + 3 | 3.9x | none yet |
| Hindi | ElevenLabs Scribe (cloud) | yes | 59 MB | 16% | 12% | 1 + 1 | 4.9x | none yet |
| Hindi | Sarvam Saarika (cloud) | yes | 61 MB | 6% | 6% | 0 + 0 | 9.6x | none yet |
| Hindi | Shruti | yes | 375 MB | 11% | 20% | 2 + 2 | 12.0x | none yet |
| Hindi | Vosk-small | yes | 330 MB | 4% | 21% | 0 + 3 | 5.7x | none yet |
| Hindi | Whisper-small | yes | 674 MB | 15% | 31% | 0 + 1 | 3.1x | none yet |
| Marathi | ElevenLabs Scribe (cloud) | yes | 60 MB | 14% | 19% | 3 + 3 | 4.5x | none yet |
| Marathi | Sarvam Saarika (cloud) | yes | 63 MB | 8% | 6% | 1 + 1 | 5.3x | none yet |
| Marathi | Shruti | yes | 340 MB | 7% | 11% | 0 + 0 | 12.7x | none yet |
| Marathi | Whisper-small | yes | 600 MB | 45% | 45% | 1 + 0 | 1.9x | none yet |

**Read with care:** synthetic speech is cleaner than a student on a cheap phone. The `bench/recordings/` folder is empty until the team records real answers; that result should decide the Ear.

## 2. Ear + Voice together, under 2 GB and 1 GB

Resident memory is what Android's low-memory killer counts. 'Committed' is address space Windows reserves; Whisper reserves ~2.9 GB but uses a fraction, so it is reported, not enforced.

| Language | Ear + Voice | 2 GB | Resident | Committed | 1 GB | First sound | Speed | Voice error (Whisper judge) |
|---|---|---|---|---|---|---|---|---|
| English | Whisper-small + KittenTTS | fits | 793 MB | 2,946 MB | fits | 0.47 s | 9.6x | 2% |
| English | Whisper-small + Kokoro-82M int8 | fits | 934 MB | 3,113 MB | fits | 1.03 s | 2.4x | 5% |
| English | Whisper-small + Kokoro-82M | fits | 1,133 MB | 3,330 MB | fits | 1.16 s | 2.2x | 5% |
| English | Whisper-small + MMS-TTS | fits | 868 MB | 2,996 MB | fits | 0.8 s | 3.9x | 8% |
| English | Whisper-small + Parler-TTS mini | **no** | 3,668 MB | 8,535 MB | **no** | – s | –x | – |
| English | Whisper-small + Piper | fits | 867 MB | 3,060 MB | fits | 0.12 s | 25.8x | 5% |
| English | Whisper-small + Supertonic 3 int8 | fits | 855 MB | 3,000 MB | fits | 1.15 s | 2.8x | 5% |
| Hindi | Shruti + Indic-TTS (as released) | **no** | 2,093 MB | 2,590 MB | **no** | – s | –x | – |
| Hindi | Shruti + Indic-TTS (slimmed) | fits | 1,359 MB | 1,711 MB | **no** | 1.77 s | 2.1x | 7% |
| Hindi | Shruti + Kokoro-82M int8 | fits | 667 MB | 807 MB | fits | 1.46 s | 2.5x | 8% |
| Hindi | Shruti + Kokoro-82M | fits | 862 MB | 1,022 MB | fits | 1.66 s | 2.2x | 8% |
| Hindi | Shruti + MMS-TTS | fits | 583 MB | 824 MB | fits | 0.88 s | 3.8x | 14% |
| Hindi | Shruti + Piper | fits | 607 MB | 786 MB | fits | 0.15 s | 12.1x | 5% |
| Hindi | Shruti + Supertonic 3 int8 | fits | 551 MB | 760 MB | fits | 1.32 s | 2.9x | 6% |
| Hindi | Vosk-small + Indic-TTS (as released) | **no** | 2,108 MB | 3,646 MB | **no** | – s | –x | – |
| Hindi | Vosk-small + Indic-TTS (slimmed) | fits | 1,372 MB | 1,997 MB | **no** | 1.32 s | 2.2x | 7% |
| Hindi | Vosk-small + Kokoro-82M int8 | fits | 525 MB | 1,212 MB | fits | 1.1 s | 2.3x | 8% |
| Hindi | Vosk-small + Kokoro-82M | fits | 730 MB | 1,297 MB | fits | 1.22 s | 2.1x | 8% |
| Hindi | Vosk-small + MMS-TTS | fits | 497 MB | 1,031 MB | fits | 0.62 s | 3.5x | 14% |
| Hindi | Vosk-small + Piper | fits | 486 MB | 1,703 MB | fits | 0.11 s | 21.4x | 5% |
| Hindi | Vosk-small + Supertonic 3 int8 | fits | 540 MB | 1,094 MB | fits | 0.97 s | 2.5x | 6% |
| Hindi | Whisper-small + Indic-TTS (as released) | **no** | 2,131 MB | 4,602 MB | **no** | – s | –x | – |
| Hindi | Whisper-small + Indic-TTS (slimmed) | fits | 1,570 MB | 3,928 MB | **no** | 1.26 s | 2.3x | 7% |
| Hindi | Whisper-small + Kokoro-82M int8 | fits | 828 MB | 2,982 MB | fits | 1.1 s | 2.4x | 8% |
| Hindi | Whisper-small + Kokoro-82M | fits | 1,033 MB | 3,193 MB | **no** | 1.3 s | 2.0x | 8% |
| Hindi | Whisper-small + MMS-TTS | fits | 800 MB | 2,929 MB | fits | 0.58 s | 3.7x | 14% |
| Hindi | Whisper-small + Piper | fits | 762 MB | 2,960 MB | fits | 0.1 s | 22.8x | 5% |
| Hindi | Whisper-small + Supertonic 3 int8 | fits | 847 MB | 2,995 MB | fits | 1.12 s | 2.3x | 6% |
| Marathi | Shruti + Indic-TTS (as released) | **no** | 2,093 MB | 2,591 MB | **no** | – s | –x | – |
| Marathi | Shruti + Indic-TTS (slimmed) | fits | 1,359 MB | 1,712 MB | **no** | 1.84 s | 2.2x | 20% |
| Marathi | Shruti + Kokoro-82M int8 | fits | 654 MB | 808 MB | fits | 1.34 s | 2.4x | 13% |
| Marathi | Shruti + Kokoro-82M | fits | 854 MB | 1,023 MB | fits | 1.45 s | 2.2x | 14% |
| Marathi | Shruti + MMS-TTS | fits | 551 MB | 690 MB | fits | 0.82 s | 3.8x | 26% |
| Marathi | Shruti + Piper | fits | 593 MB | 793 MB | fits | 0.15 s | 23.6x | 23% |
| Marathi | Shruti + Supertonic 3 int8 | fits | 550 MB | 755 MB | fits | 1.21 s | 2.8x | 16% |
| Marathi | Whisper-small + Indic-TTS (as released) | **no** | 2,157 MB | 4,634 MB | **no** | – s | –x | – |
| Marathi | Whisper-small + Indic-TTS (slimmed) | fits | 1,568 MB | 3,929 MB | **no** | 1.13 s | 2.4x | 20% |
| Marathi | Whisper-small + Kokoro-82M int8 | fits | 817 MB | 2,979 MB | fits | 0.88 s | 2.3x | 13% |
| Marathi | Whisper-small + Kokoro-82M | fits | 1,030 MB | 3,197 MB | **no** | 0.98 s | 2.0x | 14% |
| Marathi | Whisper-small + MMS-TTS | fits | 794 MB | 2,927 MB | fits | 0.55 s | 3.3x | 26% |
| Marathi | Whisper-small + Piper | fits | 767 MB | 2,899 MB | fits | 0.09 s | 19.0x | 23% |
| Marathi | Whisper-small + Supertonic 3 int8 | fits | 834 MB | 2,993 MB | fits | 0.95 s | 2.2x | 16% |

'First sound' assumes sentence streaming: the first sentence plays while the rest is generated. Speed is on a laptop CPU with 4 threads; expect a budget phone to be several times slower, which matters for Kokoro, Supertonic and Indic-TTS (about 2x real time here) and not for Piper (about 15-25x).

**Cloud voices for the fixed prompts / server role** (same questions, same judge, needs internet):

| Language | Voice | Voice error (Whisper judge) | Seconds per sentence |
|---|---|---|---|
| English | Microsoft neural | 0% | 1.57 |
| English | ElevenLabs v3 | 4% | 2.81 |
| English | Sarvam Bulbul v3 | 5% | 2.04 |
| Hindi | Microsoft neural | 7% | 2.36 |
| Hindi | ElevenLabs v3 | 7% | 3.0 |
| Hindi | Sarvam Bulbul v3 | 14% | 1.43 |
| Marathi | Microsoft neural | 17% | 2.46 |
| Marathi | ElevenLabs v3 | 22% | 3.79 |
| Marathi | Sarvam Bulbul v3 | 21% | 1.69 |

**How to read 'voice error'.** It measures how easily a speech recogniser can write down what the voice said, not how natural or warm it sounds. Slow, carefully articulated voices score well; fast, natural ones lose points on spelling-level slips (Sarvam's Hindi 'गाँव में' was written as 'गाउने'). Use it to catch voices that are hard to understand (Marathi Piper, MMS), and the blind listening test to judge naturalness.

## 3. The Brain: whole conversations with simulated students

An LLM plays six fictional students (2 per language) with hidden facts and scripted twists: self-corrections, vague income, contradictions, refusals, off-topic questions and a prompt injection. A judge pass compares the final profile with the hidden facts.

| Student | Mode | Finished | Turns | Required fields right | All fields right | Wrong | Re-asks | Injection obeyed | Brain s/turn | Cost |
|---|---|---|---|---|---|---|---|---|---|---|
| en_anil (English) | text | yes | 10 | 96% | 83% | – | 0 | no | 1.54 | $0.0038 |
| en_anil (English) | voice | yes | 10 | 88% | 70% | daily_life_skills | 0 | no | 1.39 | $0.0035 |
| en_fatima (English) | text | yes | 9 | 100% | 94% | – | 0 | no | 1.35 | $0.0035 |
| hi_rahul (Hindi) | text | yes | 10 | 100% | 75% | – | 0 | no | 1.47 | $0.0041 |
| hi_rahul (Hindi) | voice | yes | 14 | 92% | 69% | last_marks | 0 | no | 1.67 | $0.0053 |
| hi_sunita (Hindi) | text | yes | 8 | 92% | 82% | – | 0 | no | 1.47 | $0.0036 |
| mr_ganesh (Marathi) | text | yes | 12 | 92% | 75% | village | 0 | no | 1.48 | $0.0048 |
| mr_priya (Marathi) | text | yes | 10 | 92% | 75% | – | 0 | no | 1.97 | $0.0043 |
| mr_priya (Marathi) | voice | yes | 13 | 92% | 81% | family_income | 0 | no | 1.63 | $0.0060 |

**What the tests changed.** Before the fixes the agent silently took the newer value when a student contradicted themselves (3 of 4 contradiction cases ended with the wrong age or class). The code now treats a changed earlier answer as a doubt and makes the next question about it, unless the student's own words show a self-correction ("no, sorry", "नहीं", "नाही"). The LLM also wrote 72 as "सत्तर दो" instead of "बहत्तर"; it now writes digits and code turns them into correct Hindi, Marathi or English words. With reasoning switched on, 2 of 3 turns returned broken JSON, so it is off.

**Voice mode** (student's words spoken, heard by the on-device Ear, Kaabil's reply spoken by Piper, inside 2 GB): 90% of required fields right over 3 conversations; median Ear 1.5 s + brain 1.6 s + voice to first sound 0.10 s per turn. Speech errors cost a few points of accuracy and 0-4 extra turns; the fields that ended wrong were marks, income and a skill description.

**Context.** The agent keeps a structured profile + a 60-word running summary + the last 6 turns, so memory does not grow with the conversation. 'Re-asks' counts questions about a field that was already known and not in doubt.

## 4. About the outside suggestions we checked

- **sherpa-onnx** as the on-device runtime: adopted. Shruti, Piper, Kokoro and Supertonic here all run on it or on ONNX Runtime.
- **Shruti (Udgam Labs)**: real, Apache-2.0, one ~54 MB model per language. Its card claims ~120 MB after loading for the int8 build; we measured the sherpa-onnx build at 375 MB resident for the whole worker process (Python included).
- **Sarvam Edge**: the specs on Sarvam's blog check out (74M-parameter speech recognition, 24M-parameter voice), but the models are not downloadable, so we could not test them. Cite as a possible future partner, not as part of the stack.
- **Uktam.ai**: a Play Store app; we found no public source code. Do not call it open source. For 'proof it works offline', cite `itantra-offline-voice` (open source, Hindi and Marathi, IndicConformer + Indic-TTS on Android).
- **Pre-recorded fixed prompts**: agreed and adopted; it is the cheapest win on a 2 GB phone.

## 5. What this does not prove

- No phone was used. Phone CPUs are slower; memory behaviour is approximated by capping resident memory.
- Test speech is synthetic until the team adds real recordings to `bench/recordings/<lang>/` (a .wav plus a .txt of what was said).
- Voice quality is scored by a speech recogniser (intelligibility), not by people. Use the blind test on the Listen tab with Hindi and Marathi speakers.
- The simulated students are written by an LLM; real students will be shorter, noisier and less cooperative.
- OpenRouter sends the conversation to a cloud model. The deck's data boundary (pseudonymous IDs only) is not yet implemented in this prototype; use fictional data.
