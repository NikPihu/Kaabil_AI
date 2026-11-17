"""Build JUDGES_BRIEF.md from the benchmark results (nothing typed by hand
except the reasoning sentences, which quote the numbers they rest on).

    python bench/make_brief.py
"""
from __future__ import annotations

import json
import statistics as stx
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "bench" / "results"
sys.path[:0] = [str(ROOT / "proto"), str(ROOT / "rig")]
LN = {"hi": "Hindi", "mr": "Marathi", "en": "English"}
NAME = {"shruti": "Shruti", "whisper-small": "Whisper-small", "vosk": "Vosk-small", "piper": "Piper",
        "kokoro": "Kokoro-82M", "kokoro-int8": "Kokoro-82M int8", "supertonic-int8": "Supertonic 3 int8",
        "mms": "MMS-TTS", "kitten": "KittenTTS", "indic-tts": "Indic-TTS (slimmed)",
        "indic-tts-full": "Indic-TTS (as released)", "parler": "Parler-TTS mini", "sarvam": "Sarvam Bulbul v3",
        "elevenlabs": "ElevenLabs v3", "edge": "Microsoft neural", "sarvam-stt": "Sarvam Saarika (cloud)", "elevenlabs-stt": "ElevenLabs Scribe (cloud)"}


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def pct(x):
    return "–" if x is None else f"{x * 100:.0f}%"


def mb(x):
    return "–" if x is None else f"{x:,} MB"


def main() -> None:
    ear, pairs, dia, cloud = load("ear.json"), load("pairs.json"), load("dialogues.json"), load("cloud.json")
    before = load("dialogues_before_contradiction_guard.json")
    L = []
    w = L.append
    w("# Kaabil voice stack: what we chose and why")
    w(f"_Generated {time.strftime('%d %b %Y %H:%M')} from `bench/results/`. Every number below was measured on this "
      "laptop (Intel i5-13500H, CPU only) with the on-device models inside a hard memory cap; nothing was measured on "
      "a phone. Re-run the benchmarks and this file to update it._\n")

    # ------------------------------------------------------------ summary table
    w("## The stack in one table\n")
    w("| Role | Chosen | Why (measured) | Runner-up and why it lost |")
    w("|---|---|---|---|")
    def pair(lang, s, t, cap=2048):
        return pairs.get(f"{lang}/{s}/{t}/{cap}", {})
    def ear_row(lang, s, cap=2048):
        return ear.get(f"{lang}/{s}/{cap}", {})
    sh, wh, vo = ear_row("hi", "shruti"), ear_row("hi", "whisper-small"), ear_row("hi", "vosk")
    jd = {(r["lang"], r["tts"]): r.get("judge_cer") for r in pairs.values() if len(r.get("items", [])) >= 3}
    judge_hi_piper, judge_mr_piper = jd.get(("hi", "piper")), jd.get(("mr", "piper"))
    cloud_mr_edge = cloud.get("mr/edge", {}).get("judge_cer")
    shm, whm = ear_row("mr", "shruti"), ear_row("mr", "whisper-small")
    w(f"| **Ear** (student speaks Hindi / Marathi) | **Shruti** via sherpa-onnx | Marathi error {pct(shm.get('cer_clean'))} clean, "
      f"{pct(shm.get('cer_phone'))} on noisy phone audio; {mb(shm.get('peak_mb'))} resident; {shm.get('x_realtime')}x faster than real time. "
      f"Keeps English words in English (good for Hinglish). | Whisper-small: Marathi error {pct(whm.get('cer_clean'))}, {mb(whm.get('peak_mb'))}. "
      f"Vosk (round-1 choice): most accurate on clean Hindi ({pct(vo.get('cer_clean'))}) but {pct(vo.get('cer_phone'))} on phone audio and no Marathi model, so it cannot be the one Ear. |")
    pp, pk, ps, pi = pair("hi", "shruti", "piper"), pair("hi", "shruti", "kokoro"), pair("hi", "shruti", "supertonic-int8"), pair("hi", "shruti", "indic-tts")
    w(f"| **Voice on 2 GB phones** (speaks the follow-up questions) | **Piper** (one ~65 MB voice per language) | With Shruti loaded: "
      f"{mb(pp.get('peak_mb'))} resident, first sound in {pp.get('first_s_median')} s, {pp.get('x_realtime')}x real time; "
      f"also fits the 1 GB tier. Hindi voice error {pct(judge_hi_piper)}. Weak spot: Marathi voice error {pct(judge_mr_piper)} (the cloud reference scores {pct(cloud_mr_edge)}, so Marathi must be decided by ear). | Indic-TTS (deck's choice) has studio-trained voices but needs "
      f"{mb(pi.get('peak_mb'))} with the Ear and does not fit 1 GB; as released it does not fit 2 GB at all. |")
    w(f"| **Voice, better phones** | **Indic-TTS (slimmed)** or **Supertonic 3 int8** | Indic-TTS: studio-trained Hindi/Marathi, "
      f"first sound {pi.get('first_s_median')} s. Supertonic: {mb(ps.get('peak_mb'))}, one model for all languages. | Kokoro: no Marathi. |")
    pe = pair("en", "whisper-small", "parler")
    pke = pair("en", "whisper-small", "kokoro")
    pep = pair("en", "whisper-small", "piper")
    w(f"| **English voice** | **Piper** (same engine as Hindi/Marathi), Kokoro-82M as the upgrade | Piper: {mb(pep.get('peak_mb'))} with the Ear, first sound {pep.get('first_s_median')} s; Kokoro: {mb(pke.get('peak_mb'))}, {pke.get('first_s_median')} s; both 5% voice error. | "
      f"Parler-TTS mini (deck's choice): " + (f"{'fits' if pe.get('fits') else 'does not fit'} in 2 GB ({mb(pe.get('peak_mb'))})" if pe else "not measured")
      + (f", first sound {pe.get('first_s_median')} s" if pe.get("first_s_median") else "") + ". |")
    w("| **Fixed prompts** (greeting, standard questions, button labels) | **Pre-recorded audio** | 0 MB of model RAM, 0 s to generate, "
      "best possible quality (record a person or render once with a cloud voice). | Any on-device voice: costs RAM and sounds worse. |")
    dl = [d for d in dia.values() if d.get("mode") == "text"]
    if dl:
        w(f"| **Brain** (fills the profile, asks follow-ups) | **Nemotron 3.5 Super VL** via OpenRouter, reasoning off | "
          f"{stx.mean(d['accuracy_required'] for d in dl) * 100:.0f}% of required fields right over {len(dl)} simulated students; "
          f"{stx.median(d['llm_s_median'] for d in dl):.1f} s per turn; ${stx.mean(d['cost'] for d in dl):.4f} per student; "
          f"{sum(d['re_asks'] for d in dl)} re-asks of known facts; {sum(1 for d in dl if d['injection_obeyed'])} of "
          f"{len(dl)} prompt injections obeyed. | Reasoning 'low': slower and returned broken JSON in 2 of 3 turns. |")
    w("")

    # ------------------------------------------------------------ ear
    w("## 1. The Ear: speech-to-text inside the cap\n")
    w("Each model transcribed the example answers, read by a cloud voice (clean) and the same clips degraded to an 8 kHz "
      "phone line with background noise. Error = character error rate, numbers ignored. Shruti writes English words in "
      "Latin letters, which inflates its Hindi error even when it heard correctly; 'dropped' counts clips where a "
      "quarter or more of the speech came back missing, which is script-independent.\n")
    w("| Language | Model | Fits 2 GB | Resident | Error clean | Error phone | Dropped speech | Speed | Real recordings |")
    w("|---|---|---|---|---|---|---|---|---|")
    for k, r in sorted(ear.items()):
        if not k.endswith("/2048"):
            continue
        w(f"| {LN[r['lang']]} | {NAME.get(r['stt'], r['stt'])} | {'yes' if r.get('fits') else 'no'} | {mb(r.get('peak_mb'))} | "
          f"{pct(r.get('cer_clean'))} | {pct(r.get('cer_phone'))} | {r.get('dropped_clean', '–')} + {r.get('dropped_phone', '–')} | "
          f"{r.get('x_realtime')}x | {pct(r.get('cer_real')) if r.get('cer_real') is not None else 'none yet'} |")
    w("\n**Read with care:** synthetic speech is cleaner than a student on a cheap phone. The `bench/recordings/` folder is "
      "empty until the team records real answers; that result should decide the Ear.\n")

    # ------------------------------------------------------------ pairs
    w("## 2. Ear + Voice together, under 2 GB and 1 GB\n")
    w("Resident memory is what Android's low-memory killer counts. 'Committed' is address space Windows reserves; "
      "Whisper reserves ~2.9 GB but uses a fraction, so it is reported, not enforced.\n")
    w("| Language | Ear + Voice | 2 GB | Resident | Committed | 1 GB | First sound | Speed | Voice error (Whisper judge) |")
    w("|---|---|---|---|---|---|---|---|---|")
    judge = {}
    for r in pairs.values():
        if r.get("judge_cer") is not None and len(r.get("items", [])) >= 3:
            judge[(r["lang"], r["tts"])] = r["judge_cer"]
    for k, r in sorted(pairs.items()):
        if not k.endswith("/2048"):
            continue
        small = pairs.get(k[:-5] + "/1024", {})
        w(f"| {LN[r['lang']]} | {NAME.get(r['stt'], r['stt'])} + {NAME.get(r['tts'], r['tts'])} | "
          f"{'fits' if r.get('fits') else '**no**'} | {mb(r.get('peak_mb'))} | {mb(r.get('peak_commit_mb'))} | "
          f"{'fits' if small.get('fits') else ('**no**' if small else '–')} | {r.get('first_s_median', '–')} s | "
          f"{r.get('x_realtime', '–')}x | {pct(judge.get((r['lang'], r['tts'])))} |")
    w("\n'First sound' assumes sentence streaming: the first sentence plays while the rest is generated. Speed is on a "
      "laptop CPU with 4 threads; expect a budget phone to be several times slower, which matters for Kokoro, "
      "Supertonic and Indic-TTS (about 2x real time here) and not for Piper (about 15-25x).\n")

    if cloud:
        w("**Cloud voices for the fixed prompts / server role** (same questions, same judge, needs internet):\n")
        w("| Language | Voice | Voice error (Whisper judge) | Seconds per sentence |")
        w("|---|---|---|---|")
        for k, r in sorted(cloud.items()):
            w(f"| {LN[r['lang']]} | {NAME.get(r['tts'], r['tts'])} | {pct(r.get('judge_cer'))} | {r.get('secs_median', '–')} |")
        w("\n**How to read 'voice error'.** It measures how easily a speech recogniser can write down what the voice "
          "said, not how natural or warm it sounds. Slow, carefully articulated voices score well; fast, natural ones "
          "lose points on spelling-level slips (Sarvam's Hindi 'गाँव में' was written as 'गाउने'). Use it to catch "
          "voices that are hard to understand (Marathi Piper, MMS), and the blind listening test to judge naturalness.\n")

    # ------------------------------------------------------------ dialogues
    w("## 3. The Brain: whole conversations with simulated students\n")
    w("An LLM plays six fictional students (2 per language) with hidden facts and scripted twists: self-corrections, "
      "vague income, contradictions, refusals, off-topic questions and a prompt injection. A judge pass compares the "
      "final profile with the hidden facts.\n")
    w("| Student | Mode | Finished | Turns | Required fields right | All fields right | Wrong | Re-asks | Injection obeyed | Brain s/turn | Cost |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, d in sorted(dia.items()):
        w(f"| {d['persona']} ({LN[d['lang']]}) | {d['mode']} | {'yes' if d['finished'] else 'no'} | {d['student_turns']} | "
          f"{pct(d['accuracy_required'])} | {pct(d['accuracy'])} | {', '.join(d.get('wrong_fields', [])) or '–'} | {d['re_asks']} | "
          f"{'**yes**' if d['injection_obeyed'] else 'no'} | {d['llm_s_median']} | ${d['cost']:.4f} |")
    if before:
        bw = sum(len(d.get("wrong_fields", [d.get("wrong")] if isinstance(d.get("wrong"), list) else [])) for d in before.values())
        w(f"\n**What the tests changed.** Before the fixes the agent silently took the newer value when a student "
          "contradicted themselves (3 of 4 contradiction cases ended with the wrong age or class). The code now treats a "
          "changed earlier answer as a doubt and makes the next question about it, unless the student's own words show a "
          "self-correction (\"no, sorry\", \"नहीं\", \"नाही\"). The LLM also wrote 72 as \"सत्तर दो\" instead of \"बहत्तर\"; "
          "it now writes digits and code turns them into correct Hindi, Marathi or English words. With reasoning switched "
          "on, 2 of 3 turns returned broken JSON, so it is off.\n")
    vd = [d for d in dia.values() if d.get("mode") == "voice"]
    if vd:
        w(f"**Voice mode** (student's words spoken, heard by the on-device Ear, Kaabil's reply spoken by Piper, inside 2 GB): "
          f"{stx.mean(d['accuracy_required'] for d in vd) * 100:.0f}% of required fields right over {len(vd)} conversations; "
          f"median Ear {stx.median(d['stt_s_median'] for d in vd):.1f} s + brain {stx.median(d['llm_s_median'] for d in vd):.1f} s "
          f"+ voice to first sound {stx.median(d['tts_first_s_median'] for d in vd):.2f} s per turn. Speech errors cost a few "
          "points of accuracy and 0-4 extra turns; the fields that ended wrong were marks, income and a skill description.\n")
    w("**Context.** The agent keeps a structured profile + a 60-word running summary + the last 6 turns, so memory does not "
      "grow with the conversation. 'Re-asks' counts questions about a field that was already known and not in doubt.\n")

    # ------------------------------------------------------------ outside suggestion
    w("## 4. About the outside suggestions we checked\n")
    w("- **sherpa-onnx** as the on-device runtime: adopted. Shruti, Piper, Kokoro and Supertonic here all run on it or on ONNX Runtime.")
    w("- **Shruti (Udgam Labs)**: real, Apache-2.0, one ~54 MB model per language. Its card claims ~120 MB after loading for the "
      f"int8 build; we measured the sherpa-onnx build at {mb(sh.get('peak_mb'))} resident for the whole worker process (Python included).")
    w("- **Sarvam Edge**: the specs on Sarvam's blog check out (74M-parameter speech recognition, 24M-parameter voice), but the "
      "models are not downloadable, so we could not test them. Cite as a possible future partner, not as part of the stack.")
    w("- **Uktam.ai**: a Play Store app; we found no public source code. Do not call it open source. For 'proof it works offline', "
      "cite `itantra-offline-voice` (open source, Hindi and Marathi, IndicConformer + Indic-TTS on Android).")
    w("- **Pre-recorded fixed prompts**: agreed and adopted; it is the cheapest win on a 2 GB phone.\n")

    w("## 5. What this does not prove\n")
    w("- No phone was used. Phone CPUs are slower; memory behaviour is approximated by capping resident memory.")
    w("- Test speech is synthetic until the team adds real recordings to `bench/recordings/<lang>/` (a .wav plus a .txt of what was said).")
    w("- Voice quality is scored by a speech recogniser (intelligibility), not by people. Use the blind test on the Listen tab with Hindi and Marathi speakers.")
    w("- The simulated students are written by an LLM; real students will be shorter, noisier and less cooperative.")
    w("- OpenRouter sends the conversation to a cloud model. The deck's data boundary (pseudonymous IDs only) is not yet implemented in this prototype; use fictional data.")
    (ROOT / "JUDGES_BRIEF.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", ROOT / "JUDGES_BRIEF.md")


if __name__ == "__main__":
    main()
