/* Conversation tab: mic / typing / example chips -> Ear -> Brain -> Voice.
   Uses h(), post(), LANGS and RENDER from index.html. */
const C = {setup:null, sid:null, turns:[], state:null, box:null, lang:"hi", stt:"shruti", tts:"piper", voice:null,
           limit:"2048", threads:"4", cpu:"", echo:false, phone:false, busy:"", autoplay:true, rec:null, changed:[]};
const TCOL = {stt_s:"#2563eb", llm_s:"#7c3aed", tts_first_s:"#059669"};
let lastAudio = null;
function playUrl(u){ if(lastAudio) lastAudio.pause(); lastAudio = new Audio(u); lastAudio.play().catch(()=>{}); }
async function loadSetup(){ C.setup = await (await fetch("/api/setup")).json(); if(!C.setup.llm.available) C.echo = true; }

function setupPanel(){
  const st = C.setup, need = st.keys.filter(k=>k.required&&!k.set).length, miss = st.items.filter(i=>!i.present).length;
  return h("details",{class:"setup",open:(need||miss)?true:null},
    h("summary",{}, "SETUP  ·  ", h("b",{}, need?`${need} required key missing`:"keys ok"), "  ·  ",
      h("b",{}, miss?`${miss} model(s) missing`:"all models present"), "  ·  brain: ", h("b",{}, st.llm.model)),
    h("div",{class:"grid"},
      h("div",{}, h("h4",{},"API KEYS · paste into .env in the project folder, then restart the app"),
        h("div",{class:"chk"}, st.keys.map(k=>[h("span",{class:k.set?"y":k.required?"n":"o"}, k.set?"set":k.required?"missing":"empty"),
          h("code",{},k.name), h("small",{}, (k.required?"required · ":"optional · ")+k.unlocks)]))),
      h("div",{}, h("h4",{},"MODELS · run  python setup_models.py  to fetch anything missing"),
        h("div",{class:"chk"}, st.items.map(i=>[h("span",{class:i.present?"y":"n"}, i.present?"ok":"missing"), h("code",{},i.id),
          h("small",{}, i.role+(i.size_mb?` · ${i.size_mb} MB`:"")+(!i.present&&i.manual?` · by hand: ${i.manual}`:""))])))));
}

function tbar(t){
  const parts = [["stt_s",t.stt_s||0],["llm_s",t.llm_s||0],["tts_first_s",t.tts_first_s||0]];
  const tot = parts.reduce((a,b)=>a+b[1],0)||1;
  return [h("div",{class:"tbar",title:"What the student waits before Kaabil starts speaking"},
            parts.map(([k,v])=>h("i",{style:`width:${v/tot*100}%;background:${TCOL[k]}`}))),
    h("div",{class:"legend"},
      h("span",{style:`--c:${TCOL.stt_s}`}, `ear ${(t.stt_s||0).toFixed(2)}s`),
      h("span",{style:`--c:${TCOL.llm_s}`}, `brain ${(t.llm_s||0).toFixed(2)}s`+(t.llm_ttft_s?` (first token ${t.llm_ttft_s.toFixed(2)}s)`:"")),
      h("span",{style:`--c:${TCOL.tts_first_s}`}, `voice ${(t.tts_first_s||0).toFixed(2)}s to first sound`),
      h("b",{style:"color:var(--ink)"}, `student waits ${t.wait_s}s`))];
}

function msg(t){
  if(t.role==="student") return h("div",{class:"msg s"},
    h("div",{class:"who"}, "student"+(t.audio?" · heard by the Ear":""), t.audio&&h("button",{class:"mplay",onclick:()=>playUrl(t.audio)},"play original")),
    t.text, t.reference&&h("div",{class:"ref"}, "actually said: "+t.reference));
  return h("div",{class:"msg k"},
    h("div",{class:"who"}, "kaabil"+(t.fixed_prompt?" · fixed prompt":""), t.audio&&h("button",{class:"mplay",onclick:()=>playUrl(t.audio)},"play")),
    t.text, t.timing&&tbar(t.timing),
    t.tts&&t.tts.ok&&h("div",{class:"meta"}, `voice: ${t.tts.duration}s of audio made in ${t.tts.secs}s`
      +(t.llm&&t.llm.tokens?` · ${t.llm.tokens} tokens`+(t.llm.cost?` · $${t.llm.cost.toFixed(5)}`:""):"")),
    t.tts&&!t.tts.ok&&h("div",{class:"err"}, "voice failed: "+t.tts.error));
}

function profileCard(){
  const st = C.state;
  if(!st) return h("div",{class:"card2"}, h("h3",{},"Kaabil profile"), h("p",{class:"lede"},"Fills in as the student talks."));
  const bad = st.events.filter(e=>e.type==="re_ask"||e.type==="asked_skipped").length;
  return h("div",{class:"card2"},
    h("div",{class:"phase"}, st.phase==="collect"?"collecting":st.phase==="readback"?"reading back for confirmation":"confirmed"),
    h("h3",{},"Kaabil profile"),
    h("div",{class:"legend",style:"margin-bottom:6px"}, h("span",{style:"--c:var(--teal)"},"filled"), h("span",{style:"--c:var(--gold)"},"doubt"),
      h("span",{style:"--c:var(--coral)"},"missing, required"), h("span",{style:"--c:var(--line2)"},"optional")),
    st.fields.map(f=>h("div",{class:`fld ${f.status} ${f.required?"req":""} ${C.changed.includes(f.id)?"flash":""}`,
                              title:f.evidence?`heard: "${f.evidence}"`:""},
      h("span",{class:"d"}), h("span",{class:"l"}, f.label),
      h("span",{class:"v"}, f.status==="skipped"?"skipped by student":(f.value||"—"),
        f.confidence!=null&&f.confidence<0.6&&h("small",{}, `unsure (${f.confidence.toFixed(1)})`),
        f.doubt&&h("small",{}, "doubt: "+f.doubt)))),
    h("div",{class:"evs"},
      h("div",{}, `${st.student_turns} student turns · `, h("span",{style:bad?"color:var(--coral)":""}, `${bad} re-asks of known fields`), ` · $${(st.cost||0).toFixed(4)}`),
      st.events.slice(-8).map(e=>h("div",{class:e.type}, `turn ${e.turn}: ${e.type.replace(/_/g," ")} · ${e.field}`+(e.old?`: "${e.old}" → "${e.new}"`:"")))),
    st.summary&&h("p",{class:"hint",style:"margin-top:10px"}, "Memory summary: "+st.summary),
    C.sid&&st.fields.filter(f=>f.value).length>=6&&h("button",{class:"go",style:"margin-top:12px;width:100%",
      onclick:()=>openCareerFor("session",C.sid)}, st.phase==="done"?"Plan my path":"Plan my path (profile not confirmed yet)"));
}

async function startConvo(){
  C.busy = "Loading the Ear and the Voice inside the memory cap…"; renderTalk();
  try{
    const r = await post("/api/convo/start",{lang:C.lang,stt:C.stt,tts:C.tts,voice:C.voice,limit_mb:+C.limit,threads:+C.threads,cpu_percent:C.cpu?+C.cpu:null});
    if(!r.fits){ C.box={error:r.error}; C.sid=null; C.turns=[]; C.state=null; }
    else { C.sid=r.sid; C.box=r.box; C.turns=[r.turn]; C.state=r.state; C.changed=[]; if(C.autoplay&&r.turn.audio) playUrl(r.turn.audio); }
  }catch(x){ C.box={error:x.message}; }
  C.busy=""; renderTalk();
}

async function sendTurn(body){
  if(!C.sid){ C.busy = "Press Start first."; renderTalk(); return; }
  C.busy = body.audio_b64?"Hearing your recording…":body.example?"Hearing the example clip…":C.echo?"Speaking…":"Thinking…"; renderTalk();
  try{
    const r = await post("/api/convo/turn",{sid:C.sid,echo:C.echo,...body});
    C.turns.push(r.student, r.kaabil); C.state=r.state; C.box=r.box; C.changed=r.kaabil.changes||[];
    if(C.autoplay&&r.kaabil.audio) playUrl(r.kaabil.audio);
  }catch(x){ C.turns.push({role:"kaabil", text:"Error: "+x.message}); }
  C.busy=""; renderTalk();
}

/* Mic: record raw PCM and build a 16 kHz WAV in the browser, so the server needs no ffmpeg. */
async function toggleMic(btn){
  if(C.rec){ const r=C.rec; C.rec=null; r.stop(); return; }
  let stream;
  try{ stream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true}}); }
  catch(e){ C.busy = "Microphone not available here ("+e.message+"). Use typing or the audio buttons on the examples."; renderTalk(); return; }
  const ac = new AudioContext(), src = ac.createMediaStreamSource(stream), proc = ac.createScriptProcessor(4096,1,1), chunks = [];
  proc.onaudioprocess = e=>chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(proc); proc.connect(ac.destination);
  btn.classList.add("rec"); btn.textContent="Stop"; C.busy="Recording. Click Stop to send."; document.querySelector("#talk .busy").textContent=C.busy;
  C.rec = {stop:async()=>{
    proc.disconnect(); src.disconnect(); stream.getTracks().forEach(t=>t.stop());
    const sr = ac.sampleRate; await ac.close();
    const all = new Float32Array(chunks.reduce((a,c)=>a+c.length,0)); let o=0; for(const c of chunks){ all.set(c,o); o+=c.length; }
    const ratio = sr/16000, n = Math.floor(all.length/ratio), pcm = new Int16Array(n);
    for(let i=0;i<n;i++){ let a=0,c=0; for(let j=Math.floor(i*ratio);j<Math.floor((i+1)*ratio);j++){ a+=all[j]; c++; } pcm[i]=Math.max(-1,Math.min(1,a/(c||1)))*32767; }
    if(n < 8000){ C.busy="Too short; hold for at least a second."; renderTalk(); return; }
    const buf = new ArrayBuffer(44+n*2), v = new DataView(buf), w=(o,s)=>[...s].forEach((ch,i)=>v.setUint8(o+i,ch.charCodeAt(0)));
    w(0,"RIFF"); v.setUint32(4,36+n*2,true); w(8,"WAVE"); w(12,"fmt "); v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,1,true);
    v.setUint32(24,16000,true); v.setUint32(28,32000,true); v.setUint16(32,2,true); v.setUint16(34,16,true); w(36,"data"); v.setUint32(40,n*2,true);
    new Int16Array(buf,44).set(pcm);
    let bin=""; const bytes=new Uint8Array(buf); for(let i=0;i<bytes.length;i+=8192) bin+=String.fromCharCode(...bytes.subarray(i,i+8192));
    sendTurn({audio_b64:btoa(bin)});
  }};
}

function renderTalk(){
  const root = document.getElementById("talk"); root.innerHTML="";
  if(!C.setup){ root.append(h("div",{class:"empty"},"Loading setup…")); loadSetup().then(renderTalk).catch(e=>root.append(h("div",{class:"err"},e.message))); return; }
  const st = C.setup, L = C.lang;
  const sttOpts = st.stt.filter(s=>s.langs.includes(L)), ttsOpts = st.tts.filter(e=>e.voices[L]);
  if(!sttOpts.find(s=>s.id===C.stt&&s.available)) C.stt = (sttOpts.find(s=>s.available)||sttOpts[0]||{}).id;
  if(!ttsOpts.find(e=>e.id===C.tts&&e.available)) C.tts = (ttsOpts.find(e=>e.available)||ttsOpts[0]||{}).id;
  const te = ttsOpts.find(e=>e.id===C.tts);
  const sel = (val,opts,on)=>h("select",{onchange:e=>{on(e.target.value); renderTalk();}},
    opts.map(([v,l,dis])=>h("option",{value:v,selected:v===val,disabled:dis},l)));
  const cfg = h("div",{class:"cfg"},
    h("div",{class:"seg",style:"margin:0;width:260px"}, Object.entries(LANGS).map(([k,v])=>h("button",{class:L===k?"on":"",onclick:()=>{C.lang=k;C.voice=null;renderTalk();}},v))),
    h("label",{}, "Ear", sel(C.stt, sttOpts.map(s=>[s.id, s.name+(s.available?"":" (missing)"), !s.available]), v=>C.stt=v)),
    h("label",{}, "Voice", sel(C.tts, ttsOpts.map(e=>[e.id, e.name+(e.available?"":" (missing)")+(e.experimental.includes(L)?" · not trained on "+L:""), !e.available]), v=>{C.tts=v;C.voice=null;})),
    te&&h("label",{}, "speaker", sel(C.voice||te.voices[L][0].id, te.voices[L].map(v=>[v.id,`${v.label} · ${v.gender}`]), v=>C.voice=v)),
    h("label",{}, "RAM cap", sel(C.limit, [["2048","2 GB"],["1024","1 GB"],["768","768 MB"],["4096","4 GB"]], v=>C.limit=v)),
    h("label",{title:"CPU threads each model may use. A budget phone has 4 slow cores."}, "threads", sel(C.threads, [["2","2"],["4","4"],["8","8"]], v=>C.threads=v)),
    h("label",{title:"Hard CPU cap for both workers, as % of this laptop, to imitate a slow phone"}, "CPU cap", sel(C.cpu, [["","none"],["35","35%"],["20","20%"]], v=>C.cpu=v)),
    h("button",{class:"go",style:"margin:0",onclick:startConvo}, C.sid?"Restart":"Start"));
  const bi = C.box && (C.box.error ? h("div",{class:"boxinfo"}, h("span",{class:"bad"}, "Does not fit: "+C.box.error))
    : h("div",{class:"boxinfo"}, "Ear + Voice loaded. Peak memory ", h("b",{}, `${C.box.peak_mb} MB`), ` of ${C.box.limit_mb} MB`,
        ` · Ear loaded in ${C.box.load.stt?.secs}s · Voice in ${C.box.load.tts?.secs}s · ${C.box.threads} threads`+(C.box.cpu_percent?` · CPU capped at ${C.box.cpu_percent}%`:"")));
  const chat = h("div",{class:"chat",id:"chatbox"}, C.turns.length ? C.turns.map(msg)
    : h("div",{class:"empty"}, st.llm.available ? "Pick a language, an Ear and a Voice, then press Start."
        : "OPENROUTER_API_KEY is not set yet, so echo mode is on: Kaabil repeats what the Ear heard. That still tests the speech-to-text and the voice inside the memory cap."));
  const ta = h("textarea",{placeholder:"Type what the student says (Enter to send), use the mic, or click a chip below…"});
  ta.addEventListener("keydown",e=>{ if(e.key==="Enter"&&!e.shiftKey){ e.preventDefault(); if(ta.value.trim()) sendTurn({text:ta.value.trim()}); }});
  const mic = h("button",{class:"mic",title:"Click to record, click again to send",onclick:e=>toggleMic(e.currentTarget)},"Record");
  const ex = st.examples.examples[L]||[];
  const chip = it=>h("span",{class:"chip"+(it.kind==="hard"?" hard":"")},
    h("button",{title:(it.tests?`Tests: ${it.tests}\n\n`:"")+it.text+"\n\nClick to send as typed text.", onclick:()=>sendTurn({text:it.text})}, it.label),
    h("button",{class:"sp",title:"Send the spoken clip through the Ear instead"+(C.phone?" (8 kHz phone line + noise)":""),
                onclick:()=>sendTurn({example:`${L}/${it.id}${C.phone?"_phone":""}`})},"audio"));
  const hero = h("div",{class:"hero"}, h("div",{class:"eyebrow"},"Kaabil profile agent"),
    h("h1",{}, "Talk, and watch the ", h("em",{},"profile"), " fill in"),
    h("p",{}, "Speak or type in English, Hindi or Marathi. The Ear and the Voice run on the device inside a memory cap; the Brain runs on the server."));
  root.append(hero, setupPanel(), cfg, bi||"",
    h("div",{class:"convo"},
      h("div",{},
        chat, h("div",{class:"busy"}, C.busy),
        h("div",{class:"composer"}, mic, ta, h("button",{class:"go",style:"margin:0",onclick:()=>ta.value.trim()&&sendTurn({text:ta.value.trim()})},"Send")),
        h("div",{class:"ctl"},
          h("label",{}, h("input",{type:"checkbox",checked:C.autoplay,onchange:e=>C.autoplay=e.target.checked}), "auto-play replies"),
          h("label",{title:"Skip the LLM: Kaabil repeats what the Ear heard. Tests the voice loop without an API key."},
            h("input",{type:"checkbox",checked:C.echo,onchange:e=>C.echo=e.target.checked}), "echo mode (no LLM)"),
          h("label",{title:"The audio buttons send the 8 kHz phone-line copy with background noise"},
            h("input",{type:"checkbox",checked:C.phone,onchange:e=>{C.phone=e.target.checked; renderTalk();}}), "send audio as a noisy phone line")),
        h("div",{class:"chips2"},
          h("p",{class:"hint",style:"margin:4px 0 0"}, "Example student: "+st.examples.personas[L]),
          h("h4",{}, "Normal answers · click the label to send text, audio to send the spoken clip through the Ear"),
          h("div",{class:"row2"}, ex.filter(x=>x.kind==="normal").map(chip)),
          h("h4",{}, "Hard cases · hover to see what each one tests"),
          h("div",{class:"row2"}, ex.filter(x=>x.kind==="hard").map(chip)))),
      profileCard()));
  const cb = document.getElementById("chatbox"); if(cb) cb.scrollTop = cb.scrollHeight;
}
RENDER.talk = renderTalk;
