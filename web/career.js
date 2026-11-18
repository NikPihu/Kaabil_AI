/* Career recommendations tab. Uses h(), post(), LANGS and RENDER from index.html,
   and C (the Conversation state) from talk.js for the "Plan my path" hand-off. */
const K = {profiles: null, sel: null, res: null, busy: "", filter: "all", open: {}, review: null, pending: null,
           askWhy: null, audio: null};
const FIT = {strong: ["Strong fit", "fit-strong"], good: ["Good fit", "fit-good"], possible: ["Possible", "fit-possible"]};
const WHERE = {local: "in your village", district: "in your district", city: "in a big city", online: "online",
               distance: "by distance learning"};
const STATUS = {now: "You can start now", after_current: "Opens after your current class", with_step: "Needs a step first"};
const FILTERS = [["all", "All"], ["nonstem", "Non-STEM"], ["stem", "STEM"], ["government", "Government"],
                 ["self", "Own business"], ["quick", "Earn soon"]];

const money = c => !c ? "" : (c.min === 0 ? "free" : `₹${c.min.toLocaleString("en-IN")}`) +
  (c.max ? `–₹${c.max.toLocaleString("en-IN")}` : "");
const levelText = s => !s ? "" : s.current ? s.current.replace("class_", "Class ").replace("graduate", "in college")
  .replace("iti", "ITI").replace("diploma", "diploma") : `completed Class ${s.completed}`;

async function loadProfiles(){
  K.profiles = (await (await fetch("/api/career/profiles")).json());
  if (!K.sel) K.sel = K.profiles.profiles[0] && (K.profiles.profiles[0].source + "|" + K.profiles.profiles[0].id);
}

async function findPaths(extra = {}){
  if (!K.sel) return;
  const [source, id] = K.sel.split("|");
  K.busy = "Reading the profile, checking 34 paths against the rules, asking the Career Agent to explain…";
  K.open = {}; renderCareer();
  try {
    K.res = await post("/api/career/suggest", {source, id, ...extra});
  } catch (x) { K.res = {error: x.message}; }
  K.busy = ""; renderCareer();
}

async function speakSummary(){
  if (!K.res || !K.res.summary_local) return;
  K.busy = "Speaking the summary…"; renderCareer();
  try {
    const r = await post("/api/synth", {engine: "piper", lang: K.res.lang, text: K.res.summary_local, speed: 0.95});
    if (K.audio) K.audio.pause();
    K.audio = new Audio(r.url); K.audio.play().catch(() => {});
  } catch (x) { alert("Could not speak: " + x.message); }
  K.busy = ""; renderCareer();
}

async function sendFeedback(card, kind, reason = ""){
  await post("/api/career/feedback", {key: K.res.key, career_id: card.id, kind, reason});
  if (kind === "not_fit") {
    const ex = (K.res.feedback || []).filter(f => f.kind === "not_fit").map(f => f.career_id).concat([card.id]);
    const why = reason ? `${card.title.en}: ${reason}` : "";
    await findPaths({exclude: [...new Set(ex)], feedback: why});
  } else {
    K.res.feedback = (K.res.feedback || []).filter(f => f.career_id !== card.id).concat([{career_id: card.id, kind}]);
    renderCareer();
  }
}

function pill(text, cls){ return h("span", {class: "cpill " + (cls || "")}, text); }

function studentPanel(){
  const r = K.res, s = r.situation, st = r.student, p = st.profile, ev = st.evidence || {};
  const row = (label, val, quote) => val ? h("div", {class: "srow"}, h("span", {class: "sl"}, label),
    h("span", {class: "sv"}, val, quote && h("q", {}, quote))) : "";
  const blocked = r.blocked || [];
  return h("aside", {class: "spanel"},
    h("div", {class: "eyebrow"}, st.phase === "demo" ? "demo student" : "from the conversation"),
    h("h3", {}, st.name || "Student"),
    h("p", {class: "pread"}, r.profile_read),
    h("div", {class: "sgrid"},
      row("Studying", [p.class_or_year, p.stream_or_course].filter(Boolean).join(", ")),
      row("Place", [p.village, p.district, p.state].filter(Boolean).join(", ")),
      row("Age", p.age),
      row("Last marks", p.last_marks),
      row("Likes", p.favourite_subjects, ev.favourite_subjects),
      row("Does at home", p.daily_life_skills, ev.daily_life_skills),
      row("Wants to be", p.aspiration, ev.aspiration),
      row("Limits", p.constraints, ev.constraints),
      row("Family income", p.family_income)),
    h("div", {class: "rulechips"},
      pill(levelText(s)), s.stream && pill(s.stream.toUpperCase() + " stream"),
      s.local_only && pill("stays near home", "warn"), s.low_income && pill("money is tight", "warn")),
    h("div", {class: "funnel"},
      h("div", {}, h("b", {}, r.considered), " paths in the knowledge base"),
      h("div", {}, h("b", {}, r.considered - blocked.length), " pass the entry rules for ", st.name || "this student"),
      h("div", {}, h("b", {}, r.shortlisted), " shortlisted from their own words"),
      h("div", {}, h("b", {}, r.cards.length), " recommended, with reasons")),
    blocked.length ? h("details", {class: "blocked"}, h("summary", {}, `${blocked.length} paths ruled out by the rules`),
      blocked.map(b => h("div", {}, h("b", {}, b.title), " — ", b.why))) : "",
    r.summary_local && h("div", {class: "spoken"},
      h("div", {class: "eyebrow"}, "Kaabil says, in " + (LANGS[r.lang] || r.lang)),
      h("p", {}, r.summary_local),
      h("button", {class: "go small", onclick: speakSummary}, "Listen")),
    h("div", {class: "meta"}, `Career Agent: ${r.llm.model} · ${r.seconds}s · knowledge base v${r.kb_version}` + (r.cached ? " · saved result" : "")));
}

function quickFacts(c){
  const r = c.route;
  return h("div", {class: "facts"},
    h("div", {}, h("span", {}, "Duration"), h("b", {}, r.duration)),
    h("div", {}, h("span", {}, "Cost"), h("b", {}, money(r.cost)), r.cost.note && h("small", {}, r.cost.note)),
    h("div", {}, h("span", {}, "Starts earning"), h("b", {}, c.earn_after)),
    h("div", {}, h("span", {}, "Where you study"), h("b", {}, WHERE[r.where] || r.where)));
}

function cardDetail(c){
  const off = c.official;
  return h("div", {class: "cdetail"},
    h("section", {}, h("h4", {}, "Your route: " + c.route.label),
      h("ol", {}, c.route.steps.map(s => h("li", {}, s))),
      c.other_routes.length ? h("p", {class: "muted"}, "Other ways in: " + c.other_routes.map(o => o.label).join(" · ")) : ""),
    c.first_steps && c.first_steps.length ? h("section", {class: "next"}, h("h4", {}, "Your next steps (next 6 months)"),
      h("ol", {}, c.first_steps.map(s => h("li", {}, s)))) : "",
    h("section", {}, h("h4", {}, "What you will study"),
      h("ul", {class: "learn"}, c.learn.map(x => h("li", {}, x))),
      h("p", {class: "muted"}, c.learn_source === "official" ? "From the official curriculum (reviewed)." : "Typical syllabus."),
      off && off.modules.length ? h("div", {class: "official"},
        h("div", {class: "eyebrow"}, "From the government curriculum PDF · " + off.status),
        h("ul", {}, off.modules.map(x => h("li", {}, x))),
        off.entry ? h("p", {}, h("b", {}, "Entry rule in the document: "), off.entry) : "",
        h("a", {href: off.source, target: "_blank", rel: "noopener"}, "Open the source PDF")) : ""),
    h("section", {}, h("h4", {}, "Where it leads: what you can become"),
      h("div", {class: "ladder"}, c.ladder.map((x, i) => [i ? h("span", {class: "arr"}, "→") : "", h("span", {class: "rung"}, x)])),
      off && off.roles.length ? h("p", {class: "muted"}, "Job roles named in the official document: " + off.roles.join(", ")) : ""),
    h("section", {}, h("h4", {}, "Money"),
      h("p", {}, h("b", {}, "Pay: "), c.pay.start, " ", pill(c.pay.basis === "official" ? "official figure" : "estimate", c.pay.basis === "official" ? "ok" : "warn")),
      c.pay.note && h("p", {class: "muted"}, c.pay.note),
      h("p", {}, h("b", {}, "Course cost: "), money(c.route.cost), c.route.cost.note ? " — " + c.route.cost.note : "")),
    h("section", {}, h("h4", {}, "Start building these skills now"),
      h("ul", {}, c.skills_now.map(x => h("li", {}, x))),
      c.free_learning.length ? h("p", {}, "Free to learn: ", c.free_learning.map((f, i) => [i ? " · " : "", h("a", {href: f.url, target: "_blank", rel: "noopener"}, f.name)])) : ""),
    h("section", {class: "src"}, h("h4", {}, "Sources"),
      c.state_link ? h("p", {}, "Your state: ", h("a", {href: c.state_link.url, target: "_blank", rel: "noopener"}, c.state_link.name)) : "",
      c.sources.map(s => h("p", {}, h("a", {href: s.url, target: "_blank", rel: "noopener"}, s.name), h("small", {}, " — " + s.checked)))));
}

function card(c, i){
  const [fitText, fitCls] = FIT[c.fit] || FIT.possible;
  const fb = (K.res.feedback || []).find(f => f.career_id === c.id);
  const open = K.open[c.id];
  // the agent's watch-out first; a rule's warning only if the agent did not already say the same thing
  const said = (c.watch_out || "").toLowerCase();
  const sameIdea = w => { const ws = w.toLowerCase().match(/[a-z]{5,}/g) || []; return ws.filter(x => said.includes(x)).length >= Math.min(2, ws.length); };
  const watch = [c.watch_out].concat((c.watch_rules || []).filter(w => !said || !sameIdea(w))).filter(Boolean);
  return h("article", {class: "ccard" + (open ? " open" : "") + (fb && fb.kind === "interested" ? " liked" : "")},
    h("div", {class: "chead"},
      h("span", {class: "rank"}, i + 1),
      h("div", {class: "ctitle"},
        h("h3", {}, c.title.en, K.res.lang !== "en" && c.title[K.res.lang] ? h("span", {class: "local"}, c.title[K.res.lang]) : ""),
        h("div", {class: "tags2"}, pill(fitText, fitCls), pill(c.sector), pill(c.stem ? "STEM" : "Non-STEM", c.stem ? "stem" : "nonstem"),
          c.work_type.includes("government") && pill("Government", "gov"), c.work_type.includes("self-employment") && pill("Own business", "self"),
          c.added_by && pill(c.added_by === "you named this" ? "you named this" : "added for balance", "muted"))),
    ),
    c.headline_local && h("p", {class: "hl"}, c.headline_local),
    h("p", {class: "status"}, h("b", {}, STATUS[c.status] || c.status), c.status_reasons.length ? " — " + c.status_reasons.join("; ") : ""),
    c.why && c.why.length ? h("ul", {class: "why"}, c.why.map(w => h("li", {}, w.point, w.quote ? h("q", {}, w.quote) : ""))) :
      h("p", {class: "muted"}, c.summary),
    watch.length ? h("div", {class: "watch"}, h("b", {}, "Watch out: "), watch.join(" · ")) : "",
    quickFacts(c),
    h("div", {class: "cactions"},
      h("button", {class: "go small", onclick: () => { K.open[c.id] = !open; renderCareer(); }}, open ? "Close path" : "Open full path"),
      h("button", {class: "ghost" + (fb && fb.kind === "interested" ? " on" : ""), onclick: () => sendFeedback(c, "interested")},
        fb && fb.kind === "interested" ? "Interested (saved)" : "I'm interested"),
      K.askWhy === c.id ? h("span", {class: "askwhy"},
          h("input", {placeholder: "Why not? (optional)", id: "why-" + c.id}),
          h("button", {class: "ghost warn", onclick: () => { const v = document.getElementById("why-" + c.id).value; K.askWhy = null; sendFeedback(c, "not_fit", v); }}, "Show other paths"))
        : h("button", {class: "ghost", onclick: () => { K.askWhy = c.id; renderCareer(); }}, "This doesn't fit")),
    open ? cardDetail(c) : "");
}

function passes(c){
  const f = K.filter;
  return f === "all" || (f === "stem" && c.stem) || (f === "nonstem" && !c.stem) ||
    (f === "government" && c.work_type.includes("government")) || (f === "self" && c.work_type.includes("self-employment")) ||
    (f === "quick" && /month/.test(c.earn_after) && !/year/.test(c.earn_after));
}

async function renderReview(box){
  const r = await (await fetch("/api/career/review")).json();
  box.innerHTML = "";
  box.append(h("summary", {}, `Data review: ${r.items.length} government curriculum extraction(s) · knowledge base v${r.kb_version}`),
    h("p", {class: "muted"}, "career/ingest.py fetches official PDFs, extracts the syllabus with quotes, and keeps only items whose quote is found in the PDF. A person approves before it replaces the typical syllabus."),
    r.items.map(it => h("div", {class: "rv"},
      h("div", {}, h("b", {}, it.title), h("small", {}, ` · ${it.status} · fetched ${it.fetched_on}`)),
      h("div", {class: "muted"}, it.modules.map(m => m.name).join(" · ")),
      h("a", {href: it.source_url, target: "_blank", rel: "noopener"}, "source PDF"),
      it.status.startsWith("published") ? "" : h("button", {class: "ghost", onclick: async () => {
        await post("/api/career/approve", {id: it.id}); renderReview(box); }}, "Approve and publish"))));
}

function renderCareer(){
  const root = document.getElementById("career"); root.innerHTML = "";
  if (!K.profiles) { root.append(h("div", {class: "empty"}, "Loading profiles…")); loadProfiles().then(renderCareer); return; }
  if (K.pending) { K.sel = K.pending; K.pending = null; K.res = null; setTimeout(() => findPaths(), 0); }
  const hero = h("div", {class: "hero"}, h("div", {class: "eyebrow"}, "Kaabil career agent"),
    h("h1", {}, "Paths that fit ", h("em", {}, "you")),
    h("p", {}, "Built from the student's profile: their class and stream, where they live and how far they can go, money at home, and what they already do and like. Facts come from official sources; the agent explains why each path fits."));
  const groups = [["Conversations", "session"], ["Demo students", "persona"]];
  const sel = h("select", {onchange: e => { K.sel = e.target.value; K.res = null; renderCareer(); }},
    groups.map(([label, src]) => h("optgroup", {label},
      K.profiles.profiles.filter(p => p.source === src).map(p => h("option", {value: p.source + "|" + p.id, selected: K.sel === p.source + "|" + p.id},
        `${p.name} · ${LANGS[p.lang] || p.lang} · ${p.fields} fields` + (p.source === "session" ? ` · ${p.phase === "done" ? "confirmed" : p.phase}` : ""))))));
  const bar = h("div", {class: "cbar"}, h("label", {}, "Student", sel),
    h("button", {class: "go", onclick: () => findPaths()}, K.res && !K.res.error ? "Find again" : "Find paths"),
    K.res && !K.res.error ? h("button", {class: "ghost", onclick: () => findPaths({refresh: true})}, "Ask the agent afresh") : "",
    !K.profiles.llm.available && h("span", {class: "bad"}, "OPENROUTER_API_KEY is not set"));
  root.append(hero, bar, h("div", {class: "busy"}, K.busy));
  if (K.res && K.res.error) root.append(h("div", {class: "err"}, K.res.error));
  if (K.res && !K.res.error) {
    const cards = K.res.cards.filter(passes);
    root.append(h("div", {class: "cgrid"}, studentPanel(),
      h("div", {},
        h("div", {class: "filters"}, FILTERS.map(([k, label]) => h("button", {class: "chip-f" + (K.filter === k ? " on" : ""),
          onclick: () => { K.filter = k; renderCareer(); }}, label + (k === "all" ? ` (${K.res.cards.length})` : "")))),
        cards.length ? cards.map((c, i) => card(c, K.res.cards.indexOf(c))) : h("div", {class: "empty"}, "No paths in this filter."),
        K.res.explore && K.res.explore.length ? h("div", {class: "explore"},
          h("h4", {}, "Also worth exploring (not in the knowledge base yet; unverified, ask a teacher)"),
          K.res.explore.map(e => h("p", {}, h("b", {}, e.title), " — ", e.why))) : "")));
  } else if (!K.busy && !(K.res && K.res.error)) {
    root.append(h("div", {class: "empty"}, "Pick a student and press Find paths. Finished conversations from the Conversation tab appear here too."));
  }
  const rv = h("details", {class: "review"}); rv.addEventListener("toggle", () => rv.open && renderReview(rv));
  rv.append(h("summary", {}, "Data review: government curriculum extractions"));
  root.append(rv);
}

RENDER.career = renderCareer;
(function addCareerTab(){
  const nav = document.getElementById("nav"), b = document.createElement("button");
  b.dataset.tab = "career"; b.textContent = "Career recommendations"; nav.insertBefore(b, nav.children[1]);
  const m = document.createElement("main"); m.id = "career"; document.getElementById("talk").after(m);
})();
function openCareerFor(source, id){
  K.pending = source + "|" + id; K.profiles = null;
  document.querySelector('nav button[data-tab="career"]').click();
}
