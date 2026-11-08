// lineclean GUI 화면 (M7, D-20). 계산은 모두 Python(GuiApi)이 한다.
"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const VIEW_TITLES = {
  result: "결과",
  diff: "차이 — 빨강: 생긴 잉크, 파랑: 사라진 잉크",
  topology: "위상 — 빨강 X, 주황 T/Y, 초록 끝점",
  candidates: "병합 후보 — 선이나 번호를 클릭해 고르기",
};

const state = {
  schema: null,
  values: {},       // {섹션: {키: 값}}
  defaults: {},
  rows: {},         // "섹션.키" → {input, val, name, spec}
  mode: "auto", smooth: true, render: true,
  selection: [],
  view: "result",
  reqId: 0,
  result: null,     // 마지막 process 응답
  checkedId: 0,
  opened: false,
  imgW: 0, imgH: 0,
  z: { s: 1, tx: 0, ty: 0 },
  advanced: false,
};

// ======================================================== API 연결
let API = null;
async function connect() {
  if (document.documentElement.dataset.mode === "browser") {
    API = new Proxy({}, {
      get: (_, method) => (...args) =>
        fetch("/api/" + method, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ args }) })
          .then((r) => r.json()),
    });
    return;
  }
  if (!(window.pywebview && window.pywebview.api && window.pywebview.api.schema)) {
    await new Promise((resolve) => window.addEventListener("pywebviewready", resolve, { once: true }));
  }
  API = window.pywebview.api;
}

// ======================================================== 알림
let toastTimer;
function toast(text, bad = false) {
  const el = $("#toast");
  el.textContent = text;
  el.classList.toggle("bad", bad);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), bad ? 6000 : 3200);
}

let tipTimer;
function attachTip(el, textFn) {
  el.addEventListener("mouseenter", () => {
    clearTimeout(tipTimer);
    tipTimer = setTimeout(() => {
      const text = textFn();
      if (!text) return;
      const tip = $("#tip");
      tip.textContent = text;
      tip.hidden = false;
      const r = el.getBoundingClientRect();
      const w = tip.offsetWidth, h = tip.offsetHeight;
      let left = r.left - w - 12;
      if (left < 8) left = Math.min(r.left, innerWidth - w - 8);
      let top = Math.min(Math.max(8, r.top - 4), innerHeight - h - 8);
      if (left === Math.min(r.left, innerWidth - w - 8)) top = Math.min(r.bottom + 6, innerHeight - h - 8);
      tip.style.left = left + "px";
      tip.style.top = top + "px";
    }, 450);
  });
  el.addEventListener("mouseleave", () => { clearTimeout(tipTimer); $("#tip").hidden = true; });
}

// ======================================================== 파라미터 패널
function fmt(v, p) {
  let t;
  if (p.integer) t = String(Math.round(v));
  else if (p.step >= 1) t = Number(v).toFixed(1);
  else t = Number(v).toFixed(2).replace(/0$/, "").replace(/\.0$/, ".0");
  return t + (p.unit || "");
}

function paintRange(input) {
  const p = ((input.value - input.min) / (input.max - input.min)) * 100;
  input.style.setProperty("--p", p + "%");
}

function buildPanel(schema) {
  const host = $("#param-cards");
  host.innerHTML = "";
  for (const sec of schema.sections) {
    state.values[sec.name] = {};
    state.defaults[sec.name] = {};
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.section = sec.name;
    const title = document.createElement("div");
    title.className = "card-title";
    title.textContent = sec.label;
    attachTip(title, () => sec.title);
    card.appendChild(title);
    let recommended = 0;
    for (const p of sec.params) {
      state.values[sec.name][p.key] = p.value;
      state.defaults[sec.name][p.key] = p.default;
      const row = document.createElement("div");
      row.className = "param";
      row.dataset.level = p.level;
      if (p.level === "recommended") recommended++;
      const head = document.createElement("div");
      head.className = "param-head";
      const name = document.createElement("div");
      name.className = "param-name";
      const val = document.createElement("div");
      val.className = "val";
      head.append(name);
      if (p.level === "caution") {
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = "주의";
        head.append(tag);
      }
      head.append(val);
      const input = document.createElement("input");
      input.type = "range";
      input.min = 0;
      input.max = Math.round((p.high - p.low) / p.step);
      input.step = 1;
      input.setAttribute("aria-label", p.label);
      row.append(head, input);
      card.appendChild(row);
      const id = sec.name + "." + p.key;
      state.rows[id] = { input, val, name, p, section: sec.name };
      setRowValue(id, p.value);
      input.addEventListener("input", () => {
        const v = p.low + Number(input.value) * p.step;
        state.values[sec.name][p.key] = p.integer ? Math.round(v) : Math.round(v * 1e6) / 1e6;
        refreshRow(id);
        schedule(false);
      });
      input.addEventListener("change", () => schedule(true));
      name.addEventListener("dblclick", () => {
        if (state.values[sec.name][p.key] !== p.default) { setRowValue(id, p.default); schedule(true); }
      });
      attachTip(name, () => `${p.doc}\n\n${sec.name}.${p.key} · 기본값 ${p.default} · 더블클릭하면 기본값`);
    }
    card.dataset.recommended = recommended;
    host.appendChild(card);
  }
  applyAdvanced();
}

function setRowValue(id, value) {
  const r = state.rows[id];
  state.values[r.section][r.p.key] = value;
  // 설정 파일 값이 칸 사이에 있어도 값은 그대로 두고, 슬라이더만 가장 가까운 칸에
  const tick = Math.max(0, Math.min(Number(r.input.max), Math.round((value - r.p.low) / r.p.step)));
  r.input.value = tick;
  refreshRow(id);
}

function refreshRow(id) {
  const r = state.rows[id];
  const v = state.values[r.section][r.p.key];
  r.val.textContent = fmt(v, r.p);
  const changed = v !== r.p.default;
  r.name.innerHTML = "";
  if (changed) { const d = document.createElement("span"); d.className = "dot"; r.name.append(d); }
  r.name.append(document.createTextNode(r.p.label));
  paintRange(r.input);
}

function applyAdvanced() {
  for (const card of $$("#param-cards .card")) {
    let visible = 0;
    for (const row of card.querySelectorAll(".param")) {
      const show = state.advanced || row.dataset.level === "recommended";
      row.hidden = !show;
      if (show) visible++;
    }
    card.hidden = visible === 0;
  }
  $("#advanced-btn").textContent = state.advanced ? "고급 설정 숨기기" : "고급 설정 보기";
}

function setMode(mode) {
  state.mode = mode;
  for (const b of $$("#merge-seg button")) b.classList.toggle("on", b.dataset.mode === mode);
  const hint = $("#pick-hint");
  if (mode === "pick") {
    const refused = (state.result && state.result.refused || []).map((r) => r.n);
    hint.textContent = "고른 후보: " + (state.selection.length ? state.selection.join(", ") : "없음") +
      (refused.length ? ` (병합 안 됨: ${refused.join(", ")})` : "") +
      " · '병합 후보' 보기에서 선을 클릭해 켜고 끕니다";
  } else if (mode === "auto") {
    hint.textContent = "확실한 후보만 자동으로 병합합니다";
  } else hint.textContent = "";
}

// ======================================================== 계산
let debounce;
function schedule(now) {
  clearTimeout(debounce);
  if (now) run(); else debounce = setTimeout(run, 300);
}

function setBusy(on) {
  $("#busy").hidden = !on;
  $("#right-vp").classList.toggle("dim", on && !!state.result);
}

async function run() {
  if (!state.opened) return;
  const id = ++state.reqId;
  setBusy(true);
  const res = await API.process({
    id, values: state.values, selection: state.selection,
    options: { merge: state.mode, smooth: state.smooth, render: state.render },
  });
  if (!res || res.stale || id !== state.reqId) return;
  if (res.error) { setBusy(false); toast(res.error, true); return; }
  state.result = res;
  state.selection = res.selection;
  setMode(res.mode);
  const refusedNote = res.refused.length
    ? "고른 후보 중 병합하지 않은 것 (애매하면 합치지 않음):\n" + res.refused.map((r) => `${r.n}번 — ${r.reason}`).join("\n")
    : "";
  if (res.notice || refusedNote) toast([res.notice, refusedNote].filter(Boolean).join("\n"), !!refusedNote);
  await showView(id);
  if (id !== state.reqId) return;
  setBusy(false);
  const took = Object.entries(res.timings).map(([k, v]) => `${k} ${v.toFixed(2)}초`).join(" · ");
  $("#timing").textContent = took || "지난 결과 재사용";
  $("#notes").textContent = res.notes.join(" · ");
  renderChips(null);
  const ch = await API.checks(id);
  if (!ch || ch.stale || id !== state.reqId) return;
  state.checkedId = id;
  renderChips(ch.checks);
}

function renderChips(checks) {
  const host = $("#chips");
  host.innerHTML = "";
  if (!checks) {
    const c = document.createElement("span");
    c.className = "chip";
    c.textContent = "검사 중…";
    host.append(c);
    return;
  }
  for (const k of checks) {
    const c = document.createElement("span");
    c.className = "chip " + (k.ok ? "ok" : "bad");
    c.textContent = (k.ok ? "✓ " : "✕ ") + k.label;
    attachTip(c, () => k.tip);
    host.append(c);
  }
}

async function showView(id = state.reqId) {
  const res = state.result;
  $("#view-title").textContent = VIEW_TITLES[state.view];
  $("#labels").innerHTML = "";
  if (!res) return;
  let src = res.image;
  if (state.view !== "result") {
    if (state.view === "candidates" && state.mode === "off") {
      $("#view-title").textContent = "병합 후보 — 병합을 '자동'이나 '고르기'로 켜면 보입니다";
    } else {
      const v = await API.view(res.id, state.view);
      if (!v || v.stale || id !== state.reqId) return;
      src = v.image;
    }
  }
  setImage("#right-vp", src);
  $("#right-empty").hidden = true;
  if (state.view === "candidates") drawLabels();
}

// ======================================================== 이미지·확대
function setImage(vp, src) {
  const canvas = $(vp + " .canvas");
  const img = canvas.querySelector("img");
  img.src = src;
  canvas.classList.add("has-img");
}

function applyZoom() {
  const { s, tx, ty } = state.z;
  for (const c of $$(".canvas")) {
    c.style.transform = `translate(${tx}px, ${ty}px) scale(${s})`;
    c.style.setProperty("--cell", 12 / s + "px");
    c.classList.toggle("pixels", s >= 2);
  }
  $("#zoom-read").textContent = Math.round(s * 100) + "%";
  if (state.view === "candidates") drawLabels();
}

function vpSize() {
  const r = $("#right-vp").getBoundingClientRect();
  return { w: r.width, h: r.height };
}

function fit() {
  if (!state.imgW) return;
  const { w, h } = vpSize();
  if (w < 2 || h < 2) { requestAnimationFrame(fit); return; }  // 아직 배치 전
  state.userZoomed = false;
  const s = Math.min(w / state.imgW, h / state.imgH) * 0.96;
  state.z = { s, tx: (w - state.imgW * s) / 2, ty: (h - state.imgH * s) / 2 };
  applyZoom();
}

function zoomAt(factor, cx, cy) {
  state.userZoomed = true;
  const { s, tx, ty } = state.z;
  const ns = Math.min(32, Math.max(0.05, s * factor));
  state.z = { s: ns, tx: cx - (cx - tx) * (ns / s), ty: cy - (cy - ty) * (ns / s) };
  applyZoom();
}

function actual() {
  const { w, h } = vpSize();
  zoomAt(1 / state.z.s, w / 2, h / 2);
}

function drawLabels() {
  const host = $("#labels");
  host.innerHTML = "";
  const res = state.result;
  if (!res || state.view !== "candidates" || state.mode === "off") return;
  const { s, tx, ty } = state.z;
  for (const c of res.candidates) {
    const el = document.createElement("div");
    const refused = c.picked && !c.merged;
    el.className = "label" + (c.merged ? " merged" : refused ? " refused" : "");
    el.textContent = refused ? `${c.n} ✕` : c.n;
    el.style.left = tx + (c.x + 0.5) * s + "px";
    el.style.top = ty + (c.y + 0.5) * s + "px";
    el.title = c.merged ? `후보 ${c.n}: 병합됨`
      : refused ? `후보 ${c.n}: 골랐지만 병합하지 않음 — ${c.reason}`
      : `후보 ${c.n}: 병합 안 함 (클릭해서 고르기)`;
    el.addEventListener("pointerdown", (e) => e.stopPropagation());
    el.addEventListener("click", (e) => { e.stopPropagation(); toggleCandidate({ number: c.n }); });
    host.append(el);
  }
}

async function toggleCandidate({ number = null, y = null, x = null }) {
  if (!state.result || state.reqId !== state.result.id) return;
  const r = await API.toggle_candidate(number, y, x, state.selection);
  if (!r || r.none) return;
  if (r.error) { toast(r.error, true); return; }
  state.selection = r.selection;
  setMode("pick");
  if (r.notice) toast(r.notice);
  schedule(true);
}

function setupViewport(vp) {
  let drag = null;
  vp.addEventListener("wheel", (e) => {
    if (!state.imgW) return;
    e.preventDefault();
    const r = vp.getBoundingClientRect();
    zoomAt(Math.exp(-e.deltaY * 0.0022), e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });
  vp.addEventListener("pointerdown", (e) => {
    if (!state.imgW || e.button !== 0) return;
    drag = { x: e.clientX, y: e.clientY, tx: state.z.tx, ty: state.z.ty, moved: 0 };
    vp.setPointerCapture(e.pointerId);
    vp.classList.add("dragging");
  });
  vp.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.moved = Math.max(drag.moved, Math.abs(dx) + Math.abs(dy));
    state.userZoomed = true;
    state.z.tx = drag.tx + dx;
    state.z.ty = drag.ty + dy;
    applyZoom();
  });
  vp.addEventListener("pointerup", (e) => {
    if (!drag) return;
    vp.classList.remove("dragging");
    const click = drag.moved < 4;
    drag = null;
    if (click && vp.id === "right-vp" && state.view === "candidates" && state.mode !== "off") {
      const r = vp.getBoundingClientRect();
      const x = (e.clientX - r.left - state.z.tx) / state.z.s;
      const y = (e.clientY - r.top - state.z.ty) / state.z.s;
      toggleCandidate({ y, x });
    }
  });
}

// ======================================================== 열기·저장
function opened(res) {
  if (!res || res.cancelled || res.empty) return;
  if (res.error) { toast(res.error, true); return; }
  state.opened = true;
  state.result = null;
  state.selection = [];
  state.imgW = res.width;
  state.imgH = res.height;
  setMode(state.mode === "pick" ? "pick" : state.mode);
  $("#drop").hidden = true;
  $("#file-chip").hidden = false;
  $("#file-chip").textContent = `${res.name} · ${res.width}×${res.height}`;
  $("#chips").innerHTML = "";
  $("#notes").textContent = "";
  $("#timing").textContent = "";
  $("#labels").innerHTML = "";
  $("#right-vp .canvas img").removeAttribute("src");
  $("#right-vp .canvas").classList.remove("has-img");
  $("#right-empty").hidden = false;
  setImage("#left-vp", res.image);
  requestAnimationFrame(fit);
  if (res.notice) toast(res.notice);
  schedule(true);
}

async function openDialog() {
  opened(await API.open_dialog());
}

async function save() {
  if (!state.result) { toast("먼저 PNG를 여세요.", true); return; }
  if (state.checkedId !== state.reqId) { toast("계산이 끝난 뒤 저장할 수 있습니다.", true); return; }
  const r = await API.save($("#sheet-sw").checked, $("#settings-sw").checked, null);
  if (!r || r.cancelled) return;
  if (r.error) { toast(r.error, true); return; }
  const names = r.files.map((f) => f.split(/[\\/]/).pop());
  toast(`저장했습니다: ${names.join(", ")}\n${r.folder}`);
}

function setupDrop() {
  // 끌어다 놓기는 원본 칸에서만 받는다. 결과 칸은 금지 표시만 보이고 아무것도 열지 않는다.
  // 창 모드에서는 Python이 원본 칸의 drop을 받는다 (실제 파일 경로가 필요). 브라우저 모드만 여기서 연다.
  const browser = document.documentElement.dataset.mode === "browser";
  const vp = $("#left-vp");
  vp.addEventListener("dragenter", (e) => { e.preventDefault(); vp.classList.add("over"); });
  vp.addEventListener("dragover", (e) => { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; vp.classList.add("over"); });
  vp.addEventListener("dragleave", (e) => { if (!vp.contains(e.relatedTarget)) vp.classList.remove("over"); });
  vp.addEventListener("drop", (e) => {
    vp.classList.remove("over");
    if (!browser) return;
    e.preventDefault();
    e.stopPropagation();
    const file = Array.from(e.dataTransfer.files).find((f) => f.name.toLowerCase().endsWith(".png"));
    if (!file) { toast("PNG 파일만 열 수 있습니다.", true); return; }
    const reader = new FileReader();
    reader.onload = async () => opened(await API.open_bytes(file.name, reader.result));
    reader.readAsDataURL(file);
  });
  // 그 밖의 곳(결과 칸 포함): 브라우저가 파일을 열어 버리지 않게 막고, 놓을 수 없음을 표시
  document.addEventListener("dragover", (e) => {
    if (vp.contains(e.target)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "none";
  });
  document.addEventListener("drop", (e) => { if (!vp.contains(e.target)) e.preventDefault(); });
}

// ======================================================== 시작
async function main() {
  await connect();
  state.schema = await API.schema();
  buildPanel(state.schema);
  setMode("auto");

  for (const b of $$("#view-seg button")) {
    b.addEventListener("click", () => {
      state.view = b.dataset.view;
      for (const o of $$("#view-seg button")) o.classList.toggle("on", o === b);
      showView();
    });
  }
  for (const b of $$("#merge-seg button")) {
    b.addEventListener("click", () => { if (state.mode !== b.dataset.mode) { setMode(b.dataset.mode); schedule(true); } });
  }
  $("#smooth-sw").addEventListener("change", (e) => { state.smooth = e.target.checked; schedule(true); });
  $("#render-sw").addEventListener("change", (e) => { state.render = e.target.checked; schedule(true); });
  $("#advanced-btn").addEventListener("click", () => { state.advanced = !state.advanced; applyAdvanced(); });
  $("#reset-btn").addEventListener("click", () => {
    for (const id in state.rows) setRowValue(id, state.rows[id].p.default);
    schedule(true);
    toast("모든 파라미터를 기본값으로 되돌렸습니다");
  });
  $("#open-btn").addEventListener("click", openDialog);
  $("#drop-open").addEventListener("click", openDialog);
  $("#save-btn").addEventListener("click", save);
  $("#save-top").addEventListener("click", save);
  $("#zoom-fit").addEventListener("click", fit);
  $("#zoom-100").addEventListener("click", actual);
  $("#zoom-in").addEventListener("click", () => { const { w, h } = vpSize(); zoomAt(1.25, w / 2, h / 2); });
  $("#zoom-out").addEventListener("click", () => { const { w, h } = vpSize(); zoomAt(0.8, w / 2, h / 2); });
  setupViewport($("#left-vp"));
  setupViewport($("#right-vp"));
  setupDrop();
  addEventListener("resize", () => { if (state.imgW) (state.userZoomed ? applyZoom() : fit()); });
  addEventListener("keydown", (e) => {
    const mod = e.metaKey || e.ctrlKey;
    if (mod && e.key.toLowerCase() === "o") { e.preventDefault(); openDialog(); }
    else if (mod && e.key.toLowerCase() === "s") { e.preventDefault(); save(); }
    else if (!mod && e.target.tagName !== "INPUT") {
      if (e.key === "0") fit();
      if (e.key === "1") actual();
    }
  });
  opened(await API.current());
}

window.lc = { opened, toast };
main().catch((e) => toast("화면을 시작하지 못했습니다: " + e, true));
