// Desktop viewer: pairs with the phone (QR + encrypted MQTT handshake + LAN WebRTC, from stopmotion),
// triggers a fast focus sweep, runs depth-from-focus on the GPU (WebGPU), and renders a point cloud levelled
// with the phone's orientation sensor.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RemoteLink } from "./net/link.js";
import { pairString, parsePairString } from "./net/signal.js";
import { laggedDiopters } from "./lag.js";
import { focalPx, cloudMask } from "./cloud.js";

const $ = (id) => document.getElementById(id);

const ui = {
  log(...a) { const s = a.map((x) => (typeof x === "string" ? x : JSON.stringify(x))).join(" "); $("log").textContent = (s + "\n" + $("log").textContent).slice(0, 4000); console.log("[viewer]", s); },
  status(text, cls = "") { $("status-text").textContent = text; $("status").className = "status " + cls; },
};
window.viewerLog = ui.log;

// ---------------------------------------------------------------- 3D scene
// Point colours are camera sRGB bytes: pass them straight through instead of colour-managing them.
THREE.ColorManagement.enabled = false;
const renderer = new THREE.WebGLRenderer({ canvas: $("gl"), antialias: true });
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.setPixelRatio(devicePixelRatio);
const scene = new THREE.Scene();
// Scene background follows the page theme (--scene-bg in site.css).
const sceneBg = () => { scene.background = new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue("--scene-bg").trim() || "#0f1115"); };
sceneBg();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", sceneBg);
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 50);
camera.position.set(0, 0.2, 0.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
let cloud = null, grid = null, phoneGizmo = null;

function resize() {
  const r = $("gl").getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(1, r.height);
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();
renderer.setAnimationLoop(() => { controls.update(); measure.frame(); renderer.render(scene, camera); });

// ---------------------------------------------------------------- capture state
// Depth runs in a module worker (web/dff-worker.js): frames are decoded off the main thread and
// processed as they arrive, so the page stays responsive and little work is left after the sweep.
const worker = new Worker(new URL("./dff-worker.js", import.meta.url), { type: "module" });
let capture = null; // { meta, result, colors }
let pending = null; // capture in progress (frames arriving over the wire)
let phoneOrientation = null; // from the phone's hello; frame 0 of each sweep carries a fresh one
let onWorker = () => {};
worker.onmessage = ({ data }) => onWorker(data);
worker.onerror = (e) => ui.log("worker error:", e.message);

// URL overrides for comparisons, e.g. ?halvings=1 for half resolution, ?agg=0&norm=0 without aggregation.
function urlOpts() {
  const q = new URLSearchParams(location.search), opts = {};
  for (const k of ["win", "halvings", "peak", "agg", "eps", "norm"])
    if (q.has(k)) opts[{ peak: "peakR", agg: "aggR", eps: "aggEps" }[k] ?? k] = +q.get(k);
  return opts;
}

function newIngest({ lag = true, orientation = null } = {}) {
  let chain = Promise.resolve(), pushed = 0, workMs = 0, lastPush = 0, tFinish = 0;
  worker.postMessage({ type: "begin", opts: urlOpts(), lag });
  startLive(orientation);
  const progress = new Promise((res) => {
    onWorker = (m) => {
      if (m.type === "points") addLivePoints(m);
      if (m.type === "pushed") { pushed = m.n; workMs += m.ms; lastPush = performance.now(); $("s-wasm").textContent = `frame ${m.n} · ${(live?.count ?? 0).toLocaleString()} points so far`; }
      if (m.type === "result") res(m);
      if (m.type === "error") { ui.log("worker:", m.message); ui.status(`Depth failed: ${m.message}`, "bad"); }
    };
  });
  return {
    // Decode in capture order (createImageBitmap runs off-thread) and hand the bitmap to the worker.
    add(i, blob, meta) {
      if (i === 0 && meta?.orientation && live) live.orientation = meta.orientation;
      chain = chain.then(async () => {
        const bitmap = await createImageBitmap(blob);
        worker.postMessage({ type: "frame", i, bitmap, meta }, [bitmap]);
      });
      return chain;
    },
    async finish(meta) {
      await chain;
      const t = meta.frames.every((f) => f.t_req_ms != null) && meta.params?.settle === 0;
      const dio = t ? laggedDiopters(meta.frames) : meta.frames.map((f) => 1 / f.focus_requested);
      tFinish = performance.now();
      worker.postMessage({ type: "finish", diopters: dio });
      const m = await progress;
      const waited = performance.now() - tFinish;
      const liveCount = live?.count ?? 0;
      stopLive();
      // Re-frame on the refined cloud unless the user has taken over the camera: the live framing
      // was based on the first points to arrive (the nearest surfaces), so it is off-centre.
      capture = { meta, result: m.result, colors: m.colours, framed: framedOnce && userMoved };
      const r = m.result;
      $("s-sweep").textContent = `${meta.frames.length} frames · ${(meta.total_ms / 1000).toFixed(1)} s`;
      $("s-wasm").textContent = `+${(waited / 1000).toFixed(2)} s${t ? ", lag-corr." : ""}`;
      $("s-ori").textContent = meta.orientation ? meta.orientation.source.replace("Sensor", "") : "none";
      ui.log(`${liveCount.toLocaleString()} points streamed during the sweep, then refined`);
      ui.log(`depth ${r.w}x${r.h}: per-frame ${(r.pushMs / 1000).toFixed(1)} s in background, volume ${(r.finishMs / 1000).toFixed(2)} s, waited ${(waited / 1000).toFixed(2)} s after last frame`);
      build();
      return { waited, r };
    },
  };
}

// ---------------------------------------------------------------- live points during a sweep
// Points arrive in batches as soon as each pixel's focus peak is behind the sweep (decided from
// raw curves); they are drawn into a preallocated buffer and replaced by the refined cloud at the end.
let live = null, framedOnce = false, userMoved = false;
controls.addEventListener("start", () => { userMoved = true; });

function startLive(orientation) {
  stopLive();
  measure.clear();
  if (cloud) cloud.visible = false;
  live = { orientation, count: 0 };
}

function stopLive() {
  if (live?.points) { scene.remove(live.points); live.points.geometry.dispose(); }
  live = null;
  if (cloud) cloud.visible = true;
}

function addLivePoints(m) {
  if (!live) return;
  if (!live.points) {
    const n = m.w * m.h;
    live.pos = new Float32Array(n * 3);
    live.col = new Float32Array(n * 3);
    live.f = focalPx(m.w, m.h);
    live.cx = m.w / 2; live.cy = m.h / 2; live.w = m.w;
    const T = makeTransform({ orientation: live.orientation });
    live.M = [T(1, 0, 0), T(0, 1, 0), T(0, 0, 1)];
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(live.pos, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute("color", new THREE.BufferAttribute(live.col, 3).setUsage(THREE.DynamicDrawUsage));
    g.setDrawRange(0, 0);
    live.points = new THREE.Points(g, new THREE.PointsMaterial({ size: view.size, sizeAttenuation: false, vertexColors: true }));
    live.points.frustumCulled = false;
    scene.add(live.points);
    $("empty").hidden = true;
    if (phoneGizmo) scene.remove(phoneGizmo);
    phoneGizmo = frustum(T, live.f, m.w, m.h);
    scene.add(phoneGizmo);
  }
  const { pos, col, f, cx, cy, w, M: [c0, c1, c2] } = live;
  const start = live.count;
  let k = start;
  for (let i = 0; i < m.px.length; i++) {
    const z = m.z[i];
    if (!(z > 0 && z < 6)) continue;
    const p = m.px[i], x = p % w, y = (p - x) / w;
    const X = ((x + 0.5 - cx) / f) * z, Y = ((y + 0.5 - cy) / f) * z, o = k * 3;
    pos[o] = c0[0] * X + c1[0] * Y + c2[0] * z;
    pos[o + 1] = c0[1] * X + c1[1] * Y + c2[1] * z;
    pos[o + 2] = c0[2] * X + c1[2] * Y + c2[2] * z;
    col[o] = m.rgb[i * 3] / 255; col[o + 1] = m.rgb[i * 3 + 1] / 255; col[o + 2] = m.rgb[i * 3 + 2] / 255;
    k++;
  }
  live.count = k;
  const g = live.points.geometry;
  for (const name of ["position", "color"]) {
    const a = g.getAttribute(name);
    a.clearUpdateRanges?.();
    a.addUpdateRange(start * 3, (k - start) * 3);
    a.needsUpdate = true;
  }
  g.setDrawRange(0, k);
  // Frame the view on the first capture once there is enough to look at.
  if (!framedOnce && k > 4000) {
    let sx = 0, sy = 0, sz = 0;
    for (let i = 0; i < k; i++) { sx += pos[i * 3]; sy += pos[i * 3 + 1]; sz += pos[i * 3 + 2]; }
    controls.target.set(sx / k, sy / k, sz / k);
    camera.position.set(0.3, 0.22, 0.05); // just beside and above the phone, so depth reads
    framedOnce = true;
  }
}

// ---------------------------------------------------------------- geometry
const view = { depth: "smooth", conf: 1.6, size: 2, level: true, edge: 0.06, blob: 0.002 };

function quatRotate([qx, qy, qz, qw], [x, y, z]) {
  // v' = q v q*
  const tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
  return [x + qw * tx + (qy * tz - qz * ty), y + qw * ty + (qz * tx - qx * tz), z + qw * tz + (qx * ty - qy * tx)];
}

// Camera coords (x right, y down, z forward) -> three.js world (y up).
// Device frame: x right, y up (portrait top), z out of the screen; the back camera looks along -z.
// World (sensor): x east, y north, z up -> three: (east, up, -north). Then yaw so the phone faces -z.
function makeTransform(meta) {
  const q = view.level && meta.orientation?.quaternion;
  if (!q) return (X, Y, Z) => [X, -Y, -Z];
  const toThree = ([e, n, u]) => [e, u, -n];
  const fwd = toThree(quatRotate(q, [0, 0, -1]));
  const yaw = Math.atan2(fwd[0], -fwd[2]); // rotate so forward points to -z
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  return (X, Y, Z) => {
    const [x, y, z] = toThree(quatRotate(q, [X, -Y, -Z]));
    return [x * cy - z * sy, y, x * sy + z * cy];
  };
}

function build() {
  if (!capture) return;
  $("empty").hidden = true;
  const { result: r, colors, meta } = capture;
  const d = view.depth === "smooth" ? r.diopterSmooth : r.diopter;
  const f = focalPx(r.w, r.h), cx = r.w / 2, cy = r.h / 2;
  const T = makeTransform(meta);
  // T is linear (rotation + axis flips), so bake it into a 3x3 matrix for the inner loop.
  const [c0, c1, c2] = [T(1, 0, 0), T(0, 1, 0), T(0, 0, 1)];
  const n = r.w * r.h;
  const { Z, keep } = cloudMask(r, d, view);
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), zs = new Float32Array(n);
  let k = 0;
  for (let y = 0; y < r.h; y++) {
    const v = (y + 0.5 - cy) / f;
    for (let x = 0; x < r.w; x++) {
      const p = y * r.w + x;
      if (!keep[p]) continue;
      const z = Z[p], X = ((x + 0.5 - cx) / f) * z, Y = v * z, o = k * 3;
      pos[o] = c0[0] * X + c1[0] * Y + c2[0] * z;
      pos[o + 1] = c0[1] * X + c1[1] * Y + c2[1] * z;
      pos[o + 2] = c0[2] * X + c1[2] * Y + c2[2] * z;
      col[o] = colors[p * 3] / 255; col[o + 1] = colors[p * 3 + 1] / 255; col[o + 2] = colors[p * 3 + 2] / 255;
      zs[k++] = z;
    }
  }
  if (cloud) { scene.remove(cloud); cloud.geometry.dispose(); }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos.subarray(0, k * 3), 3));
  g.setAttribute("color", new THREE.BufferAttribute(col.subarray(0, k * 3), 3));
  const mat = new THREE.PointsMaterial({ size: view.size, sizeAttenuation: false, vertexColors: true });
  cloud = new THREE.Points(g, mat);
  scene.add(cloud);
  $("s-pts").textContent = k.toLocaleString();
  // Percentiles from every 16th point: plenty for a range readout, and cheap at 540x960.
  const sample = (get) => { const a = new Float32Array(Math.ceil(k / 16)); for (let i = 0, j = 0; i < k; i += 16) a[j++] = get(i); return a.sort(); };
  const sorted = sample((i) => zs[i]);
  const q = (t) => sorted[Math.floor(t * (sorted.length - 1))] ?? 0;
  $("s-range").textContent = sorted.length ? `${(q(0.02) * 100).toFixed(0)}–${(q(0.98) * 100).toFixed(0)} cm` : "–";

  // Horizontal grid at the lowest dense level (the table when levelled), and the phone itself.
  const ys = sample((i) => pos[i * 3 + 1]);
  const floor = ys[Math.floor(0.03 * (ys.length - 1))] ?? -0.3;
  if (grid) scene.remove(grid);
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  grid = new THREE.GridHelper(2, 40, dark ? 0x3a3f2a : 0xb9b3cc, dark ? 0x22252b : 0xdcd9e4);
  grid.position.y = floor;
  scene.add(grid);
  if (phoneGizmo) scene.remove(phoneGizmo);
  phoneGizmo = frustum(T, f, r.w, r.h);
  scene.add(phoneGizmo);

  if (!capture.framed) {
    framedOnce = true;
    const c = new THREE.Box3().setFromBufferAttribute(g.getAttribute("position")).getCenter(new THREE.Vector3());
    controls.target.copy(c);
    camera.position.set(0.3, 0.22, 0.05); // just beside and above the phone, so depth reads
    capture.framed = true;
  }
}

function frustum(T, f, w, h) {
  const s = 0.06, o = T(0, 0, 0);
  const corners = [[0, 0], [w, 0], [w, h], [0, h]].map(([x, y]) => T(((x - w / 2) / f) * s, ((y - h / 2) / f) * s, s));
  const pts = [];
  for (const c of corners) pts.push(...o, ...c);
  for (let i = 0; i < 4; i++) pts.push(...corners[i], ...corners[(i + 1) % 4]);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  const accent = matchMedia("(prefers-color-scheme: dark)").matches ? 0xe6e08a : 0x8a6fd1;
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: accent }));
}

// ---------------------------------------------------------------- controls
const bindRange = (id, out, fmt, fn) => { const el = $(id); const upd = () => { $(out).textContent = fmt(+el.value); fn(+el.value); }; el.addEventListener("input", upd); upd(); };
bindRange("r-near", "o-near", (v) => `${v} cm`, () => {});
bindRange("r-far", "o-far", (v) => `${v} cm`, () => {});
bindRange("r-conf", "o-conf", (v) => v.toFixed(2), (v) => { view.conf = v; build(); });
bindRange("r-edge", "o-edge", (v) => (v > 0 ? `${v.toFixed(1)}%` : "off"), (v) => { view.edge = v / 100; build(); });
bindRange("r-blob", "o-blob", (v) => (v > 0 ? `${v.toFixed(2)}%` : "off"), (v) => { view.blob = v / 100; build(); });
bindRange("r-size", "o-size", (v) => v.toFixed(2), (v) => { view.size = v; if (cloud) cloud.material.size = v; });
$("c-level").addEventListener("change", (e) => { view.level = e.target.checked; measure.clear(); if (capture) capture.framed = false; build(); });
document.querySelectorAll("[data-depth]").forEach((b) => b.addEventListener("click", () => {
  view.depth = b.dataset.depth;
  document.querySelectorAll("[data-depth]").forEach((o) => { o.setAttribute("aria-pressed", o === b); o.classList.toggle("outline", o !== b); });
  build();
}));

// ---------------------------------------------------------------- replay a stored stack (?stack=id)
async function replay(id) {
  ui.status(`Loading ${id}…`);
  const meta = await fetch(`stacks/${id}/meta.json`).then((r) => r.json());
  const blobs = await Promise.all(meta.frames.map((f) => fetch(`stacks/${id}/${f.file}`).then((r) => r.blob())));
  const lag = meta.frames.every((f) => f.t_req_ms != null) && meta.params?.settle === 0;
  const ingest = newIngest({ lag, orientation: meta.orientation });
  // ?realtime feeds frames at their original capture times, like a live sweep.
  const realtime = new URLSearchParams(location.search).has("realtime");
  const t0 = performance.now();
  for (const [i, b] of blobs.entries()) {
    if (realtime) await new Promise((r) => setTimeout(r, Math.max(0, meta.frames[i].t_ms - (performance.now() - t0))));
    ingest.add(i, b, meta.frames[i]);
  }
  window.viewer.lastReplay = await ingest.finish(meta);
  ui.status(`Replaying ${id}`, "linked");
}

// ---------------------------------------------------------------- pairing
const link = new RemoteLink();
const LINK_TEXT = { starting: "Starting…", waiting: "Waiting for phone", connecting: "Linking…", linked: "Phone linked", error: "No pairing broker" };

function phoneUrl(pairing) {
  const u = new URL("capture.html", location.href);
  u.hash = "pair=" + pairString(pairing);
  return u.href;
}

link.addEventListener("state", () => {
  ui.status(LINK_TEXT[link.state] ?? link.state, link.state === "linked" ? "linked" : link.state === "error" ? "bad" : "");
  $("pairbox").hidden = link.state === "linked" || !link.pairing;
  $("livebox").hidden = link.state !== "linked";
  $("b-sweep").disabled = link.state !== "linked";
});
link.addEventListener("track", (e) => { $("preview").srcObject = e.detail; $("preview").play().catch(() => {}); });
link.addEventListener("wire", (e) => {
  const wire = e.detail;
  wire.addEventListener("message", async ({ detail: m }) => {
    if (m.type === "hello") phoneOrientation = m.orientation;
    if (m.type === "hello") ui.log("phone:", m.settings?.width + "x" + m.settings?.height, m.orientation?.source ?? "no orientation");
    if (m.type === "progress") { $("prog").max = m.n; $("prog").value = m.k; }
    if (m.type === "frame") pending.add(m.i, m.file, m.meta);
    if (m.type === "meta") {
      const ingest = pending;
      pending = null;
      ui.status("Computing depth…", "linked");
      await new Promise((r) => setTimeout(r, 20));
      await ingest.finish(m.meta);
      ui.status("Phone linked", "linked");
      $("b-sweep").disabled = false;
    }
    if (m.type === "error") { ui.log("phone error:", m.message); $("b-sweep").disabled = false; }
  });
});

$("b-sweep").addEventListener("click", () => {
  if (!link.wire) return;
  pending = newIngest({ lag: true, orientation: phoneOrientation });
  $("b-sweep").disabled = true;
  $("prog").value = 0;
  capture && (capture.framed = capture.framed); // keep the current view while re-capturing
  const lo = +$("r-near").value / 100, hi = +$("r-far").value / 100;
  ui.status("Sweeping…", "linked");
  link.wire.send({ type: "sweep", params: { n: 40, lo, hi, settle: 0, skip: 0 } });
});
// ---------------------------------------------------------------- measuring
// Click two points on the cloud to measure the straight-line distance between them (metres, in
// the levelled world frame, so the vertical component is height above/below). Picking is done in
// screen space: the nearest-to-camera point within a few pixels of the cursor.
const measure = (() => {
  const accent = () => (matchMedia("(prefers-color-scheme: dark)").matches ? 0xe6e08a : 0x8a6fd1);
  const segs = []; // { a, b, line, label }
  let on = false, first = null, firstDot = null, hover = null, hoverXY = null;
  const dotMat = () => new THREE.PointsMaterial({ size: 9, sizeAttenuation: false, color: accent(), depthTest: false, transparent: true });
  const dot = (v) => {
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute([v.x, v.y, v.z], 3));
    const d = new THREE.Points(g, dotMat()); d.renderOrder = 10; scene.add(d); return d;
  };
  hover = dot(new THREE.Vector3()); hover.visible = false; hover.material.opacity = 0.6;

  function activeCloud() {
    if (live?.points) return { pos: live.pos, n: live.count };
    if (cloud?.visible) { const a = cloud.geometry.getAttribute("position"); return { pos: a.array, n: a.count }; }
    return null;
  }

  function pick(clientX, clientY, radius = 7) {
    const c = activeCloud(); if (!c) return null;
    const rect = renderer.domElement.getBoundingClientRect();
    const px = clientX - rect.left, py = clientY - rect.top, W = rect.width, H = rect.height;
    camera.updateMatrixWorld();
    const e = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).elements;
    const r2 = radius * radius, { pos, n } = c;
    let best = -1, bestW = Infinity;
    for (let i = 0; i < n; i++) {
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      const w = e[3] * x + e[7] * y + e[11] * z + e[15];
      if (w <= 0 || w >= bestW) continue;
      const sx = ((e[0] * x + e[4] * y + e[8] * z + e[12]) / w * 0.5 + 0.5) * W;
      const sy = (0.5 - (e[1] * x + e[5] * y + e[9] * z + e[13]) / w * 0.5) * H;
      const dx = sx - px, dy = sy - py;
      if (dx * dx + dy * dy <= r2) { best = i; bestW = w; }
    }
    return best < 0 ? null : new THREE.Vector3(pos[best * 3], pos[best * 3 + 1], pos[best * 3 + 2]);
  }

  const fmt = (m) => (m < 1 ? `${(m * 100).toFixed(1)} cm` : `${m.toFixed(3)} m`);
  function addSeg(a, b) {
    const g = new THREE.BufferGeometry().setFromPoints([a, b]);
    const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: accent(), depthTest: false, transparent: true }));
    line.renderOrder = 10; scene.add(line);
    const label = document.createElement("div"); label.className = "mlabel"; label.textContent = fmt(a.distanceTo(b));
    $("labels").append(label);
    segs.push({ a, b, line, label, dots: [dot(a), dot(b)] });
    list();
  }
  function list() {
    $("m-list").innerHTML = segs.map(({ a, b }) => {
      const d = b.clone().sub(a), hz = Math.hypot(d.x, d.z);
      return `<li>${fmt(a.distanceTo(b))} <span>· horiz ${fmt(hz)} · vert ${fmt(Math.abs(d.y))}</span></li>`;
    }).join("");
    $("b-mclear").disabled = !segs.length && !first;
  }
  function clearFirst() { if (firstDot) { scene.remove(firstDot); firstDot.geometry.dispose(); } first = firstDot = null; }
  function clear() {
    clearFirst();
    for (const s of segs) { scene.remove(s.line, ...s.dots); s.line.geometry.dispose(); s.dots.forEach((d) => d.geometry.dispose()); s.label.remove(); }
    segs.length = 0; list();
  }
  function setOn(v) {
    on = v; if (!on) { clearFirst(); hover.visible = false; list(); }
    const b = $("b-measure"); b.setAttribute("aria-pressed", on); b.classList.toggle("outline", !on);
    document.querySelector("main.view").classList.toggle("measuring", on);
    $("hint-text").textContent = on ? "click two points to measure · drag still orbits · Esc to stop"
      : "drag to orbit · right-drag to pan · scroll to zoom · M to measure";
  }
  function click(x, y) {
    const p = pick(x, y); if (!p) return;
    if (!first) { first = p; firstDot = dot(p); list(); return; }
    const a = first; clearFirst(); addSeg(a, p);
  }
  // Labels follow their segment midpoints; the hover dot snaps to the point under the cursor.
  const v = new THREE.Vector3();
  function frame() {
    const rect = renderer.domElement.getBoundingClientRect();
    for (const s of segs) {
      v.copy(s.a).add(s.b).multiplyScalar(0.5).project(camera);
      const vis = v.z < 1 && Math.abs(v.x) < 1.2 && Math.abs(v.y) < 1.2;
      s.label.hidden = !vis;
      if (vis) { s.label.style.left = `${(v.x * 0.5 + 0.5) * rect.width}px`; s.label.style.top = `${(0.5 - v.y * 0.5) * rect.height}px`; }
    }
    if (on && hoverXY) {
      const p = pick(hoverXY[0], hoverXY[1]); hoverXY = null;
      hover.visible = !!p;
      if (p) { hover.geometry.getAttribute("position").setXYZ(0, p.x, p.y, p.z); hover.geometry.getAttribute("position").needsUpdate = true; }
    }
  }

  const el = renderer.domElement;
  let down = null;
  el.addEventListener("pointerdown", (e) => { if (e.button === 0) down = [e.clientX, e.clientY]; });
  el.addEventListener("pointerup", (e) => {
    if (on && down && e.button === 0 && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 4) click(e.clientX, e.clientY);
    down = null;
  });
  el.addEventListener("pointermove", (e) => { if (on && e.buttons === 0) hoverXY = [e.clientX, e.clientY]; });
  el.addEventListener("pointerleave", () => { hover.visible = false; });
  $("b-measure").addEventListener("click", () => setOn(!on));
  $("b-mclear").addEventListener("click", clear);
  addEventListener("keydown", (e) => {
    if (e.target.closest?.("input, textarea")) return;
    if (e.key === "m" || e.key === "M") setOn(!on);
    else if (e.key === "Escape") { if (first) { clearFirst(); list(); } else setOn(false); }
  });
  return { clear, frame, get segments() { return segs.map(({ a, b }) => ({ a: a.toArray(), b: b.toArray(), d: a.distanceTo(b) })); }, pick, click, setOn };
})();
// Main-thread long tasks (>50 ms), for checking that depth work stays off the page.
const longTasks = [];
try { new PerformanceObserver((l) => longTasks.push(...l.getEntries().map((e) => Math.round(e.duration)))).observe({ type: "longtask" }); } catch {}
window.viewer = { longTasks, measure, sweep: () => $("b-sweep").click(), link, camera, controls, get capture() { return capture; } };

const stackId = new URLSearchParams(location.search).get("stack");
if (stackId) {
  replay(stackId).catch((e) => ui.log("replay failed:", String(e)));
} else {
  link.start(parsePairString(sessionStorage.getItem("dff.pair"))).then((pairing) => {
    sessionStorage.setItem("dff.pair", pairString(pairing));
    const url = phoneUrl(pairing);
    const qr = qrcode(0, "M"); qr.addData(url); qr.make();
    $("qr").innerHTML = qr.createSvgTag({ cellSize: 4, margin: 0, scalable: true });
    $("pairurl").textContent = url;
    $("pairbox").hidden = false;
    window.viewer.phoneUrl = url;
  }).catch((e) => { ui.status(e.message, "bad"); });
}
