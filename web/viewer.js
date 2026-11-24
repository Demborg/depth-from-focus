// Desktop viewer: pairs with the phone (QR + encrypted MQTT handshake + LAN WebRTC, from stopmotion),
// triggers a fast focus sweep, runs depth-from-focus in wasm, and renders a point cloud levelled
// with the phone's orientation sensor.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RemoteLink } from "./net/link.js";
import { pairString, parsePairString } from "./net/signal.js";
import { laggedDiopters } from "./dff-wasm.js";

const $ = (id) => document.getElementById(id);
const PW = 540, PH = 960; // colour copy kept per frame for the all-in-focus colours (half of 1080x1920)
// Pixel 9 main camera, 1080x1920 preview stream. Calibrated from the cutting mat's 1 cm grid
// (tools/calib_fov.py): f = 890 px at 540x960, i.e. ~57 deg across the long side.
const F_LONG_PX = 1780;

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
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

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

// URL overrides for comparisons, e.g. ?agg=0&peak=0&smooth=4&norm=0 approximates the old pipeline.
function urlOpts() {
  const q = new URLSearchParams(location.search), opts = {};
  for (const k of ["win", "halvings", "smooth", "measure", "peak", "agg", "norm"])
    if (q.has(k)) opts[{ smooth: "smoothR", peak: "peakR", agg: "aggR" }[k] ?? k] = +q.get(k);
  return opts;
}

function newIngest({ lag = true, orientation = null } = {}) {
  let chain = Promise.resolve(), pushed = 0, workMs = 0, lastPush = 0;
  worker.postMessage({ type: "begin", opts: urlOpts(), lag });
  startLive(orientation);
  const progress = new Promise((res) => {
    onWorker = (m) => {
      if (m.type === "points") addLivePoints(m);
      if (m.type === "pushed") { pushed = m.n; workMs += m.ms; lastPush = performance.now(); $("s-wasm").textContent = `frame ${m.n} · ${(live?.count ?? 0).toLocaleString()} points so far`; }
      if (m.type === "result") res(m);
      if (m.type === "error") ui.log("worker:", m.message);
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
      const tFinish = performance.now();
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
    live.f = F_LONG_PX * (m.h / 1920);
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
const view = { depth: "smooth", conf: 1.6, size: 2, level: true, edge: 0.06 };

// Flying pixels: drop points whose 3x3 neighbourhood spans more than `rel` of their depth.
// Separable 3x3 min/max on depth (computed once), so it is a few linear passes.
function discontinuities(Z, w, h, rel) {
  const out = new Uint8Array(w * h);
  if (!(rel > 0)) return out;
  const lo = new Float32Array(w * h), hi = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 1; x < w - 1; x++) {
    const p = y * w + x, a = Z[p - 1], b = Z[p], c = Z[p + 1];
    lo[p] = Math.min(a, b, c); hi[p] = Math.max(a, b, c);
  }
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const p = y * w + x;
    const l = Math.min(lo[p - w], lo[p], lo[p + w]), u = Math.max(hi[p - w], hi[p], hi[p + w]);
    out[p] = u - l > rel * l ? 1 : 0;
  }
  return out;
}

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
  const f = F_LONG_PX * (r.h / 1920), cx = r.w / 2, cy = r.h / 2;
  const T = makeTransform(meta);
  // T is linear (rotation + axis flips), so bake it into a 3x3 matrix for the inner loop.
  const [c0, c1, c2] = [T(1, 0, 0), T(0, 1, 0), T(0, 0, 1)];
  const n = r.w * r.h;
  const Z = new Float32Array(n);
  for (let p = 0; p < n; p++) Z[p] = 1 / Math.max(1e-3, d[p]);
  const flying = discontinuities(Z, r.w, r.h, view.edge);
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), zs = new Float32Array(n);
  let k = 0;
  for (let y = 0; y < r.h; y++) {
    const v = (y + 0.5 - cy) / f;
    for (let x = 0; x < r.w; x++) {
      const p = y * r.w + x;
      if (r.conf[p] < view.conf || !(d[p] > 0.2) || flying[p] || Z[p] > 6) continue;
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
  const sorted = zs.subarray(0, k).slice().sort();
  const q = (t) => sorted[Math.floor(t * (sorted.length - 1))] ?? 0;
  $("s-range").textContent = sorted.length ? `${(q(0.02) * 100).toFixed(0)}–${(q(0.98) * 100).toFixed(0)} cm` : "–";

  // Horizontal grid at the lowest dense level (the table when levelled), and the phone itself.
  const ys = new Float32Array(k); for (let i = 0; i < k; i++) ys[i] = pos[i * 3 + 1];
  ys.sort();
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
bindRange("r-size", "o-size", (v) => v.toFixed(2), (v) => { view.size = v; if (cloud) cloud.material.size = v; });
$("c-level").addEventListener("change", (e) => { view.level = e.target.checked; if (capture) capture.framed = false; build(); });
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
// Main-thread long tasks (>50 ms), for checking that depth work stays off the page.
const longTasks = [];
try { new PerformanceObserver((l) => longTasks.push(...l.getEntries().map((e) => Math.round(e.duration)))).observe({ type: "longtask" }); } catch {}
window.viewer = { longTasks, sweep: () => $("b-sweep").click(), link, camera, controls, get capture() { return capture; } };

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
