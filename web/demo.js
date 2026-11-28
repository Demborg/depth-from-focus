// Demo loop for the video: a stored capture (dev server /stacks/<id>) played as
//   first frame -> focus sweep, with a laser-like band where the scene is in focus and each pixel
//   turning into its 3D point once the band has passed it -> orbit around the cloud -> back at the
//   capture pose, the band sweeps back and the points dissolve into the first frame again, where
//   the loop starts over.
// From the capture pose the cloud lines up with the photo pixel for pixel, which is what makes the
// hand-overs seamless.
//
//   demo.html?stack=<id>            plays the loop in real time
//   demo.html?stack=<id>&export     waits for demo.exportFrames(name, from, to) (tools/demo_video.sh)
// Framing: &aspect= (output width/height, default 1 = square) cropped from the capture at &at=
//          (0 = left edge .. 1 = right edge, default 0.85) and &aty= (0 = top .. 1 = bottom, 0.5);
//          &aspect=0 keeps the whole frame.
// Cloud filters: &edge= &blob= (fractions), as in the viewer.
// Tuning: &scan= &orbit= (seconds) &yaw= &pitch= (degrees) &sigma= (band, diopters) &core= (white
//         line width / sigma) &scale= &fps=
//         and the pipeline options as in the viewer (&halvings=1, ...).
import * as THREE from "three";
import { createDff } from "./dff-gpu.js";
import { laggedDiopters } from "./lag.js";
import { focalPx, cloudMask } from "./cloud.js";

const q = new URLSearchParams(location.search);
const num = (k, d) => (q.has(k) ? +q.get(k) : d);
const id = q.get("stack") ?? "lag_up";
const FPS = num("fps", 30), SCALE = num("scale", 1), SIGMA = num("sigma", 0.09);
const GLOW = "vec3(0.24, 1.0, 0.47)"; // the capture page's "in focus now" green
const CORE = num("core", 0.18);        // width of the white-hot centre line, as a fraction of sigma
// The band, "laser" style: a green glow of width sigma with a thin white core. a = distance from
// the band in diopters, w = mask (0..1). Shared by both shaders.
const LASER = `
  uniform float core;
  vec3 laser(vec3 c, float a, float sigma, float w) {
    float g = exp(-0.5 * a * a / (sigma * sigma)) * w;
    float s = core * sigma, h = exp(-0.5 * a * a / (s * s)) * w;
    c = mix(c, ${GLOW}, clamp(0.75 * g, 0.0, 0.85));
    c += ${GLOW} * 0.6 * h;               // green bloom hugging the core
    return mix(c, vec3(1.0), clamp(1.4 * h, 0.0, 1.0));
  }`;
const statusEl = document.getElementById("status");
const status = (t) => { statusEl.textContent = t; console.log("[demo]", t); };

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
const smoother = (t) => t * t * t * (t * (6 * t - 15) + 10);

async function main() {
  const meta = await fetch(`stacks/${id}/meta.json`).then((r) => r.json());
  const bitmap = (file) => fetch(`stacks/${id}/${file}`).then((r) => r.blob()).then((b) => createImageBitmap(b));

  // Depth, exactly as the phone computes it.
  status("computing depth…");
  const opts = {};
  for (const k of ["win", "halvings", "peak", "agg", "eps", "norm", "sharp"])
    if (q.has(k)) opts[{ peak: "peakR", agg: "aggR", eps: "aggEps" }[k] ?? k] = +q.get(k);
  const session = (await createDff()).begin(opts);
  for (const f of meta.frames) { const b = await bitmap(f.file); await session.push(b); b.close(); }
  const lag = meta.params.settle === 0 && meta.frames.every((f) => f.t_req_ms != null);
  const labels = lag ? laggedDiopters(meta.frames) : meta.frames.map((f) => 1 / f.focus_requested);
  const r = await session.finish(labels);

  // Frames as textures (rows top-down, so shaders sample at 1 - v).
  status("loading frames…");
  const texture = async (file) => {
    const t = new THREE.Texture(await bitmap(file));
    t.flipY = false; t.generateMipmaps = false; t.minFilter = t.magFilter = THREE.LinearFilter; t.needsUpdate = true;
    return t;
  };
  const sweepTex = await Promise.all(meta.frames.map((f) => texture(f.file)));
  const firstTex = sweepTex[0];

  THREE.ColorManagement.enabled = false; // camera sRGB bytes straight through
  // Crop window, as fractions of the capture frame.
  const aspect = num("aspect", 1) || meta.width / meta.height;
  const cw = Math.min(1, (meta.height * aspect) / meta.width), ch = Math.min(1, meta.width / aspect / meta.height);
  const crop = { x: num("at", 0.85) * (1 - cw), y: num("aty", 0.5) * (1 - ch), w: cw, h: ch };
  const W = Math.round(meta.width * cw * SCALE), H = Math.round(meta.height * ch * SCALE);
  const canvas = document.createElement("canvas");
  document.body.append(canvas);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.setPixelRatio(1);
  renderer.setSize(W, H, false);
  renderer.autoClear = false;

  // Per-pixel diopter + confidence, for the band on not-yet-converted pixels.
  const n = r.w * r.h, dc = new Float32Array(n * 4);
  for (let p = 0; p < n; p++) { dc[p * 4] = r.diopter[p]; dc[p * 4 + 1] = r.conf[p]; }
  const depthTex = new THREE.DataTexture(dc, r.w, r.h, THREE.RGBAFormat, THREE.FloatType);
  depthTex.needsUpdate = true;

  const d0 = labels[0], d1 = labels[labels.length - 1], sgn = Math.sign(d0 - d1) || 1;
  const common = { core: { value: CORE }, sgn: { value: sgn }, sigma: { value: SIGMA }, dioScan: { value: 0 }, glowScan: { value: 0 } };

  // Background: the current frame (crossfaded), dimmed by `bright`, with the in-focus band.
  const bg = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
    depthTest: false, depthWrite: false,
    uniforms: { ...common, crop: { value: new THREE.Vector4(crop.x, crop.y, crop.w, crop.h) }, texA: { value: firstTex }, texB: { value: firstTex }, ab: { value: 0 }, bright: { value: 1 }, depth: { value: depthTex } },
    vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
    fragmentShader: `
      uniform sampler2D texA, texB, depth; uniform float ab, bright, dioScan, sgn, sigma, glowScan; uniform vec4 crop; varying vec2 vUv;
      ${LASER}
      void main() {
        vec2 t = crop.xy + vec2(vUv.x, 1.0 - vUv.y) * crop.zw;
        vec3 c = mix(texture2D(texA, t).rgb, texture2D(texB, t).rgb, ab) * bright;
        vec4 d = texture2D(depth, t);
        float a = sgn * (d.r - dioScan);
        gl_FragColor = vec4(laser(c, a, sigma, glowScan * smoothstep(1.2, 2.0, d.g)), 1.0);
      }`,
  }));
  const bgScene = new THREE.Scene(), bgCam = new THREE.Camera();
  bgScene.add(bg);

  // The cloud. Each point appears once the sweep's focus has passed its depth (flashing with the
  // band) and disappears when the return band passes it, drifting up a little as it goes.
  const tm = performance.now();
  const f = focalPx(r.w, r.h), { Z, keep } = cloudMask(r, r.diopter, { edge: num("edge", 0.02), blob: num("blob", 0.002) });
  const maskMs = performance.now() - tm;
  const pos = [], col = [], ps = [], dio = [], rnd = [], centre = [];
  for (let y = 0; y < r.h; y++) for (let x = 0; x < r.w; x++) {
    const p = y * r.w + x;
    if (!keep[p]) continue;
    const z = Z[p];
    pos.push(((x + 0.5 - r.w / 2) / f) * z, -((y + 0.5 - r.h / 2) / f) * z, -z);
    col.push(r.colours[p * 3] / 255, r.colours[p * 3 + 1] / 255, r.colours[p * 3 + 2] / 255);
    ps.push((1.2 * z) / f); dio.push(r.diopter[p]); rnd.push(Math.random());
    const u = (x / r.w - crop.x) / crop.w, v = (y / r.h - crop.y) / crop.h; // within the crop
    if (Math.abs(u - 0.5) < 0.25 && Math.abs(v - 0.5) < 0.25 && (p & 7) === 0) centre.push(z);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("col", new THREE.Float32BufferAttribute(col, 3));
  for (const [k, v] of Object.entries({ psize: ps, dio, rnd })) g.setAttribute(k, new THREE.Float32BufferAttribute(v, 1));
  const JITTER = 0.25 * Math.abs(d0 - d1) / 3; // dissolve front raggedness, in diopters
  const pointsMat = new THREE.ShaderMaterial({
    uniforms: { ...common, sizeK: { value: 1 }, dioGone: { value: 0 }, glowGone: { value: 0 }, jitter: { value: JITTER } },
    vertexShader: `
      attribute vec3 col; attribute float psize, dio, rnd;
      uniform float sizeK, dioScan, dioGone, sgn, sigma, glowScan, glowGone, jitter; varying vec3 vCol;
      ${LASER}
      void main() {
        float a = sgn * (dio - dioScan);                   // > 0: the sweep's focus has passed it
        float b = sgn * (dioGone - dio - jitter * (rnd - 0.5)); // > 0: the return band has passed it
        float s = smoothstep(-0.3 * sigma, 0.6 * sigma, a) * (1.0 - smoothstep(0.0, 1.5 * sigma, b));
        vec3 p = position;
        p.y += smoothstep(0.0, 2.0 * sigma, b) * 0.03 * (0.5 + rnd) * -position.z;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_Position = s < 0.01 ? vec4(2.0, 2.0, 2.0, 1.0) : projectionMatrix * mv;
        gl_PointSize = s * psize * sizeK / -mv.z;
        // The return band itself stays a clean line; only the dissolving behind it is ragged.
        vCol = laser(laser(col, a, sigma, glowScan), sgn * (dioGone - dio), sigma, glowGone);
      }`,
    fragmentShader: `varying vec3 vCol; void main() { gl_FragColor = vec4(vCol, 1.0); }`,
  });
  const scene = new THREE.Scene();
  scene.add(new THREE.Points(g, pointsMat));
  const camera = new THREE.PerspectiveCamera((2 * Math.atan(r.h / 2 / f) * 180) / Math.PI, r.w / r.h, 0.01, 50);
  // The crop is an off-axis window of the capture camera, so the capture pose still lines up with
  // the photo exactly.
  camera.setViewOffset(r.w, r.h, crop.x * r.w, crop.y * r.h, crop.w * r.w, crop.h * r.h);
  pointsMat.uniforms.sizeK.value = (meta.height * SCALE) / 2 / Math.tan((camera.fov * Math.PI) / 360);
  centre.sort((a, b) => a - b);
  // Orbit about the point at the crop's centre (median depth there). The camera turns rigidly about
  // it rather than looking at it, so it stays at the centre of the crop throughout.
  const zt = num("tz", centre[centre.length >> 1] ?? 0.5); // &tz= pins the pivot depth (comparisons)
  const target = new THREE.Vector3((((crop.x + crop.w / 2) - 0.5) * r.w / f) * zt, -(((crop.y + crop.h / 2) - 0.5) * r.h / f) * zt, -zt);

  // Timeline (seconds).
  const T = {};
  let total = 0;
  const seg = (name, d) => { T[name] = [total, total + d]; total += d; };
  seg("intro", 0.8);
  seg("scan", num("scan", 3.2));
  seg("hold", 0.5);
  seg("orbit", num("orbit", 6));
  seg("settle", 0.4);
  seg("dissolve", 2.0);
  seg("rest", 0.5);
  const at = (name, t) => clamp((t - T[name][0]) / (T[name][1] - T[name][0]));
  const inSeg = (name, t) => t >= T[name][0] && t < T[name][1];
  const frames = Math.round(total * FPS);

  // Focus label at fractional frame k, extrapolated past the ends so the band can enter and leave.
  const labelAt = (k) => {
    const i = clamp(Math.floor(k), 0, labels.length - 2), u = k - i;
    return labels[i] * (1 - u) + labels[i + 1] * u;
  };
  const MARGIN = 3; // frames of band travel beyond each end of the sweep
  const YAW = (num("yaw", 28) * Math.PI) / 180, PITCH = (num("pitch", 10) * Math.PI) / 180;
  const offset = new THREE.Vector3(), euler = new THREE.Euler(0, 0, 0, "YXZ");

  function renderAt(t) {
    t = ((t % total) + total) % total;
    const U = bg.material.uniforms, P = pointsMat.uniforms;
    let texA = firstTex, texB = firstTex, ab = 0, bright = 1, glowScan = 0, glowGone = 0;
    let dioScan = d0 + sgn * 10, dioGone = d1 - sgn * 10, yaw = 0, pitch = 0;
    if (inSeg("intro", t)) {
      // the first frame (the default textures)
    } else if (inSeg("scan", t)) {
      const k = -MARGIN + at("scan", t) * (labels.length - 1 + 2 * MARGIN);
      const kc = clamp(k, 0, labels.length - 1), i = Math.min(Math.floor(kc), labels.length - 2);
      texA = sweepTex[i]; texB = sweepTex[i + 1]; ab = kc - i;
      dioScan = labelAt(k); glowScan = 1;
    } else {
      dioScan = d1 - sgn * 10;
      texA = texB = sweepTex[labels.length - 1];
      if (t >= T.orbit[0]) bright = 1 - smooth(T.orbit[0], T.orbit[0] + 0.8, t);
      if (inSeg("orbit", t)) {
        const s = smoother(at("orbit", t));
        yaw = YAW * Math.sin(2 * Math.PI * s); pitch = PITCH * Math.sin(4 * Math.PI * s);
      }
      if (t >= T.dissolve[0]) {
        texA = texB = firstTex;
        bright = smooth(T.dissolve[0], T.dissolve[0] + 0.5, t);
        const span = JITTER + 3 * SIGMA, u = smooth(T.dissolve[0] + 0.3, T.dissolve[1] - 0.1, t);
        dioGone = (d1 - sgn * span) * (1 - u) + (d0 + sgn * span) * u;
        glowGone = 1;
      }
    }
    U.texA.value = texA; U.texB.value = texB; U.ab.value = ab; U.bright.value = bright;
    U.dioScan.value = P.dioScan.value = dioScan; U.glowScan.value = P.glowScan.value = glowScan;
    P.dioGone.value = dioGone; P.glowGone.value = glowGone;
    euler.set(pitch, yaw, 0);
    offset.set(0, 0, 0).sub(target).applyEuler(euler);
    camera.position.copy(target).add(offset);
    camera.quaternion.setFromEuler(euler);
    renderer.clear();
    renderer.render(bgScene, bgCam);
    renderer.render(scene, camera);
  }

  async function exportFrames(name, from = 0, to = frames) {
    for (let i = from; i < Math.min(to, frames); i++) {
      renderAt(i / FPS);
      const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.95));
      await fetch(`/api/render?name=${encodeURIComponent(name)}&i=${i}`, { method: "POST", body: blob });
    }
    return Math.min(to, frames);
  }

  window.demo = { total, fps: FPS, frames, segments: T, points: dio.length, tz: zt, finishMs: r.finishMs, pushMs: r.pushMs, maskMs, renderAt, exportFrames };
  status(`${id}: ${dio.length.toLocaleString()} points · loop ${total.toFixed(1)} s`);
  setTimeout(() => (statusEl.hidden = true), 2500);
  if (!q.has("export")) {
    const t0 = performance.now();
    renderer.setAnimationLoop(() => renderAt((performance.now() - t0) / 1000));
  }
}

main().catch((e) => { status(`failed: ${e.message}`); throw e; });
