// The phone's own 3D view of a capture. It replaces the camera preview in place and starts from
// exactly the capture pose: the cloud is levelled with phone gravity (when available) and the
// virtual camera starts at the origin with the matching orientation and calibrated FOV, so the
// first rendered frame lines up with the live image. From there, drag to orbit, tilt the phone
// (IMU), jump to an overhead Top view, or tap two points to measure directly on the cloud.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { focalPx, cloudMask, makeLevelTransform } from "./cloud.js";

// Tilting the phone by an angle orbits the view by GAIN times that angle around the scene centre.
const GAIN = 1.5;
const FLY_MS = 320;
const fmtDist = (m) => (m < 1 ? `${(m * 100).toFixed(1)} cm` : `${m.toFixed(3)} m`);

export function createPhone3D(stage, orientation, { labelsEl = null } = {}) {
  THREE.ColorManagement.enabled = false; // point colours are camera sRGB bytes
  const canvas = document.createElement("canvas");
  canvas.className = "view3d";
  canvas.hidden = true;
  stage.append(canvas);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.setPixelRatio(devicePixelRatio);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);
  const camera = new THREE.PerspectiveCamera(50, 9 / 16, 0.01, 50);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.enabled = false;
  controls.rotateSpeed = 0.5; // a phone-width drag is about a half turn at 1.0

  let img = null; // { w, h, f } of the capture, for the field of view
  let cloud = null, grid = null, gizmo = null, levelled = false;
  let tilt = false, topView = false, measuring = false, atHome = true, q0 = null;
  let onTilt = () => {}, onTop = () => {}, onMeasure = () => {};
  let fly = null; // { t0, p0, p1, c0, c1, q0, q1 }

  const target = new THREE.Vector3(); // median focus point along capture optical axis
  const cloudBox = new THREE.Box3();
  const homeQuat = new THREE.Quaternion();
  const homeQuatInv = new THREE.Quaternion();
  const toImage = new THREE.Quaternion(), zAxis = new THREE.Vector3(0, 0, 1);
  const offset0 = new THREE.Vector3();

  function resize() {
    const r = stage.getBoundingClientRect();
    renderer.setSize(r.width, r.height, false);
    camera.aspect = r.width / Math.max(1, r.height);
    // Fit the capture's image inside the stage like the feed (object-fit: contain): it spans the
    // full height when the stage is wider than the image, otherwise the full width.
    if (img) {
      const halfH = camera.aspect > img.w / img.h ? img.h / 2 : img.w / 2 / camera.aspect;
      camera.fov = (2 * Math.atan(halfH / img.f) * 180) / Math.PI;
    }
    camera.updateProjectionMatrix();
    material.size = splatSize();
  }
  addEventListener("resize", () => { if (!canvas.hidden) resize(); });

  // Delta rotation of the phone since q0, conjugated into the image frame and then into world.
  const qa = new THREE.Quaternion(), qb = new THREE.Quaternion(), axis = new THREE.Vector3();
  function tiltWorldRotation() {
    if (!q0 || !orientation.q) return null;
    qa.fromArray(q0).invert();
    qb.fromArray(orientation.q);
    qa.multiply(qb).normalize();
    if (qa.w < 0) qa.set(-qa.x, -qa.y, -qa.z, -qa.w);
    const angle = 2 * Math.acos(Math.min(1, qa.w));
    const s = Math.sqrt(Math.max(1e-12, 1 - qa.w * qa.w));
    axis.set(qa.x / s, qa.y / s, qa.z / s);
    qa.setFromAxisAngle(axis, angle * GAIN);
    // Image-frame delta -> world-frame delta via homeQuat.
    qa.premultiply(toImage).multiply(qb.copy(toImage).invert());
    return qa.premultiply(homeQuat).multiply(homeQuatInv);
  }

  function startFly(p1, c1, q1 = null) {
    fly = {
      t0: performance.now(),
      p0: camera.position.clone(), p1: p1.clone(),
      c0: controls.target.clone(), c1: c1.clone(),
      q0: camera.quaternion.clone(), q1: q1 ? q1.clone() : null,
    };
  }

  function home(animate = false) {
    if (animate) {
      atHome = true;
      startFly(new THREE.Vector3(0, 0, 0), target, homeQuat);
      return;
    }
    fly = null;
    atHome = true;
    camera.position.set(0, 0, 0);
    camera.quaternion.copy(homeQuat);
    camera.up.set(0, 1, 0);
    offset0.copy(camera.position).sub(target);
    controls.target.copy(target);
  }

  function setTop(on) {
    if (!cloud) return;
    if (on) {
      if (tilt) setTilt(false);
      topView = true;
      atHome = false;
      controls.enabled = true;
      camera.up.set(0, 1, 0);
      const c = cloudBox.getCenter(new THREE.Vector3());
      const span = cloudBox.getSize(new THREE.Vector3());
      const halfV = Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
      const halfH = halfV * camera.aspect;
      // Offset slightly behind the scene centre along the capture's horizontal forward direction so
      // camera-forward points straight UP on screen (matching the photo's left-to-right layout)
      // and OrbitControls avoids its polar singularity.
      const fx = c.x, fz = c.z, fl = Math.hypot(fx, fz) || 1;
      const hx = fx / fl, hz = fz / fl;
      // Rotate horizontal span into screen (right, up) axes to fit cleanly in the stage.
      const spanScreenW = Math.abs(hz) * span.x + Math.abs(hx) * span.z;
      const spanScreenH = Math.abs(hx) * span.x + Math.abs(hz) * span.z;
      const dist = Math.max((spanScreenH * 0.58) / halfV, (spanScreenW * 0.58) / halfH, 0.35);
      const yTop = Math.max(c.y + dist, cloudBox.max.y + 0.18);
      startFly(new THREE.Vector3(c.x - hx * 0.002, yTop, c.z - hz * 0.002), c);
    } else {
      topView = false;
      home(true);
    }
    onTop(topView);
  }

  function setTilt(on) {
    tilt = on && !!orientation.q;
    controls.enabled = !tilt;
    if (tilt) {
      if (topView) { topView = false; onTop(false); }
      if (measuring) setMeasure(false);
      q0 = [...orientation.q];
      home();
    } else {
      controls.target.copy(controls.target);
    }
    onTilt(tilt);
  }

  // ---------------------------------------------------------------- measuring (touch + mouse)
  const accentHex = () => (matchMedia("(prefers-color-scheme: dark)").matches ? 0xe6e08a : 0x8a6fd1);
  const segs = []; // { a, b, line, label, dots }
  let first = null, firstDot = null;
  const makeDot = (v) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute([v.x, v.y, v.z], 3));
    const d = new THREE.Points(g, new THREE.PointsMaterial({ size: 11, sizeAttenuation: false, color: accentHex(), depthTest: false, transparent: true }));
    d.renderOrder = 10;
    scene.add(d);
    return d;
  };

  function emitMeasure() {
    const last = segs[segs.length - 1];
    let summary = null;
    if (last) {
      const d = last.b.clone().sub(last.a);
      summary = {
        total: fmtDist(last.a.distanceTo(last.b)),
        horiz: fmtDist(Math.hypot(d.x, d.z)),
        vert: fmtDist(Math.abs(d.y)),
        levelled,
      };
    }
    onMeasure({ on: measuring, count: segs.length, pending: !!first, summary });
  }

  function clearFirst() {
    if (firstDot) { scene.remove(firstDot); firstDot.geometry.dispose(); }
    first = firstDot = null;
  }
  function clearMeasure() {
    clearFirst();
    for (const s of segs) {
      scene.remove(s.line, ...s.dots);
      s.line.geometry.dispose();
      s.dots.forEach((d) => d.geometry.dispose());
      s.label?.remove();
    }
    segs.length = 0;
    emitMeasure();
  }
  function setMeasure(on) {
    measuring = !!on;
    if (measuring && tilt) setTilt(false);
    if (!measuring) clearFirst();
    emitMeasure();
  }

  // Two-tier screen-space pick: nearest-to-camera within tight radius (accurate on dense surfaces),
  // falling back to closest-in-2D within a wider touch radius so fingers reliably hit sparse edges.
  function pick(clientX, clientY, isTouch) {
    if (!cloud?.visible) return null;
    const rect = canvas.getBoundingClientRect();
    const px = clientX - rect.left, py = clientY - rect.top, W = rect.width, H = rect.height;
    camera.updateMatrixWorld();
    const e = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).elements;
    const pos = cloud.geometry.getAttribute("position").array, n = cloud.geometry.getAttribute("position").count;
    const rTight2 = (isTouch ? 13 : 8) ** 2;
    const rWide2 = (isTouch ? 28 : 14) ** 2;
    let bestTight = -1, bestW = Infinity;
    let bestWide = -1, bestD2 = rWide2;
    for (let i = 0; i < n; i++) {
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      const w = e[3] * x + e[7] * y + e[11] * z + e[15];
      if (w <= 0) continue;
      const sx = (((e[0] * x + e[4] * y + e[8] * z + e[12]) / w) * 0.5 + 0.5) * W;
      const sy = (0.5 - ((e[1] * x + e[5] * y + e[9] * z + e[13]) / w) * 0.5) * H;
      const d2 = (sx - px) ** 2 + (sy - py) ** 2;
      if (d2 <= rTight2 && w < bestW) { bestTight = i; bestW = w; }
      else if (bestTight < 0 && d2 < bestD2) { bestWide = i; bestD2 = d2; }
    }
    const idx = bestTight >= 0 ? bestTight : bestWide;
    return idx < 0 ? null : new THREE.Vector3(pos[idx * 3], pos[idx * 3 + 1], pos[idx * 3 + 2]);
  }

  function tapAt(clientX, clientY, isTouch) {
    const p = pick(clientX, clientY, isTouch);
    if (!p) return;
    if (!first) { first = p; firstDot = makeDot(p); emitMeasure(); return; }
    const a = first; clearFirst();
    const g = new THREE.BufferGeometry().setFromPoints([a, p]);
    const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: accentHex(), depthTest: false, transparent: true }));
    line.renderOrder = 10;
    scene.add(line);
    let label = null;
    if (labelsEl) {
      label = document.createElement("div");
      label.className = "mlabel";
      label.textContent = fmtDist(a.distanceTo(p));
      labelsEl.append(label);
    }
    segs.push({ a, b: p, line, label, dots: [makeDot(a), makeDot(p)] });
    emitMeasure();
  }

  // Track active pointers so multi-touch pinch/pan never drops a measure pin.
  const activePtrs = new Set();
  let down = null;
  canvas.addEventListener("pointerdown", (e) => {
    activePtrs.add(e.pointerId);
    if (tilt) setTilt(false);
    if (fly) fly = null;
    if (activePtrs.size > 1) atHome = false;
    if (activePtrs.size === 1 && e.button === 0) {
      down = { id: e.pointerId, x: e.clientX, y: e.clientY, t: performance.now(), touch: e.pointerType === "touch" };
    } else {
      down = null;
    }
  });
  canvas.addEventListener("pointermove", (e) => {
    if (down && down.id === e.pointerId && Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) {
      atHome = false;
    }
    if (down && down.id === e.pointerId && Math.hypot(e.clientX - down.x, e.clientY - down.y) > 10) {
      if (topView) { topView = false; onTop(false); }
      down = null;
    }
  });
  canvas.addEventListener("wheel", () => { atHome = false; }, { passive: true });
  const endPtr = (e, isUp) => {
    activePtrs.delete(e.pointerId);
    if (isUp && measuring && down && down.id === e.pointerId && performance.now() - down.t < 450 && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 10) {
      tapAt(e.clientX, e.clientY, down.touch);
    }
    if (down?.id === e.pointerId) down = null;
  };
  canvas.addEventListener("pointerup", (e) => endPtr(e, true));
  canvas.addEventListener("pointercancel", (e) => endPtr(e, false));

  // ---------------------------------------------------------------- animation loop
  const projV = new THREE.Vector3();
  function updateLabels() {
    if (!labelsEl || !segs.length) return;
    const rect = canvas.getBoundingClientRect();
    for (const s of segs) {
      if (!s.label) continue;
      projV.copy(s.a).add(s.b).multiplyScalar(0.5).project(camera);
      const vis = projV.z < 1 && Math.abs(projV.x) < 1.2 && Math.abs(projV.y) < 1.2;
      s.label.hidden = !vis;
      if (vis) {
        s.label.style.left = `${(projV.x * 0.5 + 0.5) * rect.width}px`;
        s.label.style.top = `${(0.5 - projV.y * 0.5) * rect.height}px`;
      }
    }
  }

  function frame() {
    if (canvas.hidden) return;
    if (tilt) {
      const R = tiltWorldRotation();
      if (R) {
        camera.quaternion.copy(R).multiply(homeQuat);
        camera.position.copy(offset0).applyQuaternion(R).add(target);
      }
    } else if (fly) {
      const u = Math.min(1, (performance.now() - fly.t0) / FLY_MS);
      const e = 1 - (1 - u) ** 3; // cubic ease-out
      camera.position.lerpVectors(fly.p0, fly.p1, e);
      controls.target.lerpVectors(fly.c0, fly.c1, e);
      if (fly.q1) camera.quaternion.slerpQuaternions(fly.q0, fly.q1, e);
      else camera.lookAt(controls.target);
      if (u >= 1) {
        const doneQ = fly.q1;
        fly = null;
        if (!doneQ) controls.update();
      }
    } else if (!atHome) {
      controls.update();
    }
    // Keep capture-pose and IMU Tilt peek clean; show grid & camera frustum once orbiting or in Top.
    const moved = !tilt && !atHome && camera.position.lengthSq() > 0.0025;
    if (gizmo) gizmo.visible = moved;
    if (grid) grid.visible = moved && levelled;
    updateLabels();
    renderer.render(scene, camera);
  }
  renderer.setAnimationLoop(frame);

  // Points are splats sized to their pixel's footprint on the surface (z / f), so they tile at the
  // capture pose at any resolution and keep their physical size as you zoom or orbit.
  const splatSize = () => 1 / Math.tan((camera.fov * Math.PI) / 360);
  const material = new THREE.PointsMaterial({ size: 1, sizeAttenuation: true, vertexColors: true });
  material.onBeforeCompile = (sh) => {
    sh.vertexShader = sh.vertexShader
      .replace("uniform float size;", "uniform float size;\nattribute float psize;")
      .replace("gl_PointSize = size;", "gl_PointSize = size * psize;");
  };

  function makeFrustum(T, f, w, h) {
    const s = 0.055, o = T(0, 0, 0);
    const corners = [[0, 0], [w, 0], [w, h], [0, h]].map(([x, y]) => T(((x - w / 2) / f) * s, ((y - h / 2) / f) * s, s));
    const pts = [];
    for (const c of corners) pts.push(...o, ...c);
    for (let i = 0; i < 4; i++) pts.push(...corners[i], ...corners[(i + 1) % 4]);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: accentHex() }));
  }

  function build(r, colours, depth, captureOrientation) {
    const d = depth === "smooth" ? r.diopterSmooth : r.diopter;
    const f = focalPx(r.w, r.h), cx = r.w / 2, cy = r.h / 2, n = r.w * r.h;
    levelled = !!captureOrientation?.quaternion;
    const T = makeLevelTransform(captureOrientation);
    const [c0, c1, c2] = [T(1, 0, 0), T(0, 1, 0), T(0, 0, 1)];
    // Camera home rotation in levelled world: right = T(1,0,0), up = T(0,-1,0), +z(back) = T(0,0,-1).
    const m = new THREE.Matrix4().makeBasis(
      new THREE.Vector3(c0[0], c0[1], c0[2]),
      new THREE.Vector3(-c1[0], -c1[1], -c1[2]),
      new THREE.Vector3(-c2[0], -c2[1], -c2[2]),
    );
    homeQuat.setFromRotationMatrix(m);
    homeQuatInv.copy(homeQuat).invert();

    const { Z, keep } = cloudMask(r, d);
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), ps = new Float32Array(n), zs = [], ys = [];
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
        col[o] = colours[p * 3] / 255; col[o + 1] = colours[p * 3 + 1] / 255; col[o + 2] = colours[p * 3 + 2] / 255;
        ps[k] = (1.2 * z) / f;
        if ((k & 15) === 0) { zs.push(z); ys.push(pos[o + 1]); }
        k++;
      }
    }
    if (cloud) { scene.remove(cloud); cloud.geometry.dispose(); }
    if (grid) { scene.remove(grid); grid = null; }
    if (gizmo) { scene.remove(gizmo); gizmo.geometry.dispose(); gizmo = null; }

    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos.subarray(0, k * 3), 3));
    g.setAttribute("color", new THREE.BufferAttribute(col.subarray(0, k * 3), 3));
    g.setAttribute("psize", new THREE.BufferAttribute(ps.subarray(0, k), 1));
    img = { w: r.w, h: r.h, f };
    canvas.hidden = false;
    resize();
    cloud = new THREE.Points(g, material);
    scene.add(cloud);
    cloudBox.setFromBufferAttribute(g.getAttribute("position"));

    ys.sort((a, b) => a - b);
    const floor = ys[Math.floor(0.03 * (ys.length - 1))] ?? -0.3;
    const center = cloudBox.getCenter(new THREE.Vector3());
    grid = new THREE.GridHelper(2, 40, 0x3a3f2a, 0x22252b);
    grid.position.set(center.x, floor, center.z);
    grid.visible = false;
    scene.add(grid);
    gizmo = makeFrustum(T, f, r.w, r.h);
    gizmo.visible = false;
    scene.add(gizmo);

    return { k, zs, fwd: new THREE.Vector3(c2[0], c2[1], c2[2]) };
  }

  return {
    get tilt() { return tilt; },
    get top() { return topView; },
    get measuring() { return measuring; },
    get segments() { return segs.map(({ a, b }) => ({ a: a.toArray(), b: b.toArray(), d: a.distanceTo(b) })); },
    set onTilt(fn) { onTilt = fn; },
    set onTop(fn) { onTop = fn; },
    set onMeasure(fn) { onMeasure = fn; },
    setTilt,
    setTop,
    setMeasure,
    clearMeasure,
    tapAt,
    /** result/colours from dff-worker.js: builds the gravity-levelled cloud and starts at the exact
     *  capture pose. */
    show(r, colours, { depth = "smooth", screenAngle = 0, orientation: capOri = null } = {}) {
      clearMeasure();
      topView = false;
      measuring = false;
      const angle = capOri?.angle ?? screenAngle;
      toImage.setFromAxisAngle(zAxis, (angle * Math.PI) / 180);
      const { k, zs, fwd } = build(r, colours, depth, capOri);
      zs.sort((a, b) => a - b);
      const zMed = zs[zs.length >> 1] ?? 0.5;
      target.copy(fwd).multiplyScalar(zMed);
      home();
      setTilt(!!orientation.q);
      onTop(false);
      emitMeasure();
      return { points: k };
    },
    hide() {
      clearMeasure();
      fly = null; topView = false; measuring = false; tilt = false;
      controls.enabled = false;
      canvas.hidden = true;
    },
    get visible() { return !canvas.hidden; },
  };
}
