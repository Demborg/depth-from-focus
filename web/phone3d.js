// The phone's own 3D view of a capture. It replaces the camera preview in place and starts from
// exactly the capture pose: the cloud is levelled with phone gravity (when available) and the
// virtual camera starts at the origin with the matching orientation and calibrated FOV, so the
// first rendered frame lines up with the live image. From there, drag to orbit, tilt the phone
// (IMU), switch to a locked 1:1 Top-down map view, or tap/drag handles to measure on the cloud.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { focalPx, cloudMask, makeLevelTransform } from "./cloud.js";

// Tilting the phone by an angle orbits the view by GAIN times that angle around the scene centre.
const GAIN = 1.5;
const FLY_MS = 280;
const TOP_MIN_DIST = 0.06;
const TOP_MAX_DIST = 8.0;
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
  controls.dampingFactor = 0.14;
  controls.enabled = false;
  controls.rotateSpeed = 0.5;
  controls.minDistance = 0.12;
  controls.maxDistance = 10;

  let img = null; // { w, h, f } of the capture, for the field of view
  let cloud = null, grid = null, gizmo = null, levelled = false;
  let tilt = false, topView = false, measuring = false, atHome = true, q0 = null;
  let onTilt = () => {}, onTop = () => {}, onMeasure = () => {};
  let fly = null; // { t0, p0, p1, c0, c1, q0, q1 }

  // Locked top-down map state: camera sits directly above topCenter along +Y, oriented with
  // capture-forward (topUp) pointing straight UP on screen and topRight pointing right.
  const topCenter = new THREE.Vector3();
  const topUp = new THREE.Vector3(0, 0, -1);    // horizontal unit vector pointing up on screen
  const topRight = new THREE.Vector3(1, 0, 0);  // horizontal unit vector pointing right on screen
  const topQuat = new THREE.Quaternion();       // fixed straight-down orientation for Top view
  let topDist = 0.8, topMinCamDist = 0.15;

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
    if (img) {
      const halfH = camera.aspect > img.w / img.h ? img.h / 2 : img.w / 2 / camera.aspect;
      camera.fov = (2 * Math.atan(halfH / img.f) * 180) / Math.PI;
    }
    camera.updateProjectionMatrix();
    material.size = splatSize();
  }
  addEventListener("resize", () => { if (!canvas.hidden) resize(); });

  function setCamZoom(z) {
    if (camera.zoom === z) return;
    camera.zoom = z;
    camera.updateProjectionMatrix();
    material.size = splatSize();
  }

  // Apply locked top-down camera pose immediately (1:1, zero damping lag).
  // Keeps the camera safely above cloudBox.max.y and switches smoothly to optical zoom when
  // zoomed in close, so tall scenes never clip behind the camera while world-per-px stays 1:1.
  function applyTopPose() {
    const camDist = Math.max(topDist, topMinCamDist);
    camera.position.set(topCenter.x, topCenter.y + camDist, topCenter.z);
    camera.quaternion.copy(topQuat);
    controls.target.copy(topCenter);
    setCamZoom(camDist / topDist);
  }
  const topWorldPerPx = () => {
    const h = Math.max(1, canvas.getBoundingClientRect().height);
    return (2 * topDist * Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5))) / h;
  };

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
    qa.premultiply(toImage).multiply(qb.copy(toImage).invert());
    return qa.premultiply(homeQuat).multiply(homeQuatInv);
  }

  function startFly(p1, c1, q1 = null, z1 = 1) {
    fly = {
      t0: performance.now(),
      p0: camera.position.clone(), p1: p1.clone(),
      c0: controls.target.clone(), c1: c1.clone(),
      q0: camera.quaternion.clone(), q1: q1 ? q1.clone() : null,
      z0: camera.zoom, z1,
    };
  }

  function home(animate = false) {
    if (animate) {
      atHome = true;
      controls.enabled = false;
      startFly(new THREE.Vector3(0, 0, 0), target, homeQuat, 1);
      return;
    }
    fly = null;
    atHome = true;
    setCamZoom(1);
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
      controls.enabled = false; // our direct 1:1 2D map handler drives Top view
      const c = cloudBox.getCenter(new THREE.Vector3());
      const span = cloudBox.getSize(new THREE.Vector3());
      const halfV = Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
      const halfH = halfV * camera.aspect;
      // Align screen-up with capture's horizontal forward direction so left/right match the photo.
      const fl = Math.hypot(c.x, c.z) || 1;
      topUp.set(c.x / fl, 0, c.z / fl);
      // Right = topUp x (0, 1, 0) = (-topUp.z, 0, topUp.x)
      topRight.set(-topUp.z, 0, topUp.x);
      // Camera basis looking straight down (-Y): right = topRight, up = topUp, +z(back) = +Y.
      topQuat.setFromRotationMatrix(new THREE.Matrix4().makeBasis(topRight, topUp, new THREE.Vector3(0, 1, 0)));
      const spanScreenW = Math.abs(topRight.x) * span.x + Math.abs(topRight.z) * span.z;
      const spanScreenH = Math.abs(topUp.x) * span.x + Math.abs(topUp.z) * span.z;
      const dist = Math.max((spanScreenH * 0.58) / halfV, (spanScreenW * 0.58) / halfH, 0.35);
      topCenter.copy(c);
      topMinCamDist = Math.max(0.15, cloudBox.max.y - c.y + 0.08);
      topDist = Math.max(dist, topMinCamDist + 0.10);
      startFly(new THREE.Vector3(topCenter.x, topCenter.y + topDist, topCenter.z), topCenter, topQuat, 1);
    } else {
      topView = false;
      home(true);
    }
    onTop(topView);
  }

  function setTilt(on) {
    tilt = on && !!orientation.q && !topView && !measuring;
    controls.enabled = !tilt && !topView;
    if (tilt) {
      q0 = [...orientation.q];
      home();
    }
    onTilt(tilt);
  }

  // ---------------------------------------------------------------- measuring + draggable handles
  const accentHex = () => (matchMedia("(prefers-color-scheme: dark)").matches ? 0xe6e08a : 0x8a6fd1);
  const segs = []; // { a, b, line, label, handles: [elA, elB] }
  let first = null, firstHandle = null;
  let activeSeg = null; // last touched/added segment for status readout
  let dragHandle = null; // { ptrId, isTouch, getPt, setPt, el, seg, pendingXY }

  function makeHandleEl() {
    if (!labelsEl) return null;
    const el = document.createElement("div");
    el.className = "mhandle";
    labelsEl.append(el);
    return el;
  }

  function emitMeasure() {
    const s = activeSeg && segs.includes(activeSeg) ? activeSeg : segs[segs.length - 1];
    let summary = null;
    if (s) {
      const d = s.b.clone().sub(s.a);
      summary = {
        total: fmtDist(s.a.distanceTo(s.b)),
        horiz: fmtDist(Math.hypot(d.x, d.z)),
        vert: fmtDist(Math.abs(d.y)),
        levelled,
      };
    }
    onMeasure({ on: measuring, count: segs.length, pending: !!first, summary });
  }

  function syncSegVisual(s) {
    const pos = s.line.geometry.getAttribute("position");
    pos.setXYZ(0, s.a.x, s.a.y, s.a.z);
    pos.setXYZ(1, s.b.x, s.b.y, s.b.z);
    pos.needsUpdate = true;
    if (s.label) s.label.textContent = fmtDist(s.a.distanceTo(s.b));
  }

  function clearFirst() {
    firstHandle?.remove();
    first = firstHandle = null;
  }
  function clearMeasure() {
    dragHandle = null;
    clearFirst();
    for (const s of segs) {
      scene.remove(s.line);
      s.line.geometry.dispose();
      s.label?.remove();
      s.handles.forEach((h) => h?.remove());
    }
    segs.length = 0;
    activeSeg = null;
    emitMeasure();
  }
  function setMeasure(on) {
    measuring = !!on;
    if (measuring && tilt) setTilt(false);
    if (!measuring) clearFirst();
    emitMeasure();
  }

  // Two-tier screen-space pick with optional stride for fast 60fps live handle dragging.
  function pick(clientX, clientY, isTouch, stride = 1) {
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
    for (let i = 0; i < n; i += stride) {
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

  const hitV = new THREE.Vector3();
  function hitHandle(clientX, clientY, isTouch) {
    if (!segs.length && !first) return null;
    const rect = canvas.getBoundingClientRect();
    const px = clientX - rect.left, py = clientY - rect.top;
    const maxR = isTouch ? (measuring ? 34 : 24) : (measuring ? 18 : 14);
    let best = null, bestD2 = maxR * maxR;
    const test = (pt, el, seg, setPt) => {
      hitV.copy(pt).project(camera);
      if (hitV.z >= 1) return;
      const sx = (hitV.x * 0.5 + 0.5) * rect.width;
      const sy = (0.5 - hitV.y * 0.5) * rect.height;
      const d2 = (sx - px) ** 2 + (sy - py) ** 2;
      if (d2 < bestD2) { bestD2 = d2; best = { el, seg, setPt }; }
    };
    if (first) test(first, firstHandle, null, (p) => first.copy(p));
    for (const s of segs) {
      test(s.a, s.handles[0], s, (p) => { s.a.copy(p); syncSegVisual(s); });
      test(s.b, s.handles[1], s, (p) => { s.b.copy(p); syncSegVisual(s); });
    }
    return best;
  }

  function tapAt(clientX, clientY, isTouch) {
    const p = pick(clientX, clientY, isTouch, 1);
    if (!p) return;
    if (!first) { first = p; firstHandle = makeHandleEl(); emitMeasure(); return; }
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
    const seg = { a, b: p, line, label, handles: [makeHandleEl(), makeHandleEl()] };
    segs.push(seg);
    activeSeg = seg;
    emitMeasure();
  }

  // ---------------------------------------------------------------- unified pointer / 2D map input
  // Capture-phase listeners let handle dragging and locked 1:1 Top-down pan/pinch intercept
  // cleanly before OrbitControls sees the events.
  const ptrs = new Map(); // pointerId -> { x, y }
  let down = null, lastTap = 0;

  function pinchStats() {
    const [[, p0], [, p1]] = [...ptrs.entries()];
    return {
      mx: 0.5 * (p0.x + p1.x),
      my: 0.5 * (p0.y + p1.y),
      dist: Math.hypot(p1.x - p0.x, p1.y - p0.y),
    };
  }
  let prevPinch = null;

  canvas.addEventListener("pointerdown", (e) => {
    ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (tilt) setTilt(false);
    if (fly) {
      fly = null;
      if (topView) applyTopPose();
      else { setCamZoom(1); controls.enabled = true; }
    }
    // 1) Check draggable measure handles first (single pointer only).
    if (ptrs.size === 1 && e.button === 0) {
      const h = hitHandle(e.clientX, e.clientY, e.pointerType === "touch");
      if (h) {
        dragHandle = { ptrId: e.pointerId, isTouch: e.pointerType === "touch", ...h, pendingXY: null };
        h.el?.classList.add("dragging");
        if (h.seg) { activeSeg = h.seg; emitMeasure(); }
        down = null;
        try { canvas.setPointerCapture(e.pointerId); } catch {}
        e.stopImmediatePropagation();
        e.preventDefault();
        return;
      }
    }
    if (ptrs.size === 2) {
      if (dragHandle) { dragHandle.el?.classList.remove("dragging"); dragHandle = null; }
      atHome = false;
      down = null;
      if (topView) prevPinch = pinchStats();
    } else if (ptrs.size === 1 && e.button === 0) {
      down = { id: e.pointerId, x: e.clientX, y: e.clientY, t: performance.now(), touch: e.pointerType === "touch" };
    } else {
      down = null;
    }
    if (topView) {
      try { canvas.setPointerCapture(e.pointerId); } catch {}
      e.stopImmediatePropagation();
      e.preventDefault();
    }
  }, { capture: true });

  canvas.addEventListener("pointermove", (e) => {
    const prev = ptrs.get(e.pointerId);
    if (!prev) return;
    const dx = e.clientX - prev.x, dy = e.clientY - prev.y;
    ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });

    // Live handle dragging (picked at most once per animation frame).
    if (dragHandle && dragHandle.ptrId === e.pointerId) {
      dragHandle.pendingXY = [e.clientX, e.clientY];
      e.stopImmediatePropagation();
      e.preventDefault();
      return;
    }

    if (down && down.id === e.pointerId) {
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
      if (moved > 4) atHome = false;
      if (moved > 10) down = null;
    }

    // Locked 1:1 Top-down 2D map pan & pinch-zoom (zero damping lag).
    if (topView) {
      if (ptrs.size === 1) {
        const wpp = topWorldPerPx();
        topCenter.addScaledVector(topRight, -dx * wpp).addScaledVector(topUp, dy * wpp);
        applyTopPose();
      } else if (ptrs.size === 2) {
        const cur = pinchStats();
        if (prevPinch && prevPinch.dist > 8 && cur.dist > 8) {
          const rect = canvas.getBoundingClientRect();
          const oldWpp = topWorldPerPx();
          // Pan midpoint first
          topCenter.addScaledVector(topRight, -(cur.mx - prevPinch.mx) * oldWpp)
                   .addScaledVector(topUp, (cur.my - prevPinch.my) * oldWpp);
          // Scale distance 1:1 with finger spread, keeping the pinch midpoint anchored!
          const nextDist = Math.min(TOP_MAX_DIST, Math.max(TOP_MIN_DIST, topDist * (prevPinch.dist / cur.dist)));
          const ox = cur.mx - (rect.left + rect.width * 0.5);
          const oy = (rect.top + rect.height * 0.5) - cur.my;
          const newWpp = (oldWpp / topDist) * nextDist;
          topCenter.addScaledVector(topRight, ox * (oldWpp - newWpp))
                   .addScaledVector(topUp, oy * (oldWpp - newWpp));
          topDist = nextDist;
          applyTopPose();
        }
        prevPinch = cur;
      }
      e.stopImmediatePropagation();
      e.preventDefault();
    }
  }, { capture: true });

  const endPtr = (e, isUp) => {
    ptrs.delete(e.pointerId);
    if (ptrs.size < 2) prevPinch = null;

    if (dragHandle && dragHandle.ptrId === e.pointerId) {
      if (isUp) {
        // Final full-resolution snap on release.
        const p = pick(e.clientX, e.clientY, dragHandle.isTouch, 1);
        if (p) { dragHandle.setPt(p); emitMeasure(); }
      }
      dragHandle.el?.classList.remove("dragging");
      dragHandle = null;
      e.stopImmediatePropagation();
      return;
    }

    const now = performance.now();
    if (isUp && down && down.id === e.pointerId && now - down.t < 450 && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 10) {
      if (measuring) {
        tapAt(e.clientX, e.clientY, down.touch);
      } else if (!topView && now - lastTap < 320) {
        lastTap = 0;
        if (orientation.q) setTilt(true); else home(true);
      } else {
        lastTap = now;
      }
    }
    if (down?.id === e.pointerId) down = null;
  };
  canvas.addEventListener("pointerup", (e) => endPtr(e, true), { capture: true });
  canvas.addEventListener("pointercancel", (e) => endPtr(e, false), { capture: true });
  addEventListener("pointerup", (e) => { ptrs.delete(e.pointerId); if (ptrs.size < 2) prevPinch = null; });
  addEventListener("pointercancel", (e) => { ptrs.delete(e.pointerId); if (ptrs.size < 2) prevPinch = null; });

  canvas.addEventListener("wheel", (e) => {
    atHome = false;
    if (!topView) return;
    e.stopImmediatePropagation();
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const oldWpp = topWorldPerPx();
    const factor = Math.exp(e.deltaY * 0.0018);
    const nextDist = Math.min(TOP_MAX_DIST, Math.max(TOP_MIN_DIST, topDist * factor));
    const ox = e.clientX - (rect.left + rect.width * 0.5);
    const oy = (rect.top + rect.height * 0.5) - e.clientY;
    const newWpp = (oldWpp / topDist) * nextDist;
    topCenter.addScaledVector(topRight, ox * (oldWpp - newWpp))
             .addScaledVector(topUp, oy * (oldWpp - newWpp));
    topDist = nextDist;
    applyTopPose();
  }, { capture: true, passive: false });

  // ---------------------------------------------------------------- animation loop
  const projV = new THREE.Vector3();
  function placeOverlay(el, worldPt, rect) {
    if (!el) return;
    projV.copy(worldPt).project(camera);
    const vis = projV.z < 1 && Math.abs(projV.x) < 1.2 && Math.abs(projV.y) < 1.2;
    el.hidden = !vis;
    if (vis) {
      el.style.left = `${(projV.x * 0.5 + 0.5) * rect.width}px`;
      el.style.top = `${(0.5 - projV.y * 0.5) * rect.height}px`;
    }
  }
  const midV = new THREE.Vector3();
  function updateOverlays() {
    if (!labelsEl || (!segs.length && !first)) return;
    const rect = canvas.getBoundingClientRect();
    if (first) placeOverlay(firstHandle, first, rect);
    for (const s of segs) {
      placeOverlay(s.handles[0], s.a, rect);
      placeOverlay(s.handles[1], s.b, rect);
      midV.copy(s.a).add(s.b).multiplyScalar(0.5);
      placeOverlay(s.label, midV, rect);
    }
  }

  function frame() {
    if (canvas.hidden) return;
    if (dragHandle?.pendingXY) {
      const [cx, cy] = dragHandle.pendingXY;
      dragHandle.pendingXY = null;
      const nPts = cloud?.geometry?.getAttribute("position")?.count ?? 0;
      const stride = Math.max(1, Math.floor(nPts / 180000));
      const p = pick(cx, cy, dragHandle.isTouch, stride);
      if (p) { dragHandle.setPt(p); emitMeasure(); }
    }
    if (tilt) {
      const R = tiltWorldRotation();
      if (R) {
        camera.quaternion.copy(R).multiply(homeQuat);
        camera.position.copy(offset0).applyQuaternion(R).add(target);
      }
    } else if (fly) {
      const u = Math.min(1, (performance.now() - fly.t0) / FLY_MS);
      const e = 1 - (1 - u) ** 3;
      camera.position.lerpVectors(fly.p0, fly.p1, e);
      controls.target.lerpVectors(fly.c0, fly.c1, e);
      if (fly.q1) camera.quaternion.slerpQuaternions(fly.q0, fly.q1, e);
      else camera.lookAt(controls.target);
      if (fly.z0 !== fly.z1) setCamZoom(fly.z0 + (fly.z1 - fly.z0) * e);
      if (u >= 1) {
        const doneQ = fly.q1;
        fly = null;
        if (topView) {
          applyTopPose();
        } else {
          controls.enabled = true;
          controls.enableDamping = false;
          if (!doneQ) controls.update();
          controls.enableDamping = true;
          if (doneQ && !measuring && orientation.q) setTilt(true);
        }
      }
    } else if (!atHome && !topView) {
      controls.update();
    }
    const moved = !tilt && !atHome && (topView || camera.position.lengthSq() > 0.0025);
    if (gizmo) gizmo.visible = moved;
    if (grid) grid.visible = moved && levelled;
    updateOverlays();
    renderer.render(scene, camera);
  }
  renderer.setAnimationLoop(frame);

  // Points are splats sized to their pixel's footprint on the surface (z / f), so they tile at the
  // capture pose at any resolution and keep their physical size as you zoom or orbit.
  const splatSize = () => camera.zoom / Math.tan((camera.fov * Math.PI) / 360);
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
      setCamZoom(1);
      controls.enabled = false;
      canvas.hidden = true;
    },
    get visible() { return !canvas.hidden; },
  };
}
