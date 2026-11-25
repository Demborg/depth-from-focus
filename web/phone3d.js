// The phone's own 3D view of a capture. It replaces the camera preview in place and starts from
// exactly the capture pose: the virtual camera sits at the origin of the camera frame with the
// calibrated field of view, so the first rendered frame lines up with the live image. From there,
// drag to orbit (OrbitControls touch), or tilt the phone to orbit around the scene (IMU).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { F_LONG_PX, discontinuities } from "./cloud.js";

const CONF = 1.6, EDGE = 0.06;
// Tilting the phone by an angle orbits the view by GAIN times that angle around the scene centre.
const GAIN = 1.5;

export function createPhone3D(stage, orientation) {
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

  let cloud = null, target = new THREE.Vector3(), tilt = false, q0 = null, onTilt = () => {};
  const offset0 = new THREE.Vector3();

  function resize() {
    const r = stage.getBoundingClientRect();
    renderer.setSize(r.width, r.height, false);
    camera.aspect = r.width / Math.max(1, r.height);
    camera.updateProjectionMatrix();
  }
  addEventListener("resize", () => { if (!canvas.hidden) resize(); });

  // Delta rotation of the phone since q0, in device coordinates (x right, y up, z out of the
  // screen), which in portrait are also the three.js camera axes.
  const qa = new THREE.Quaternion(), qb = new THREE.Quaternion(), axis = new THREE.Vector3();
  function tiltRotation() {
    if (!q0 || !orientation.q) return null;
    qa.fromArray(q0).invert();
    qb.fromArray(orientation.q);
    qa.multiply(qb).normalize();
    if (qa.w < 0) qa.set(-qa.x, -qa.y, -qa.z, -qa.w);
    // Scale the angle by GAIN.
    const angle = 2 * Math.acos(Math.min(1, qa.w));
    const s = Math.sqrt(Math.max(1e-12, 1 - qa.w * qa.w));
    axis.set(qa.x / s, qa.y / s, qa.z / s);
    return qa.setFromAxisAngle(axis, angle * GAIN);
  }

  function frame() {
    if (canvas.hidden) return;
    if (tilt) {
      const R = tiltRotation();
      if (R) {
        camera.quaternion.copy(R);
        camera.position.copy(offset0).applyQuaternion(R).add(target);
      }
    } else {
      controls.update();
    }
    renderer.render(scene, camera);
  }
  renderer.setAnimationLoop(frame);

  function home() {
    camera.position.set(0, 0, 0);
    camera.quaternion.identity();
    camera.up.set(0, 1, 0);
    offset0.copy(camera.position).sub(target);
    controls.target.copy(target);
  }

  function setTilt(on) {
    tilt = on && !!orientation.q;
    controls.enabled = !tilt;
    if (tilt) { q0 = [...orientation.q]; home(); }
    else controls.target.copy(target);
    onTilt(tilt);
  }
  // Touching the view hands the camera over to touch orbiting from wherever tilt left it.
  canvas.addEventListener("pointerdown", () => { if (tilt) setTilt(false); });

  return {
    get tilt() { return tilt; },
    set onTilt(fn) { onTilt = fn; },
    setTilt,
    /** result/colours from dff-worker.js. Builds the cloud in the camera frame and shows it. */
    show(r, colours, { depth = "smooth" } = {}) {
      const d = depth === "smooth" ? r.diopterSmooth : r.diopter;
      const f = F_LONG_PX * (r.h / 1920), cx = r.w / 2, cy = r.h / 2, n = r.w * r.h;
      const Z = new Float32Array(n);
      for (let p = 0; p < n; p++) Z[p] = 1 / Math.max(1e-3, d[p]);
      const flying = discontinuities(Z, r.w, r.h, EDGE);
      const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), zs = [];
      let k = 0;
      for (let y = 0; y < r.h; y++) for (let x = 0; x < r.w; x++) {
        const p = y * r.w + x;
        if (r.conf[p] < CONF || !(d[p] > 0.2) || flying[p] || Z[p] > 6) continue;
        const z = Z[p], o = k * 3;
        // Camera frame (x right, y down, z forward) to three.js (x right, y up, z backward).
        pos[o] = ((x + 0.5 - cx) / f) * z; pos[o + 1] = -((y + 0.5 - cy) / f) * z; pos[o + 2] = -z;
        col[o] = colours[p * 3] / 255; col[o + 1] = colours[p * 3 + 1] / 255; col[o + 2] = colours[p * 3 + 2] / 255;
        if ((k & 15) === 0) zs.push(z);
        k++;
      }
      if (cloud) { scene.remove(cloud); cloud.geometry.dispose(); }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos.subarray(0, k * 3), 3));
      g.setAttribute("color", new THREE.BufferAttribute(col.subarray(0, k * 3), 3));
      canvas.hidden = false;
      resize();
      // Point size in device pixels so neighbouring depth pixels just touch at the capture pose.
      const size = Math.ceil(renderer.domElement.width / r.w) + 1;
      cloud = new THREE.Points(g, new THREE.PointsMaterial({ size, sizeAttenuation: false, vertexColors: true }));
      scene.add(cloud);
      // Same vertical field of view as the preview (the stage has the stream's 9:16 aspect).
      camera.fov = (2 * Math.atan(r.h / 2 / f) * 180) / Math.PI;
      camera.updateProjectionMatrix();
      zs.sort((a, b) => a - b);
      target.set(0, 0, -(zs[zs.length >> 1] ?? 0.5));
      home();
      setTilt(!!orientation.q);
      return { points: k };
    },
    hide() { canvas.hidden = true; tilt = false; controls.enabled = false; },
    get visible() { return !canvas.hidden; },
  };
}
