#!/usr/bin/env node
// Drive the capture page in Chrome on the phone over the DevTools protocol.
//   node tools/cdp.mjs open                 # adb reverse/forward, open page in Chrome, grant camera
//   node tools/cdp.mjs eval '<js expr>'     # evaluate (awaits promises), prints JSON result
//   node tools/cdp.mjs reload
import { execFileSync } from "node:child_process";

const ADB = process.env.ADB || `${process.env.HOME}/Android/Sdk/platform-tools/adb`;
const PORT = 8000;
const CDP = 9222;
const ORIGIN = `http://localhost:${PORT}`;
const adb = (...a) => execFileSync(ADB, a, { encoding: "utf8" });

function setupPorts() {
  adb("reverse", `tcp:${PORT}`, `tcp:${PORT}`);
  adb("forward", `tcp:${CDP}`, "localabstract:chrome_devtools_remote");
}

async function json(path) {
  const r = await fetch(`http://127.0.0.1:${CDP}${path}`, { signal: AbortSignal.timeout(5000) });
  return r.json();
}

function connect(wsUrl) {
  const ws = new WebSocket(wsUrl);
  let id = 0;
  const pending = new Map();
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) {
      const { res, rej } = pending.get(m.id);
      pending.delete(m.id);
      m.error ? rej(new Error(JSON.stringify(m.error))) : res(m.result);
    }
  };
  const ready = new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  const send = (method, params = {}) =>
    ready.then(() => new Promise((res, rej) => {
      const i = ++id;
      pending.set(i, { res, rej });
      ws.send(JSON.stringify({ id: i, method, params }));
    }));
  return { send, close: () => ws.close() };
}

async function page() {
  const pages = (await json("/json")).filter((p) => p.type === "page" && p.url.startsWith(ORIGIN));
  if (!pages.length) throw new Error("capture page not open in Chrome");
  return connect(pages[0].webSocketDebuggerUrl);
}

async function grant() {
  const v = await json("/json/version");
  const b = connect(v.webSocketDebuggerUrl);
  await b.send("Browser.grantPermissions", { origin: ORIGIN, permissions: ["videoCapture"] }).catch((e) => console.error("grant:", e.message));
  b.close();
}

const [cmd, arg] = process.argv.slice(2);
if (cmd === "open") {
  setupPorts();
  adb("shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", `${ORIGIN}/capture.html?t=${Date.now()}`, "com.android.chrome");
  await new Promise((r) => setTimeout(r, 2500));
  await grant();
  console.log("opened");
} else if (cmd === "eval") {
  setupPorts();
  const p = await page();
  const r = await p.send("Runtime.evaluate", { expression: arg, awaitPromise: true, returnByValue: true, userGesture: true, timeout: 600000 });
  p.close();
  if (r.exceptionDetails) {
    console.error("EXCEPTION", JSON.stringify(r.exceptionDetails.exception?.description ?? r.exceptionDetails));
    process.exit(1);
  }
  console.log(JSON.stringify(r.result.value, null, 1));
} else if (cmd === "reload") {
  setupPorts();
  const p = await page();
  await p.send("Page.reload", { ignoreCache: true });
  p.close();
  console.log("reloaded");
} else {
  console.log("usage: cdp.mjs open | eval <expr> | reload");
}
