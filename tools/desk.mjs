#!/usr/bin/env node
// Drive a local (headless) desktop Chrome over CDP for testing the viewer.
//   google-chrome --headless=new --remote-debugging-port=9333 ...
//   node tools/desk.mjs open <url> | eval '<expr>' | shot <file.png>
import { writeFileSync } from "node:fs";

const PORT = process.env.DESK_PORT || 9333;
const json = (p, method = "GET") => fetch(`http://127.0.0.1:${PORT}${p}`, { method, signal: AbortSignal.timeout(5000) }).then((r) => r.json());

function connect(url) {
  const ws = new WebSocket(url);
  let id = 0;
  const pending = new Map();
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)) { const { res, rej } = pending.get(m.id); pending.delete(m.id); m.error ? rej(new Error(JSON.stringify(m.error))) : res(m.result); }
  };
  const ready = new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  return {
    send: (method, params = {}) => ready.then(() => new Promise((res, rej) => { const i = ++id; pending.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params })); })),
    close: () => ws.close(),
  };
}

async function page() {
  const pages = (await json("/json")).filter((p) => p.type === "page");
  return connect(pages[0].webSocketDebuggerUrl);
}

const [cmd, arg] = process.argv.slice(2);
const p = await page();
if (cmd === "open") {
  await p.send("Page.navigate", { url: arg });
  console.log("opened", arg);
} else if (cmd === "eval") {
  const r = await p.send("Runtime.evaluate", { expression: arg, awaitPromise: true, returnByValue: true, timeout: 120000 });
  if (r.exceptionDetails) { console.error("EXCEPTION", r.exceptionDetails.exception?.description ?? JSON.stringify(r.exceptionDetails)); process.exit(1); }
  console.log(JSON.stringify(r.result.value, null, 1));
} else if (cmd === "shot") {
  const r = await p.send("Page.captureScreenshot", { format: "png" });
  writeFileSync(arg, Buffer.from(r.data, "base64"));
  console.log("saved", arg);
}
p.close();
