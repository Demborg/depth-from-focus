// Injected before page scripts (debug only): records RTCPeerConnection state changes in window.__rtc.
(() => {
  const Orig = window.RTCPeerConnection;
  window.__rtc = [];
  const t0 = performance.now();
  const log = (...a) => window.__rtc.push([Math.round(performance.now() - t0), ...a]);
  window.RTCPeerConnection = function (...args) {
    const pc = new Orig(...args);
    const id = Math.random().toString(36).slice(2, 6);
    log(id, "new");
    for (const ev of ["connectionstatechange", "iceconnectionstatechange", "signalingstatechange"])
      pc.addEventListener(ev, () => log(id, ev.replace("statechange", ""), pc.connectionState, pc.iceConnectionState));
    pc.addEventListener("datachannel", (e) => e.channel.addEventListener("close", () => log(id, "dc-close")));
    const cdc = pc.createDataChannel.bind(pc);
    pc.createDataChannel = (...a) => { const c = cdc(...a); c.addEventListener("close", () => log(id, "dc-close")); return c; };
    const close = pc.close.bind(pc);
    pc.close = () => { log(id, "close() called", new Error().stack.split("\n").slice(2, 4).join(" | ")); close(); };
    return pc;
  };
  window.RTCPeerConnection.prototype = Orig.prototype;
})();
