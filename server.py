#!/usr/bin/env python3
"""Tiny dev server: serves web/ and accepts focal-stack uploads from the phone.

Phone reaches it via `adb reverse tcp:8000 tcp:8000` -> http://localhost:8000 (secure context).

  POST /api/frame?stack=<id>&i=<n>   body: image/jpeg   -> data/stacks/<id>/<n>.jpg
  POST /api/meta?stack=<id>          body: json          -> data/stacks/<id>/meta.json
  POST /api/log                      body: json          -> appended to data/log.jsonl + printed
"""
import json
import re
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
DATA = ROOT / "data"
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args):
        if "/api/frame" not in str(args[0] if args else ""):
            sys.stderr.write("%s\n" % (format % args))

    def translate_path(self, path):
        # /stacks/<id>/... is served from data/stacks so the viewer can replay captured sweeps.
        p = urlparse(path).path
        if p.startswith("/stacks/"):
            parts = [x for x in p.split("/")[2:] if x]
            if all(SAFE.match(x) for x in parts):
                return str(DATA / "stacks" / Path(*parts)) if parts else str(DATA / "stacks")
        return super().translate_path(path)

    def _reply(self, code=200, obj=None):
        body = json.dumps(obj or {"ok": True}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if u.path == "/api/log":
            DATA.mkdir(exist_ok=True)
            rec = {"t": time.time(), **json.loads(body or b"{}")}
            with open(DATA / "log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print("LOG", json.dumps(rec)[:2000], flush=True)
            return self._reply()
        stack = q.get("stack", "")
        if not SAFE.match(stack):
            return self._reply(400, {"error": "bad stack id"})
        d = DATA / "stacks" / stack
        d.mkdir(parents=True, exist_ok=True)
        if u.path == "/api/frame":
            i = int(q["i"])
            (d / f"{i:03d}.jpg").write_bytes(body)
            return self._reply()
        if u.path == "/api/meta":
            (d / "meta.json").write_bytes(body)
            print("META", stack, flush=True)
            return self._reply()
        self._reply(404, {"error": "unknown"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"serving {WEB} on :{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
