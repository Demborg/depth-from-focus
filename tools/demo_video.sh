#!/usr/bin/env bash
# Render the demo loop (web/demo.html) of a stored capture to an mp4.
#   tools/demo_video.sh <stack> [extra demo.html query, e.g. "&orbit=7&yaw=20"]
# Needs the dev server (python3 server.py) and a headless Chrome with WebGPU on $DESK_PORT
# (default 9336), e.g.
#   google-chrome --headless=new --remote-debugging-port=9336 --user-data-dir=/tmp/chrome-gpu \
#     --enable-unsafe-webgpu --enable-features=Vulkan,WebGPUService --use-angle=vulkan --ignore-gpu-blocklist
# Frames land in data/render/<stack>/, the video in out/<stack>-loop.mp4.
set -euo pipefail
cd "$(dirname "$0")/.."
stack=${1:?usage: tools/demo_video.sh <stack> [query]}
extra=${2:-}
export DESK_PORT=${DESK_PORT:-9336}
rm -rf "data/render/$stack"
node tools/desk.mjs open "http://localhost:8000/demo.html?stack=$stack&export$extra" >/dev/null
for _ in $(seq 60); do
  frames=$(node tools/desk.mjs eval 'window.demo?.frames ?? 0' 2>/dev/null || echo 0)
  [ "$frames" != 0 ] && break
  sleep 1
done
[ "$frames" != 0 ] || { echo "demo page did not load"; exit 1; }
fps=$(node tools/desk.mjs eval 'demo.fps')
echo "$stack: $frames frames at $fps fps"
for ((i = 0; i < frames; i += 60)); do
  node tools/desk.mjs eval "demo.exportFrames('$stack', $i, $((i + 60)))" >/dev/null
  echo -ne "\r$((i + 60 < frames ? i + 60 : frames))/$frames"
done
echo
mkdir -p out
ffmpeg -loglevel error -y -framerate "$fps" -i "data/render/$stack/%05d.jpg" \
  -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -movflags +faststart "out/$stack-loop.mp4"
echo "out/$stack-loop.mp4"
