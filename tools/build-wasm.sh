#!/usr/bin/env bash
# Build the wasm core into web/dff.wasm (needs: rustup target add wasm32-unknown-unknown).
set -euo pipefail
cd "$(dirname "$0")/.."
RUSTFLAGS="-C target-feature=+simd128" cargo build --release -p dff-wasm --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/dff_wasm.wasm web/dff.wasm
ls -l web/dff.wasm
