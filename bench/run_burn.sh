#!/usr/bin/env bash
# Run the Burn (WGPU/Metal) throughput benchmark.
#
# Builds with wgpu (fusion + autotune) for best performance.
#
# Environment variables (all optional):
#   MODEL_DIR      path to Qwen3.5-0.8B model dir  (default: ../models/Qwen3.5-0.8B)
#   DECODE_TOKENS  tokens to decode per run         (default: 100)
#   TEMPERATURE    sampling temperature 0=greedy    (default: 0.7)
#   WARMUP         warmup runs; autotune needs ≥3   (default: 5)
#   RUNS           measured runs                    (default: 5)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/Qwen3.5-0.8B}"
DECODE_TOKENS="${DECODE_TOKENS:-100}"
TEMPERATURE="${TEMPERATURE:-0.7}"
WARMUP="${WARMUP:-5}"
RUNS="${RUNS:-5}"
OUTPUT="$SCRIPT_DIR/results/burn.json"

mkdir -p "$SCRIPT_DIR/results"

export CUBECL_WGPU_MAX_TASKS=8
export CUBECL_AUTOTUNE_LEVEL="full"  # minimal(0), balanced(1), extensive(2), full(3)

echo "=== Burn benchmark (wgpu: fusion + autotune) ==="
echo "Building..."
cd "$PROJECT_DIR"
cargo build --release --features wgpu --example bench 2>&1 | tail -3

echo "Running..."
./target/release/examples/bench \
    --model-dir     "$MODEL_DIR"     \
    --backend       wgpu             \
    --decode-tokens "$DECODE_TOKENS" \
    --temperature   "$TEMPERATURE"   \
    --warmup        "$WARMUP"        \
    --runs          "$RUNS"          \
    --output        "$OUTPUT"

echo "Done → $OUTPUT"
