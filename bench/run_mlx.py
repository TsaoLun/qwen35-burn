#!/usr/bin/env python3
"""
Benchmark Qwen3.5-0.8B using mlx-lm.

Requirements:
    pip install mlx-lm

mlx-lm's --verbose flag prints timing in the format:
    Prompt: N tokens, X.XX tokens-per-second
    Generation: N tokens, X.XX tokens-per-second
This script parses those lines and records the results as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_MODEL_DIR = SCRIPT_DIR.parent / "models" / "Qwen3.5-0.8B"

# ~100-token prompt matching the Rust bench binary
DEFAULT_PROMPT = (
    "Explain the following algorithms and data structures in detail, covering "
    "time complexity, space complexity, and practical use cases: quicksort, "
    "mergesort, binary search trees, hash tables, dynamic programming, BFS/DFS "
    "graph traversal, and heap sort. For each, give a concrete code example."
)


def _mlx_cmd() -> list[str]:
    """Return the command prefix for mlx_lm generate."""
    # Prefer the mlx_lm entry-point binary if it's on PATH
    bin_path = shutil.which("mlx_lm")
    if bin_path:
        return [bin_path, "generate"]
    # Fall back to python -m mlx_lm generate
    return [sys.executable, "-m", "mlx_lm", "generate"]


def run_once(
    model_dir: str, prompt: str, max_tokens: int, temperature: float = 0.7
) -> dict:
    """Run mlx_lm generate once and parse timing output."""
    # mlx-lm >= 0.20 uses subcommand form: `mlx_lm generate`
    # --verbose requires an explicit boolean value: --verbose true
    cmd = [
        *_mlx_cmd(),
        "--model",
        model_dir,
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--verbose",
        "true",
        "--temp",
        str(temperature),
        "--ignore-chat-template",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Parse lines like:
    #   Prompt: 96 tokens, 234.56 tokens-per-sec
    #   Generation: 100 tokens, 45.67 tokens-per-sec
    prompt_m = re.search(
        r"Prompt:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec", output
    )
    gen_m = re.search(
        r"Generation:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec", output
    )

    if not prompt_m or not gen_m:
        sys.stderr.write("--- mlx_lm output ---\n" + output + "\n---\n")
        raise RuntimeError(
            "Could not parse mlx_lm timing output.  "
            "Make sure mlx-lm is installed: pip install mlx-lm"
        )

    prompt_tokens = int(prompt_m.group(1))
    prefill_tps = float(prompt_m.group(2))
    decode_tokens = int(gen_m.group(1))
    decode_tps = float(gen_m.group(2))
    ttft_ms = prompt_tokens / prefill_tps * 1000.0

    return {
        "prompt_tokens": prompt_tokens,
        "decode_tokens": decode_tokens,
        "ttft_ms": ttft_ms,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX Qwen3.5-0.8B benchmark")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--decode-tokens", type=int, default=100)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0=greedy, default 0.7 matches chat)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", default=str(SCRIPT_DIR / "results" / "mlx.json"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    # Pass the raw user message; mlx-lm applies the model's chat template itself.
    # We use --ignore-chat-template so the ChatML wrapping is identical to the
    # Burn bench binary's manually built prompt.
    formatted = f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"

    print(f"=== MLX benchmark ===")
    print(f"Model : {args.model_dir}")
    print(f"Warmup: {args.warmup} run(s); Measured: {args.runs} run(s)")

    # Warmup uses greedy (temp=0) for speed; MLX sampling runs on-device so
    # temperature does not affect which kernels are compiled.
    print("Warming up...")
    for _ in range(args.warmup):
        run_once(args.model_dir, formatted, 32, temperature=0.0)

    ttft_all, prefill_all, decode_all = [], [], []
    last: dict = {}
    for i in range(args.runs):
        r = run_once(
            args.model_dir, formatted, args.decode_tokens, temperature=args.temperature
        )
        print(
            f"  run {i + 1}/{args.runs}: "
            f"TTFT={r['ttft_ms']:.0f}ms  "
            f"prefill={r['prefill_tps']:.1f} tok/s  "
            f"decode={r['decode_tps']:.1f} tok/s"
        )
        ttft_all.append(r["ttft_ms"])
        prefill_all.append(r["prefill_tps"])
        decode_all.append(r["decode_tps"])
        last = r

    def avg(lst: list[float]) -> float:
        return sum(lst) / len(lst)

    result = {
        "framework": "mlx-lm",
        "model": "Qwen3.5-0.8B",
        "backend": "MLX/Metal",
        "device": "Apple M3",
        "prompt_tokens": last.get("prompt_tokens", 0),
        "decode_tokens": args.decode_tokens,
        "temperature": args.temperature,
        "warmup_runs": args.warmup,
        "bench_runs": args.runs,
        "ttft_ms_avg": avg(ttft_all),
        "prefill_tps_avg": avg(prefill_all),
        "decode_tps_avg": avg(decode_all),
        "ttft_ms_all": ttft_all,
        "prefill_tps_all": prefill_all,
        "decode_tps_all": decode_all,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
