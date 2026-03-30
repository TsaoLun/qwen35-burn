#!/usr/bin/env python3
"""
Read benchmark results from bench/results/*.json and print a Markdown table.

Usage:
    python3 bench/compare.py
    python3 bench/compare.py --markdown   # same, explicit
    python3 bench/compare.py --csv        # CSV output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"

# Display order for known frameworks
DISPLAY_ORDER = ["burn-wgpu", "burn-ndarray", "mlx-lm"]

FRAMEWORK_LABELS: dict[str, str] = {
    "burn-wgpu": "**burn-wgpu** (this project)",
    "burn-ndarray": "burn-ndarray (CPU)",
    "mlx-lm": "mlx-lm",
}


def load_results() -> dict[str, dict]:
    results: dict[str, dict] = {}
    if not RESULTS_DIR.exists():
        return results
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        fw = data.get("framework", path.stem)
        results[fw] = data
    return results


def fmt(val: object, fmt_spec: str = ".1f", fallback: str = "–") -> str:
    try:
        return format(float(val), fmt_spec)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def print_markdown(results: dict[str, dict]) -> None:
    if not results:
        print("No results found in bench/results/. Run the benchmark scripts first.")
        return

    ordered: list[tuple[str, dict]] = []
    for fw in DISPLAY_ORDER:
        if fw in results:
            ordered.append((fw, results[fw]))
    for fw, data in results.items():
        if fw not in DISPLAY_ORDER:
            ordered.append((fw, data))

    def fmt_temp(t: object) -> str:
        try:
            v = float(t)  # type: ignore[arg-type]
            return "greedy" if v <= 0.0 else f"temp={v:.1f}"
        except (TypeError, ValueError):
            return "?"

    # Header
    print()
    print(
        "| Framework | Model | Backend | Sampling | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) |"
    )
    print(
        "|-----------|-------|---------|----------|----------------:|---------------:|----------:|"
    )

    for fw, d in ordered:
        label = FRAMEWORK_LABELS.get(fw, fw)
        model = d.get("model", "?")
        backend = d.get("backend", "?")
        sampling = fmt_temp(d.get("temperature"))
        prefill = (
            fmt(d.get("prefill_tps_avg")) if d.get("prefill_tps_avg", 0) > 0 else "–"
        )
        decode = fmt(d.get("decode_tps_avg"))
        ttft = fmt(d.get("ttft_ms_avg"), ".0f")
        print(
            f"| {label} | {model} | {backend} | {sampling} | {prefill} | {decode} | {ttft} |"
        )

    print()


def print_csv(results: dict[str, dict]) -> None:
    print("framework,model,backend,temperature,prefill_tps,decode_tps,ttft_ms")
    for fw, d in results.items():
        row = [
            fw,
            d.get("model", ""),
            d.get("backend", ""),
            fmt(d.get("temperature", 0.0)),
            fmt(d.get("prefill_tps_avg", 0)),
            fmt(d.get("decode_tps_avg", 0)),
            fmt(d.get("ttft_ms_avg", 0), ".0f"),
        ]
        print(",".join(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare inference benchmark results")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--markdown",
        action="store_true",
        default=True,
        help="Output Markdown table (default)",
    )
    group.add_argument("--csv", action="store_true", help="Output CSV")
    args = parser.parse_args()

    results = load_results()

    if args.csv:
        print_csv(results)
    else:
        print_markdown(results)


if __name__ == "__main__":
    main()
