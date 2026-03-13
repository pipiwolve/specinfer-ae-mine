#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt

from autodl_single.common import ensure_output_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["success"] != "1":
                continue
            row["latency_seconds"] = float(row["latency_seconds"])
            rows.append(row)
    return rows


def p95(values):
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


def aggregate_by_id(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["id"], []).append(row["latency_seconds"])
    return {key: median(values) for key, values in grouped.items()}


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    baseline_rows = load_rows(args.baseline)
    spec_rows = load_rows(args.spec)

    baseline_by_id = aggregate_by_id(baseline_rows)
    spec_by_id = aggregate_by_id(spec_rows)
    shared_ids = sorted(set(baseline_by_id) & set(spec_by_id))

    summary_rows = []
    speedups = []
    for prompt_id in shared_ids:
        base_latency = baseline_by_id[prompt_id]
        spec_latency = spec_by_id[prompt_id]
        speedup = base_latency / spec_latency if spec_latency else math.nan
        speedups.append(speedup)
        summary_rows.append(
            {
                "id": prompt_id,
                "baseline_latency_seconds": f"{base_latency:.6f}",
                "spec_latency_seconds": f"{spec_latency:.6f}",
                "speedup": f"{speedup:.6f}",
            }
        )

    summary_path = output_dir / "latency_ab.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "baseline_latency_seconds", "spec_latency_seconds", "speedup"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    baseline_latencies = [row["latency_seconds"] for row in baseline_rows]
    spec_latencies = [row["latency_seconds"] for row in spec_rows]

    figure_path = output_dir / "latency_speedup.png"
    plt.figure(figsize=(8, 5))
    plt.bar(["Incremental", "SpecInfer"], [mean(baseline_latencies), mean(spec_latencies)], color=["#c65d3b", "#2f7d64"])
    plt.ylabel("Average latency (seconds)")
    plt.title("SpecInfer latency comparison")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)

    report_path = output_dir / "latency_summary.txt"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(f"baseline_mean={mean(baseline_latencies):.6f}\n")
        handle.write(f"baseline_p50={median(baseline_latencies):.6f}\n")
        handle.write(f"baseline_p95={p95(baseline_latencies):.6f}\n")
        handle.write(f"spec_mean={mean(spec_latencies):.6f}\n")
        handle.write(f"spec_p50={median(spec_latencies):.6f}\n")
        handle.write(f"spec_p95={p95(spec_latencies):.6f}\n")
        handle.write(f"average_speedup={mean(speedups):.6f}\n")

    print(f"Wrote latency summary to {summary_path}")


if __name__ == "__main__":
    main()
