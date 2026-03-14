#!/usr/bin/env python3
import argparse
import csv
from collections import Counter, defaultdict

from autodl_single.io_utils import ensure_output_dir


QUALITY_ORDER = ["ok", "truncated", "garbled", "off_topic", "runtime_error"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    rows = load_rows(args.inputs)

    grouped = defaultdict(Counter)
    totals = Counter()
    for row in rows:
        mode = row["mode"]
        quality_flag = row.get("quality_flag", "runtime_error")
        grouped[mode][quality_flag] += 1
        totals[mode] += 1

    summary_rows = []
    for mode in sorted(grouped):
        total = totals[mode]
        for quality_flag in QUALITY_ORDER:
            count = grouped[mode][quality_flag]
            summary_rows.append(
                {
                    "mode": mode,
                    "quality_flag": quality_flag,
                    "count": count,
                    "ratio": f"{(count / total) if total else 0:.6f}",
                }
            )

    summary_path = output_dir / "failure_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "quality_flag", "count", "ratio"])
        writer.writeheader()
        writer.writerows(summary_rows)

    notes_path = output_dir / "failure_summary.txt"
    with open(notes_path, "w", encoding="utf-8") as handle:
        for mode in sorted(grouped):
            handle.write(f"[{mode}] total={totals[mode]}\n")
            for quality_flag in QUALITY_ORDER:
                handle.write(
                    f"{quality_flag}={grouped[mode][quality_flag]} "
                    f"ratio={(grouped[mode][quality_flag] / totals[mode]) if totals[mode] else 0:.6f}\n"
                )
            handle.write("\n")

    print(f"Wrote failure summary to {summary_path}")


if __name__ == "__main__":
    main()
