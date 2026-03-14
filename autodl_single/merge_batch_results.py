#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

from autodl_single.io_utils import ensure_output_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def resolve_input_paths(inputs):
    resolved = []
    missing = []
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_dir():
            path = path / "results.csv"
        if path.exists():
            resolved.append(path)
        else:
            missing.append(raw_path)
    if missing:
        raise FileNotFoundError(
            "Missing input results files. Pass either existing results.csv files or directories "
            f"that contain results.csv. Missing: {', '.join(missing)}"
        )
    return resolved


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    rows = []
    input_paths = resolve_input_paths(args.inputs)

    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))

    rows.sort(key=lambda row: (row["mode"], row["id"], int(row["repeat"])))

    results_path = output_dir / "results.csv"
    results_jsonl_path = output_dir / "results.jsonl"
    fieldnames = list(rows[0].keys()) if rows else []

    if rows:
        with open(results_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        with open(results_jsonl_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Merged {len(rows)} rows from {len(input_paths)} inputs into {results_path}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise
