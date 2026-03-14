#!/usr/bin/env python3
import argparse
import csv
import json

from autodl_single.io_utils import ensure_output_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    rows = []

    for input_path in args.inputs:
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

    print(f"Merged {len(rows)} rows into {results_path}")


if __name__ == "__main__":
    main()
