#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict

from autodl_single.io_utils import ensure_output_dir


QUALITY_RANK = {"ok": 0, "truncated": 1, "garbled": 2, "off_topic": 3, "runtime_error": 4}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=15)
    return parser.parse_args()


def load_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def choose_ids(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["id"]].append(row)

    selected = {}
    for prompt_id, group in grouped.items():
        if all(row["success"] == "1" and row.get("quality_flag") != "runtime_error" for row in group):
            score = min(QUALITY_RANK.get(row.get("quality_flag", "runtime_error"), 99) for row in group)
            selected[prompt_id] = score
    return selected


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    baseline_rows = load_rows(args.baseline)
    spec_rows = load_rows(args.spec)

    baseline_ids = choose_ids(baseline_rows)
    spec_ids = choose_ids(spec_rows)
    shared_ids = sorted(set(baseline_ids) & set(spec_ids), key=lambda item: (baseline_ids[item] + spec_ids[item], item))
    selected_ids = shared_ids[: args.limit]

    dataset_rows = []
    with open(args.dataset, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["id"] in selected_ids:
                    dataset_rows.append(row)

    subset_path = output_dir / "stable_subset.jsonl"
    with open(subset_path, "w", encoding="utf-8") as handle:
        for row in dataset_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = output_dir / "stable_subset_ids.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(selected_ids, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(selected_ids)} stable ids to {subset_path}")


if __name__ == "__main__":
    main()
