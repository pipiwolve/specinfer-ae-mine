#!/usr/bin/env python3
import argparse
import csv
import json
import traceback
from pathlib import Path

from autodl_single.common import FlexFlowRunner, ensure_output_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["incr", "spec"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def load_dataset(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def make_prompt(record: dict) -> str:
    return (
        "You are a concise medical QA assistant. "
        "Provide a short, cautious answer and avoid overclaiming.\n"
        f"Question: {record['question']}\n"
        "Answer:"
    )


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    results_path = output_dir / "results.csv"
    results_jsonl_path = output_dir / "results.jsonl"
    dataset = load_dataset(args.dataset)

    rows = []
    runner = FlexFlowRunner(args.config)
    runner.start()
    try:
        for record in dataset:
            prompt = make_prompt(record)
            for repeat in range(1, args.repeats + 1):
                row = {
                    "id": record["id"],
                    "category": record["category"],
                    "mode": args.mode,
                    "repeat": repeat,
                    "question": record["question"],
                    "reference_answer": record["reference_answer"],
                    "source": record["source"],
                }
                try:
                    result = runner.generate_text(prompt, max_length=args.max_length)
                    row.update(
                        {
                            "success": 1,
                            "latency_seconds": f"{result['latency_seconds']:.6f}",
                            "prompt_tokens": result["prompt_tokens"],
                            "completion_tokens": result["completion_tokens"],
                            "total_tokens": result["total_tokens"],
                            "output_text": result["completion_text"].replace("\n", "\\n"),
                            "error": "",
                        }
                    )
                except Exception as exc:
                    row.update(
                        {
                            "success": 0,
                            "latency_seconds": "",
                            "prompt_tokens": "",
                            "completion_tokens": "",
                            "total_tokens": "",
                            "output_text": "",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    traceback.print_exc()
                rows.append(row)
    finally:
        runner.stop()

    fieldnames = [
        "id",
        "category",
        "mode",
        "repeat",
        "question",
        "reference_answer",
        "source",
        "success",
        "latency_seconds",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "output_text",
        "error",
    ]
    with open(results_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(results_jsonl_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {results_path}")


if __name__ == "__main__":
    main()
