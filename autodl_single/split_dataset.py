#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--parts", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="medical_qa")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [line for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunk_size = math.ceil(len(lines) / args.parts)
    width = max(2, len(str(args.parts)))

    for index in range(args.parts):
        chunk = lines[index * chunk_size : (index + 1) * chunk_size]
        if not chunk:
            continue
        output_path = output_dir / f"{args.prefix}_{index + 1:0{width}d}.jsonl"
        output_path.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        print(f"Wrote {len(chunk)} rows to {output_path}")


if __name__ == "__main__":
    main()
