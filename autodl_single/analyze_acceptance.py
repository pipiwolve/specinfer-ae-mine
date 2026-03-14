#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from autodl_single.common import ensure_output_dir


ACCEPT_PATTERNS = [
    re.compile(r"accepted[_ ]tokens[:=]\s*(\d+)", re.IGNORECASE),
    re.compile(r"tokens[_ ]to[_ ]commit[:=]\s*(\d+)", re.IGNORECASE),
]
REJECT_PATTERNS = [
    re.compile(r"rejected[_ ]tokens[:=]\s*(\d+)", re.IGNORECASE),
]
VERIFIED_PATTERNS = [
    re.compile(r"Number of Verified Tokens\s*=\s*(\d+)", re.IGNORECASE),
]
ROW_END_PATTERN = re.compile(
    r"\[BatchEnd\]\s+mode=(?P<mode>\w+)\s+id=(?P<id>[^ ]+)\s+repeat=(?P<repeat>\d+)\s+"
    r"success=(?P<success>\d+)\s+quality_flag=(?P<quality_flag>[^ ]+)",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-results", required=True)
    parser.add_argument("--baseline-results", default="")
    parser.add_argument("--log-glob", default="")
    parser.add_argument("--tokenizer-model", default="huggyllama/llama-7b")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_rows(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_logs(log_glob: str):
    accepted = []
    rejected = []
    verified = []
    per_prompt_verified = []
    if not log_glob:
        return accepted, rejected, verified, per_prompt_verified
    for path in glob.glob(log_glob):
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        current_verified = []
        for line in text.splitlines():
            for pattern in VERIFIED_PATTERNS:
                current_verified.extend(int(match) for match in pattern.findall(line))
            row_end = ROW_END_PATTERN.search(line)
            if row_end:
                if current_verified:
                    per_prompt_verified.append(
                        {
                            "id": row_end.group("id"),
                            "repeat": row_end.group("repeat"),
                            "metric_type": "verified_tokens_per_step",
                            "verified_tokens_mean": mean(current_verified),
                            "acceptance_ratio": math.nan,
                        }
                    )
                    verified.extend(current_verified)
                current_verified = []
        for pattern in ACCEPT_PATTERNS:
            accepted.extend(int(match) for match in pattern.findall(text))
        for pattern in REJECT_PATTERNS:
            rejected.extend(int(match) for match in pattern.findall(text))
    return accepted, rejected, verified, per_prompt_verified


def compute_output_agreement(spec_rows, baseline_rows, tokenizer_model):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
    baseline_by_key = {(row["id"], row["repeat"]): row for row in baseline_rows if row["success"] == "1"}
    metrics = []
    for row in spec_rows:
        if row["success"] != "1":
            continue
        key = (row["id"], row["repeat"])
        baseline = baseline_by_key.get(key)
        if baseline is None:
            continue
        spec_tokens = tokenizer.encode(row["output_text"].replace("\\n", "\n"))
        base_tokens = tokenizer.encode(baseline["output_text"].replace("\\n", "\n"))
        common = 0
        for spec_token, base_token in zip(spec_tokens, base_tokens):
            if spec_token != base_token:
                break
            common += 1
        denominator = max(len(base_tokens), 1)
        metrics.append(
            {
                "id": row["id"],
                "repeat": row["repeat"],
                "metric_type": "output_agreement_proxy",
                "acceptance_ratio": common / denominator,
            }
        )
    return metrics


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    spec_rows = load_rows(args.spec_results)
    baseline_rows = load_rows(args.baseline_results) if args.baseline_results else []

    accepted, rejected, verified, per_prompt_verified = parse_logs(args.log_glob)
    metrics = []

    if accepted:
        total_accepted = sum(accepted)
        total_rejected = sum(rejected)
        denom = total_accepted + total_rejected
        ratio = total_accepted / denom if denom else math.nan
        metrics.append(
            {
                "id": "aggregate",
                "repeat": "all",
                "metric_type": "log_parsed_acceptance",
                "acceptance_ratio": ratio,
                "verified_tokens_mean": math.nan,
            }
        )
    if per_prompt_verified:
        metrics.extend(per_prompt_verified)
    elif baseline_rows:
        metrics = compute_output_agreement(spec_rows, baseline_rows, args.tokenizer_model)

    metrics_path = output_dir / "acceptance_metrics.csv"
    with open(metrics_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "repeat", "metric_type", "acceptance_ratio", "verified_tokens_mean"],
        )
        writer.writeheader()
        writer.writerows(metrics)

    by_prompt = {}
    for metric in metrics:
        by_prompt.setdefault(metric["id"], []).append(metric)
    per_prompt = {
        key: {
            "metric_types": sorted({value["metric_type"] for value in values}),
            "mean_acceptance_ratio": mean(
                value["acceptance_ratio"] for value in values if not math.isnan(value.get("acceptance_ratio", math.nan))
            )
            if any(not math.isnan(value.get("acceptance_ratio", math.nan)) for value in values)
            else math.nan,
            "median_acceptance_ratio": median(
                value["acceptance_ratio"] for value in values if not math.isnan(value.get("acceptance_ratio", math.nan))
            )
            if any(not math.isnan(value.get("acceptance_ratio", math.nan)) for value in values)
            else math.nan,
            "mean_verified_tokens": mean(
                value["verified_tokens_mean"] for value in values if not math.isnan(value.get("verified_tokens_mean", math.nan))
            )
            if any(not math.isnan(value.get("verified_tokens_mean", math.nan)) for value in values)
            else math.nan,
        }
        for key, values in by_prompt.items()
    }

    json_path = output_dir / "acceptance_by_prompt.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(per_prompt, handle, ensure_ascii=False, indent=2)

    ratios = [metric["acceptance_ratio"] for metric in metrics if not math.isnan(metric["acceptance_ratio"])]
    figure_path = output_dir / "acceptance_distribution.png"
    plt.figure(figsize=(8, 5))
    plt.hist(ratios if ratios else [0.0], bins=10, color="#2f7d64", edgecolor="white")
    plt.xlabel("Acceptance ratio")
    plt.ylabel("Count")
    plt.title("SpecInfer acceptance-style metric distribution")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)

    summary_path = output_dir / "acceptance_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        if ratios:
            handle.write(f"mean_acceptance_ratio={mean(ratios):.6f}\n")
            handle.write(f"median_acceptance_ratio={median(ratios):.6f}\n")
        else:
            handle.write("mean_acceptance_ratio=\n")
            handle.write("median_acceptance_ratio=\n")
        if verified:
            handle.write(f"mean_verified_tokens={mean(verified):.6f}\n")
            handle.write(f"median_verified_tokens={median(verified):.6f}\n")
        else:
            handle.write("mean_verified_tokens=\n")
            handle.write("median_verified_tokens=\n")
        handle.write(f"metric_source={metrics[0]['metric_type'] if metrics else 'unavailable'}\n")

    print(f"Wrote acceptance summary to {metrics_path}")


if __name__ == "__main__":
    main()
