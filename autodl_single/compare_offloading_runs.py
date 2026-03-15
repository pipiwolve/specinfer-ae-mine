import argparse
import csv
from pathlib import Path


def load_rows(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            batch_size = int(row["batch_size"])
            rows[batch_size] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--spec-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline_rows = load_rows(Path(args.baseline_summary))
    spec_rows = load_rows(Path(args.spec_summary))

    fieldnames = [
        "batch_size",
        "baseline_mean_us",
        "baseline_p50_us",
        "spec_mean_us",
        "spec_p50_us",
        "speedup_mean",
        "speedup_p50",
        "baseline_finished",
        "spec_finished",
    ]

    output_rows = []
    for batch_size in sorted(set(baseline_rows) & set(spec_rows)):
        base = baseline_rows[batch_size]
        spec = spec_rows[batch_size]
        base_mean = float(base["latency_mean_us"]) if base["latency_mean_us"] else 0.0
        base_p50 = float(base["latency_p50_us"]) if base["latency_p50_us"] else 0.0
        spec_mean = float(spec["latency_mean_us"]) if spec["latency_mean_us"] else 0.0
        spec_p50 = float(spec["latency_p50_us"]) if spec["latency_p50_us"] else 0.0
        output_rows.append(
            {
                "batch_size": batch_size,
                "baseline_mean_us": f"{base_mean:.3f}" if base_mean else "",
                "baseline_p50_us": f"{base_p50:.3f}" if base_p50 else "",
                "spec_mean_us": f"{spec_mean:.3f}" if spec_mean else "",
                "spec_p50_us": f"{spec_p50:.3f}" if spec_p50 else "",
                "speedup_mean": f"{(base_mean / spec_mean):.6f}" if base_mean and spec_mean else "",
                "speedup_p50": f"{(base_p50 / spec_p50):.6f}" if base_p50 and spec_p50 else "",
                "baseline_finished": base["finished"],
                "spec_finished": spec["finished"],
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
