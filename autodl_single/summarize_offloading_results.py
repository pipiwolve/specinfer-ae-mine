import argparse
import csv
import re
from pathlib import Path


PROFILE_RE = re.compile(r"latency\(([\d.]+)\)")


def parse_latencies(path: Path) -> list[float]:
    latencies = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "[Profile]" not in line:
                continue
            match = PROFILE_RE.search(line)
            if match:
                latencies.append(float(match.group(1)))
    return latencies


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="FlexFlow/inference/output/autodl_single/offloading_single_a100/default",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows = []
    for path in sorted(input_dir.glob("offloading_*_*.out")):
        name = path.stem
        parts = name.split("_")
        if len(parts) < 3:
            continue
        mode = "_".join(parts[:-1])
        batch_size = int(parts[-1])
        latencies = parse_latencies(path)
        finished = "----------inference finished--------------" in path.read_text(
            encoding="utf-8", errors="ignore"
        )
        rows.append(
            {
                "mode": mode,
                "batch_size": batch_size,
                "finished": int(finished),
                "num_profiles": len(latencies),
                "latency_mean_us": f"{mean(latencies):.3f}" if latencies else "",
                "latency_p50_us": f"{sorted(latencies)[len(latencies)//2]:.3f}" if latencies else "",
                "path": str(path),
            }
        )

    output_path = input_dir / "summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "batch_size",
                "finished",
                "num_profiles",
                "latency_mean_us",
                "latency_p50_us",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
