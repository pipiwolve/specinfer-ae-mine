import argparse
import csv
import re
from pathlib import Path


PROFILE_RE = re.compile(r"latency\(([\d.]+)\)")
VERIFIED_RE = re.compile(r"Number of Verified Tokens = (\d+)")


def parse_out(path: Path) -> tuple[list[float], list[int]]:
    latencies = []
    verified = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "[Profile]" in line:
                match = PROFILE_RE.search(line)
                if match:
                    latencies.append(float(match.group(1)))
            if "Number of Verified Tokens =" in line:
                match = VERIFIED_RE.search(line)
                if match:
                    verified.append(int(match.group(1)))
    return latencies, verified


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="FlexFlow/inference/output/autodl_single/server_gpu_single_a100",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows = []

    for path in sorted(input_dir.glob("server_small-*_batchsize-*.out")):
        name = path.name
        batch_size = int(name.split("_batchsize-")[0].split("server_small-")[1])
        if "sequence_specinfer" in name:
            mode = "sequence"
        elif "tree_specinfer" in name:
            mode = "tree"
        elif "incr_dec" in name:
            mode = "incr"
        else:
            continue

        latencies, verified = parse_out(path)
        rows.append(
            {
                "batch_size": batch_size,
                "mode": mode,
                "num_profiles": len(latencies),
                "latency_mean_us": f"{mean(latencies):.3f}",
                "latency_p50_us": f"{sorted(latencies)[len(latencies)//2]:.3f}" if latencies else "",
                "verified_mean": f"{mean(verified):.3f}" if verified else "",
                "verified_count": len(verified),
                "path": str(path),
            }
        )

    output_path = input_dir / "summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch_size",
                "mode",
                "num_profiles",
                "latency_mean_us",
                "latency_p50_us",
                "verified_mean",
                "verified_count",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
