from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VERIFIED_RE = re.compile(r"Number of Verified Tokens = (\d+)")
LAT_RE = re.compile(r"latency\(([\d.]+)\)")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def parse_verified_tokens(paths: list[Path]) -> list[int]:
    values: list[int] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        values.extend(int(m.group(1)) for m in VERIFIED_RE.finditer(text))
    return values


def parse_mean_latency_us(path: Path) -> float | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    vals = [float(m.group(1)) for m in LAT_RE.finditer(text)]
    return mean(vals) if vals else None


def plot_resident_latency(final_dir: Path, out_dir: Path) -> None:
    rows = read_csv(final_dir / "server_gpu_single_a100" / "summary.csv")
    wanted_batches = [1, 2, 4]
    modes = ["incr", "sequence", "tree"]
    labels = {"incr": "Incremental", "sequence": "Sequence-Spec", "tree": "Tree-Spec"}
    data = {mode: [] for mode in modes}
    for bs in wanted_batches:
        batch_rows = {r["mode"]: r for r in rows if int(r["batch_size"]) == bs}
        for mode in modes:
            value = float(batch_rows[mode]["latency_mean_us"]) / 1_000_000 if mode in batch_rows else math.nan
            data[mode].append(value)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = range(len(wanted_batches))
    width = 0.24
    offsets = {"incr": -width, "sequence": 0.0, "tree": width}
    colors = {"incr": "#2C6E49", "sequence": "#D17B0F", "tree": "#A63D40"}
    for mode in modes:
        ax.bar([i + offsets[mode] for i in x], data[mode], width=width, label=labels[mode], color=colors[mode])
    ax.set_xticks(list(x), [str(b) for b in wanted_batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Mean latency (s)")
    ax.set_title("Single-GPU resident inference latency")
    ax.legend(frameon=False)
    savefig(out_dir / "fig_4_1_resident_latency.png")


def plot_english_stability(final_dir: Path, out_dir: Path) -> None:
    rows = read_csv(final_dir / "english_len64" / "stability_summary_en" / "failure_summary.csv")
    modes = ["incr", "spec"]
    flags = ["ok", "truncated", "garbled", "off_topic", "runtime_error"]
    colors = {
        "ok": "#2C6E49",
        "truncated": "#D17B0F",
        "garbled": "#A63D40",
        "off_topic": "#6C757D",
        "runtime_error": "#3B5B92",
    }
    ratio_map = {(r["mode"], r["quality_flag"]): float(r["ratio"]) for r in rows}
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bottoms = [0.0, 0.0]
    x = [0, 1]
    for flag in flags:
        vals = [ratio_map.get((mode, flag), 0.0) for mode in modes]
        ax.bar(x, vals, bottom=bottoms, label=flag, color=colors[flag], width=0.55)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(x, ["Incremental", "Speculative"])
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1.02)
    ax.set_title("English medical QA stability distribution")
    ax.legend(frameon=False, ncol=2)
    savefig(out_dir / "fig_6_1_english_stability.png")


def plot_offloading_speedup(final_dir: Path, out_dir: Path) -> None:
    rows = read_csv(final_dir / "offloading" / "compare.csv")
    batch_sizes = [int(r["batch_size"]) for r in rows]
    speedups = [float(r["speedup_mean"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.bar([str(b) for b in batch_sizes], speedups, color="#2B59C3", width=0.55)
    for i, v in enumerate(speedups):
        ax.text(i, v + 0.03, f"{v:.2f}x", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Mean speedup over baseline")
    ax.set_title("Offloading baseline vs speculative speedup")
    ax.set_ylim(0, max(speedups) + 0.35)
    savefig(out_dir / "fig_5_1_offloading_speedup.png")


def plot_offloading_latency_compare(final_dir: Path, out_dir: Path) -> None:
    rows = read_csv(final_dir / "offloading" / "compare.csv")
    batch_sizes = [int(r["batch_size"]) for r in rows]
    base_vals = [float(r["baseline_mean_us"]) / 1_000_000 for r in rows]
    spec_vals = [float(r["spec_mean_us"]) / 1_000_000 for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = range(len(batch_sizes))
    width = 0.33
    ax.bar([i - width / 2 for i in x], base_vals, width=width, label="Baseline offload", color="#8D99AE")
    ax.bar([i + width / 2 for i in x], spec_vals, width=width, label="SpecInfer offload", color="#2C6E49")
    ax.set_xticks(list(x), [str(b) for b in batch_sizes])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Mean latency (s)")
    ax.set_title("Offloading latency comparison")
    ax.legend(frameon=False)
    savefig(out_dir / "fig_5_2_offloading_latency.png")


def plot_offloading_verified(final_dir: Path, out_dir: Path) -> None:
    spec_dir = final_dir / "offloading" / "spec_opt13b_r1"
    vals = parse_verified_tokens(sorted(spec_dir.glob("offloading_small_*.out")))
    counter = Counter(vals)
    xs = sorted(counter)
    ys = [counter[x] for x in xs]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.bar(xs, ys, color="#A63D40", width=0.8)
    ax.set_xlabel("Verified tokens per step")
    ax.set_ylabel("Frequency")
    ax.set_title("Offloading verified-token distribution")
    savefig(out_dir / "fig_5_3_offloading_verified_distribution.png")

    summary_path = out_dir / "fig_5_3_offloading_verified_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["count", len(vals)])
        writer.writerow(["mean_verified_tokens", f"{mean(vals):.6f}" if vals else ""])
        writer.writerow(["median_verified_tokens", sorted(vals)[len(vals) // 2] if vals else ""])
        writer.writerow(["max_verified_tokens", max(vals) if vals else ""])
        writer.writerow(["gt1_ratio", f"{sum(v > 1 for v in vals) / len(vals):.6f}" if vals else ""])


def plot_english_acceptance(final_dir: Path, out_dir: Path) -> None:
    rows = read_csv(final_dir / "english_len64" / "acceptance_analysis_en" / "acceptance_metrics.csv")
    vals = [
        float(r["acceptance_ratio"])
        for r in rows
        if r.get("acceptance_ratio") not in (None, "", "nan", "NaN")
    ]
    vals = sorted(vals)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(vals, bins=10, color="#3B5B92", edgecolor="white")
    ax.set_xlabel("Agreement ratio")
    ax.set_ylabel("Prompt count")
    ax.set_title("English medical QA agreement-ratio distribution")
    savefig(out_dir / "fig_6_2_english_acceptance_proxy.png")


def plot_legacy_opt13b(legacy_dir: Path, out_dir: Path) -> None:
    base = legacy_dir / "opt-13b 全量实验结果"
    if not base.exists():
        return
    modes = ["incr", "seq", "tree"]
    labels = {"incr": "Incremental", "seq": "Sequence-Spec", "tree": "Tree-Spec"}
    batch_sizes = [1, 2, 4, 8, 16]
    data = defaultdict(list)
    for bs in batch_sizes:
        for mode in modes:
            path = base / f"offloading_opt13b_bs{bs}_{mode}.out"
            val = parse_mean_latency_us(path) if path.exists() else None
            data[mode].append(val / 1_000_000 if val is not None else math.nan)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = range(len(batch_sizes))
    width = 0.24
    offsets = {"incr": -width, "seq": 0.0, "tree": width}
    colors = {"incr": "#8D99AE", "seq": "#2C6E49", "tree": "#A63D40"}
    for mode in modes:
        ax.bar([i + offsets[mode] for i in x], data[mode], width=width, label=labels[mode], color=colors[mode])
    ax.set_xticks(list(x), [str(b) for b in batch_sizes])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Mean latency (s)")
    ax.set_title("Legacy OPT-13B offloading results (appendix)")
    ax.legend(frameon=False)
    savefig(out_dir / "fig_app_1_legacy_opt13b_latency.png")


def plot_legacy_tree_width(legacy_dir: Path, out_dir: Path) -> None:
    configs = [
        ("tree", "bs=1"),
        ("tree_bs2", "bs=2"),
    ]
    labels = []
    verified_means = []
    lat_means = []
    for folder, tag in configs:
        base = legacy_dir / folder
        if not base.exists():
            continue
        for width in [1, 2]:
            path = base / f"offloading_opt13b_tree_w{width}.out"
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            verified = [int(m.group(1)) for m in VERIFIED_RE.finditer(text)]
            lats = [float(m.group(1)) for m in LAT_RE.finditer(text)]
            labels.append(f"{tag}, w={width}")
            verified_means.append(mean(verified) if verified else math.nan)
            lat_means.append(mean(lats) / 1_000_000 if lats else math.nan)

    if not labels:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(labels, verified_means, color="#2C6E49")
    axes[0].set_ylabel("Mean verified tokens")
    axes[0].set_title("Legacy tree-width ablation")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(labels, lat_means, color="#D17B0F")
    axes[1].set_ylabel("Mean latency (s)")
    axes[1].set_title("Legacy tree-width latency")
    axes[1].tick_params(axis="x", rotation=20)
    savefig(out_dir / "fig_app_2_legacy_tree_width.png")


def write_figure_notes(out_dir: Path) -> None:
    text = """# 论文图表说明

## 正文建议使用

- fig_4_1_resident_latency.png：第4章，单卡常驻 `incr/sequence/tree` 对比
- fig_5_1_offloading_speedup.png：第5章，offloading 主结果图
- fig_5_2_offloading_latency.png：第5章，baseline vs spec latency
- fig_5_3_offloading_verified_distribution.png：第5章，verified token 分布
- fig_6_1_english_stability.png：第6章，英文医疗问答稳定性
- fig_6_2_english_acceptance_proxy.png：第6章，英文医疗问答输出一致性代理

## 附录候选

- fig_app_1_legacy_opt13b_latency.png：旧版 offloading 全量结果
- fig_app_2_legacy_tree_width.png：旧版 tree width 消融
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", default="final_review_20260316_035130")
    parser.add_argument("--legacy-dir", default="毕设实验结果")
    parser.add_argument("--output-dir", default="论文图表")
    args = parser.parse_args()

    final_dir = Path(args.final_dir)
    legacy_dir = Path(args.legacy_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    plot_resident_latency(final_dir, out_dir)
    plot_english_stability(final_dir, out_dir)
    plot_offloading_speedup(final_dir, out_dir)
    plot_offloading_latency_compare(final_dir, out_dir)
    plot_offloading_verified(final_dir, out_dir)
    plot_english_acceptance(final_dir, out_dir)
    plot_legacy_opt13b(legacy_dir, out_dir)
    plot_legacy_tree_width(legacy_dir, out_dir)
    write_figure_notes(out_dir)
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
