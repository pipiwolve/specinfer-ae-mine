#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def run(command: list[str], cwd: str | None = None) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        return output.strip()
    except Exception as exc:
        return f"<error: {exc}>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interesting_env = [
        "AUTODL_TMP_ROOT",
        "HF_HOME",
        "FF_CACHE_DIR",
        "TMPDIR",
        "FF_HOME",
        "UCX_DIR",
        "FF_SERVER_CPUS",
        "FF_SERVER_UTILS",
        "FF_SERVER_GPUS",
        "FF_SERVER_FSIZE_MB",
        "FF_SERVER_ZSIZE_MB",
        "FF_SERVER_MAX_SEQ_LEN",
        "FF_SERVER_MAX_TOKENS",
        "FF_SERVER_BATCH_SIZES",
        "FF_SERVER_RUN_TAG",
        "FF_SSM_MODEL",
        "FF_LLM_MODEL",
        "FF_OFFLOAD_CPUS",
        "FF_OFFLOAD_UTILS",
        "FF_OFFLOAD_GPUS",
        "FF_OFFLOAD_FSIZE_MB",
        "FF_OFFLOAD_ZSIZE_MB",
        "FF_OFFLOAD_MAX_SEQ_LEN",
        "FF_OFFLOAD_BATCH_SIZES",
        "FF_OFFLOAD_RUN_TAG",
        "FF_OFFLOAD_SMALL_LLM",
        "FF_OFFLOAD_LARGE_LLM",
        "FF_OFFLOAD_SSM",
        "FF_OFFLOAD_RUN_SMALL",
        "FF_OFFLOAD_RUN_LARGE",
        "FF_OFFLOAD_SMALL_RESERVE_MB",
        "FF_OFFLOAD_LARGE_RESERVE_MB",
        "FF_OFFLOAD_PROMPT_FILE",
        "FF_OFFLOAD_SEQUENCE_MODE",
    ]

    payload = {
        "experiment_name": args.experiment_name,
        "timestamp": datetime.now().astimezone().isoformat(),
        "repo_root": args.repo_root,
        "cwd": os.getcwd(),
        "env": {key: os.environ.get(key, "") for key in interesting_env if os.environ.get(key)},
        "extra": dict(item.split("=", 1) for item in args.extra if "=" in item),
        "git_branch": run(["git", "branch", "--show-current"], cwd=args.repo_root),
        "git_commit": run(["git", "rev-parse", "HEAD"], cwd=args.repo_root),
        "git_status": run(["git", "status", "-sb"], cwd=args.repo_root),
        "nvidia_smi": run(["nvidia-smi"]),
        "df_h": run(["df", "-h"]),
        "free_h": run(["free", "-h"]) if shutil.which("free") else "",
    }

    json_path = output_dir / "manifest.json"
    txt_path = output_dir / "manifest.txt"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"experiment_name={payload['experiment_name']}",
        f"timestamp={payload['timestamp']}",
        f"repo_root={payload['repo_root']}",
        f"cwd={payload['cwd']}",
        f"git_branch={payload['git_branch']}",
        f"git_commit={payload['git_commit']}",
        "",
        "[env]",
    ]
    lines.extend(f"{k}={v}" for k, v in payload["env"].items())
    if payload["extra"]:
        lines.append("")
        lines.append("[extra]")
        lines.extend(f"{k}={v}" for k, v in payload["extra"].items())
    lines.append("")
    lines.append("[git_status]")
    lines.append(payload["git_status"])
    lines.append("")
    lines.append("[nvidia_smi]")
    lines.append(payload["nvidia_smi"])
    lines.append("")
    lines.append("[df_h]")
    lines.append(payload["df_h"])
    lines.append("")
    lines.append("[free_h]")
    lines.append(payload["free_h"])
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote manifest to {json_path}")


if __name__ == "__main__":
    main()
