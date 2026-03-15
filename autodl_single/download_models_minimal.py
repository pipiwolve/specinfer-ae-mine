#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

import flexflow.serve as ff


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-folder", default=os.environ.get("FF_CACHE_DIR", ""))
    parser.add_argument("--llm-model", default="huggyllama/llama-7b")
    parser.add_argument("--ssm-model", default="JackFram/llama-68m")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("models", nargs="*")
    return parser.parse_args()


def tokenizer_dir(cache_folder: str, model_name: str) -> Path:
    return Path(cache_folder).expanduser() / "tokenizers" / model_name.lower()


def hf_snapshot_dir(model_name: str) -> Path:
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    repo_dir = hf_home / "hub" / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"HuggingFace snapshot directory missing: {snapshots_dir}")
    snapshots = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), key=lambda p: p.name)
    if not snapshots:
        raise FileNotFoundError(f"No HuggingFace snapshots found for {model_name} in {snapshots_dir}")
    return snapshots[-1]


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def ensure_tokenizer_files(cache_folder: str, model_name: str, filenames: list[str]) -> None:
    dst_dir = tokenizer_dir(cache_folder, model_name)
    dst_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = hf_snapshot_dir(model_name)
    copied_any = False
    missing = []
    for filename in filenames:
        src = snapshot_dir / filename
        dst = dst_dir / filename
        if dst.exists():
            copied_any = True
            continue
        if copy_if_exists(src, dst):
            copied_any = True
            print(f"Copied tokenizer file from {src} to {dst}")
        else:
            missing.append(filename)
    if not copied_any:
        raise FileNotFoundError(
            f"No tokenizer files copied for {model_name}; missing candidates: {missing}"
        )


def ensure_sentencepiece_tokenizer(cache_folder: str, model_name: str) -> None:
    dst_dir = tokenizer_dir(cache_folder, model_name)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_file = dst_dir / "tokenizer.model"
    if dst_file.exists():
        return
    src_file = hf_snapshot_dir(model_name) / "tokenizer.model"
    if not src_file.exists():
        raise FileNotFoundError(f"Tokenizer model missing in HF snapshot: {src_file}")
    shutil.copy2(src_file, dst_file)
    print(f"Copied sentencepiece tokenizer from {src_file} to {dst_file}")


def ensure_opt_tokenizer(cache_folder: str, model_name: str) -> None:
    ensure_tokenizer_files(
        cache_folder,
        model_name,
        [
            "vocab.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    )


def ensure_tokenizer_fallback(cache_folder: str, source_model: str, target_model: str) -> None:
    src = tokenizer_dir(cache_folder, source_model)
    dst = tokenizer_dir(cache_folder, target_model)
    if not src.exists():
        raise FileNotFoundError(f"Tokenizer fallback source missing: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"Copied tokenizer fallback from {src} to {dst}")


def download_model(model_name: str, cache_folder: str, refresh_cache: bool, fallback_model: str = "") -> None:
    llm = ff.LLM(
        model_name,
        data_type=ff.DataType.DT_HALF,
        cache_path=cache_folder,
        refresh_cache=refresh_cache,
    )
    llm.download_hf_weights_if_needed()
    try:
        llm.download_hf_tokenizer_if_needed()
    except Exception as exc:
        if not fallback_model:
            raise
        print(f"Tokenizer download failed for {model_name}: {exc}")
        ensure_tokenizer_fallback(cache_folder, fallback_model, model_name)
    if "llama" in model_name.lower():
        ensure_sentencepiece_tokenizer(cache_folder, model_name)
    if "opt-" in model_name.lower():
        ensure_opt_tokenizer(cache_folder, model_name)
    llm.download_hf_config()


def main():
    args = parse_args()
    cache_folder = args.cache_folder or os.environ.get("FF_CACHE_DIR", "~/.cache/flexflow")
    llm_model = args.llm_model
    ssm_model = args.ssm_model

    if args.models:
        if len(args.models) != 2:
            raise SystemExit("Expected exactly two positional models: <llm_model> <ssm_model>")
        llm_model, ssm_model = args.models

    download_model(llm_model, cache_folder, args.refresh_cache)
    download_model(
        ssm_model,
        cache_folder,
        args.refresh_cache,
        fallback_model=llm_model,
    )


if __name__ == "__main__":
    main()
