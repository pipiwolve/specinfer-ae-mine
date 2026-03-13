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
    return parser.parse_args()


def tokenizer_dir(cache_folder: str, model_name: str) -> Path:
    return Path(cache_folder).expanduser() / "tokenizers" / model_name.lower()


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
    llm.download_hf_config()


def main():
    args = parse_args()
    cache_folder = args.cache_folder or os.environ.get("FF_CACHE_DIR", "~/.cache/flexflow")

    download_model(args.llm_model, cache_folder, args.refresh_cache)
    download_model(
        args.ssm_model,
        cache_folder,
        args.refresh_cache,
        fallback_model=args.llm_model,
    )


if __name__ == "__main__":
    main()
