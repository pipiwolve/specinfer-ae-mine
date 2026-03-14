#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import flexflow.serve as ff
from transformers import AutoTokenizer


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _apply_cache_env_overrides(config: Dict) -> Dict:
    ff_cache_dir = os.environ.get("FF_CACHE_DIR")
    if not ff_cache_dir:
        return config

    config["cache_path"] = ff_cache_dir
    for ssm_config in config.get("ssms", []):
        ssm_config["cache_path"] = ff_cache_dir
    return config


def default_generation_config() -> ff.GenerationConfig:
    return ff.GenerationConfig(do_sample=False, temperature=0.9, topp=0.8, topk=1)


def _runtime_limit(config: Dict, key: str, default: int) -> int:
    server_config = config.get("server", {})
    if key in server_config:
        return server_config[key]
    return config.get(key, default)


def build_prompt(messages: List[Dict[str, str]], context: str = "") -> str:
    lines: List[str] = []
    if context:
        lines.append("Context:")
        lines.append(context.strip())
        lines.append("")
    for message in messages:
        role = message.get("role", "user").strip().upper()
        content = message.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines).strip()


class FlexFlowRunner:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_dict = _apply_cache_env_overrides(load_json(config_path))
        self.config = SimpleNamespace(**self.config_dict)
        self.server_config = SimpleNamespace(**self.config_dict.get("server", {}))
        self.mode = self.config_dict.get("server", {}).get(
            "mode",
            "specinfer" if self.config_dict.get("ssms") else "incr",
        )
        self.llm = None
        self.ssms = []
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model, trust_remote_code=True)

    def start(self) -> None:
        ff.init(self.config_dict)
        llm_dtype = ff.DataType.DT_FLOAT if self.config.full_precision else ff.DataType.DT_HALF
        self.llm = ff.LLM(
            self.config.llm_model,
            data_type=llm_dtype,
            cache_path=self.config.cache_path,
            refresh_cache=self.config.refresh_cache,
            output_file=self.config.output_file,
        )
        for ssm_config in self.config_dict.get("ssms", []):
            ssm_ns = SimpleNamespace(**ssm_config)
            ssm_dtype = ff.DataType.DT_FLOAT if ssm_ns.full_precision else ff.DataType.DT_HALF
            ssm = ff.SSM(
                ssm_ns.ssm_model,
                data_type=ssm_dtype,
                cache_path=ssm_ns.cache_path,
                refresh_cache=ssm_ns.refresh_cache,
                output_file=self.config.output_file,
            )
            ssm.compile(
                default_generation_config(),
                max_requests_per_batch=_runtime_limit(self.config_dict, "max_requests_per_batch", 1),
                max_seq_length=_runtime_limit(self.config_dict, "max_seq_length", 64),
                max_tokens_per_batch=_runtime_limit(self.config_dict, "max_tokens_per_batch", 64),
            )
            self.ssms.append(ssm)

        self.llm.compile(
            default_generation_config(),
            max_requests_per_batch=_runtime_limit(self.config_dict, "max_requests_per_batch", 1),
            max_seq_length=_runtime_limit(self.config_dict, "max_seq_length", 64),
            max_tokens_per_batch=_runtime_limit(self.config_dict, "max_tokens_per_batch", 64),
            ssms=self.ssms,
        )
        self.llm.start_server()

    def stop(self) -> None:
        if self.llm is not None:
            self.llm.stop_server()

    def generate_text(self, prompt: str, max_length: Optional[int] = None) -> Dict:
        if self.llm is None:
            raise RuntimeError("FlexFlowRunner.start() must be called before generate_text().")
        generation_max_length = max_length or _runtime_limit(self.config_dict, "default_max_length", 64)
        started_at = time.perf_counter()
        results = self.llm.generate([prompt], generation_max_length)
        latency_seconds = time.perf_counter() - started_at
        output_text = results[0].output_text.decode("utf-8")
        completion_text = output_text[len(prompt):].strip() if output_text.startswith(prompt) else output_text.strip()
        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = len(self.tokenizer.encode(completion_text)) if completion_text else 0
        return {
            "prompt": prompt,
            "raw_output_text": output_text,
            "completion_text": completion_text,
            "latency_seconds": latency_seconds,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }


def ensure_output_dir(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
