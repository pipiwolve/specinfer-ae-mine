#!/usr/bin/env python3
import os
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from autodl_single.common import FlexFlowRunner, build_prompt


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0)
    stream: bool = False
    context: str = ""


def create_app() -> FastAPI:
    config_file = os.getenv("CONFIG_FILE")
    if not config_file:
        config_file = str(
            Path(__file__).resolve().parent / "configs" / "api_specinfer_single_a100.json"
        )

    runner = FlexFlowRunner(config_file)
    lock = threading.Lock()
    app = FastAPI(title="SpecInfer OpenAI Compatibility Layer")

    cors_origins = runner.config_dict.get("server", {}).get("cors_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event() -> None:
        runner.start()

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        runner.stop()

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "mode": runner.mode, "model": runner.config.llm_model}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> dict:
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported by this server.")
        prompt = build_prompt([message.model_dump() for message in request.messages], context=request.context)

        def _run() -> dict:
            with lock:
                return runner.generate_text(prompt, max_length=request.max_tokens)

        result = await run_in_threadpool(_run)
        created = int(time.time())
        model_name = request.model or runner.config.llm_model
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": result["completion_text"],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
            },
        }

    return app


app = create_app()
