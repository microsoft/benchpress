"""Unified LLM client helpers for BenchPress experiments.

The public release supports two backend families:

- OpenAI-compatible chat APIs configured by environment variables.
- Local vLLM inference for open-weight models.

No provider credentials or private deployment endpoints are stored here.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional


OPENAI_COMPAT_MODELS: dict[str, str] = {}

VLLM_MODELS = {
    "phi-4-reasoning": "microsoft/Phi-4-reasoning",
    "phi-4-mini-reason": "microsoft/Phi-4-mini-reasoning",
    "phi-reasoning-plus": "microsoft/Phi-4-reasoning-plus",
}

ALL_MODELS = {**OPENAI_COMPAT_MODELS, **VLLM_MODELS}


@dataclass
class LLMResponse:
    """Standardized response from any LLM backend."""

    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    finish_reason: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def usage(self):
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


def parse_json_response(text: str) -> dict:
    """Extract JSON from an LLM response, including markdown-fenced JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


class JsonResponseParserMixin:
    def parse_json_response(self, text: str) -> dict:
        return parse_json_response(text)


class OpenAICompatibleClient(JsonResponseParserMixin):
    """Client for OpenAI-compatible chat-completion APIs.

    Configure with:
      - ``OPENAI_API_KEY`` (required)
      - ``OPENAI_BASE_URL`` (optional; defaults to the OpenAI public API)
      - ``OPENAI_API_VERSION`` (optional; passed as a default query parameter)
      - ``BENCHPRESS_OPENAI_REQUEST_TIMEOUT`` (optional seconds)
    """

    def __init__(self, *, base_url: str | None = None, api_version: str | None = None):
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_version = api_version or os.environ.get("OPENAI_API_VERSION")
        self.timeout = float(os.environ.get("BENCHPRESS_OPENAI_REQUEST_TIMEOUT", "90.0"))
        self._client = None

    @staticmethod
    def _ensure_openai():
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ImportError("openai package required: pip install openai") from exc

    def _get_client(self):
        if self._client is not None:
            return self._client
        self._ensure_openai()
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set OPENAI_API_KEY, and optionally "
                "OPENAI_BASE_URL / OPENAI_API_VERSION for an OpenAI-compatible endpoint."
            )
        kwargs = {"api_key": api_key, "max_retries": 0}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.api_version:
            kwargs["default_query"] = {"api-version": self.api_version}
        self._client = OpenAI(**kwargs)
        return self._client

    def _resolve_model(self, model: str) -> str:
        return OPENAI_COMPAT_MODELS.get(model, model)

    def chat(
        self,
        model: str,
        user_message: str,
        system_message: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        retries: int = 2,
        retry_delay: float = 5.0,
    ) -> LLMResponse:
        import time

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        last_error = None
        for attempt in range(retries + 1):
            try:
                return self._call_api(
                    self._resolve_model(model),
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(retry_delay)
        raise last_error

    def _call_api(
        self,
        model: str,
        messages: list,
        *,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        client = self._get_client()
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = client.chat.completions.create(**kwargs, timeout=self.timeout)
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=response.model or model,
            finish_reason=choice.finish_reason or "",
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )


class VLLMClient(JsonResponseParserMixin):
    """Client for local vLLM inference with open-weight models."""

    def __init__(self, model: str | None = None, tensor_parallel_size: int = 1):
        self._loaded_model: Optional[str] = None
        self._engine = None
        self._tp = tensor_parallel_size
        if model:
            self._load_model(model)

    def _resolve_model(self, model: str) -> str:
        return VLLM_MODELS.get(model, model)

    def _load_model(self, model: str):
        hf_id = self._resolve_model(model)
        if self._loaded_model == hf_id and self._engine is not None:
            return
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError("vllm package required: pip install vllm") from exc

        if self._engine is not None:
            del self._engine
            self._engine = None
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._engine = LLM(
            model=hf_id,
            trust_remote_code=True,
            max_model_len=16384,
            gpu_memory_utilization=0.90,
            tensor_parallel_size=self._tp,
        )
        self._loaded_model = hf_id

    def chat(
        self,
        model: str,
        user_message: str,
        system_message: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        retries: int = 0,
        retry_delay: float = 0.0,
    ) -> LLMResponse:
        from vllm import SamplingParams

        self._load_model(model)
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        outputs = self._engine.chat(
            messages=[messages],
            sampling_params=SamplingParams(
                temperature=max(temperature, 0.01),
                max_tokens=max_tokens,
            ),
        )
        output = outputs[0]
        text = output.outputs[0].text
        return LLMResponse(
            content=text,
            input_tokens=len(output.prompt_token_ids) if output.prompt_token_ids else 0,
            output_tokens=len(output.outputs[0].token_ids) if output.outputs[0].token_ids else 0,
            model=self._loaded_model or model,
            finish_reason=output.outputs[0].finish_reason or "",
        )


_client_cache: dict = {}


def get_client(model: str, tensor_parallel_size: int = 1):
    """Return a cached client for ``model``."""
    if model in VLLM_MODELS:
        if model not in _client_cache:
            _client_cache[model] = VLLMClient(model, tensor_parallel_size=tensor_parallel_size)
        return _client_cache[model]
    if "_openai_compatible" not in _client_cache:
        _client_cache["_openai_compatible"] = OpenAICompatibleClient()
    return _client_cache["_openai_compatible"]


def list_models() -> dict:
    """Return known shorthand model aliases."""
    return dict(ALL_MODELS)


def list_openai_compatible_models() -> dict:
    """Return configured OpenAI-compatible shorthand aliases."""
    return dict(OPENAI_COMPAT_MODELS)


def list_vllm_models() -> dict:
    """Return configured vLLM shorthand aliases."""
    return dict(VLLM_MODELS)
