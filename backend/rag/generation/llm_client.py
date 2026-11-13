"""Lớp trừu tượng LLM: đổi nhà cung cấp chỉ bằng config, không sửa pipeline.

- anthropic: Claude qua SDK chính thức (ANTHROPIC_API_KEY).
- openai:    mọi API tương thích OpenAI (OpenAI, Gemini, Ollama, ...) qua
             LLM_BASE_URL + LLM_API_KEY.
API key chỉ đọc từ biến môi trường / file .env, không bao giờ nằm trong code.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any


class LLMError(RuntimeError):
    pass


class BaseLLM(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, system: str, user: str) -> str: ...


class AnthropicLLM(BaseLLM):
    name = "anthropic"
    # Các model hỗ trợ server-side fallback khi bị bộ lọc an toàn từ chối
    _FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1"}

    def __init__(self, model: str = "claude-opus-5-5", max_tokens: int = 4000, effort: str | None = "medium") -> None:
        import anthropic

        self.client = anthropic.Anthropic()   # tự đọc ANTHROPIC_API_KEY
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    def generate(self, system: str, user: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}

        if self.model in self._FALLBACK_MODELS:
            response = self.client.beta.messages.create(
                **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        else:
            response = self.client.messages.create(**kwargs)

        if response.stop_reason == "refusal":
            raise LLMError("Mô hình từ chối trả lời yêu cầu này.")
        return "".join(block.text for block in response.content if block.type == "text").strip()


class OpenAICompatibleLLM(BaseLLM):
    name = "openai"

    def __init__(self, model: str, max_tokens: int = 4000, temperature: float = 0.0,
                 base_url: str | None = None) -> None:
        from openai import OpenAI

        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "not-needed"  # Ollama không cần key
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (response.choices[0].message.content or "").strip()


def get_llm(cfg: dict[str, Any]) -> BaseLLM:
    llm = cfg.get("llm", {})
    provider = llm.get("provider", "anthropic")
    if provider == "anthropic":
        return AnthropicLLM(llm.get("model", "claude-opus-5-5"), llm.get("max_tokens", 4000), llm.get("effort"))
    if provider == "openai":
        return OpenAICompatibleLLM(llm["model"], llm.get("max_tokens", 4000), llm.get("temperature", 0.0),
                                   llm.get("base_url"))
    raise ValueError(f"LLM provider không hợp lệ: {provider}")
