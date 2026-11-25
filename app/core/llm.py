"""LLM 抽象层：统一 OpenAI 兼容接口、重试、Token 统计与 Mock 离线实现。"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from app.config import ConfigError, LLMSettings

logger = logging.getLogger("insight.llm")


class LLMError(RuntimeError):
    """LLM 调用失败。"""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0
    finish_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    others = max(0, len(text) - cjk)
    return max(1, int(cjk + others / 4))


class BaseLLM(ABC):
    provider = "base"
    model = "base"

    def __init__(self, settings: LLMSettings):
        self.settings = settings

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stage: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"


class OpenAICompatibleLLM(BaseLLM):
    provider = "openai-compatible"

    def __init__(self, settings: LLMSettings):
        super().__init__(settings)
        self.model = settings.model
        key = os.getenv(settings.api_key_env or "", "").strip()
        if not key:
            raise ConfigError(
                f"未找到 {settings.api_key_env}。请在 .env 中配置，或切换到 provider=mock。"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 依赖缺失
            raise LLMError("未安装 openai，请执行 pip install -r requirements.txt") from exc

        self.client = OpenAI(
            api_key=key,
            base_url=settings.base_url or None,
            timeout=settings.timeout,
            max_retries=0,
        )

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status in {408, 409, 429, 500, 502, 503, 504}:
            return True
        message = str(exc).lower()
        return any(word in message for word in ("timeout", "timed out", "connection", "rate limit"))

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stage: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        attempts = self.settings.max_retries + 1
        last_error: Optional[Exception] = None
        use_json_mode = json_mode or self.settings.json_mode

        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.settings.temperature
                    if temperature is None
                    else temperature,
                    "max_tokens": max_tokens or self.settings.max_tokens,
                }
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                if self.settings.extra_body:
                    kwargs["extra_body"] = self.settings.extra_body
                response = self.client.chat.completions.create(**kwargs)
                latency_ms = (time.perf_counter() - started) * 1000
                choice = response.choices[0]
                text = choice.message.content or ""
                finish_reason = getattr(choice, "finish_reason", "") or ""
                if finish_reason == "length":
                    logger.warning(
                        "LLM 输出被 max_tokens=%d 截断（stage=%s），结果可能不完整",
                        max_tokens or self.settings.max_tokens,
                        stage,
                    )
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or estimate_tokens(
                    "\n".join(message["content"] for message in messages)
                )
                completion_tokens = getattr(usage, "completion_tokens", 0) or estimate_tokens(text)
                logger.debug(
                    "LLM 调用成功 stage=%s attempt=%d latency=%.0fms tokens=%s/%s",
                    stage,
                    attempt + 1,
                    latency_ms,
                    prompt_tokens,
                    completion_tokens,
                )
                return LLMResponse(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    model=self.model,
                    latency_ms=latency_ms,
                    finish_reason=finish_reason,
                )
            except Exception as exc:  # noqa: BLE001 - 统一转换为带上下文的 LLMError
                last_error = exc
                message = str(exc)
                if use_json_mode and "response_format" in message.lower():
                    use_json_mode = False
                    logger.warning("模型不支持 response_format=json_object，已自动降级")
                    continue
                if attempt < attempts - 1 and self._retryable(exc):
                    wait = min(2**attempt, 8)
                    logger.warning(
                        "LLM 调用失败（stage=%s），%.1fs 后重试：%s", stage, wait, message[:200]
                    )
                    time.sleep(wait)
                    continue
                break

        raise LLMError(f"LLM 调用失败（stage={stage}）：{last_error}") from last_error


def create_llm(settings: LLMSettings) -> BaseLLM:
    if settings.provider == "mock":
        from app.core.mock_llm import MockLLM

        return MockLLM(settings)
    return OpenAICompatibleLLM(settings)
