from __future__ import annotations

import json
import time
from json import JSONDecodeError
from types import ModuleType
from typing import Any, cast

from core.config import AppConfig


class LLMClient:
    """SiliconFlow 聊天接口的轻量封装。"""

    _MAX_LENGTH_JSON_RETRIES = 4
    _MAX_LENGTH_RETRY_TOKENS = 16384

    def __init__(self, config: AppConfig):
        self.config = config
        self._next_request_not_before = 0.0

    def _build_extra_body(self) -> dict[str, Any] | None:
        """构造 SiliconFlow 扩展字段。"""

        if self.config.enable_thinking == "omit":
            return None
        return {"enable_thinking": self.config.enable_thinking == "enabled"}

    def _build_client(self):
        """延迟创建 OpenAI 客户端，避免无关模块导入时强依赖第三方包。"""

        if not self.config.api_key:
            raise RuntimeError("未配置 SiliconFlow API Key，无法调用聊天接口。")

        from openai import OpenAI

        return OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.request_timeout_seconds,
        )

    def _content_to_text(self, content: Any) -> str:
        """统一提取 OpenAI 兼容响应中的文本内容。"""

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    def _parse_response_content(self, content: Any) -> dict[str, Any]:
        """解析模型原始文本为 JSON 对象。"""

        text = self._content_to_text(content).strip()
        if not text:
            return {}

        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("LLM 返回内容不是 JSON 对象。")
        return payload

    def _extract_status_code(self, exc: Exception) -> int | None:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        response = getattr(exc, "response", None)
        response_status_code = getattr(response, "status_code", None)
        if isinstance(response_status_code, int):
            return response_status_code
        return None

    def _extract_retry_after_seconds(self, exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return None

        retry_after_ms = headers.get("retry-after-ms")
        if retry_after_ms is not None:
            try:
                delay_ms = float(retry_after_ms)
            except (TypeError, ValueError):
                delay_ms = 0.0
            if delay_ms > 0:
                return delay_ms / 1000.0

        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                delay_seconds = float(retry_after)
            except (TypeError, ValueError):
                delay_seconds = 0.0
            if delay_seconds > 0:
                return delay_seconds
        return None

    def _retry_delay_seconds(
        self, retry_delay_count: int, exc: Exception | None = None
    ) -> float:
        delay = self.config.api_retry_base_delay_seconds * (2**retry_delay_count)
        if exc is None or self._extract_status_code(exc) != 429:
            return delay
        retry_after = self._extract_retry_after_seconds(exc)
        if retry_after is not None:
            return max(delay, retry_after)
        return max(delay, 5.0)

    def _wait_for_request_slot(self) -> None:
        min_interval = max(0.0, self.config.api_min_interval_seconds)
        if min_interval <= 0:
            return
        now = time.monotonic()
        if now < self._next_request_not_before:
            time.sleep(self._next_request_not_before - now)

    def _mark_request_complete(self) -> None:
        min_interval = max(0.0, self.config.api_min_interval_seconds)
        if min_interval <= 0:
            self._next_request_not_before = 0.0
            return
        self._next_request_not_before = time.monotonic() + min_interval

    def call_messages(
        self, messages: list[dict[str, Any]], temperature: float, max_tokens: int
    ) -> dict:
        """直接调用 messages 结构并返回 JSON 结果。"""

        client = self._build_client()
        last_error: Exception | None = None
        request_max_tokens = max_tokens
        transient_error_count = 0
        length_retry_count = 0
        retry_delay_count = 0

        while True:
            response = None
            finish_reason = ""
            request_sent = False
            try:
                request_payload: dict[str, Any] = {
                    "model": self.config.llm_model,
                    "messages": cast(Any, messages),
                    "temperature": temperature,
                    "max_tokens": request_max_tokens,
                    "response_format": {"type": "json_object"},
                }
                extra_body = self._build_extra_body()
                if extra_body is not None:
                    request_payload["extra_body"] = extra_body

                self._wait_for_request_slot()
                request_sent = True
                response = client.chat.completions.create(
                    **cast(Any, request_payload),
                )
                choice = response.choices[0]
                finish_reason = str(getattr(choice, "finish_reason", "") or "")
                return self._parse_response_content(choice.message.content)
            except JSONDecodeError as exc:
                content = (
                    self._content_to_text(
                        getattr(response.choices[0].message, "content", "")
                    )
                    if response is not None
                    else ""
                )
                preview = content[:240].replace("\n", "\\n")
                if finish_reason == "length":
                    next_max_tokens = min(
                        max(
                            request_max_tokens * 2,
                            int(self.config.max_tokens),
                        ),
                        self._MAX_LENGTH_RETRY_TOKENS,
                    )
                    last_error = RuntimeError(
                        "LLM 返回的 JSON 被截断。"
                        f" finish_reason=length, 已请求 max_tokens={request_payload['max_tokens']},"
                        f" 下次重试提升到 {next_max_tokens}。"
                        f" 内容预览: {preview}"
                    )
                    if (
                        next_max_tokens > request_max_tokens
                        and length_retry_count < self._MAX_LENGTH_JSON_RETRIES
                    ):
                        length_retry_count += 1
                        request_max_tokens = next_max_tokens
                        time.sleep(self._retry_delay_seconds(retry_delay_count))
                        retry_delay_count += 1
                        continue

                    last_error = RuntimeError(
                        "LLM 返回的 JSON 被截断，且已达到长度重试上限。"
                        f" finish_reason=length, max_tokens={request_payload['max_tokens']},"
                        f" token_cap={self._MAX_LENGTH_RETRY_TOKENS}."
                        f" 内容预览: {preview}"
                    )
                    break
                last_error = RuntimeError(
                    "LLM 返回了无效 JSON。" f" 解析错误: {exc}. 内容预览: {preview}"
                )
                transient_error_count += 1
                if transient_error_count >= max(1, self.config.api_retry_count):
                    break
                time.sleep(self._retry_delay_seconds(retry_delay_count))
                retry_delay_count += 1
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                transient_error_count += 1
                if transient_error_count >= max(1, self.config.api_retry_count):
                    break
                time.sleep(self._retry_delay_seconds(retry_delay_count, exc=exc))
                retry_delay_count += 1
            finally:
                if request_sent:
                    self._mark_request_complete()

        raise RuntimeError(f"LLM 调用失败: {last_error}") from last_error

    def call(
        self,
        prompt_module: ModuleType,
        context: dict,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """通过提示词模块调用 LLM。"""

        messages = prompt_module.build_prompt(context)
        payload = self.call_messages(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        return prompt_module.parse_response(json.dumps(payload, ensure_ascii=False))
