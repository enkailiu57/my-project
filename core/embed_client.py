from __future__ import annotations

import time

from core.config import AppConfig


class EmbeddingClient:
    """SiliconFlow Embedding 接口封装，负责分批调用。"""

    def __init__(self, config: AppConfig):
        self.config = config

    def _build_client(self):
        if not self.config.api_key:
            raise RuntimeError("未配置 SiliconFlow API Key，无法调用 Embedding 接口。")

        from openai import OpenAI

        return OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.request_timeout_seconds,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把文本列表编码为向量列表。"""

        if not texts:
            return []

        client = self._build_client()
        batch_size = min(self.config.embed_batch_size, 64)
        vectors: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            last_error: Exception | None = None
            for attempt in range(self.config.api_retry_count):
                try:
                    response = client.embeddings.create(
                        model=self.config.embed_model,
                        input=chunk,
                        dimensions=self.config.embed_dimensions,
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if attempt == self.config.api_retry_count - 1:
                        raise RuntimeError(
                            f"Embedding 调用失败: {last_error}"
                        ) from last_error
                    time.sleep(self.config.api_retry_base_delay_seconds * (2**attempt))
            vectors.extend(item.embedding for item in response.data)

        return vectors
