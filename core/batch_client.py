from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any, cast

from core.config import AppConfig
from utils.io_utils import ensure_parent_dir


class BatchClient:
    """SiliconFlow Batch API 的基础封装。

    SiliconFlow 官方 Batch 当前只支持 /v1/chat/completions。
    """

    def __init__(self, config: AppConfig):
        self.config = config

    def _build_client(self):
        if not self.config.api_key:
            raise RuntimeError("未配置 SiliconFlow API Key，无法调用 Batch 接口。")

        from openai import OpenAI

        return OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.request_timeout_seconds,
        )

    def write_requests(self, file_path: str | Path, requests: list[dict]) -> Path:
        """把批请求列表写成 JSONL 文件。"""

        path = ensure_parent_dir(file_path)
        with path.open("w", encoding="utf-8") as handle:
            for request in requests:
                handle.write(json.dumps(request, ensure_ascii=False) + "\n")
        return path

    def _extract_uploaded_file_id(self, file_obj: Any) -> str:
        """兼容 SiliconFlow 文件上传响应中嵌套在 data 里的文件 ID。"""

        direct_id = getattr(file_obj, "id", None)
        if direct_id:
            return str(direct_id)

        nested_data = getattr(file_obj, "data", None)
        if isinstance(nested_data, dict) and nested_data.get("id"):
            return str(nested_data["id"])

        if hasattr(file_obj, "model_dump"):
            payload = file_obj.model_dump()
            nested = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(nested, dict) and nested.get("id"):
                return str(nested["id"])

        raise RuntimeError(f"无法从文件上传响应中解析 file_id: {file_obj}")

    def submit(self, request_file: str | Path, endpoint: str) -> str:
        """上传请求文件并创建 Batch 任务。"""

        client = self._build_client()
        with Path(request_file).open("rb") as handle:
            file_obj = client.files.create(file=handle, purpose="batch")
        input_file_id = self._extract_uploaded_file_id(file_obj)
        batch = client.batches.create(
            input_file_id=input_file_id,
            endpoint=cast(Any, endpoint),
            completion_window=cast(Any, self.config.batch_completion_window),
        )
        return str(batch.id)

    def poll_until_complete(self, batch_id: str, poll_interval: int = 30) -> dict:
        """轮询批任务直到完成。"""

        client = self._build_client()
        while True:
            batch = client.batches.retrieve(batch_id)
            if batch.status == "completed":
                return {
                    "status": batch.status,
                    "output_file_id": batch.output_file_id,
                    "error_file_id": getattr(batch, "error_file_id", None),
                }
            if batch.status in {"failed", "expired", "cancelled"}:
                raise RuntimeError(f"Batch 任务失败: {batch.status}")
            time.sleep(poll_interval)

    def download_file(self, file_id: str, target_path: str | Path) -> Path:
        """下载 Batch 输出文件到本地。"""

        target = ensure_parent_dir(target_path)
        if file_id.startswith("http://") or file_id.startswith("https://"):
            with urllib.request.urlopen(
                file_id, timeout=self.config.request_timeout_seconds
            ) as response:
                target.write_bytes(response.read())
            return target

        client = self._build_client()
        client.files.content(file_id).write_to_file(str(target))
        return target

    def list_batches(self, limit: int = 20, after: str | None = None):
        """获取当前账号的 Batch 任务列表。"""

        client = self._build_client()
        params: dict[str, Any] = {"limit": limit}
        if after:
            params["after"] = after
        return client.batches.list(**params)

    def cancel(self, batch_id: str):
        """取消指定 Batch 任务。"""

        client = self._build_client()
        return client.batches.cancel(batch_id)
