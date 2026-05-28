from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from core.batch_client import BatchClient
from core.config import AppConfig
from core.embed_client import EmbeddingClient
from core.llm_client import LLMClient
from utils.io_utils import read_json, write_json


@dataclass(slots=True)
class PromptTask:
    """单个提示词任务。"""

    custom_id: str
    context: dict[str, Any]


@dataclass(slots=True)
class EmbeddingTask:
    """单个嵌入任务。"""

    custom_id: str
    text: str


class TaskExecutor:
    """统一的 sync / batch 双通道任务执行器。"""

    def __init__(self, config: AppConfig, logger):
        self.config = config
        self.logger = logger
        self.llm_client = LLMClient(config)
        self.embedding_client = EmbeddingClient(config)
        self.batch_client = BatchClient(config)

    def run_prompt_tasks(
        self,
        stage_name: str,
        task_name: str,
        prompt_module: ModuleType,
        tasks: list[PromptTask],
        temperature: float,
        max_tokens: int,
        mode: str,
    ) -> dict[str, dict]:
        """执行一组 prompt 任务并返回 custom_id 到结果的映射。"""

        if not tasks:
            return {}

        if mode == "sync":
            return {
                task.custom_id: self.llm_client.call(
                    prompt_module,
                    task.context,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                for task in tasks
            }

        batch_dir = self._prepare_batch_dir(stage_name, task_name)
        request_file = batch_dir / "requests.jsonl"
        status_file = batch_dir / "status.json"
        output_file = batch_dir / "output.jsonl"
        error_file = batch_dir / "errors.jsonl"

        requests: list[dict[str, Any]] = []
        for task in tasks:
            body: dict[str, Any] = {
                "model": self.config.batch_llm_model,
                "messages": prompt_module.build_prompt(task.context),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            if self.config.enable_thinking != "omit":
                body["enable_thinking"] = self.config.enable_thinking == "enabled"

            requests.append(
                {
                    "custom_id": task.custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )

        self.batch_client.write_requests(request_file, requests)
        status = self._submit_or_resume(
            request_file=request_file,
            status_file=status_file,
            output_file=output_file,
            error_file=error_file,
            endpoint="/v1/chat/completions",
        )
        return self._parse_prompt_batch_output(
            output_file=Path(status["output_path"]),
            prompt_module=prompt_module,
            expected_ids=[task.custom_id for task in tasks],
        )

    def run_embedding_tasks(
        self,
        stage_name: str,
        tasks: list[EmbeddingTask],
        mode: str,
    ) -> dict[str, list[float]]:
        """执行一组 embedding 任务。"""

        if not tasks:
            return {}

        if mode == "sync":
            vectors = self.embedding_client.embed([task.text for task in tasks])
            return {
                task.custom_id: vector
                for task, vector in zip(tasks, vectors, strict=True)
            }

        self.logger.warning(
            "SiliconFlow 官方 Batch 当前仅支持 /v1/chat/completions；"
            "阶段 %s 的 Embedding 任务将自动回退为同步分批调用。",
            stage_name,
        )
        vectors = self.embedding_client.embed([task.text for task in tasks])
        return {
            task.custom_id: vector for task, vector in zip(tasks, vectors, strict=True)
        }

    def _prepare_batch_dir(self, stage_name: str, task_name: str) -> Path:
        path = self.config.batch_dir / stage_name / task_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _submit_or_resume(
        self,
        request_file: Path,
        status_file: Path,
        output_file: Path,
        error_file: Path,
        endpoint: str,
    ) -> dict[str, Any]:
        """提交 batch 任务或从现有状态恢复。"""

        if status_file.exists():
            status = read_json(status_file)
            if status.get("completed") and Path(status.get("output_path", "")).exists():
                return status
            batch_id = status.get("batch_id")
        else:
            status = {
                "endpoint": endpoint,
                "request_path": str(request_file),
                "completed": False,
            }
            batch_id = None

        if not batch_id:
            batch_id = self.batch_client.submit(
                request_file=request_file, endpoint=endpoint
            )
            status["batch_id"] = batch_id
            write_json(status_file, status)

        result = self.batch_client.poll_until_complete(
            batch_id=batch_id,
            poll_interval=self.config.batch_poll_interval,
        )
        downloaded_output = self.batch_client.download_file(
            result["output_file_id"], output_file
        )
        status["output_path"] = str(downloaded_output)

        error_file_id = result.get("error_file_id")
        if error_file_id:
            downloaded_error = self.batch_client.download_file(
                error_file_id, error_file
            )
            status["error_path"] = str(downloaded_error)

        status["completed"] = True
        write_json(status_file, status)
        return status

    def _parse_prompt_batch_output(
        self,
        output_file: Path,
        prompt_module: ModuleType,
        expected_ids: list[str],
    ) -> dict[str, dict]:
        results: dict[str, dict] = {}
        with output_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                content = record["response"]["body"]["choices"][0]["message"]["content"]
                results[record["custom_id"]] = prompt_module.parse_response(content)

        missing = [custom_id for custom_id in expected_ids if custom_id not in results]
        if missing:
            raise RuntimeError(f"Batch 输出缺少结果: {missing}")
        return results

    def _parse_embedding_batch_output(
        self,
        output_file: Path,
        manifest: dict[str, list[str]],
        expected_ids: list[str],
    ) -> dict[str, list[float]]:
        results: dict[str, list[float]] = {}
        with output_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                chunk_id = record["custom_id"]
                member_ids = manifest.get(chunk_id, [])
                embeddings = record["response"]["body"]["data"]
                if len(member_ids) != len(embeddings):
                    raise RuntimeError(
                        f"Embedding batch 输出长度不匹配: {chunk_id}, 期望 {len(member_ids)}，实际 {len(embeddings)}"
                    )
                for member_id, embedding in zip(member_ids, embeddings, strict=True):
                    results[member_id] = embedding["embedding"]

        missing = [custom_id for custom_id in expected_ids if custom_id not in results]
        if missing:
            raise RuntimeError(f"Batch embedding 输出缺少结果: {missing}")
        return results


def resolve_task_executor(runtime) -> TaskExecutor:
    """从运行时对象中解析任务执行器。"""

    if runtime.executor_factory is not None:
        return runtime.executor_factory(runtime.config, runtime.logger)
    return TaskExecutor(runtime.config, runtime.logger)
