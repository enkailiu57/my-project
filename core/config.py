from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

SILICONFLOW_DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_DEFAULT_CHAT_MODEL = "Qwen/Qwen3-32B"
SILICONFLOW_DEFAULT_BATCH_CHAT_MODEL = "deepseek-ai/DeepSeek-V3"
SILICONFLOW_DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"
SILICONFLOW_DEFAULT_ENABLE_THINKING = "disabled"


def _parse_bool(value: object | None, default: bool) -> bool:
    """把配置值解析为布尔值。"""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_thinking_mode(value: object | None, default: str) -> str:
    """把 enable_thinking 配置解析为 enabled/disabled/omit 三态。"""

    if value is None:
        return default
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    normalized = str(value).strip().lower()
    if normalized in {"enabled", "disabled", "omit"}:
        return normalized
    raise ValueError("配置项 enable_thinking 只支持 enabled、disabled、omit 或布尔值。")


def _load_yaml(path: Path) -> dict:
    """读取 YAML 配置文件。"""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件必须是对象结构: {path}")
    return payload


@dataclass(slots=True)
class AppConfig:
    """全局配置对象。

    当前工程统一使用 SiliconFlow 作为聊天、Embedding 和 Batch 的 API 提供方。
    为兼容已存在的本机环境变量，如果 config.yaml 中 api_key 留空，
    会临时回退读取环境变量 SILICONFLOW_API_KEY。
    """

    api_key: str
    base_url: str
    llm_model: str
    enable_thinking: str
    batch_llm_model: str
    embed_model: str
    embed_dimensions: int
    request_timeout_seconds: int
    api_retry_count: int
    api_retry_base_delay_seconds: float
    api_min_interval_seconds: float
    temperature_extract: float
    temperature_describe: float
    max_tokens: int
    embed_batch_size: int
    tuple_concept_top_k: int
    dedup_cluster_size: int
    dedup_top_k: int
    dedup_bm25_weight: float
    dedup_max_iterations: int
    min_confidence: float
    context_token_budget: int
    context_window_size: int
    batch_poll_interval: int
    batch_completion_window: str
    random_seed: int
    execution_mode: str
    strict_dependency_check: bool
    strict_canonical_resolution: bool
    data_loader: str
    project_root: Path
    data_dir: Path
    output_dir: Path
    prompts_dir: Path
    batch_dir: Path
    reports_dir: Path
    debug_dir: Path
    cache_db_path: Path

    @classmethod
    def load(
        cls, project_root: str | Path, config_name: str = "config.yaml"
    ) -> "AppConfig":
        """从项目根目录加载统一配置文件。"""

        root = Path(project_root).resolve()
        config_path = root / config_name
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        raw = _load_yaml(config_path)
        api_key = str(
            raw.get("api_key") or os.getenv("SILICONFLOW_API_KEY") or ""
        ).strip()
        base_url = str(raw.get("base_url") or SILICONFLOW_DEFAULT_BASE_URL).rstrip("/")
        llm_model = str(raw.get("llm_model") or SILICONFLOW_DEFAULT_CHAT_MODEL)
        batch_llm_model = str(
            raw.get("batch_llm_model") or SILICONFLOW_DEFAULT_BATCH_CHAT_MODEL
        )
        embed_model = str(raw.get("embed_model") or SILICONFLOW_DEFAULT_EMBED_MODEL)

        output_dir = root / "output"
        batch_dir = output_dir / "batch"
        reports_dir = output_dir / "reports"
        debug_dir = output_dir / "debug"

        return cls(
            api_key=api_key,
            base_url=base_url,
            llm_model=llm_model,
            enable_thinking=_parse_thinking_mode(
                raw.get("enable_thinking"),
                SILICONFLOW_DEFAULT_ENABLE_THINKING,
            ),
            batch_llm_model=batch_llm_model,
            embed_model=embed_model,
            embed_dimensions=int(raw.get("embed_dimensions", 512)),
            request_timeout_seconds=int(raw.get("request_timeout_seconds", 300)),
            api_retry_count=max(1, int(raw.get("api_retry_count", 3))),
            api_retry_base_delay_seconds=float(
                raw.get("api_retry_base_delay_seconds", 1.0)
            ),
            api_min_interval_seconds=max(
                0.0,
                float(raw.get("api_min_interval_seconds", 0.0)),
            ),
            temperature_extract=float(raw.get("temperature_extract", 0.1)),
            temperature_describe=float(raw.get("temperature_describe", 0.3)),
            max_tokens=int(raw.get("max_tokens", 1024)),
            embed_batch_size=int(raw.get("embed_batch_size", 64)),
            tuple_concept_top_k=int(raw.get("tuple_concept_top_k", 16)),
            dedup_cluster_size=int(raw.get("dedup_cluster_size", 128)),
            dedup_top_k=int(raw.get("dedup_top_k", 16)),
            dedup_bm25_weight=float(raw.get("dedup_bm25_weight", 0.5)),
            dedup_max_iterations=int(raw.get("dedup_max_iterations", 10)),
            min_confidence=float(raw.get("min_confidence", 0.6)),
            context_token_budget=int(raw.get("context_token_budget", 6000)),
            context_window_size=int(raw.get("context_window_size", 5)),
            batch_poll_interval=int(raw.get("batch_poll_interval", 30)),
            batch_completion_window=str(raw.get("batch_completion_window", "24h")),
            random_seed=int(raw.get("random_seed", 42)),
            execution_mode=str(raw.get("execution_mode", "sync")),
            strict_dependency_check=_parse_bool(
                raw.get("strict_dependency_check"),
                True,
            ),
            strict_canonical_resolution=_parse_bool(
                raw.get("strict_canonical_resolution"),
                False,
            ),
            data_loader=str(raw.get("data_loader", "news_txt")),
            project_root=root,
            data_dir=root / "data",
            output_dir=output_dir,
            prompts_dir=root / "prompts",
            batch_dir=batch_dir,
            reports_dir=reports_dir,
            debug_dir=debug_dir,
            cache_db_path=root / "cache.db",
        )

    def ensure_runtime_dirs(self) -> None:
        """确保运行所需目录存在。"""

        for path in (
            self.data_dir,
            self.output_dir,
            self.batch_dir,
            self.reports_dir,
            self.debug_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def validate_execution_mode(self) -> None:
        """校验 execution_mode 是否合法。"""

        if self.execution_mode not in {"sync", "batch"}:
            raise ValueError(f"不支持的 execution_mode: {self.execution_mode}")

    def validate_for_stage_names(
        self, stage_names: Sequence[str], dry_run: bool = False
    ) -> None:
        """根据待执行阶段校验必要配置。"""

        self.validate_execution_mode()
        self.ensure_runtime_dirs()
        if dry_run:
            return

        api_stages = {"s0", "s1", "s2", "s3", "s4", "s5"}
        if any(stage in api_stages for stage in stage_names) and not self.api_key:
            raise RuntimeError(
                "当前执行阶段依赖 SiliconFlow API，但未在 config.yaml 的 api_key 字段中配置密钥。"
                "如果你还没把密钥迁入配置文件，也可以临时设置环境变量 SILICONFLOW_API_KEY 兼容运行。"
            )
