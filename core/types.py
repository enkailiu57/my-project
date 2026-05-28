from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from core.config import AppConfig

StageName = Literal["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]
ExecutionMode = Literal["sync", "batch"]
ElementType = Literal["entity", "relation", "location"]
ExecutorFactory = Callable[[AppConfig, logging.Logger], Any]


@dataclass(slots=True)
class DocumentRecord:
    """统一文档对象。

    由于原始数据格式尚未确定，所有输入适配器最终都要产出这一统一结构。
    """

    doc_id: str
    text: str
    meta: dict[str, Any] = field(default_factory=lambda: {})


@dataclass(slots=True)
class StageRuntime:
    """阶段运行时上下文。

    所有阶段函数都通过该对象获取配置、路径、日志器和执行模式，
    减少参数四处透传的复杂度。
    """

    config: AppConfig
    stage_name: StageName
    mode: ExecutionMode
    logger: logging.Logger
    project_root: Path
    data_dir: Path
    output_dir: Path
    news_scope: str = "all"
    file_scope: str = "all"
    force: bool = False
    dry_run: bool = False
    executor_factory: ExecutorFactory | None = None


@dataclass(slots=True)
class StageSpec:
    """阶段注册信息。"""

    name: StageName
    title: str
    deps: tuple[str, ...]
    outputs: tuple[str, ...]
    runner: Callable[[StageRuntime], None]


@dataclass(slots=True)
class UnresolvedAliasRecord:
    """S7 未命中 alias_map 时的诊断记录。"""

    raw_id: str
    element_type: ElementType
    raw_str: str
