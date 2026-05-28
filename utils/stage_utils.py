from __future__ import annotations

from pathlib import Path
from typing import Iterable

from core.types import StageSpec
from utils.io_utils import is_jsonl_done


def resolve_stage_artifact_path(
    output_dir: str | Path,
    artifact: str,
    project_root: str | Path | None = None,
) -> Path:
    normalized = artifact.replace("\\", "/")
    if "/" in normalized:
        base_dir = Path(project_root) if project_root is not None else Path(".")
        return base_dir / Path(normalized)
    return Path(output_dir) / artifact


def describe_stage_artifact(artifact: str) -> str:
    normalized = artifact.replace("\\", "/")
    if "/" in normalized:
        return normalized
    return f"output/{artifact}"


def is_stage_output_complete(
    output_dir: str | Path,
    filename: str,
    project_root: str | Path | None = None,
) -> bool:
    """检查阶段输出文件是否存在且完整。"""

    path = resolve_stage_artifact_path(output_dir, filename, project_root)
    if not path.exists():
        return False

    if path.suffix == ".jsonl":
        return is_jsonl_done(path)

    return path.stat().st_size > 0


def collect_missing_deps(
    output_dir: str | Path,
    deps: Iterable[str],
    project_root: str | Path | None = None,
) -> list[str]:
    """收集当前阶段缺失或不完整的依赖文件。"""

    missing: list[str] = []
    for dep in deps:
        if not is_stage_output_complete(output_dir, dep, project_root):
            missing.append(dep)
    return missing


def should_skip_stage(
    spec: StageSpec,
    output_dir: str | Path,
    force: bool,
    project_root: str | Path | None = None,
) -> bool:
    """根据输出文件完成态判断是否跳过阶段。"""

    if force:
        return False

    return all(
        is_stage_output_complete(output_dir, name, project_root)
        for name in spec.outputs
    )
