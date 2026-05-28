from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from core.config import AppConfig
from core.task_executor import PromptTask, TaskExecutor
from prompts import tuple_discourse_sort as tuple_discourse_sort_prompt
from utils.io_utils import chunked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "调用 LLM 基于篇章理解对 data/tuple_pocessed 中的五元组进行先后顺序排序，"
            "并将结果写入 data/tuple_pocessed&sorted。"
        )
    )
    parser.add_argument("--root", default=".", help="项目根目录，默认当前目录。")
    parser.add_argument(
        "--timeline-dir",
        default="data/processed_timeline",
        help="篇章文本目录，默认 data/processed_timeline。",
    )
    parser.add_argument(
        "--tuple-dir",
        default="data/tuple_pocessed",
        help="待排序五元组目录，默认 data/tuple_pocessed。",
    )
    parser.add_argument(
        "--output-dir",
        default="data/tuple_pocessed&sorted",
        help="排序后五元组目录，默认 data/tuple_pocessed&sorted。",
    )
    parser.add_argument(
        "--scope",
        default="all",
        help=(
            "文件范围，默认 all。支持按文件名、stem 或 doc_id 精确匹配，多个值用逗号分隔，"
            "例如 M066 或 'M066 解放军核潜艇台湾周边水下活动＜2025.12.30＞'。"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["sync", "batch"],
        help="覆盖 config.yaml 中的 execution_mode。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM 温度参数，默认 0.0。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="LLM 最大输出 token 数，默认 2048。",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="单批调用的文档数量，默认 8。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖 output-dir 中已存在的排序结果；默认跳过已有文件以支持续跑。",
    )
    return parser.parse_args()


def build_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger("kg-tuple-discourse-sort")


def _read_json_array(file_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{file_path} 不是 JSON 数组。")
    return [item for item in payload if isinstance(item, dict)]


def _write_json_array(file_path: Path, rows: list[dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["["]
    for index, row in enumerate(rows):
        suffix = "," if index < len(rows) - 1 else ""
        lines.append(f"  {json.dumps(row, ensure_ascii=False)}{suffix}")
    lines.append("]")
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_scope_tokens(scope: str) -> set[str]:
    tokens = {token.strip().lower() for token in scope.split(",") if token.strip()}
    return tokens or {"all"}


def _matches_scope(file_path: Path, scope_tokens: set[str]) -> bool:
    if "all" in scope_tokens:
        return True
    stem = file_path.stem
    doc_id = stem.split(" ", 1)[0]
    candidates = {file_path.name.lower(), stem.lower(), doc_id.lower()}
    return any(token in candidates for token in scope_tokens)


def _load_sort_jobs(
    tuple_dir: Path,
    timeline_dir: Path,
    output_dir: Path,
    scope: str,
    overwrite: bool = False,
) -> tuple[list[PromptTask], dict[str, dict[str, Any]], int, int]:
    tasks: list[PromptTask] = []
    job_map: dict[str, dict[str, Any]] = {}
    empty_file_count = 0
    skipped_file_count = 0
    scope_tokens = _normalize_scope_tokens(scope)

    for tuple_file in sorted(tuple_dir.glob("*.json")):
        if not _matches_scope(tuple_file, scope_tokens):
            continue

        output_file = output_dir / tuple_file.name
        if output_file.exists() and not overwrite:
            skipped_file_count += 1
            continue

        tuple_rows = _read_json_array(tuple_file)
        document_file = timeline_dir / f"{tuple_file.stem}.txt"
        if not document_file.exists():
            raise FileNotFoundError(f"未找到对应篇章文件: {document_file}")

        custom_id = tuple_file.stem
        job_map[custom_id] = {
            "tuple_file": tuple_file,
            "document_file": document_file,
            "output_file": output_file,
            "tuple_rows": tuple_rows,
        }
        if not tuple_rows:
            empty_file_count += 1
            continue

        tasks.append(
            PromptTask(
                custom_id=custom_id,
                context={
                    "document_file": document_file.name,
                    "document_text": document_file.read_text(encoding="utf-8"),
                    "tuple_file": tuple_file.name,
                    "tuple_rows": tuple_rows,
                },
            )
        )
    return tasks, job_map, empty_file_count, skipped_file_count


def _apply_sorted_tuple_ids(
    tuple_rows: list[dict[str, Any]],
    sorted_tuple_ids: list[str],
    source_name: str,
) -> list[dict[str, Any]]:
    expected_ids = [str(row.get("tuple_id", "")).strip() for row in tuple_rows]
    if any(not tuple_id for tuple_id in expected_ids):
        raise ValueError(f"{source_name} 中存在缺失 tuple_id 的五元组。")

    input_counter = Counter(expected_ids)
    output_counter = Counter(sorted_tuple_ids)
    duplicates = [tuple_id for tuple_id, count in output_counter.items() if count > 1]
    extras = [
        tuple_id for tuple_id in sorted_tuple_ids if tuple_id not in input_counter
    ]
    missing = [tuple_id for tuple_id in expected_ids if tuple_id not in output_counter]

    if duplicates or extras or missing or len(sorted_tuple_ids) != len(expected_ids):
        raise ValueError(
            f"{source_name} 的排序结果非法："
            f"duplicates={duplicates or []}, extras={extras or []}, missing={missing or []}"
        )

    row_map = {row["tuple_id"]: row for row in tuple_rows}
    return [row_map[tuple_id] for tuple_id in sorted_tuple_ids]


def sort_processed_tuple_files(
    project_root: Path,
    timeline_dir: Path,
    tuple_dir: Path,
    output_dir: Path,
    executor: Any,
    mode: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    scope: str = "all",
    chunk_size: int = 8,
    overwrite: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    active_logger = logger or build_logger()

    output_dir.mkdir(parents=True, exist_ok=True)
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0。")

    tasks, job_map, empty_file_count, skipped_file_count = _load_sort_jobs(
        tuple_dir,
        timeline_dir,
        output_dir,
        scope,
        overwrite=overwrite,
    )

    if not tasks and not job_map:
        active_logger.info("未匹配到任何待排序五元组文件。")
        return {
            "tuple_dir": tuple_dir.relative_to(project_root).as_posix(),
            "timeline_dir": timeline_dir.relative_to(project_root).as_posix(),
            "output_dir": output_dir.relative_to(project_root).as_posix(),
            "document_count": 0,
            "tuple_count": 0,
            "empty_document_count": 0,
            "skipped_document_count": skipped_file_count,
            "written_files": {},
        }

    written_files: dict[str, int] = {}
    total_tuple_count = 0
    failures: list[str] = []
    for custom_id, job in job_map.items():
        tuple_rows = job["tuple_rows"]
        output_file = job["output_file"]
        if not tuple_rows:
            _write_json_array(output_file, [])
            written_files[output_file.name] = 0
            continue

    completed_batches = 0
    total_batches = len(list(chunked(tasks, chunk_size))) if tasks else 0
    for task_batch in chunked(tasks, chunk_size):
        results = executor.run_prompt_tasks(
            stage_name="tuple_discourse_sort",
            task_name="document_ordering",
            prompt_module=tuple_discourse_sort_prompt,
            tasks=task_batch,
            temperature=temperature,
            max_tokens=max_tokens,
            mode=mode,
        )

        for task in task_batch:
            job = job_map[task.custom_id]
            try:
                sorted_rows = _apply_sorted_tuple_ids(
                    job["tuple_rows"],
                    results[task.custom_id]["sorted_tuple_ids"],
                    job["tuple_file"].name,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{job['tuple_file'].name}: {exc}")
                continue

            _write_json_array(job["output_file"], sorted_rows)
            written_files[job["output_file"].name] = len(sorted_rows)
            total_tuple_count += len(sorted_rows)

        completed_batches += 1
        active_logger.info(
            "篇章级排序进度：已完成 %s/%s 批，累计写入 %s 个文档。",
            completed_batches,
            total_batches,
            len(written_files),
        )

    relative_output_dir = output_dir.relative_to(project_root).as_posix()
    active_logger.info(
        "已为 %s 个文档、%s 条五元组生成篇章级排序结果，写入 %s；跳过已有文件 %s 个。",
        len(written_files),
        total_tuple_count,
        relative_output_dir,
        skipped_file_count,
    )
    if failures:
        raise RuntimeError("部分文档排序失败：" + "；".join(failures[:10]))

    return {
        "tuple_dir": tuple_dir.relative_to(project_root).as_posix(),
        "timeline_dir": timeline_dir.relative_to(project_root).as_posix(),
        "output_dir": relative_output_dir,
        "document_count": len(written_files),
        "tuple_count": total_tuple_count,
        "empty_document_count": empty_file_count,
        "skipped_document_count": skipped_file_count,
        "written_files": written_files,
    }


def main() -> int:
    args = parse_args()
    logger = build_logger()
    config = AppConfig.load(project_root=args.root)
    config.ensure_runtime_dirs()

    mode = args.mode or config.execution_mode
    if mode not in {"sync", "batch"}:
        raise ValueError(f"不支持的执行模式: {mode}")
    if not config.api_key:
        raise RuntimeError("未配置 SiliconFlow API Key，无法调用篇章级排序模型。")

    executor = TaskExecutor(config, logger)
    project_root = Path(args.root).resolve()
    timeline_dir = (project_root / args.timeline_dir).resolve()
    tuple_dir = (project_root / args.tuple_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    if not timeline_dir.exists():
        raise FileNotFoundError(f"篇章目录不存在: {timeline_dir}")
    if not tuple_dir.exists():
        raise FileNotFoundError(f"五元组目录不存在: {tuple_dir}")

    sort_processed_tuple_files(
        project_root=project_root,
        timeline_dir=timeline_dir,
        tuple_dir=tuple_dir,
        output_dir=output_dir,
        executor=executor,
        mode=mode,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        scope=args.scope,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
        logger=logger,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
