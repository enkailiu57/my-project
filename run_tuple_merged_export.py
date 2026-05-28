from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from run_tuple_time_normalize import normalize_time_text
from utils.io_utils import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从 data/tuple_extrct 导出保留主体/客体列表结构的新五元组 JSONL，"
            "不再按笛卡尔积拆分为多条记录。"
        )
    )
    parser.add_argument(
        "--root",
        default=".",
        help="项目根目录，默认当前目录。",
    )
    parser.add_argument(
        "--tuple-dir",
        default="data/tuple_extrct",
        help="句级五元组目录，默认 data/tuple_extrct。",
    )
    parser.add_argument(
        "--output-file",
        default="output/s2_raw_tuples_merged.jsonl",
        help="导出 JSONL 路径，默认 output/s2_raw_tuples_merged.jsonl。",
    )
    parser.add_argument(
        "--tuple-field",
        default="规范化五元组",
        help="从句级结果中读取的五元组字段名，默认 规范化五元组。",
    )
    return parser.parse_args()


def build_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger("kg-tuple-merged-export")


def _read_json_array(file_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{file_path} 不是 JSON 数组。")
    return [item for item in payload if isinstance(item, dict)]


def _build_tuple_id(sent_id: str, index: int) -> str:
    return f"{sent_id}_t{index:03d}"


def _normalize_concept_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("concept") or value.get("概念")
    if isinstance(value, list):
        concepts = []
        for item in value:
            normalized_item = _normalize_concept_value(item)
            if isinstance(normalized_item, str) and normalized_item.strip():
                concepts.append(normalized_item.strip())
        return concepts
    if isinstance(value, str):
        return value.strip()
    return value


def _resolve_doc_id(sentence_row: dict[str, Any], sent_id: str) -> str:
    doc_id = sentence_row.get("doc_id")
    if isinstance(doc_id, str) and doc_id.strip():
        return doc_id.strip()
    if "_s" in sent_id:
        return sent_id.split("_s", 1)[0]
    raise ValueError(f"句子记录缺失 doc_id 且无法从句子编号推断: {sentence_row}")


def _normalize_time_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = normalize_time_text(value)
    return normalized if normalized is not None else value


def _freeze_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_value(item)) for key, item in value.items()))
    return value


def _append_unique_values(
    target: list[Any],
    value: Any,
    *,
    include_empty: bool = True,
) -> bool:
    added = False
    values = value if isinstance(value, list) else [value]
    seen = {_freeze_value(item) for item in target}
    for item in values:
        if not include_empty and item is None:
            continue
        if not include_empty and isinstance(item, str) and not item.strip():
            continue
        frozen_item = _freeze_value(item)
        if frozen_item in seen:
            continue
        seen.add(frozen_item)
        target.append(item)
        added = True
    return added


def _merge_location_values(existing_value: Any, new_value: Any) -> tuple[Any, bool]:
    merged_values: list[Any] = []
    _append_unique_values(merged_values, existing_value)
    added = _append_unique_values(merged_values, new_value)
    if len(merged_values) == 1:
        return merged_values[0], added
    return merged_values, added


def _has_mergeable_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    return True


def _rows_share_subject_or_object(
    left_row: dict[str, Any],
    right_row: dict[str, Any],
) -> bool:
    left_subject = left_row.get("subject")
    right_subject = right_row.get("subject")
    if (
        _has_mergeable_value(left_subject)
        and _has_mergeable_value(right_subject)
        and _freeze_value(left_subject) == _freeze_value(right_subject)
    ):
        return True

    left_object = left_row.get("object")
    right_object = right_row.get("object")
    return (
        _has_mergeable_value(left_object)
        and _has_mergeable_value(right_object)
        and _freeze_value(left_object) == _freeze_value(right_object)
    )


def _collapse_values(values: list[Any], *, include_empty: bool) -> Any:
    merged_values: list[Any] = []
    for value in values:
        _append_unique_values(merged_values, value, include_empty=include_empty)
    if not merged_values:
        return values[0] if values else None
    if len(merged_values) == 1:
        return merged_values[0]
    return merged_values


def _merge_sentence_component(component_rows: list[dict[str, Any]]) -> dict[str, Any]:
    first_row = component_rows[0]
    return {
        **first_row,
        "tuple_id": first_row["tuple_id"],
        "sent_id": first_row["tuple_id"],
        "subject": _collapse_values(
            [row.get("subject") for row in component_rows],
            include_empty=False,
        ),
        "object": _collapse_values(
            [row.get("object") for row in component_rows],
            include_empty=False,
        ),
        "location": _collapse_values(
            [row.get("location") for row in component_rows],
            include_empty=True,
        ),
    }


def _merge_sentence_tuple_rows(
    sentence_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    grouped_rows: dict[tuple[Any, Any], list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(sentence_rows):
        group_key = (row.get("relation"), _freeze_value(row.get("time")))
        grouped_rows.setdefault(group_key, []).append((index, row))

    merged_rows: list[tuple[int, dict[str, Any]]] = []
    merged_tuple_count = 0
    for group_items in grouped_rows.values():
        visited: set[int] = set()
        for item_index, (row_index, row) in enumerate(group_items):
            if item_index in visited:
                continue

            stack = [item_index]
            component_indices: list[int] = []
            while stack:
                current_index = stack.pop()
                if current_index in visited:
                    continue
                visited.add(current_index)
                component_indices.append(current_index)
                _, current_row = group_items[current_index]
                for candidate_index, (_, candidate_row) in enumerate(group_items):
                    if candidate_index in visited:
                        continue
                    if _rows_share_subject_or_object(current_row, candidate_row):
                        stack.append(candidate_index)

            component_items = [
                group_items[index] for index in sorted(component_indices)
            ]
            component_rows = [component_row for _, component_row in component_items]
            first_component_index = component_items[0][0]
            if len(component_rows) == 1:
                merged_rows.append((first_component_index, component_rows[0]))
                continue

            merged_rows.append(
                (first_component_index, _merge_sentence_component(component_rows))
            )
            merged_tuple_count += len(component_rows) - 1

    merged_rows.sort(key=lambda item: item[0])
    return [row for _, row in merged_rows], merged_tuple_count


def build_merged_tuple_rows(
    tuple_dir: Path,
    tuple_field: str = "规范化五元组",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    grouped_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    stats = {
        "sentence_count": 0,
        "tuple_count": 0,
        "duplicate_tuple_count": 0,
        "location_merged_tuple_count": 0,
        "sentence_merged_tuple_count": 0,
        "subject_list_tuple_count": 0,
        "object_list_tuple_count": 0,
        "subject_or_object_list_tuple_count": 0,
    }

    if not tuple_dir.exists():
        return rows, stats

    for file_path in sorted(tuple_dir.glob("*.json")):
        sentence_rows = _read_json_array(file_path)
        for sentence_row in sentence_rows:
            sent_id = str(sentence_row.get("句子编号", "")).strip()
            if not sent_id:
                continue
            stats["sentence_count"] += 1
            doc_id = _resolve_doc_id(sentence_row, sent_id)
            tuple_items = sentence_row.get(tuple_field)
            if not isinstance(tuple_items, list):
                continue

            sentence_candidate_rows: list[dict[str, Any]] = []

            for index, item in enumerate(tuple_items, start=1):
                if not isinstance(item, dict):
                    continue

                subject = _normalize_concept_value(item.get("subject"))
                object_ = _normalize_concept_value(item.get("object"))
                location = _normalize_concept_value(item.get("location"))
                relation = _normalize_concept_value(item.get("relation"))
                time_value = _normalize_time_value(item.get("time"))
                tuple_id = _build_tuple_id(sent_id, index)
                sentence_candidate_rows.append(
                    {
                        "tuple_id": tuple_id,
                        "doc_id": doc_id,
                        "sent_id": tuple_id,
                        "subject": subject,
                        "relation": relation,
                        "object": object_,
                        "location": location,
                        "time": time_value,
                        "source_sent_id": sent_id,
                        "confidence": float(item.get("confidence", 0.8)),
                    }
                )

            merged_sentence_rows, merged_tuple_count = _merge_sentence_tuple_rows(
                sentence_candidate_rows
            )
            stats["sentence_merged_tuple_count"] += merged_tuple_count

            for row in merged_sentence_rows:
                signature = (
                    doc_id,
                    _freeze_value(row["subject"]),
                    row["relation"],
                    _freeze_value(row["object"]),
                    _freeze_value(row["time"]),
                )
                if signature in grouped_rows:
                    merged_location, added = _merge_location_values(
                        grouped_rows[signature]["location"],
                        row["location"],
                    )
                    if added:
                        grouped_rows[signature]["location"] = merged_location
                        stats["location_merged_tuple_count"] += 1
                    else:
                        stats["duplicate_tuple_count"] += 1
                    continue

                grouped_rows[signature] = row

                rows.append(row)
                stats["tuple_count"] += 1

                subject_is_list = isinstance(row["subject"], list)
                object_is_list = isinstance(row["object"], list)
                if subject_is_list:
                    stats["subject_list_tuple_count"] += 1
                if object_is_list:
                    stats["object_list_tuple_count"] += 1
                if subject_is_list or object_is_list:
                    stats["subject_or_object_list_tuple_count"] += 1

    return rows, stats


def export_merged_tuples(
    project_root: Path,
    tuple_dir: Path,
    output_file: Path,
    tuple_field: str = "规范化五元组",
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    active_logger = logger or build_logger()
    rows, stats = build_merged_tuple_rows(tuple_dir, tuple_field=tuple_field)
    write_jsonl(output_file, rows, append_done=True)

    relative_output_file = output_file.relative_to(project_root).as_posix()
    active_logger.info(
        (
            "已从 %s 导出 %s 条五元组到 %s；其中主体为列表 %s 条，"
            "客体为列表 %s 条，主体或客体为列表合计 %s 条，"
            "同句主客体合并 %s 条，按地点合并 %s 条，仅完全重复跳过 %s 条。"
        ),
        tuple_dir.relative_to(project_root).as_posix(),
        stats["tuple_count"],
        relative_output_file,
        stats["subject_list_tuple_count"],
        stats["object_list_tuple_count"],
        stats["subject_or_object_list_tuple_count"],
        stats["sentence_merged_tuple_count"],
        stats["location_merged_tuple_count"],
        stats["duplicate_tuple_count"],
    )
    return {
        "tuple_dir": tuple_dir.relative_to(project_root).as_posix(),
        "output_file": relative_output_file,
        **stats,
    }


def main() -> int:
    args = parse_args()
    project_root = Path(args.root).resolve()
    tuple_dir = (project_root / args.tuple_dir).resolve()
    output_file = (project_root / args.output_file).resolve()
    if not tuple_dir.exists():
        raise FileNotFoundError(f"句级五元组目录不存在: {tuple_dir}")

    export_merged_tuples(
        project_root=project_root,
        tuple_dir=tuple_dir,
        output_file=output_file,
        tuple_field=args.tuple_field,
        logger=build_logger(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
