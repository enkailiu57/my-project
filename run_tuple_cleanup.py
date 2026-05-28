from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from run_tuple_time_normalize import normalize_time_text
from utils.io_utils import (
    DONE_SENTINEL,
    is_jsonl_done,
    read_jsonl,
    write_json,
    write_jsonl,
)

TRACE_ID_FIELDS = ("tuple_id", "raw_id", "sent_id")
DEFAULT_TRACE_DIRS = ("output", "data/tuple_extrct")
DEFAULT_EXCLUDED_TRACE_FILES = ("output/tuple_time_normalize_report.json",)
_REMOVE = object()
_CANONICAL_YEAR_DAY_PATTERN = re.compile(
    r"^(?P<year>\d{4})(?:年|[./-])(?P<month>\d{1,2})(?:月|[./-])(?P<day>\d{1,2})日?$"
)
_CANONICAL_YEAR_MONTH_PATTERN = re.compile(
    r"^(?P<year>\d{4})(?:年|[./-])(?P<month>\d{1,2})月?$"
)
_CANONICAL_YEAR_PATTERN = re.compile(r"^(?P<year>\d{4})年?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="清理 S2 非法五元组及其基于 tuple_id 的关联记录。"
    )
    parser.add_argument(
        "--root",
        default=".",
        help="项目根目录，默认当前目录。",
    )
    parser.add_argument(
        "--s2-file",
        default="output/s2_raw_tuples.jsonl",
        help="S2 扁平五元组文件，默认 output/s2_raw_tuples.jsonl。",
    )
    parser.add_argument(
        "--trace-dir",
        dest="trace_dirs",
        action="append",
        default=None,
        help=(
            "需要按 tuple_id 追溯清理的目录，可重复传入；"
            "默认清理 output 和 data/tuple_extrct。"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计将删除哪些记录，不实际写回文件。",
    )
    parser.add_argument(
        "--report-file",
        default="output/tuple_cleanup_report.json",
        help="删除报告输出路径，默认 output/tuple_cleanup_report.json。",
    )
    return parser.parse_args()


def build_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger("kg-tuple-cleanup")


def _normalize_time_for_signature(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return text

    if matched := _CANONICAL_YEAR_DAY_PATTERN.fullmatch(text):
        return (
            f"{int(matched.group('year')):04d}."
            f"{int(matched.group('month')):02d}."
            f"{int(matched.group('day')):02d}"
        )

    if matched := _CANONICAL_YEAR_MONTH_PATTERN.fullmatch(text):
        return f"{int(matched.group('year')):04d}.{int(matched.group('month')):02d}"

    if matched := _CANONICAL_YEAR_PATTERN.fullmatch(text):
        return f"{int(matched.group('year')):04d}"

    return text


def _tuple_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("doc_id"),
        row.get("subject"),
        row.get("relation"),
        row.get("object"),
        row.get("location"),
        _normalize_time_for_signature(row.get("time")),
    )


def _is_same_subject_object(row: dict[str, Any]) -> bool:
    subject = row.get("subject")
    object_ = row.get("object")
    return isinstance(subject, str) and isinstance(object_, str) and subject == object_


def _has_unreliable_time_anchor(row: dict[str, Any]) -> bool:
    time_value = row.get("time")
    if not isinstance(time_value, str) or not time_value.strip():
        return False
    return normalize_time_text(time_value) is None


def collect_tuple_delete_reasons(s2_file: Path) -> dict[str, list[str]]:
    tuple_delete_reasons: dict[str, list[str]] = {}
    seen_signatures: dict[tuple[Any, ...], str] = {}
    for row in read_jsonl(s2_file):
        tuple_id = row.get("tuple_id")
        if not isinstance(tuple_id, str) or not tuple_id:
            raise ValueError(f"{s2_file} 中存在缺失 tuple_id 的记录: {row}")

        reasons: list[str] = []
        signature = _tuple_signature(row)
        if signature in seen_signatures:
            reasons.append("duplicate_in_doc")
        else:
            seen_signatures[signature] = tuple_id

        if _is_same_subject_object(row):
            reasons.append("subject_equals_object")

        if _has_unreliable_time_anchor(row):
            reasons.append("unreliable_time_anchor")

        if reasons:
            tuple_delete_reasons[tuple_id] = reasons
    return tuple_delete_reasons


def _matched_traced_tuple_ids(payload: dict[str, Any], tuple_ids: set[str]) -> set[str]:
    matched_ids: set[str] = set()
    for field_name in TRACE_ID_FIELDS:
        value = payload.get(field_name)
        if isinstance(value, str) and value in tuple_ids:
            matched_ids.add(value)
        if isinstance(value, list) and any(
            isinstance(item, str) and item in tuple_ids for item in value
        ):
            matched_ids.update(
                item for item in value if isinstance(item, str) and item in tuple_ids
            )
    return matched_ids


def _prune_payload(payload: Any, tuple_ids: set[str]) -> tuple[Any, int, set[str]]:
    if isinstance(payload, dict):
        matched_ids = _matched_traced_tuple_ids(payload, tuple_ids)
        if matched_ids:
            return _REMOVE, 1, matched_ids

        removed_count = 0
        removed_ids: set[str] = set()
        pruned: dict[str, Any] = {}
        for key, value in payload.items():
            next_value, child_removed, child_ids = _prune_payload(value, tuple_ids)
            removed_count += child_removed
            removed_ids.update(child_ids)
            if next_value is _REMOVE:
                continue
            pruned[key] = next_value
        return pruned, removed_count, removed_ids

    if isinstance(payload, list):
        removed_count = 0
        removed_ids: set[str] = set()
        pruned_list: list[Any] = []
        for item in payload:
            next_value, child_removed, child_ids = _prune_payload(item, tuple_ids)
            removed_count += child_removed
            removed_ids.update(child_ids)
            if next_value is _REMOVE:
                continue
            pruned_list.append(next_value)
        return pruned_list, removed_count, removed_ids

    return payload, 0, set()


def _iter_trace_files(
    project_root: Path,
    trace_dirs: list[str],
    excluded_files: set[Path] | None = None,
) -> list[Path]:
    files: set[Path] = set()
    ignored = {path.resolve() for path in (excluded_files or set())}
    for trace_dir in trace_dirs:
        base_dir = project_root / trace_dir
        if not base_dir.exists():
            continue
        for pattern in ("**/*.json", "**/*.jsonl"):
            files.update(
                path
                for path in base_dir.glob(pattern)
                if path.is_file() and path.resolve() not in ignored
            )
    return sorted(files)


def _cleanup_jsonl_file(
    file_path: Path,
    tuple_ids: set[str],
    dry_run: bool,
) -> tuple[int, set[str]]:
    had_done = is_jsonl_done(file_path)
    kept_rows: list[dict[str, Any]] = []
    removed_count = 0
    removed_ids: set[str] = set()

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if payload == DONE_SENTINEL:
                continue
            next_payload, child_removed, child_ids = _prune_payload(payload, tuple_ids)
            removed_count += child_removed
            removed_ids.update(child_ids)
            if next_payload is _REMOVE:
                continue
            if not isinstance(next_payload, dict):
                raise ValueError(f"JSONL 记录清理后不是对象: {file_path}")
            kept_rows.append(next_payload)

    if removed_count and not dry_run:
        write_jsonl(file_path, kept_rows, append_done=had_done)
    return removed_count, removed_ids


def _cleanup_json_file(
    file_path: Path,
    tuple_ids: set[str],
    dry_run: bool,
) -> tuple[int, set[str]]:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    next_payload, removed_count, removed_ids = _prune_payload(payload, tuple_ids)
    if removed_count and not dry_run:
        if next_payload is _REMOVE:
            next_payload = []
        file_path.write_text(
            json.dumps(next_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return removed_count, removed_ids


def _build_report_rows(
    deleted_tuple_ids: list[str],
    delete_reasons: dict[str, list[str]],
    hit_files_by_tuple_id: dict[str, list[str]],
) -> list[dict[str, Any]]:
    return [
        {
            "tuple_id": tuple_id,
            "delete_reasons": delete_reasons.get(tuple_id, []),
            "hit_files": hit_files_by_tuple_id.get(tuple_id, []),
        }
        for tuple_id in deleted_tuple_ids
    ]


def cleanup_duplicate_tuples(
    project_root: Path,
    s2_file: Path,
    trace_dirs: list[str] | None = None,
    report_file: Path | None = None,
    dry_run: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    active_logger = logger or build_logger()
    active_trace_dirs = trace_dirs or list(DEFAULT_TRACE_DIRS)
    tuple_delete_reasons = collect_tuple_delete_reasons(s2_file)
    deleted_tuple_ids = list(tuple_delete_reasons.keys())
    duplicate_tuple_ids = [
        tuple_id
        for tuple_id, reasons in tuple_delete_reasons.items()
        if "duplicate_in_doc" in reasons
    ]
    self_loop_tuple_ids = [
        tuple_id
        for tuple_id, reasons in tuple_delete_reasons.items()
        if "subject_equals_object" in reasons
    ]
    unreliable_time_tuple_ids = [
        tuple_id
        for tuple_id, reasons in tuple_delete_reasons.items()
        if "unreliable_time_anchor" in reasons
    ]
    tuple_id_set = set(deleted_tuple_ids)
    excluded_files = {
        (project_root / relative_path).resolve()
        for relative_path in DEFAULT_EXCLUDED_TRACE_FILES
        if (project_root / relative_path).exists()
    }
    if report_file is not None:
        excluded_files.add(report_file.resolve())

    active_logger.info(
        "在 %s 中识别到 %s 条需删除的非法五元组。",
        s2_file,
        len(deleted_tuple_ids),
    )

    trace_files = _iter_trace_files(
        project_root,
        active_trace_dirs,
        excluded_files=excluded_files,
    )
    removed_records: dict[str, int] = {}
    hit_files_by_tuple_id: dict[str, list[str]] = {
        tuple_id: [] for tuple_id in deleted_tuple_ids
    }
    for file_path in trace_files:
        if file_path.suffix == ".jsonl":
            removed_count, hit_ids = _cleanup_jsonl_file(
                file_path, tuple_id_set, dry_run
            )
        else:
            removed_count, hit_ids = _cleanup_json_file(
                file_path, tuple_id_set, dry_run
            )

        if removed_count:
            relative_path = file_path.relative_to(project_root).as_posix()
            removed_records[relative_path] = removed_count
            for tuple_id in deleted_tuple_ids:
                if tuple_id in hit_ids:
                    hit_files_by_tuple_id[tuple_id].append(relative_path)
            active_logger.info(
                "%s %s 删除 %s 条关联记录。",
                "将" if dry_run else "已",
                relative_path,
                removed_count,
            )

    report_rows = _build_report_rows(
        deleted_tuple_ids,
        tuple_delete_reasons,
        hit_files_by_tuple_id,
    )
    report_path_str = None
    if report_file is not None:
        resolved_report_file = report_file.resolve()
        report_path_str = resolved_report_file.relative_to(project_root).as_posix()
        report_payload = {
            "s2_file": s2_file.relative_to(project_root).as_posix(),
            "deleted_tuple_count": len(deleted_tuple_ids),
            "duplicate_tuple_count": len(duplicate_tuple_ids),
            "self_loop_tuple_count": len(self_loop_tuple_ids),
            "unreliable_time_tuple_count": len(unreliable_time_tuple_ids),
            "deleted_tuples": report_rows,
            "removed_records": removed_records,
            "dry_run": dry_run,
        }
        if not dry_run:
            write_json(resolved_report_file, report_payload)
            active_logger.info("删除报告已写入 %s", report_path_str)

    active_logger.info(
        "%s扫描 %s 个文件，命中 %s 个文件。",
        "dry-run：" if dry_run else "",
        len(trace_files),
        len(removed_records),
    )
    return {
        "deleted_tuple_ids": deleted_tuple_ids,
        "duplicate_tuple_ids": duplicate_tuple_ids,
        "self_loop_tuple_ids": self_loop_tuple_ids,
        "unreliable_time_tuple_ids": unreliable_time_tuple_ids,
        "deleted_tuples": report_rows,
        "scanned_files": len(trace_files),
        "changed_files": len(removed_records),
        "removed_records": removed_records,
        "report_file": report_path_str,
    }


def main() -> int:
    args = parse_args()
    project_root = Path(args.root).resolve()
    s2_file = (project_root / args.s2_file).resolve()
    report_file = (project_root / args.report_file).resolve()
    if not s2_file.exists():
        raise FileNotFoundError(f"S2 五元组文件不存在: {s2_file}")

    cleanup_duplicate_tuples(
        project_root=project_root,
        s2_file=s2_file,
        trace_dirs=args.trace_dirs,
        report_file=report_file,
        dry_run=args.dry_run,
        logger=build_logger(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
