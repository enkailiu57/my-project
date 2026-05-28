from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from utils.io_utils import read_jsonl

OUTPUT_FIELDS = (
    "tuple_id",
    "subject",
    "relation",
    "object",
    "location",
    "time",
)

TIME_POINT_RE = re.compile(
    r"^(?P<year>\d{4})(?:-(?P<month>\d{2})(?:-(?P<day>\d{2}))?)?$"
)
SENTENCE_INDEX_RE = re.compile(r"_s(?P<index>\d+)(?:$|_)")
TUPLE_INDEX_RE = re.compile(r"_t(?P<index>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按文档拆分 output/s2_raw_tuples_merged.jsonl，"
            "并将每个文档的元组记录独立保存到 data/tuple_pocessed。"
        )
    )
    parser.add_argument(
        "--root",
        default=".",
        help="项目根目录，默认当前目录。",
    )
    parser.add_argument(
        "--merged-file",
        default="output/s2_raw_tuples_merged.jsonl",
        help="合并版五元组文件，默认 output/s2_raw_tuples_merged.jsonl。",
    )
    parser.add_argument(
        "--manifest-file",
        default="output/s1_extract_plan_manifest.jsonl",
        help="文档命名映射文件，默认 output/s1_extract_plan_manifest.jsonl。",
    )
    parser.add_argument(
        "--output-dir",
        default="data/tuple_pocessed",
        help="按文档拆分后的输出目录，默认 data/tuple_pocessed。",
    )
    return parser.parse_args()


def build_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger("kg-tuple-processed-export")


def _load_doc_file_map(manifest_file: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in read_jsonl(manifest_file):
        doc_id = row.get("doc_id")
        extract_file = row.get("extract_file")
        if not isinstance(doc_id, str) or not doc_id.strip():
            continue
        if not isinstance(extract_file, str) or not extract_file.strip():
            continue
        mapping[doc_id.strip()] = extract_file.strip()
    return mapping


def _resolve_output_file_name(doc_id: str, doc_file_map: dict[str, str]) -> str:
    file_name = doc_file_map.get(doc_id)
    if file_name:
        return file_name
    return f"{doc_id}.json"


def _project_tuple_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field_name: row.get(field_name) for field_name in OUTPUT_FIELDS}


def _extract_index(value: Any, pattern: re.Pattern[str]) -> int | None:
    if not isinstance(value, str):
        return None
    matched = pattern.search(value)
    if matched is None:
        return None
    return int(matched.group("index"))


def _extract_sentence_index(row: dict[str, Any]) -> int | None:
    for field_name in ("source_sent_id", "sent_id", "tuple_id"):
        sentence_index = _extract_index(row.get(field_name), SENTENCE_INDEX_RE)
        if sentence_index is not None:
            return sentence_index
    return None


def _extract_tuple_index(row: dict[str, Any]) -> int | None:
    for field_name in ("tuple_id", "sent_id"):
        tuple_index = _extract_index(row.get(field_name), TUPLE_INDEX_RE)
        if tuple_index is not None:
            return tuple_index
    return None


def _parse_time_anchor(value: Any) -> tuple[int | None, int | None, int | None, int]:
    if not isinstance(value, str):
        return None, None, None, 0
    text = value.strip()
    if not text:
        return None, None, None, 0

    first_fragment = text.split("/", 1)[0].strip()
    matched = TIME_POINT_RE.fullmatch(first_fragment)
    if matched is None:
        return None, None, None, 0

    year = int(matched.group("year"))
    month_text = matched.group("month")
    day_text = matched.group("day")
    month = int(month_text) if month_text is not None else None
    day = int(day_text) if day_text is not None else None
    precision = 1 + int(month is not None) + int(day is not None)
    return year, month, day, precision


def _document_order_key(row: dict[str, Any], row_index: int) -> tuple[int, int, int]:
    sentence_index = _extract_sentence_index(row)
    tuple_index = _extract_tuple_index(row)
    return (
        sentence_index if sentence_index is not None else sys.maxsize,
        tuple_index if tuple_index is not None else sys.maxsize,
        row_index,
    )


def _sort_tuple_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        year, month, day, precision = _parse_time_anchor(row.get("time"))
        prepared_rows.append(
            {
                "row": row,
                "year": year,
                "month": month,
                "day": day,
                "precision": precision,
                "order_key": _document_order_key(row, row_index),
            }
        )

    timed_rows = [item for item in prepared_rows if item["year"] is not None]
    untimed_rows = [item for item in prepared_rows if item["year"] is None]

    sorted_rows: list[dict[str, Any]] = []
    year_buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in timed_rows:
        year_buckets[item["year"]].append(item)

    for year in sorted(year_buckets):
        year_rows = sorted(year_buckets[year], key=lambda item: item["order_key"])
        if any(item["precision"] == 1 for item in year_rows):
            sorted_rows.extend(year_rows)
            continue

        month_buckets: dict[int | None, list[dict[str, Any]]] = defaultdict(list)
        for item in year_rows:
            month_buckets[item["month"]].append(item)

        for month in sorted(
            month_buckets,
            key=lambda value: value if value is not None else sys.maxsize,
        ):
            month_rows = sorted(
                month_buckets[month], key=lambda item: item["order_key"]
            )
            if any(item["precision"] == 2 for item in month_rows):
                sorted_rows.extend(month_rows)
                continue

            sorted_rows.extend(
                sorted(
                    month_rows,
                    key=lambda item: (
                        item["day"] if item["day"] is not None else sys.maxsize,
                        item["order_key"],
                    ),
                )
            )

    sorted_rows.extend(sorted(untimed_rows, key=lambda item: item["order_key"]))
    return [item["row"] for item in sorted_rows]


def _write_json_array(file_path: Path, rows: list[dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["["]
    for index, row in enumerate(rows):
        suffix = "," if index < len(rows) - 1 else ""
        lines.append(f"  {json.dumps(row, ensure_ascii=False)}{suffix}")
    lines.append("]")
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_processed_tuple_files(
    project_root: Path,
    merged_file: Path,
    manifest_file: Path,
    output_dir: Path,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    active_logger = logger or build_logger()
    doc_file_map = _load_doc_file_map(manifest_file)
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in read_jsonl(merged_file):
        doc_id = row.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(f"{merged_file} 中存在缺失 doc_id 的记录: {row}")
        grouped_rows[doc_id.strip()].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    written_files: dict[str, int] = {}
    for doc_id in sorted(grouped_rows):
        file_name = _resolve_output_file_name(doc_id, doc_file_map)
        file_path = output_dir / file_name
        sorted_rows = _sort_tuple_rows(grouped_rows[doc_id])
        projected_rows = [_project_tuple_row(row) for row in sorted_rows]
        _write_json_array(file_path, projected_rows)
        written_files[file_name] = len(grouped_rows[doc_id])

    relative_output_dir = output_dir.relative_to(project_root).as_posix()
    active_logger.info(
        "已将 %s 个文档、%s 条五元组按文档拆分写入 %s。",
        len(grouped_rows),
        sum(len(rows) for rows in grouped_rows.values()),
        relative_output_dir,
    )
    return {
        "merged_file": merged_file.relative_to(project_root).as_posix(),
        "manifest_file": manifest_file.relative_to(project_root).as_posix(),
        "output_dir": relative_output_dir,
        "document_count": len(grouped_rows),
        "tuple_count": sum(len(rows) for rows in grouped_rows.values()),
        "written_files": written_files,
    }


def main() -> int:
    args = parse_args()
    project_root = Path(args.root).resolve()
    merged_file = (project_root / args.merged_file).resolve()
    manifest_file = (project_root / args.manifest_file).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    if not merged_file.exists():
        raise FileNotFoundError(f"合并版五元组文件不存在: {merged_file}")
    if not manifest_file.exists():
        raise FileNotFoundError(f"文档映射文件不存在: {manifest_file}")

    export_processed_tuple_files(
        project_root=project_root,
        merged_file=merged_file,
        manifest_file=manifest_file,
        output_dir=output_dir,
        logger=build_logger(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
