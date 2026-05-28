from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from utils.io_utils import (
    DONE_SENTINEL,
    is_jsonl_done,
    read_jsonl,
    write_json,
    write_jsonl,
)

TRACE_ID_FIELDS = ("tuple_id", "raw_id", "sent_id")
DEFAULT_TRACE_DIRS = ("output", "data/tuple_extrct")

LIST_SPLIT_RE = re.compile(r"(?:、|，|,|及|与|和)")
YEAR_MONTH_DAY_CN_RE = re.compile(
    r"^(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日$"
)
YEAR_MONTH_DAY_GENERIC_RE = re.compile(
    r"^(?P<year>\d{4})[./-](?P<month>\d{1,2})[./-](?P<day>\d{1,2})$"
)
YEAR_MONTH_CN_RE = re.compile(r"^(?P<year>\d{4})年(?P<month>\d{1,2})月$")
YEAR_MONTH_GENERIC_RE = re.compile(r"^(?P<year>\d{4})[./-](?P<month>\d{1,2})$")
YEAR_ONLY_RE = re.compile(r"^(?P<year>\d{4})年?$")
MONTH_DAY_RE = re.compile(r"^(?P<month>\d{1,2})月(?P<day>\d{1,2})日$")
MONTH_ONLY_RE = re.compile(r"^(?P<month>\d{1,2})月$")
DAY_ONLY_RE = re.compile(r"^(?P<day>\d{1,2})日$")
YEAR_RANGE_HYPHEN_RE = re.compile(r"^(?P<start>\d{4})-(?P<end>\d{4})年?$")
YEAR_MONTH_RANGE_HYPHEN_RE = re.compile(
    r"^(?P<year>\d{4})年(?P<start>\d{1,2})(?:月)?-(?P<end>\d{1,2})月$"
)
DECADE_RE = re.compile(r"^(?P<year>\d{4})年代(?:(?P<position>初|中|末))?$")
YEAR_BOUNDARY_RE = re.compile(r"^(?P<year>\d{4})年?(?:年)?初$")
YEAR_END_RE = re.compile(r"^(?P<year>\d{4})年?(?:年)?(?:末|底)$")
YEAR_MID_RE = re.compile(r"^(?P<year>\d{4})年中$")
YEAR_HALF_RE = re.compile(r"^(?P<year>\d{4})年(?P<half>上|下)半年$")
YEAR_QUARTER_RE = re.compile(r"^(?P<year>\d{4})年第(?P<quarter>[一二三四])季度$")
YEAR_SEASON_RE = re.compile(r"^(?P<year>\d{4})年(?P<season>春|夏|秋|冬)(?:季)?$")
OPEN_ENDED_SUFFIXES = (
    "以来",
    "至今",
    "之后",
    "以后",
    "之前",
    "年起",
    "起",
    "后",
    "前",
)


class TimeContext(dict):
    @property
    def year(self) -> int | None:
        value = self.get("year")
        return value if isinstance(value, int) else None

    @property
    def month(self) -> int | None:
        value = self.get("month")
        return value if isinstance(value, int) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="规范化 S2 扁平五元组时间字段，并同步更新可追溯记录。"
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
            "需要按 tuple_id / raw_id / sent_id 同步更新时间的目录，可重复传入；"
            "默认清理 output 和 data/tuple_extrct。"
        ),
    )
    parser.add_argument(
        "--report-file",
        default="output/tuple_time_normalize_report.json",
        help="时间规范化报告输出路径，默认 output/tuple_time_normalize_report.json。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计将更新时间，不实际写回文件。",
    )
    return parser.parse_args()


def build_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger("kg-tuple-time-normalize")


def _normalize_point(
    year: int, month: int | None = None, day: int | None = None
) -> str:
    if day is not None and month is not None:
        return f"{year:04d}-{month:02d}-{day:02d}"
    if month is not None:
        return f"{year:04d}-{month:02d}"
    return f"{year:04d}"


def _strip_open_ended_suffixes(text: str) -> str:
    cleaned = text
    changed = True
    while changed:
        changed = False
        for suffix in OPEN_ENDED_SUFFIXES:
            if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
                cleaned = cleaned[: -len(suffix)].strip()
                changed = True
                break
    return cleaned


def _context(year: int | None = None, month: int | None = None) -> TimeContext:
    payload = TimeContext()
    if year is not None:
        payload["year"] = year
    if month is not None:
        payload["month"] = month
    return payload


def _normalize_single_fragment(
    text: str,
    context: TimeContext | None = None,
) -> tuple[str | None, TimeContext | None]:
    context = context or TimeContext()

    if matched := YEAR_MONTH_DAY_CN_RE.fullmatch(text):
        year = int(matched.group("year"))
        month = int(matched.group("month"))
        day = int(matched.group("day"))
        return _normalize_point(year, month, day), _context(year, month)

    if matched := YEAR_MONTH_DAY_GENERIC_RE.fullmatch(text):
        year = int(matched.group("year"))
        month = int(matched.group("month"))
        day = int(matched.group("day"))
        return _normalize_point(year, month, day), _context(year, month)

    if matched := YEAR_MONTH_CN_RE.fullmatch(text):
        year = int(matched.group("year"))
        month = int(matched.group("month"))
        return _normalize_point(year, month), _context(year, month)

    if matched := YEAR_MONTH_GENERIC_RE.fullmatch(text):
        year = int(matched.group("year"))
        month = int(matched.group("month"))
        return _normalize_point(year, month), _context(year, month)

    if matched := YEAR_ONLY_RE.fullmatch(text):
        year = int(matched.group("year"))
        return _normalize_point(year), _context(year)

    if matched := MONTH_DAY_RE.fullmatch(text):
        if context.year is None:
            return None, None
        month = int(matched.group("month"))
        day = int(matched.group("day"))
        return _normalize_point(context.year, month, day), _context(context.year, month)

    if matched := MONTH_ONLY_RE.fullmatch(text):
        if context.year is None:
            return None, None
        month = int(matched.group("month"))
        return _normalize_point(context.year, month), _context(context.year, month)

    if matched := DAY_ONLY_RE.fullmatch(text):
        if context.year is None or context.month is None:
            return None, None
        day = int(matched.group("day"))
        return _normalize_point(context.year, context.month, day), _context(
            context.year,
            context.month,
        )

    if matched := YEAR_BOUNDARY_RE.fullmatch(text):
        year = int(matched.group("year"))
        return _normalize_point(year, 1), _context(year, 1)

    if matched := YEAR_END_RE.fullmatch(text):
        year = int(matched.group("year"))
        return _normalize_point(year, 12), _context(year, 12)

    if matched := YEAR_MID_RE.fullmatch(text):
        year = int(matched.group("year"))
        return _normalize_point(year, 6), _context(year, 6)

    if matched := YEAR_HALF_RE.fullmatch(text):
        year = int(matched.group("year"))
        month = 1 if matched.group("half") == "上" else 7
        return _normalize_point(year, month), _context(year, month)

    if matched := YEAR_QUARTER_RE.fullmatch(text):
        year = int(matched.group("year"))
        quarter = matched.group("quarter")
        month = {"一": 1, "二": 4, "三": 7, "四": 10}[quarter]
        return _normalize_point(year, month), _context(year, month)

    if matched := YEAR_SEASON_RE.fullmatch(text):
        year = int(matched.group("year"))
        month = {"春": 3, "夏": 6, "秋": 9, "冬": 12}[matched.group("season")]
        return _normalize_point(year, month), _context(year, month)

    if matched := DECADE_RE.fullmatch(text):
        base_year = int(matched.group("year"))
        position = matched.group("position")
        if position is None:
            return _normalize_point(base_year), _context(base_year)
        if position == "初":
            return _normalize_point(base_year, 1), _context(base_year, 1)
        if position == "中":
            return _normalize_point(base_year + 5, 6), _context(base_year + 5, 6)
        return _normalize_point(base_year + 9, 12), _context(base_year + 9, 12)

    return None, None


def _normalize_expression(
    text: str,
    context: TimeContext | None = None,
) -> tuple[str | None, TimeContext | None]:
    if matched := YEAR_RANGE_HYPHEN_RE.fullmatch(text):
        start_year = int(matched.group("start"))
        end_year = int(matched.group("end"))
        return (
            f"{_normalize_point(start_year)}/{_normalize_point(end_year)}",
            _context(end_year),
        )

    if matched := YEAR_MONTH_RANGE_HYPHEN_RE.fullmatch(text):
        year = int(matched.group("year"))
        start_month = int(matched.group("start"))
        end_month = int(matched.group("end"))
        return (
            f"{_normalize_point(year, start_month)}/{_normalize_point(year, end_month)}",
            _context(year, end_month),
        )

    if "至" in text:
        left_text, right_text = text.split("至", 1)
        left_normalized, left_context = _normalize_expression(left_text, context)
        if left_normalized is None:
            return None, None
        right_normalized, right_context = _normalize_expression(
            right_text, left_context
        )
        if right_normalized is None:
            return None, None
        return f"{left_normalized}/{right_normalized}", right_context or left_context

    return _normalize_single_fragment(text, context)


def normalize_time_text(value: str) -> str | None:
    text = value.strip()
    if not text:
        return text

    text = re.sub(r"\s+", "", text)
    text = text.replace("—", "-").replace("–", "-").replace("－", "-")
    text = _strip_open_ended_suffixes(text)

    parts = [part for part in LIST_SPLIT_RE.split(text) if part]
    if len(parts) > 1:
        normalized_parts: list[str] = []
        context = TimeContext()
        for part in parts:
            normalized, context = _normalize_expression(part, context)
            if normalized is None:
                normalized_parts = []
                break
            normalized_parts.append(normalized)
        if normalized_parts:
            return "/".join(normalized_parts)

    normalized, _ = _normalize_expression(text, None)
    return normalized


def collect_time_updates(
    s2_file: Path,
) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, Any]]]:
    updates: dict[str, str] = {}
    changed_rows: list[dict[str, str]] = []
    unresolved_counter: Counter[str] = Counter()
    unresolved_samples: defaultdict[str, list[str]] = defaultdict(list)

    for row in read_jsonl(s2_file):
        tuple_id = row.get("tuple_id")
        time_value = row.get("time")
        if not isinstance(tuple_id, str) or not tuple_id:
            raise ValueError(f"{s2_file} 中存在缺失 tuple_id 的记录: {row}")
        if not isinstance(time_value, str) or not time_value.strip():
            continue

        normalized = normalize_time_text(time_value)
        if normalized is None:
            unresolved_counter[time_value] += 1
            if len(unresolved_samples[time_value]) < 5:
                unresolved_samples[time_value].append(tuple_id)
            continue
        if normalized == time_value:
            continue

        updates[tuple_id] = normalized
        changed_rows.append(
            {
                "tuple_id": tuple_id,
                "old_time": time_value,
                "new_time": normalized,
            }
        )

    unresolved_rows = [
        {
            "time": time_text,
            "count": unresolved_counter[time_text],
            "sample_tuple_ids": unresolved_samples[time_text],
        }
        for time_text in sorted(
            unresolved_counter,
            key=lambda item: (-unresolved_counter[item], item),
        )
    ]
    return updates, changed_rows, unresolved_rows


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


def _matched_trace_id(
    payload: dict[str, Any], time_updates: dict[str, str]
) -> str | None:
    for field_name in TRACE_ID_FIELDS:
        value = payload.get(field_name)
        if isinstance(value, str) and value in time_updates:
            return value
    return None


def _update_payload_times(
    payload: Any,
    time_updates: dict[str, str],
) -> tuple[Any, int, set[str]]:
    if isinstance(payload, dict):
        updated_payload = dict(payload)
        updated_count = 0
        updated_ids: set[str] = set()

        matched_id = _matched_trace_id(payload, time_updates)
        if matched_id is not None and isinstance(payload.get("time"), str):
            new_time = time_updates[matched_id]
            if payload.get("time") != new_time:
                updated_payload["time"] = new_time
                updated_count += 1
                updated_ids.add(matched_id)

        for key, value in payload.items():
            if key == "time" and matched_id is not None and isinstance(value, str):
                continue
            next_value, child_count, child_ids = _update_payload_times(
                value, time_updates
            )
            updated_payload[key] = next_value
            updated_count += child_count
            updated_ids.update(child_ids)
        return updated_payload, updated_count, updated_ids

    if isinstance(payload, list):
        updated_items: list[Any] = []
        updated_count = 0
        updated_ids: set[str] = set()
        for item in payload:
            next_value, child_count, child_ids = _update_payload_times(
                item, time_updates
            )
            updated_items.append(next_value)
            updated_count += child_count
            updated_ids.update(child_ids)
        return updated_items, updated_count, updated_ids

    return payload, 0, set()


def _update_jsonl_file(
    file_path: Path,
    time_updates: dict[str, str],
    dry_run: bool,
) -> tuple[int, set[str]]:
    had_done = is_jsonl_done(file_path)
    kept_rows: list[dict[str, Any]] = []
    updated_count = 0
    updated_ids: set[str] = set()

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if payload == DONE_SENTINEL:
                continue
            next_payload, child_count, child_ids = _update_payload_times(
                payload,
                time_updates,
            )
            if not isinstance(next_payload, dict):
                raise ValueError(f"JSONL 记录更新时间后不是对象: {file_path}")
            kept_rows.append(next_payload)
            updated_count += child_count
            updated_ids.update(child_ids)

    if updated_count and not dry_run:
        write_jsonl(file_path, kept_rows, append_done=had_done)
    return updated_count, updated_ids


def _update_json_file(
    file_path: Path,
    time_updates: dict[str, str],
    dry_run: bool,
) -> tuple[int, set[str]]:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    next_payload, updated_count, updated_ids = _update_payload_times(
        payload, time_updates
    )
    if updated_count and not dry_run:
        file_path.write_text(
            json.dumps(next_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return updated_count, updated_ids


def normalize_tuple_times(
    project_root: Path,
    s2_file: Path,
    trace_dirs: list[str] | None = None,
    report_file: Path | None = None,
    dry_run: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    active_logger = logger or build_logger()
    active_trace_dirs = trace_dirs or list(DEFAULT_TRACE_DIRS)
    time_updates, changed_rows, unresolved_rows = collect_time_updates(s2_file)
    resolved_report_file = report_file.resolve() if report_file else None

    active_logger.info(
        "在 %s 中识别到 %s 条可规范化的时间记录。",
        s2_file,
        len(changed_rows),
    )

    trace_files = _iter_trace_files(
        project_root,
        active_trace_dirs,
        excluded_files={resolved_report_file} if resolved_report_file else None,
    )
    updated_records: dict[str, int] = {}
    hit_files_by_tuple_id: dict[str, list[str]] = {
        row["tuple_id"]: [] for row in changed_rows
    }

    if time_updates:
        for file_path in trace_files:
            if file_path.suffix == ".jsonl":
                updated_count, hit_ids = _update_jsonl_file(
                    file_path, time_updates, dry_run
                )
            else:
                updated_count, hit_ids = _update_json_file(
                    file_path, time_updates, dry_run
                )

            if updated_count:
                relative_path = file_path.relative_to(project_root).as_posix()
                updated_records[relative_path] = updated_count
                for row in changed_rows:
                    tuple_id = row["tuple_id"]
                    if tuple_id in hit_ids:
                        hit_files_by_tuple_id[tuple_id].append(relative_path)
                active_logger.info(
                    "%s %s 更新 %s 条时间记录。",
                    "将" if dry_run else "已",
                    relative_path,
                    updated_count,
                )

    report_rows = [
        {
            **row,
            "hit_files": hit_files_by_tuple_id.get(row["tuple_id"], []),
        }
        for row in changed_rows
    ]

    report_path_str = None
    if resolved_report_file is not None:
        report_path_str = resolved_report_file.relative_to(project_root).as_posix()
        report_payload = {
            "s2_file": s2_file.relative_to(project_root).as_posix(),
            "updated_tuple_count": len(changed_rows),
            "updated_tuples": report_rows,
            "unresolved_time_count": sum(item["count"] for item in unresolved_rows),
            "unresolved_times": unresolved_rows,
            "updated_records": updated_records,
            "dry_run": dry_run,
        }
        if not dry_run:
            write_json(resolved_report_file, report_payload)
            active_logger.info("时间规范化报告已写入 %s", report_path_str)

    active_logger.info(
        "%s扫描 %s 个文件，命中 %s 个文件；仍有 %s 条记录无法可靠规范化。",
        "dry-run：" if dry_run else "",
        len(trace_files),
        len(updated_records),
        sum(item["count"] for item in unresolved_rows),
    )
    return {
        "updated_tuple_count": len(changed_rows),
        "updated_tuples": report_rows,
        "unresolved_times": unresolved_rows,
        "updated_records": updated_records,
        "report_file": report_path_str,
    }


def main() -> int:
    args = parse_args()
    project_root = Path(args.root).resolve()
    s2_file = (project_root / args.s2_file).resolve()
    report_file = (project_root / args.report_file).resolve()
    if not s2_file.exists():
        raise FileNotFoundError(f"S2 五元组文件不存在: {s2_file}")

    normalize_tuple_times(
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
