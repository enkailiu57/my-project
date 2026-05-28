from __future__ import annotations

import argparse
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from utils.io_utils import read_jsonl, write_json, write_text

TUPLE_SUFFIX = "_t"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计 S2 扁平五元组文件中每个句子产生的五元组数量，并输出排序报告。"
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
        "--report-file",
        default="output/tuple_sentence_stats_report.json",
        help="JSON 统计报告输出路径，默认 output/tuple_sentence_stats_report.json。",
    )
    parser.add_argument(
        "--markdown-report-file",
        default="output/tuple_sentence_stats_report.md",
        help="Markdown 统计报告输出路径，默认 output/tuple_sentence_stats_report.md。",
    )
    return parser.parse_args()


def build_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger("kg-tuple-sentence-stats")


def _sentence_id_from_tuple_like(value: str) -> str:
    marker_index = value.rfind(TUPLE_SUFFIX)
    if marker_index <= 0:
        return value
    suffix = value[marker_index + len(TUPLE_SUFFIX) :]
    return value[:marker_index] if suffix.isdigit() else value


def _resolve_sentence_id(row: dict[str, Any]) -> str:
    source_sent_id = row.get("source_sent_id")
    if isinstance(source_sent_id, str) and source_sent_id.strip():
        return source_sent_id.strip()

    for field_name in ("sent_id", "tuple_id"):
        value = row.get(field_name)
        if isinstance(value, str) and value.strip():
            return _sentence_id_from_tuple_like(value.strip())

    raise ValueError(f"记录缺失可解析的句子 ID: {row}")


def _resolve_doc_id(row: dict[str, Any], sentence_id: str) -> str | None:
    doc_id = row.get("doc_id")
    if isinstance(doc_id, str) and doc_id.strip():
        return doc_id.strip()

    if "_s" in sentence_id:
        return sentence_id.split("_s", 1)[0]
    return None


def collect_sentence_tuple_stats(s2_file: Path) -> dict[str, Any]:
    sentence_stats_by_id: dict[str, dict[str, Any]] = {}

    for row in read_jsonl(s2_file):
        sentence_id = _resolve_sentence_id(row)
        tuple_id = row.get("tuple_id")
        entry = sentence_stats_by_id.setdefault(
            sentence_id,
            {
                "sentence_id": sentence_id,
                "doc_id": _resolve_doc_id(row, sentence_id),
                "tuple_count": 0,
                "tuple_ids": [],
            },
        )
        if entry["doc_id"] is None:
            entry["doc_id"] = _resolve_doc_id(row, sentence_id)

        entry["tuple_count"] += 1
        if isinstance(tuple_id, str) and tuple_id.strip():
            entry["tuple_ids"].append(tuple_id.strip())

    sorted_rows = sorted(
        sentence_stats_by_id.values(),
        key=lambda item: (-item["tuple_count"], item["sentence_id"]),
    )
    for index, row in enumerate(sorted_rows, start=1):
        row["rank"] = index

    counts = [row["tuple_count"] for row in sorted_rows]
    distribution = [
        {"tuple_count": tuple_count, "sentence_count": sentence_count}
        for tuple_count, sentence_count in sorted(
            Counter(counts).items(),
            key=lambda item: (-item[0], item[1]),
        )
    ]

    total_tuple_count = sum(counts)
    total_sentence_count = len(sorted_rows)
    average_tuple_count = (
        round(total_tuple_count / total_sentence_count, 4)
        if total_sentence_count
        else 0.0
    )
    median_tuple_count = statistics.median(counts) if counts else 0

    return {
        "total_tuple_count": total_tuple_count,
        "total_sentence_count": total_sentence_count,
        "max_tuple_count": max(counts, default=0),
        "min_tuple_count": min(counts, default=0),
        "average_tuple_count": average_tuple_count,
        "median_tuple_count": median_tuple_count,
        "distribution": distribution,
        "sentence_stats": sorted_rows,
    }


def _build_markdown_report(s2_file: str, stats: dict[str, Any]) -> str:
    lines = [
        "# S2 句子五元组统计报告",
        "",
        "## 概览",
        "",
        f"- 统计文件: {s2_file}",
        f"- 句子总数: {stats['total_sentence_count']}",
        f"- 五元组总数: {stats['total_tuple_count']}",
        f"- 单句最大五元组数: {stats['max_tuple_count']}",
        f"- 单句最小五元组数: {stats['min_tuple_count']}",
        f"- 平均每句五元组数: {stats['average_tuple_count']}",
        f"- 中位数: {stats['median_tuple_count']}",
        "",
        "## 数量分布",
        "",
        "| 每句五元组数 | 句子数 |",
        "| --- | --- |",
    ]
    for row in stats["distribution"]:
        lines.append(f"| {row['tuple_count']} | {row['sentence_count']} |")

    lines.extend(
        [
            "",
            "## 全量排序结果",
            "",
            "| 排名 | 句子 ID | 文档 ID | 五元组数 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in stats["sentence_stats"]:
        doc_id = row["doc_id"] or ""
        lines.append(
            f"| {row['rank']} | {row['sentence_id']} | {doc_id} | {row['tuple_count']} |"
        )

    return "\n".join(lines) + "\n"


def generate_sentence_tuple_stats_report(
    project_root: Path,
    s2_file: Path,
    report_file: Path,
    markdown_report_file: Path,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    active_logger = logger or build_logger()
    stats = collect_sentence_tuple_stats(s2_file)

    relative_s2_file = s2_file.relative_to(project_root).as_posix()
    report_payload = {
        "s2_file": relative_s2_file,
        **stats,
    }
    write_json(report_file, report_payload)
    write_text(
        markdown_report_file,
        _build_markdown_report(relative_s2_file, stats),
    )

    active_logger.info(
        "已统计 %s 个句子、%s 条五元组；JSON 报告写入 %s，Markdown 报告写入 %s。",
        stats["total_sentence_count"],
        stats["total_tuple_count"],
        report_file.relative_to(project_root).as_posix(),
        markdown_report_file.relative_to(project_root).as_posix(),
    )
    return {
        **report_payload,
        "report_file": report_file.relative_to(project_root).as_posix(),
        "markdown_report_file": markdown_report_file.relative_to(
            project_root
        ).as_posix(),
    }


def main() -> int:
    args = parse_args()
    project_root = Path(args.root).resolve()
    s2_file = (project_root / args.s2_file).resolve()
    report_file = (project_root / args.report_file).resolve()
    markdown_report_file = (project_root / args.markdown_report_file).resolve()
    if not s2_file.exists():
        raise FileNotFoundError(f"S2 五元组文件不存在: {s2_file}")

    generate_sentence_tuple_stats_report(
        project_root=project_root,
        s2_file=s2_file,
        report_file=report_file,
        markdown_report_file=markdown_report_file,
        logger=build_logger(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
