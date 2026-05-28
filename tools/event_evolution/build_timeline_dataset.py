from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from tools.event_evolution.manifest_utils import (
    build_corrected_path,
    build_corrected_stem,
    extract_node_id,
    split_title_stem,
)
from tools.event_evolution.time_utils import (
    TimeCandidate,
    TitleTimeInfo,
    build_boundary_date,
    choose_final_time_candidate,
    extract_body_time_candidates,
    get_title_time_window,
    has_explicit_body_year,
    parse_title_time_info,
)
from utils.io_utils import write_json, write_jsonl, write_text
from utils.news_preprocess_utils import list_raw_news_files


@dataclass(slots=True)
class TimelineCorrectionRecord:
    """单篇文档的校正结果。"""

    node_id: str
    source_path: Path
    corrected_path: Path
    base_title: str
    corrected_stem: str
    title_info: TitleTimeInfo
    body_candidates: list[TimeCandidate]
    final_candidate: TimeCandidate | None
    temporal_status: str
    needs_review: bool
    review_reasons: list[str]
    ambiguity_tags: list[str]

    def to_manifest_row(self) -> dict:
        return {
            "node_id": self.node_id,
            "source_file": self.source_path.name,
            "source_path": str(self.source_path),
            "corrected_file": self.corrected_path.name,
            "corrected_path": str(self.corrected_path),
            "base_title": self.base_title,
            "corrected_title": self.corrected_stem,
            "raw_title_time": self.title_info.raw_text,
            "title_info": self.title_info.to_dict(),
            "end_time": format_candidate_label(self.final_candidate),
            "end_time_precision": (
                self.final_candidate.precision
                if self.final_candidate is not None
                else "unknown"
            ),
            "confidence": (
                round(self.final_candidate.confidence, 4)
                if self.final_candidate is not None
                else 0.0
            ),
            "temporal_status": self.temporal_status,
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
            "ambiguity_tags": self.ambiguity_tags,
            "title_body_conflict": has_title_body_conflict(
                title_info=self.title_info,
                body_candidates=self.body_candidates,
                final_candidate=self.final_candidate,
            ),
            "final_candidate": (
                None if self.final_candidate is None else self.final_candidate.to_dict()
            ),
            "body_candidates": [item.to_dict() for item in self.body_candidates],
        }


def parse_args() -> argparse.Namespace:
    """解析时间线校正脚本参数。"""

    parser = argparse.ArgumentParser(description="processed 文档时间线结束时间校正脚本")
    parser.add_argument(
        "--scope",
        default="all",
        help="处理范围。支持 all、D001、E001-E014、D001,E001-E014。",
    )
    parser.add_argument(
        "--input-dir",
        default="data/processed",
        help="输入目录，默认 data/processed。",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed_timeline",
        help="校正后的派生文本目录，默认 data/processed_timeline。",
    )
    parser.add_argument(
        "--manifest-path",
        default="output/event_evolution/timeline_manifest.jsonl",
        help="全量 manifest 输出路径。",
    )
    parser.add_argument(
        "--review-path",
        default="output/event_evolution/timeline_review_manifest.jsonl",
        help="待复核清单输出路径。",
    )
    parser.add_argument(
        "--summary-path",
        default="output/event_evolution/timeline_summary.json",
        help="汇总统计输出路径。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅处理前 N 篇文件，0 表示处理全部。",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只展示匹配文件和汇总，不真正写出结果"
    )
    return parser.parse_args()


def build_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger("kg-timeline-correction")


def main() -> int:
    args = parse_args()
    logger = build_logger()
    records = build_timeline_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        manifest_path=Path(args.manifest_path),
        review_path=Path(args.review_path),
        summary_path=Path(args.summary_path),
        scope=args.scope,
        logger=logger,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    logger.info(
        "时间线校正完成：共 %s 篇，待复核 %s 篇，UNKNOWN_TIME %s 篇",
        len(records),
        sum(1 for item in records if item.needs_review),
        sum(1 for item in records if item.final_candidate is None),
    )
    return 0


def build_timeline_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    review_path: Path,
    summary_path: Path,
    scope: str,
    logger: logging.Logger,
    dry_run: bool = False,
    limit: int = 0,
) -> list[TimelineCorrectionRecord]:
    """对整个 processed 数据集执行时间线结束时间校正。"""

    files = list_raw_news_files(input_dir, scope)
    if limit > 0:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(
            f"未找到满足范围 {scope!r} 的 processed txt 文件。请检查 {input_dir}。"
        )

    logger.info("匹配到 %s 篇 processed 文档，范围=%s", len(files), scope)
    records: list[TimelineCorrectionRecord] = []
    for index, source_path in enumerate(files, start=1):
        record = correct_document(source_path=source_path, output_dir=output_dir)
        records.append(record)

        if record.needs_review or record.final_candidate is None:
            logger.info(
                "[%s/%s] 待复核 %s -> %s (%s)",
                index,
                len(files),
                source_path.name,
                record.corrected_path.name,
                ", ".join(record.review_reasons) or "unknown",
            )
        elif index == 1 or index % 25 == 0 or index == len(files):
            logger.info(
                "[%s/%s] 已校正 %s -> %s",
                index,
                len(files),
                source_path.name,
                record.corrected_path.name,
            )

    if dry_run:
        return records

    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        write_text(
            record.corrected_path, record.source_path.read_text(encoding="utf-8")
        )

    manifest_rows = [item.to_manifest_row() for item in records]
    review_rows = [row for row in manifest_rows if row["needs_review"]]

    write_jsonl(manifest_path, manifest_rows, append_done=True)
    write_jsonl(review_path, review_rows, append_done=True)
    write_json(summary_path, build_summary(records))
    return records


def correct_document(
    *, source_path: Path, output_dir: Path
) -> TimelineCorrectionRecord:
    """校正单篇 processed 文档的结束时间并生成目标文件名。"""

    stem = source_path.stem
    base_title, raw_title_time, malformed = split_title_stem(stem)
    title_info = parse_title_time_info(raw_title_time, malformed=malformed)
    body_text = source_path.read_text(encoding="utf-8")
    body_candidates, ambiguity_tags = extract_body_time_candidates(
        body_text, title_info
    )
    final_candidate = choose_final_time_candidate(
        title_info=title_info,
        body_candidates=body_candidates,
    )

    temporal_status = determine_temporal_status(
        title_info=title_info,
        final_candidate=final_candidate,
    )
    review_reasons = determine_review_reasons(
        title_info=title_info,
        body_candidates=body_candidates,
        final_candidate=final_candidate,
        ambiguity_tags=ambiguity_tags,
    )
    corrected_stem = build_corrected_stem(
        base_title, format_candidate_label(final_candidate)
    )
    corrected_path = build_corrected_path(
        output_dir=output_dir,
        base_title=base_title,
        time_label=format_candidate_label(final_candidate),
    )

    return TimelineCorrectionRecord(
        node_id=extract_node_id(stem),
        source_path=source_path,
        corrected_path=corrected_path,
        base_title=base_title,
        corrected_stem=corrected_stem,
        title_info=title_info,
        body_candidates=body_candidates,
        final_candidate=final_candidate,
        temporal_status=temporal_status,
        needs_review=bool(review_reasons),
        review_reasons=review_reasons,
        ambiguity_tags=ambiguity_tags,
    )


def determine_temporal_status(
    *, title_info: TitleTimeInfo, final_candidate: TimeCandidate | None
) -> str:
    """根据候选来源和标题标记给出时间状态。"""

    if final_candidate is None:
        return "unknown"
    if "open_ended" in title_info.flags:
        return "open_ended"
    if "repeated" in title_info.flags:
        return "repeated"
    if final_candidate.source == "title_hint":
        return "title_only"
    if any(
        tag.startswith("year_") or tag.startswith("relative_")
        for tag in final_candidate.tags
    ):
        return "inferred"
    return "exact"


def determine_review_reasons(
    *,
    title_info: TitleTimeInfo,
    body_candidates: list[TimeCandidate],
    final_candidate: TimeCandidate | None,
    ambiguity_tags: list[str],
) -> list[str]:
    """汇总该文档为什么需要人工复核。"""

    reasons: list[str] = []
    if final_candidate is None:
        reasons.append("unknown_time")
    if title_info.malformed:
        reasons.append("malformed_title_time")
    if "open_ended" in title_info.flags:
        reasons.append("open_ended_title")
    if "repeated" in title_info.flags:
        reasons.append("repeated_title")
    if final_candidate is not None and final_candidate.confidence < 0.75:
        reasons.append("low_confidence")
    if final_candidate is not None and final_candidate.source == "title_hint":
        reasons.append("title_only_fallback")
    if title_info.spans_multiple_years() and not has_explicit_body_year(
        body_candidates
    ):
        reasons.append("multi_year_title_without_explicit_body_year")
    if has_title_body_conflict(
        title_info=title_info,
        body_candidates=body_candidates,
        final_candidate=final_candidate,
    ):
        reasons.append("title_body_conflict")
    if should_review_ambiguous_relative_time(
        final_candidate=final_candidate,
        ambiguity_tags=ambiguity_tags,
    ):
        reasons.append("contains_ambiguous_relative_time")
    return dedupe_preserve_order(reasons)


def should_review_ambiguous_relative_time(
    *, final_candidate: TimeCandidate | None, ambiguity_tags: list[str]
) -> bool:
    """仅在歧义表达可能影响最终候选时要求复核。"""

    if final_candidate is None or not ambiguity_tags:
        return False
    if final_candidate.source.startswith("body_relative"):
        return True
    if any(tag.startswith("relative_") for tag in final_candidate.tags):
        return True
    if final_candidate.line_number is None:
        return False

    ambiguous_marker = f"line_{final_candidate.line_number}:ambiguous_relative_time"
    if ambiguous_marker not in ambiguity_tags:
        return False
    if final_candidate.source.startswith("body_explicit"):
        return False
    return final_candidate.confidence < 0.8


def has_title_body_conflict(
    *,
    title_info: TitleTimeInfo,
    body_candidates: list[TimeCandidate],
    final_candidate: TimeCandidate | None,
) -> bool:
    """判断标题时间提示与正文最终时间是否冲突。"""

    if title_info.candidate is None or not body_candidates or final_candidate is None:
        return False
    title_window = get_title_time_window(title_info)
    if title_window is None:
        return False

    _, title_end, _ = title_window
    final_end = build_boundary_date(
        year=final_candidate.year,
        month=final_candidate.month,
        day=final_candidate.day,
        is_end=True,
    )
    if title_window[0] <= final_end <= title_window[1]:
        return False
    if is_short_forward_extension(
        title_info=title_info,
        title_end=title_end,
        final_candidate=final_candidate,
        final_end=final_end,
    ):
        return False
    return True


def is_short_forward_extension(
    *,
    title_info: TitleTimeInfo,
    title_end,
    final_candidate: TimeCandidate,
    final_end,
) -> bool:
    """把标题时间后的短期顺延视作正常结束时间延展。"""

    title_candidate = title_info.candidate
    if title_candidate is None or final_end < title_end:
        return False
    if title_candidate.precision == "day":
        if (
            final_candidate.year != title_candidate.year
            or final_candidate.month != title_candidate.month
            or title_candidate.day is None
            or final_candidate.day is None
        ):
            return False
        return (final_end - title_end).days <= 14
    if title_candidate.precision == "month":
        if (
            final_candidate.year != title_candidate.year
            or title_candidate.month is None
            or final_candidate.month is None
        ):
            return False
        return 0 <= final_candidate.month - title_candidate.month <= 3
    return False


def format_candidate_label(candidate: TimeCandidate | None) -> str:
    """把最终候选转换成用于文件名和 manifest 的标签。"""

    if candidate is None:
        return "UNKNOWN_TIME"
    return candidate.normalized


def build_summary(records: list[TimelineCorrectionRecord]) -> dict:
    """生成整库校正的汇总统计。"""

    return {
        "total_documents": len(records),
        "corrected_documents": sum(
            1 for item in records if item.final_candidate is not None
        ),
        "unknown_time_documents": sum(
            1 for item in records if item.final_candidate is None
        ),
        "needs_review_documents": sum(1 for item in records if item.needs_review),
        "high_confidence_documents": sum(
            1
            for item in records
            if item.final_candidate is not None
            and item.final_candidate.confidence >= 0.8
            and not item.needs_review
        ),
        "open_ended_documents": sum(
            1 for item in records if "open_ended" in item.title_info.flags
        ),
        "repeated_documents": sum(
            1 for item in records if "repeated" in item.title_info.flags
        ),
        "title_body_conflict_documents": sum(
            1
            for item in records
            if has_title_body_conflict(
                title_info=item.title_info,
                body_candidates=item.body_candidates,
                final_candidate=item.final_candidate,
            )
        ),
    }


def dedupe_preserve_order(values: list[str]) -> list[str]:
    """对复核原因去重并保持顺序。"""

    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


if __name__ == "__main__":
    sys.exit(main())
