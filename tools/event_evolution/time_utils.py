from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date

YEAR_HINT_PATTERN = re.compile(r"(?<!\d)(?P<year>(?:19|20)\d{2})(?!\d)")

TITLE_SAME_YEAR_CROSS_MONTH_DAY_RANGE_PATTERN = re.compile(
    r"(?P<year>(?:19|20)\d{2})[.年](?P<start_month>\d{1,2})[.月](?P<start_day>\d{1,2})\s*[-—–~～至]\s*(?P<end_month>\d{1,2})[.月](?P<end_day>\d{1,2})(?!\d)"
)
TITLE_SAME_YEAR_DAY_RANGE_PATTERN = re.compile(
    r"(?P<year>(?:19|20)\d{2})[.年](?P<month>\d{1,2})[.月](?P<start_day>\d{1,2})\s*[-—–~～至]\s*(?P<end_day>\d{1,2})(?![.\d])"
)
TITLE_SAME_YEAR_MONTH_RANGE_PATTERN = re.compile(
    r"(?P<year>(?:19|20)\d{2})[.年](?P<start_month>\d{1,2})\s*[-—–~～至]\s*(?P<end_month>\d{1,2})(?:[.月](?P<end_day>\d{1,2}))?(?!\d)"
)
TITLE_CROSS_YEAR_RANGE_PATTERN = re.compile(
    r"(?P<start_year>(?:19|20)\d{2})(?:[.年](?P<start_month>\d{1,2})(?:[.月](?P<start_day>\d{1,2}))?)?\s*[-—–~～至]\s*(?P<end_year>(?:19|20)\d{2})(?:[.年](?P<end_month>\d{1,2})(?:[.月](?P<end_day>\d{1,2}))?)?"
)
TITLE_SINGLE_DOT_PATTERN = re.compile(
    r"(?P<year>(?:19|20)\d{2})[.年](?P<month>\d{1,2})(?:[.月](?P<day>\d{1,2}))?"
)
TITLE_YEAR_ONLY_PATTERN = re.compile(r"(?P<year>(?:19|20)\d{2})(?:年|全年)?")

YEAR_MONTH_DAY_RANGE_PATTERN = re.compile(
    r"(?<!\d)(?P<year>(?:19|20)\d{2})年(?P<start_month>\d{1,2})月(?P<start_day>\d{1,2})(?:日|号)?[^。\n]{0,24}?[至到](?:(?P<end_month>\d{1,2})月)?(?P<end_day>\d{1,2})(?:日|号)"
)
DOT_DATE_PATTERN = re.compile(
    r"(?<!\d)(?P<year>(?:19|20)\d{2})\.(?P<month>\d{1,2})(?:\.(?P<day>\d{1,2}))?(?!\d)"
)
YEAR_MONTH_DAY_PATTERN = re.compile(
    r"(?<!\d)(?P<year>(?:19|20)\d{2})年(?P<month>\d{1,2})月(?P<day>\d{1,2})(?:日|号)"
)
YEAR_MONTH_PATTERN = re.compile(
    r"(?<!\d)(?P<year>(?:19|20)\d{2})年(?P<month>\d{1,2})月"
)
YEAR_END_PATTERN = re.compile(r"(?<!\d)(?P<year>(?:19|20)\d{2})年底")
YEAR_BEGIN_PATTERN = re.compile(r"(?<!\d)(?P<year>(?:19|20)\d{2})年初")
YEAR_ONLY_PATTERN = re.compile(r"(?<!\d)(?P<year>(?:19|20)\d{2})年(?![月\d])")

MONTH_DAY_RANGE_PATTERN = re.compile(
    r"(?<!\d)(?P<start_month>\d{1,2})月(?P<start_day>\d{1,2})(?:日|号)?[^。\n]{0,24}?[至到](?:(?P<end_month>\d{1,2})月)?(?P<end_day>\d{1,2})(?:日|号)"
)
DAY_RANGE_PATTERN = re.compile(
    r"(?<!\d)(?P<start_day>\d{1,2})日[^。\n]{0,12}?[至到](?P<end_day>\d{1,2})日"
)
MONTH_DAY_PATTERN = re.compile(r"(?<!\d)(?P<month>\d{1,2})月(?P<day>\d{1,2})(?:日|号)")
MONTH_ONLY_PATTERN = re.compile(r"(?<!\d)(?P<month>\d{1,2})月(?![日号\d])")

RELATIVE_YEAR_PATTERN = re.compile(
    r"(?P<kind>今年|去年)(?P<suffix>初|底)?(?:(?P<month>\d{1,2})月(?:(?P<day>\d{1,2})(?:日|号))?)?"
)
SAME_DAY_PATTERN = re.compile(r"同日|当天")
NEXT_DAY_PATTERN = re.compile(r"次日|翌日|隔天|第二天")

AMBIGUOUS_RELATIVE_PATTERN = re.compile(r"上周|本周|近日|日前|近期|稍后|随后")
HISTORICAL_YEAR_MARKER_PATTERN = re.compile(r"年(?:来|以来|起|开始)")


@dataclass(slots=True)
class TimeCandidate:
    """单个可排序的时间候选。"""

    year: int
    month: int | None
    day: int | None
    precision: str
    normalized: str
    source: str
    matched_text: str
    line_number: int | None
    confidence: float
    tags: list[str]

    def sort_key(self) -> tuple[int, int, int, int, float]:
        precision_rank = {"unknown": 0, "year": 1, "month": 2, "day": 3}[self.precision]
        return (
            self.year,
            self.month or 0,
            self.day or 0,
            precision_rank,
            self.confidence,
        )

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "month": self.month,
            "day": self.day,
            "precision": self.precision,
            "normalized": self.normalized,
            "source": self.source,
            "matched_text": self.matched_text,
            "line_number": self.line_number,
            "confidence": round(self.confidence, 4),
            "tags": self.tags,
        }


@dataclass(slots=True)
class TitleTimeInfo:
    """文件标题中的时间提示。"""

    raw_text: str | None
    pattern: str
    start_year: int | None
    end_year: int | None
    candidate: TimeCandidate | None
    flags: list[str]
    malformed: bool

    def spans_multiple_years(self) -> bool:
        return (
            self.start_year is not None
            and self.end_year is not None
            and self.start_year != self.end_year
        )

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "pattern": self.pattern,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "candidate": None if self.candidate is None else self.candidate.to_dict(),
            "flags": self.flags,
            "malformed": self.malformed,
        }


def parse_title_time_info(
    raw_time_text: str | None, malformed: bool = False
) -> TitleTimeInfo:
    """把原标题中的括号时间提示解析成结构化信息。"""

    flags: list[str] = []
    if raw_time_text:
        if "多次" in raw_time_text:
            flags.append("repeated")
        if (
            "持续" in raw_time_text
            or "至今" in raw_time_text
            or "年起" in raw_time_text
        ):
            flags.append("open_ended")

    year_hints = [
        int(match.group("year"))
        for match in YEAR_HINT_PATTERN.finditer(raw_time_text or "")
    ]
    start_year = year_hints[0] if year_hints else None
    end_year = year_hints[-1] if year_hints else None

    candidate: TimeCandidate | None = None
    pattern = "none"
    if raw_time_text:
        if match := TITLE_SAME_YEAR_CROSS_MONTH_DAY_RANGE_PATTERN.search(raw_time_text):
            pattern = "title_same_year_day_range"
            candidate = build_time_candidate(
                year=int(match.group("year")),
                month=int(match.group("end_month")),
                day=int(match.group("end_day")),
                source="title_hint",
                matched_text=match.group(0),
                line_number=None,
                confidence=0.65,
                tags=flags.copy(),
            )
        elif match := TITLE_CROSS_YEAR_RANGE_PATTERN.search(raw_time_text):
            pattern = "title_cross_year_range"
            candidate = build_time_candidate(
                year=int(match.group("end_year")),
                month=(
                    int(match.group("end_month")) if match.group("end_month") else None
                ),
                day=int(match.group("end_day")) if match.group("end_day") else None,
                source="title_hint",
                matched_text=match.group(0),
                line_number=None,
                confidence=0.58,
                tags=flags.copy(),
            )
        elif match := TITLE_SAME_YEAR_DAY_RANGE_PATTERN.search(raw_time_text):
            pattern = "title_same_year_day_range"
            candidate = build_time_candidate(
                year=int(match.group("year")),
                month=int(match.group("month")),
                day=int(match.group("end_day")),
                source="title_hint",
                matched_text=match.group(0),
                line_number=None,
                confidence=0.65,
                tags=flags.copy(),
            )
        elif match := TITLE_SAME_YEAR_MONTH_RANGE_PATTERN.search(raw_time_text):
            pattern = "title_same_year_month_range"
            candidate = build_time_candidate(
                year=int(match.group("year")),
                month=int(match.group("end_month")),
                day=int(match.group("end_day")) if match.group("end_day") else None,
                source="title_hint",
                matched_text=match.group(0),
                line_number=None,
                confidence=0.6,
                tags=flags.copy(),
            )
        elif match := TITLE_SINGLE_DOT_PATTERN.search(raw_time_text):
            pattern = "title_single_date"
            candidate = build_time_candidate(
                year=int(match.group("year")),
                month=int(match.group("month")),
                day=int(match.group("day")) if match.group("day") else None,
                source="title_hint",
                matched_text=match.group(0),
                line_number=None,
                confidence=0.62,
                tags=flags.copy(),
            )
        elif match := TITLE_YEAR_ONLY_PATTERN.search(raw_time_text):
            pattern = "title_year_only"
            candidate = build_time_candidate(
                year=int(match.group("year")),
                month=None,
                day=None,
                source="title_hint",
                matched_text=match.group(0),
                line_number=None,
                confidence=0.5,
                tags=flags.copy(),
            )

    return TitleTimeInfo(
        raw_text=raw_time_text,
        pattern=pattern,
        start_year=start_year,
        end_year=end_year,
        candidate=candidate,
        flags=flags,
        malformed=malformed,
    )


def build_time_candidate(
    *,
    year: int,
    month: int | None,
    day: int | None,
    source: str,
    matched_text: str,
    line_number: int | None,
    confidence: float,
    tags: list[str] | None = None,
) -> TimeCandidate:
    """按统一格式创建时间候选。"""

    normalized, precision = normalize_time_components(year=year, month=month, day=day)
    return TimeCandidate(
        year=year,
        month=month,
        day=day,
        precision=precision,
        normalized=normalized,
        source=source,
        matched_text=matched_text,
        line_number=line_number,
        confidence=max(0.0, min(confidence, 0.99)),
        tags=list(tags or []),
    )


def normalize_time_components(
    *, year: int, month: int | None, day: int | None
) -> tuple[str, str]:
    """把时间分量格式化成统一字符串。"""

    if day is not None and month is not None:
        return f"{year:04d}.{month:02d}.{day:02d}", "day"
    if month is not None:
        return f"{year:04d}.{month:02d}", "month"
    return f"{year:04d}", "year"


def build_boundary_date(
    *, year: int, month: int | None, day: int | None, is_end: bool
) -> date:
    """把不完整时间转换成可比较的日期边界。"""

    if month is None:
        month = 12 if is_end else 1
    month = max(1, min(month, 12))
    if day is None:
        day = calendar.monthrange(year, month)[1] if is_end else 1
    else:
        day = max(1, min(day, calendar.monthrange(year, month)[1]))
    return date(year, month, day)


def get_title_time_window(title_info: TitleTimeInfo) -> tuple[date, date, str] | None:
    """返回标题提示可覆盖的时间窗口。"""

    raw_time_text = title_info.raw_text
    if raw_time_text:
        if match := TITLE_SAME_YEAR_CROSS_MONTH_DAY_RANGE_PATTERN.search(raw_time_text):
            year = int(match.group("year"))
            return (
                build_boundary_date(
                    year=year,
                    month=int(match.group("start_month")),
                    day=int(match.group("start_day")),
                    is_end=False,
                ),
                build_boundary_date(
                    year=year,
                    month=int(match.group("end_month")),
                    day=int(match.group("end_day")),
                    is_end=True,
                ),
                "day",
            )

        if match := TITLE_CROSS_YEAR_RANGE_PATTERN.search(raw_time_text):
            start_year = int(match.group("start_year"))
            end_year = int(match.group("end_year"))
            end_precision = (
                "day"
                if match.group("end_day")
                else "month" if match.group("end_month") else "year"
            )
            return (
                build_boundary_date(
                    year=start_year,
                    month=(
                        int(match.group("start_month"))
                        if match.group("start_month")
                        else None
                    ),
                    day=(
                        int(match.group("start_day"))
                        if match.group("start_day")
                        else None
                    ),
                    is_end=False,
                ),
                build_boundary_date(
                    year=end_year,
                    month=(
                        int(match.group("end_month"))
                        if match.group("end_month")
                        else None
                    ),
                    day=(
                        int(match.group("end_day")) if match.group("end_day") else None
                    ),
                    is_end=True,
                ),
                end_precision,
            )

        if match := TITLE_SAME_YEAR_DAY_RANGE_PATTERN.search(raw_time_text):
            year = int(match.group("year"))
            month = int(match.group("month"))
            return (
                build_boundary_date(
                    year=year,
                    month=month,
                    day=int(match.group("start_day")),
                    is_end=False,
                ),
                build_boundary_date(
                    year=year,
                    month=month,
                    day=int(match.group("end_day")),
                    is_end=True,
                ),
                "day",
            )

        if match := TITLE_SAME_YEAR_MONTH_RANGE_PATTERN.search(raw_time_text):
            year = int(match.group("year"))
            end_precision = "day" if match.group("end_day") else "month"
            return (
                build_boundary_date(
                    year=year,
                    month=int(match.group("start_month")),
                    day=None,
                    is_end=False,
                ),
                build_boundary_date(
                    year=year,
                    month=int(match.group("end_month")),
                    day=int(match.group("end_day")) if match.group("end_day") else None,
                    is_end=True,
                ),
                end_precision,
            )

    if title_info.candidate is None:
        return None

    return (
        build_boundary_date(
            year=title_info.candidate.year,
            month=title_info.candidate.month,
            day=title_info.candidate.day,
            is_end=False,
        ),
        build_boundary_date(
            year=title_info.candidate.year,
            month=title_info.candidate.month,
            day=title_info.candidate.day,
            is_end=True,
        ),
        title_info.candidate.precision,
    )


def extract_body_time_candidates(
    text: str, title_info: TitleTimeInfo
) -> tuple[list[TimeCandidate], list[str]]:
    """扫描正文，提取可排序的时间候选。"""

    current_year = title_info.start_year or title_info.end_year
    previous_month: int | None = None
    last_absolute_day: TimeCandidate | None = None
    candidates: list[TimeCandidate] = []
    ambiguity_tags: list[str] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if AMBIGUOUS_RELATIVE_PATTERN.search(line):
            ambiguity_tags.append(f"line_{line_number}:ambiguous_relative_time")

        inline_years = [
            int(match.group("year")) for match in YEAR_HINT_PATTERN.finditer(line)
        ]
        inline_year = inline_years[-1] if inline_years else None
        line_candidates = _extract_line_candidates(
            line=line,
            line_number=line_number,
            title_info=title_info,
            current_year=current_year,
            previous_month=previous_month,
            last_absolute_day=last_absolute_day,
        )

        for candidate in line_candidates:
            candidates.append(candidate)
            if candidate_updates_temporal_context(candidate, current_year):
                current_year = candidate.year
                if candidate.month is not None:
                    previous_month = candidate.month
            if candidate.precision == "day" and not candidate.source.startswith(
                "body_relative"
            ):
                last_absolute_day = candidate

    return dedupe_candidates(candidates), dedupe_strings(ambiguity_tags)


def choose_final_time_candidate(
    *, title_info: TitleTimeInfo, body_candidates: list[TimeCandidate]
) -> TimeCandidate | None:
    """按“正文优先、标题兜底”的规则选出最终结束时间。"""

    if body_candidates:
        return max(body_candidates, key=lambda item: item.sort_key())

    if "open_ended" in title_info.flags or "repeated" in title_info.flags:
        return None
    return title_info.candidate


def has_explicit_body_year(body_candidates: list[TimeCandidate]) -> bool:
    """判断正文是否存在带明确年份的候选。"""

    for candidate in body_candidates:
        if "year_fallback_from_title" in candidate.tags:
            continue
        if "year_inferred_by_rollover" in candidate.tags:
            continue
        return True
    return False


def _extract_line_candidates(
    *,
    line: str,
    line_number: int,
    title_info: TitleTimeInfo,
    current_year: int | None,
    previous_month: int | None,
    last_absolute_day: TimeCandidate | None,
) -> list[TimeCandidate]:
    occupied: list[tuple[int, int]] = []
    collected: list[tuple[int, TimeCandidate]] = []

    def add_candidate(start: int, end: int, candidate: TimeCandidate) -> None:
        if overlaps((start, end), occupied):
            return
        occupied.append((start, end))
        collected.append((start, candidate))

    for match in YEAR_MONTH_DAY_RANGE_PATTERN.finditer(line):
        end_month = (
            int(match.group("end_month"))
            if match.group("end_month")
            else int(match.group("start_month"))
        )
        add_candidate(
            match.start(),
            match.end(),
            build_time_candidate(
                year=int(match.group("year")),
                month=end_month,
                day=int(match.group("end_day")),
                source="body_explicit_range",
                matched_text=match.group(0),
                line_number=line_number,
                confidence=0.93,
            ),
        )

    for match in DOT_DATE_PATTERN.finditer(line):
        add_candidate(
            match.start(),
            match.end(),
            build_time_candidate(
                year=int(match.group("year")),
                month=int(match.group("month")),
                day=int(match.group("day")) if match.group("day") else None,
                source="body_explicit_dot",
                matched_text=match.group(0),
                line_number=line_number,
                confidence=0.9 if match.group("day") else 0.84,
            ),
        )

    for match in YEAR_MONTH_DAY_PATTERN.finditer(line):
        add_candidate(
            match.start(),
            match.end(),
            build_time_candidate(
                year=int(match.group("year")),
                month=int(match.group("month")),
                day=int(match.group("day")),
                source="body_explicit_day",
                matched_text=match.group(0),
                line_number=line_number,
                confidence=0.95,
            ),
        )

    for match in YEAR_END_PATTERN.finditer(line):
        add_candidate(
            match.start(),
            match.end(),
            build_time_candidate(
                year=int(match.group("year")),
                month=12,
                day=None,
                source="body_year_end",
                matched_text=match.group(0),
                line_number=line_number,
                confidence=0.82,
                tags=["year_boundary_keyword"],
            ),
        )

    for match in YEAR_BEGIN_PATTERN.finditer(line):
        add_candidate(
            match.start(),
            match.end(),
            build_time_candidate(
                year=int(match.group("year")),
                month=1,
                day=None,
                source="body_year_begin",
                matched_text=match.group(0),
                line_number=line_number,
                confidence=0.8,
                tags=["year_boundary_keyword"],
            ),
        )

    for match in YEAR_MONTH_PATTERN.finditer(line):
        add_candidate(
            match.start(),
            match.end(),
            build_time_candidate(
                year=int(match.group("year")),
                month=int(match.group("month")),
                day=None,
                source="body_explicit_month",
                matched_text=match.group(0),
                line_number=line_number,
                confidence=0.86,
            ),
        )

    for match in YEAR_ONLY_PATTERN.finditer(line):
        add_candidate(
            match.start(),
            match.end(),
            build_time_candidate(
                year=int(match.group("year")),
                month=None,
                day=None,
                source="body_explicit_year",
                matched_text=match.group(0),
                line_number=line_number,
                confidence=0.75,
            ),
        )

    for match in RELATIVE_YEAR_PATTERN.finditer(line):
        relative_candidate = _build_relative_year_candidate(
            match=match,
            title_info=title_info,
            current_year=current_year,
            line_number=line_number,
        )
        if relative_candidate is not None:
            add_candidate(match.start(), match.end(), relative_candidate)

    for match in MONTH_DAY_RANGE_PATTERN.finditer(line):
        nearby_year = extract_nearby_year_hint(
            line=line,
            anchor_start=match.start(),
            anchor_end=match.end(),
        )
        inferred_year, tags, confidence = infer_year_for_month_day(
            month=(
                int(match.group("end_month"))
                if match.group("end_month")
                else int(match.group("start_month"))
            ),
            title_info=title_info,
            current_year=current_year,
            previous_month=previous_month,
            inline_year=nearby_year,
        )
        if inferred_year is None:
            continue
        add_candidate(
            match.start(),
            match.end(),
            build_time_candidate(
                year=inferred_year,
                month=(
                    int(match.group("end_month"))
                    if match.group("end_month")
                    else int(match.group("start_month"))
                ),
                day=int(match.group("end_day")),
                source="body_inferred_range",
                matched_text=match.group(0),
                line_number=line_number,
                confidence=confidence,
                tags=tags,
            ),
        )

    line_month_hint = infer_line_month_hint(
        line=line, collected=collected, previous_month=previous_month
    )
    for match in DAY_RANGE_PATTERN.finditer(line):
        if overlaps((match.start(), match.end()), occupied):
            continue
        nearby_year = extract_nearby_year_hint(
            line=line,
            anchor_start=match.start(),
            anchor_end=match.end(),
        )
        inferred_year, tags, confidence = infer_year_for_month_day(
            month=line_month_hint,
            title_info=title_info,
            current_year=current_year,
            previous_month=previous_month,
            inline_year=nearby_year,
        )
        if inferred_year is None or line_month_hint is None:
            continue
        add_candidate(
            match.start(),
            match.end(),
            build_time_candidate(
                year=inferred_year,
                month=line_month_hint,
                day=int(match.group("end_day")),
                source="body_inferred_day_range",
                matched_text=match.group(0),
                line_number=line_number,
                confidence=confidence - 0.03,
                tags=tags + ["day_range_context_month"],
            ),
        )

    for match in MONTH_DAY_PATTERN.finditer(line):
        nearby_year = extract_nearby_year_hint(
            line=line,
            anchor_start=match.start(),
            anchor_end=match.end(),
        )
        inferred_year, tags, confidence = infer_year_for_month_day(
            month=int(match.group("month")),
            title_info=title_info,
            current_year=current_year,
            previous_month=previous_month,
            inline_year=nearby_year,
        )
        if inferred_year is None:
            continue
        add_candidate(
            match.start(),
            match.end(),
            build_time_candidate(
                year=inferred_year,
                month=int(match.group("month")),
                day=int(match.group("day")),
                source="body_inferred_day",
                matched_text=match.group(0),
                line_number=line_number,
                confidence=confidence,
                tags=tags,
            ),
        )

    for match in SAME_DAY_PATTERN.finditer(line):
        if last_absolute_day is None:
            continue
        add_candidate(
            match.start(),
            match.end(),
            build_time_candidate(
                year=last_absolute_day.year,
                month=last_absolute_day.month,
                day=last_absolute_day.day,
                source="body_relative_same_day",
                matched_text=match.group(0),
                line_number=line_number,
                confidence=0.68,
                tags=["relative_same_day"],
            ),
        )

    for match in NEXT_DAY_PATTERN.finditer(line):
        if (
            last_absolute_day is None
            or last_absolute_day.day is None
            or last_absolute_day.month is None
        ):
            continue
        next_day = min(last_absolute_day.day + 1, 31)
        add_candidate(
            match.start(),
            match.end(),
            build_time_candidate(
                year=last_absolute_day.year,
                month=last_absolute_day.month,
                day=next_day,
                source="body_relative_next_day",
                matched_text=match.group(0),
                line_number=line_number,
                confidence=0.64,
                tags=["relative_next_day"],
            ),
        )

    for match in MONTH_ONLY_PATTERN.finditer(line):
        nearby_year = extract_nearby_year_hint(
            line=line,
            anchor_start=match.start(),
            anchor_end=match.end(),
        )
        inferred_year, tags, confidence = infer_year_for_month_day(
            month=int(match.group("month")),
            title_info=title_info,
            current_year=current_year,
            previous_month=previous_month,
            inline_year=nearby_year,
        )
        if inferred_year is None:
            continue
        add_candidate(
            match.start(),
            match.end(),
            build_time_candidate(
                year=inferred_year,
                month=int(match.group("month")),
                day=None,
                source="body_inferred_month",
                matched_text=match.group(0),
                line_number=line_number,
                confidence=max(confidence - 0.05, 0.35),
                tags=tags + ["month_only"],
            ),
        )

    return [item[1] for item in sorted(collected, key=lambda pair: pair[0])]


def infer_year_for_month_day(
    *,
    month: int | None,
    title_info: TitleTimeInfo,
    current_year: int | None,
    previous_month: int | None,
    inline_year: int | None,
) -> tuple[int | None, list[str], float]:
    """为只含月/日的候选推断年份。"""

    tags: list[str] = []
    if inline_year is not None:
        return inline_year, ["year_from_inline_hint"], 0.85

    if current_year is not None:
        inferred_year = current_year
        confidence = 0.78
        tags.append("year_from_context")
    elif title_info.start_year is not None:
        inferred_year = title_info.start_year
        confidence = 0.72 if not title_info.spans_multiple_years() else 0.55
        tags.append("year_fallback_from_title")
    elif title_info.end_year is not None:
        inferred_year = title_info.end_year
        confidence = 0.5
        tags.append("year_fallback_from_title")
    else:
        return None, [], 0.0

    if (
        month is not None
        and previous_month is not None
        and month <= previous_month - 3
        and title_info.end_year is not None
        and inferred_year < title_info.end_year
    ):
        inferred_year += 1
        confidence = max(confidence - 0.12, 0.35)
        tags.append("year_inferred_by_rollover")

    return inferred_year, tags, confidence


def extract_nearby_year_hint(
    *, line: str, anchor_start: int, anchor_end: int
) -> int | None:
    """为月/日表达寻找更可靠的邻近年份提示。"""

    best_year: int | None = None
    best_score = -1
    for match in YEAR_HINT_PATTERN.finditer(line):
        year = int(match.group("year"))
        trailing_context = line[match.end() : match.end() + 4]
        if HISTORICAL_YEAR_MARKER_PATTERN.match(trailing_context):
            continue
        if match.end() <= anchor_start:
            distance = anchor_start - match.end()
            if distance > 20:
                continue
            score = 120 - distance
        elif match.start() >= anchor_end:
            distance = match.start() - anchor_end
            if distance > 220:
                continue
            score = 260 - distance
        else:
            score = 130

        if score > best_score:
            best_score = score
            best_year = year
    return best_year


def infer_line_month_hint(
    *, line: str, collected: list[tuple[int, TimeCandidate]], previous_month: int | None
) -> int | None:
    """为“8日至10日”这类表达寻找所在月份。"""

    month_candidates = [
        candidate.month for _, candidate in collected if candidate.month is not None
    ]
    if month_candidates:
        return month_candidates[-1]
    match = re.search(r"(?<!\d)(?P<month>\d{1,2})月", line)
    if match:
        return int(match.group("month"))
    return previous_month


def candidate_updates_temporal_context(
    candidate: TimeCandidate, current_year: int | None
) -> bool:
    """决定一个候选是否应更新后续行的年份上下文。"""

    if candidate.source.startswith("body_relative"):
        return False
    if candidate.source == "body_explicit_year":
        return current_year is None
    return True


def overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    """判断正则命中区间是否与已处理区间重叠。"""

    start, end = span
    for occ_start, occ_end in occupied:
        if max(start, occ_start) < min(end, occ_end):
            return True
    return False


def dedupe_candidates(candidates: list[TimeCandidate]) -> list[TimeCandidate]:
    """去重，避免同一时间被多种规则重复提取。"""

    seen: set[tuple] = set()
    deduped: list[TimeCandidate] = []
    for candidate in candidates:
        key = (
            candidate.year,
            candidate.month,
            candidate.day,
            candidate.precision,
            candidate.normalized,
            candidate.source,
            candidate.line_number,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def dedupe_strings(values: list[str]) -> list[str]:
    """对字符串列表去重并保持顺序。"""

    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _build_relative_year_candidate(
    *,
    match: re.Match[str],
    title_info: TitleTimeInfo,
    current_year: int | None,
    line_number: int,
) -> TimeCandidate | None:
    base_year = current_year or title_info.start_year or title_info.end_year
    if base_year is None:
        return None

    kind = match.group("kind")
    year = base_year if kind == "今年" else base_year - 1
    suffix = match.group("suffix")
    month = int(match.group("month")) if match.group("month") else None
    day = int(match.group("day")) if match.group("day") else None

    if month is None and suffix == "初":
        month = 1
    if month is None and suffix == "底":
        month = 12

    return build_time_candidate(
        year=year,
        month=month,
        day=day,
        source="body_relative_year",
        matched_text=match.group(0),
        line_number=line_number,
        confidence=0.58 if kind == "去年" else 0.62,
        tags=["relative_year_expression"],
    )
