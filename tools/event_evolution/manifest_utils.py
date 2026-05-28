from __future__ import annotations

import re
from pathlib import Path

from utils.news_preprocess_utils import build_doc_id_from_path

NODE_ID_PATTERN = re.compile(r"^(?P<code>[A-Za-z]+\d{3})(?:\b|\s)")
TRAILING_TIME_HINT_PATTERN = re.compile(r"\s*[（(](?P<raw>[^()（）]+)[)）]\s*$")
TRAILING_TIMELINE_SUFFIX_PATTERN = re.compile(r"\s*＜[^＜＞]+＞\s*$")

WINDOWS_FILENAME_REPLACEMENTS = str.maketrans(
    {
        "<": "＜",
        ">": "＞",
        ":": "：",
        '"': "＂",
        "/": "／",
        "\\": "＼",
        "|": "｜",
        "?": "？",
        "*": "＊",
    }
)


def extract_node_id(stem: str) -> str:
    """优先从文件名前缀抽取稳定节点编号。"""

    match = NODE_ID_PATTERN.match(stem)
    if match:
        return match.group("code")
    return build_doc_id_from_path(stem)


def split_title_stem(stem: str) -> tuple[str, str | None, bool]:
    """拆出无时间后缀的标题和原标题中的时间提示。"""

    cleaned_stem = TRAILING_TIMELINE_SUFFIX_PATTERN.sub("", stem).strip()
    match = TRAILING_TIME_HINT_PATTERN.search(cleaned_stem)
    if match:
        base_title = cleaned_stem[: match.start()].rstrip()
        raw_time_text = match.group("raw").strip() or None
        return base_title or cleaned_stem, raw_time_text, False

    malformed = "（" in cleaned_stem and "）" not in cleaned_stem
    return cleaned_stem, None, malformed


def sanitize_windows_stem(stem: str) -> str:
    """把文件 stem 规范化为 Windows 可写入的名称。"""

    sanitized = stem.translate(WINDOWS_FILENAME_REPLACEMENTS).strip().rstrip(".")
    return sanitized or "timeline_document"


def build_corrected_stem(base_title: str, time_label: str) -> str:
    """构造校正后的文件 stem。

    Windows 不允许文件名中直接包含 < 和 >，因此这里使用全角符号。
    """

    return sanitize_windows_stem(f"{base_title}＜{time_label}＞")


def build_corrected_path(
    output_dir: str | Path, base_title: str, time_label: str
) -> Path:
    """根据标题和时间标签生成目标 txt 路径。"""

    return Path(output_dir) / f"{build_corrected_stem(base_title, time_label)}.txt"
