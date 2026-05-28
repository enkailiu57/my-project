from __future__ import annotations

import re
from collections.abc import Iterable
from math import ceil

from utils.io_utils import stable_hash8


def split_sentences(text: str) -> list[str]:
    """对中文文本做轻量句切分。

    这里先使用规则切分，后续如果需要更稳健的效果，
    可以替换为更强的分句器而不影响阶段接口。
    """

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    parts = re.split(r"(?<=[。！？!?；;])\s*", normalized)
    return [part.strip() for part in parts if part.strip()]


def normalize_alias_key(raw_str: str | None) -> str | None:
    """为 alias_map 查询统一字符串键。"""

    if raw_str is None:
        return None
    return raw_str.strip().lower()


def normalize_surface_text(raw_str: str) -> str:
    """对元素表面字符串做轻量清洗，保留可读性。"""

    return re.sub(r"\s+", " ", raw_str).strip()


def build_element_id(element_type: str, raw_str: str) -> str:
    """根据元素类型和原始字符串生成稳定 ID。"""

    prefix_map = {
        "entity": "ent",
        "relation": "rel",
        "location": "loc",
    }
    prefix = prefix_map[element_type]
    return f"{prefix}_{stable_hash8(raw_str)}"


def join_context_lines(lines: Iterable[str]) -> str:
    """把多条上下文拼接成单个传给 LLM 的上下文字符串。"""

    return "\n".join(line.strip() for line in lines if line.strip())


def estimate_token_count(text: str) -> int:
    """粗略估算文本 token 数。"""

    if not text.strip():
        return 0
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_tokens = len(re.findall(r"[A-Za-z0-9_]+", text))
    punctuation = len(re.findall(r"[，。！？；,.!?;:]", text))
    return max(1, chinese_chars + latin_tokens * 2 + ceil(punctuation / 2))


def build_context_with_budget(
    sentences: list[str], center_index: int, token_budget: int, window_size: int
) -> tuple[str, bool]:
    """按 token 预算为中心句构造上下文。"""

    full_context = join_context_lines(sentences)
    if estimate_token_count(full_context) <= token_budget:
        return full_context, False

    start = center_index
    end = center_index
    current = join_context_lines(sentences[start : end + 1])
    for step in range(1, max(1, window_size) + 1):
        left = center_index - step
        if left >= 0:
            candidate = join_context_lines(sentences[left : end + 1])
            if estimate_token_count(candidate) <= token_budget:
                start = left
                current = candidate

        right = center_index + step
        if right < len(sentences):
            candidate = join_context_lines(sentences[start : right + 1])
            if estimate_token_count(candidate) <= token_budget:
                end = right
                current = candidate

    return current, True


def lexical_tokenize(text: str) -> list[str]:
    """对文本做轻量词法切分，用于词法相似度检索。"""

    normalized = normalize_surface_text(text).lower()
    tokens: list[str] = []
    for piece in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_]+", normalized):
        if re.fullmatch(r"[a-z0-9_]+", piece):
            tokens.append(piece)
            continue
        tokens.extend(list(piece))
        if len(piece) > 1:
            tokens.extend(piece[index : index + 2] for index in range(len(piece) - 1))
    return tokens


def normalize_score_list(scores: list[float]) -> list[float]:
    """把分数列表缩放到 0 到 1 区间。"""

    if not scores:
        return []
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        return [1.0 if max_score > 0 else 0.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]
