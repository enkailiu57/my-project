from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from utils.io_utils import stable_hash8
from utils.text_utils import normalize_surface_text, split_sentences


@dataclass(slots=True)
class RawNewsDocument:
    """原始新闻在 S0 阶段的统一表示。"""

    doc_id: str
    title: str | None
    source_path: Path
    raw_text: str
    normalized_text: str
    sentences: list[str]


@dataclass(slots=True)
class CleanupStat:
    """单条清洗规则的命中统计。"""

    description: str
    count: int


NEWS_CODE_PATTERN = re.compile(r"^(?P<prefix>[A-Za-z]+)(?P<number>\d{3})(?:\b|[^0-9])")
CITATION_BRACKET_PATTERN = re.compile(
    r"\[(?:注\s*)?[0-9０-９]+(?:\s*[-—–~～至]\s*[0-9０-９]+)?\]"
)
EMPTY_BRACKET_PATTERN = re.compile(r"[\[【]\s*[\]】]")
LEADING_NOISE_QUOTE_PATTERN = re.compile(r'^[”＂"]+')
SPACE_BEFORE_PUNCT_PATTERN = re.compile(r"\s+([，。！？；：,.!?])")
MULTI_SPACE_PATTERN = re.compile(r"[ \t]{2,}")
TW_LEADER_NAMES = (
    "蔡英文",
    "赖清德",
    "马英九",
    "陈水扁",
    "李登辉",
    "陈建仁",
    "萧美琴",
    "吕秀莲",
    "连战",
    "宋楚瑜",
)
TW_CONTEXT_HINTS = TW_LEADER_NAMES + (
    "台当局",
    "台湾地区",
    "台湾方面",
    "美国在台协会",
    "访台",
    "过境美国",
    "友邦",
    "民进党",
)
TW_LEADER_PATTERN = "|".join(
    re.escape(name) for name in sorted(TW_LEADER_NAMES, key=len, reverse=True)
)
LINE_OFFICE_PREFIX_PATTERN = r"(^|[、，,（(\s]|和|及|与|但|并|而|不过|说明)"
TW_LEADER_TITLE_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(
            rf"(?:中华民国|台湾(?:地区)?|台湾民选)?(?:前|现任|候任|当选|新任)?[“\"＂]?总统[”\"＂]?(?P<name>{TW_LEADER_PATTERN})"
        ),
        r"\g<name>",
        "删除台湾地区领导人头衔",
    ),
    (
        re.compile(
            rf"(?P<name>{TW_LEADER_PATTERN})(?:前|现任|候任|当选|新任)?[“\"＂]?总统[”\"＂]?"
        ),
        r"\g<name>",
        "删除台湾地区领导人头衔",
    ),
)
TW_WORD_REPLACEMENTS: tuple[tuple[str, str, str], ...] = (
    ("中华民国总统府发言人", "台湾地区领导人办公室发言人", "替换总统府称谓"),
    ("中华民国总统府秘书长", "台湾地区领导人办公室秘书长", "替换总统府称谓"),
    ("中华民国总统府幕僚", "台湾地区领导人办公室幕僚", "替换总统府称谓"),
    ("中华民国总统府", "台湾地区领导人办公室", "替换总统府称谓"),
    ("台“总统府”", "台湾地区领导人办公室", "替换总统府称谓"),
    ("台“外交部长”", "台湾地区外事部门负责人", "替换外交系统称谓"),
    ("台“外交部次长”", "台湾地区外事部门官员", "替换外交系统称谓"),
    ("中华民国外交部部长", "台湾地区外事部门负责人", "替换外交系统称谓"),
    ("中华民国外交部长", "台湾地区外事部门负责人", "替换外交系统称谓"),
    ("中华民国外交部发言人", "台湾地区外事部门发言人", "替换外交系统称谓"),
    ("中华民国外交部", "台湾地区外事部门", "替换外交系统称谓"),
    ("台“外交部”", "台湾地区外事部门", "替换外交系统称谓"),
    ("台“国防部”", "台湾地区防务部门", "替换防务系统称谓"),
    ("台“立法院”", "台湾地区立法机构", "替换立法机构称谓"),
    ("行政院大陆委员会", "台湾地区大陆事务主管部门", "替换涉台机构称谓"),
    ("陆委会", "台湾地区大陆事务主管部门", "替换涉台机构称谓"),
    ("中华民国政府", "台湾当局", "替换当局称谓"),
    ("中华民国方面", "台湾方面", "替换地区称谓"),
    ("中华民国访团", "台湾地区访团", "替换地区称谓"),
    ("中华民国副总统", "台湾地区副领导人", "替换涉台职务称谓"),
    ("中华民国总统", "台湾地区领导人", "替换涉台职务称谓"),
    ("中华民国是主权国家", "台湾是中国的一部分", "替换涉台不当主权表述"),
    ("中华民国为主权独立的国家", "台湾是中国的一部分", "替换涉台不当主权表述"),
    ("台湾民选总统", "台湾地区领导人", "替换涉台职务称谓"),
    ("台湾领袖", "台湾地区领导人", "替换涉台职务称谓"),
    ("台湾领导人", "台湾地区领导人", "替换涉台职务称谓"),
    ("台湾的副总统", "台湾地区副领导人", "替换涉台职务称谓"),
    ("台湾的总统", "台湾地区领导人", "替换涉台职务称谓"),
    ("台湾副总统", "台湾地区副领导人", "替换涉台职务称谓"),
    ("台湾总统", "台湾地区领导人", "替换涉台职务称谓"),
    ("台湾地区是主权国家", "台湾是中国的一部分", "替换涉台不当主权表述"),
    ("台湾是主权国家", "台湾是中国的一部分", "替换涉台不当主权表述"),
    ("台湾为主权独立的国家", "台湾是中国的一部分", "替换涉台不当主权表述"),
    ("中华民国", "台湾地区", "替换地区称谓"),
)
TW_CONTEXT_LINE_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(rf"{LINE_OFFICE_PREFIX_PATTERN}总统府发言人"),
        r"\1台湾地区领导人办公室发言人",
        "替换总统府称谓",
    ),
    (
        re.compile(rf"{LINE_OFFICE_PREFIX_PATTERN}(?:代理)?总统府秘书长"),
        r"\1台湾地区领导人办公室秘书长",
        "替换总统府称谓",
    ),
    (
        re.compile(rf"{LINE_OFFICE_PREFIX_PATTERN}总统府"),
        r"\1台湾地区领导人办公室",
        "替换总统府称谓",
    ),
    (
        re.compile(rf"{LINE_OFFICE_PREFIX_PATTERN}外交部部长"),
        r"\1台湾地区外事部门负责人",
        "替换外交系统称谓",
    ),
    (
        re.compile(rf"{LINE_OFFICE_PREFIX_PATTERN}外交部长"),
        r"\1台湾地区外事部门负责人",
        "替换外交系统称谓",
    ),
    (
        re.compile(rf"{LINE_OFFICE_PREFIX_PATTERN}外交部次长"),
        r"\1台湾地区外事部门官员",
        "替换外交系统称谓",
    ),
    (
        re.compile(rf"{LINE_OFFICE_PREFIX_PATTERN}立法院"),
        r"\1台湾地区立法机构",
        "替换立法机构称谓",
    ),
    (
        re.compile(rf"{LINE_OFFICE_PREFIX_PATTERN}行政院"),
        r"\1台湾地区行政机构",
        "替换涉台机构称谓",
    ),
    (
        re.compile(rf"{LINE_OFFICE_PREFIX_PATTERN}总统(?=[与要会过出本])"),
        r"\1台湾地区领导人",
        "替换涉台职务称谓",
    ),
)


def _append_cleanup_stat(
    stats: list[CleanupStat] | None, description: str, count: int
) -> None:
    """在命中次数大于 0 时记录清洗日志。"""

    if stats is not None and count > 0:
        stats.append(CleanupStat(description=description, count=count))


def _apply_regex_replacement(
    text: str,
    pattern: re.Pattern[str],
    repl: str,
    description: str,
    stats: list[CleanupStat] | None,
) -> str:
    """执行正则替换，并在有命中时记录统计。"""

    updated_text, count = pattern.subn(repl, text)
    _append_cleanup_stat(stats, description, count)
    return updated_text


def _apply_literal_replacement(
    text: str,
    source: str,
    target: str,
    description: str,
    stats: list[CleanupStat] | None,
) -> str:
    """执行字面量替换，并在有命中时记录统计。"""

    count = text.count(source)
    if count <= 0:
        return text
    _append_cleanup_stat(stats, description, count)
    return text.replace(source, target)


def _merge_cleanup_stats(stats: list[CleanupStat]) -> list[CleanupStat]:
    """合并同名规则的统计，便于脚本输出更紧凑的日志。"""

    merged: dict[str, int] = {}
    ordered_descriptions: list[str] = []
    for item in stats:
        if item.description not in merged:
            ordered_descriptions.append(item.description)
            merged[item.description] = 0
        merged[item.description] += item.count
    return [
        CleanupStat(description=description, count=merged[description])
        for description in ordered_descriptions
    ]


def _contains_taiwan_context(line: str) -> bool:
    """判断一行文本是否明显处于涉台语境。"""

    return any(token in line for token in TW_CONTEXT_HINTS)


def _normalize_cleaned_lines(text: str, stats: list[CleanupStat] | None) -> str:
    """清理空行与行内多余空白。"""

    removed_empty_lines = 0
    normalized_lines: list[str] = []
    for line in text.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            removed_empty_lines += 1
            continue
        stripped_line = MULTI_SPACE_PATTERN.sub(" ", stripped_line)
        normalized_lines.append(stripped_line)
    _append_cleanup_stat(stats, "删除空行", removed_empty_lines)
    return "\n".join(normalized_lines).strip()


def clean_illegal_characters(text: str) -> str:
    """清理空行、脚注中括号与悬空引号等无意义字符。"""

    return clean_illegal_characters_with_stats(text)[0]


def clean_illegal_words(text: str) -> str:
    """清理涉台不规范称谓，统一到一个中国原则表述。"""

    return clean_illegal_words_with_stats(text)[0]


def clean_illegal_characters_with_stats(text: str) -> tuple[str, list[CleanupStat]]:
    """执行字符层清洗，并返回命中统计。"""

    stats: list[CleanupStat] = []
    cleaned = text
    cleaned = _apply_regex_replacement(
        cleaned,
        CITATION_BRACKET_PATTERN,
        "",
        "删除引用中括号",
        stats,
    )
    cleaned = _apply_regex_replacement(
        cleaned,
        EMPTY_BRACKET_PATTERN,
        "",
        "删除空白中括号",
        stats,
    )
    cleaned = _apply_regex_replacement(
        cleaned,
        LEADING_NOISE_QUOTE_PATTERN,
        "",
        "删除文首悬空引号",
        stats,
    )
    cleaned = _apply_regex_replacement(
        cleaned,
        SPACE_BEFORE_PUNCT_PATTERN,
        r"\1",
        "删除标点前多余空格",
        stats,
    )
    cleaned = _normalize_cleaned_lines(cleaned, stats)
    return cleaned, _merge_cleanup_stats(stats)


def clean_illegal_words_with_stats(text: str) -> tuple[str, list[CleanupStat]]:
    """执行词汇层清洗，并返回命中统计。"""

    stats: list[CleanupStat] = []
    cleaned = text
    for pattern, repl, description in TW_LEADER_TITLE_RULES:
        cleaned = _apply_regex_replacement(cleaned, pattern, repl, description, stats)
    for source, target, description in TW_WORD_REPLACEMENTS:
        cleaned = _apply_literal_replacement(
            cleaned, source, target, description, stats
        )

    rewritten_lines: list[str] = []
    for line in cleaned.splitlines():
        current_line = line
        if _contains_taiwan_context(current_line):
            # 仅在明显涉台语境中启用行内称谓改写，避免误伤其他国家机构名。
            for pattern, repl, description in TW_CONTEXT_LINE_RULES:
                current_line = _apply_regex_replacement(
                    current_line, pattern, repl, description, stats
                )
        rewritten_lines.append(current_line)

    cleaned = "\n".join(rewritten_lines)
    return cleaned, _merge_cleanup_stats(stats)


def clean_news_text(text: str) -> tuple[str, list[CleanupStat]]:
    """执行完整新闻清洗流程，返回清洗后文本与日志统计。"""

    normalized = normalize_raw_news_text(text)
    char_cleaned, char_stats = clean_illegal_characters_with_stats(normalized)
    word_cleaned, word_stats = clean_illegal_words_with_stats(char_cleaned)
    final_text = _normalize_cleaned_lines(word_cleaned, stats=None)
    return final_text, _merge_cleanup_stats([*char_stats, *word_stats])


def extract_news_code(file_path: str | Path) -> tuple[str, int] | None:
    """从新闻文件名中提取形如 D001 / E014 的前导编号。"""

    stem = Path(file_path).stem.strip()
    match = NEWS_CODE_PATTERN.match(stem)
    if not match:
        return None
    return match.group("prefix").upper(), int(match.group("number"))


def _parse_scope_token(token: str) -> tuple[str, int]:
    code = extract_news_code(token)
    if code is None:
        raise ValueError(
            f"无法解析新闻范围标记: {token}。支持 all、D001、E001-E014、D001,E001-E014。"
        )
    return code


def _build_scope_rules(scope: str) -> list[tuple[str, str, int, int]]:
    """把范围字符串解析成规则列表。"""

    normalized_scope = (scope or "all").strip()
    if not normalized_scope or normalized_scope.lower() == "all":
        return [("all", "", 0, 0)]

    rules: list[tuple[str, str, int, int]] = []
    for raw_part in normalized_scope.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_token, end_token = (item.strip() for item in part.split("-", 1))
            start_prefix, start_number = _parse_scope_token(start_token)
            end_prefix, end_number = _parse_scope_token(end_token)
            if start_prefix != end_prefix:
                raise ValueError(
                    f"新闻范围不支持跨前缀区间: {part}。请拆成多个片段，例如 D001,D003 或 E001-E014。"
                )
            lower, upper = sorted((start_number, end_number))
            rules.append(("range", start_prefix, lower, upper))
            continue

        prefix, number = _parse_scope_token(part)
        rules.append(("single", prefix, number, number))

    if not rules:
        raise ValueError("新闻范围不能为空。")
    return rules


def select_news_files(file_paths: list[Path], scope: str = "all") -> list[Path]:
    """按范围参数筛选新闻文件。"""

    sorted_paths = sorted(file_paths)
    rules = _build_scope_rules(scope)
    if rules[0][0] == "all":
        return sorted_paths

    selected: list[Path] = []
    for file_path in sorted_paths:
        code = extract_news_code(file_path)
        if code is None:
            continue
        prefix, number = code
        for rule_type, rule_prefix, lower, upper in rules:
            if prefix != rule_prefix:
                continue
            if rule_type == "single" and number == lower:
                selected.append(file_path)
                break
            if rule_type == "range" and lower <= number <= upper:
                selected.append(file_path)
                break
    return selected


def build_doc_id_from_path(file_path: str | Path) -> str:
    """根据新闻文件路径构造稳定、较短的 doc_id。"""

    stem = Path(file_path).stem
    match = re.match(r"[A-Za-z0-9+_-]+", stem)
    prefix = match.group(0).lower() if match else "doc"
    prefix = prefix.replace("+", "_plus_").replace("-", "_")
    prefix = re.sub(r"[^a-z0-9_]+", "_", prefix).strip("_") or "doc"
    return f"{prefix}_{stable_hash8(stem)}"


def resolve_raw_news_dir(data_dir: str | Path) -> Path:
    """解析原始新闻目录。

    新约定下原始新闻放在 data/sources。
    为兼容旧目录结构，若未发现 sources 子目录，则回退到传入目录本身。
    """

    root = Path(data_dir)
    sources_dir = root / "sources"
    if sources_dir.exists():
        return sources_dir
    return root


def list_raw_news_files(data_dir: str | Path, scope: str = "all") -> list[Path]:
    """列出指定目录下符合范围的新闻 txt 文件。"""

    root = resolve_raw_news_dir(data_dir)
    raw_files = [path for path in root.glob("*.txt") if path.is_file()]
    return select_news_files(raw_files, scope)


def resolve_news_preprocess_input_files(
    data_dir: str | Path, scope: str = "all"
) -> tuple[list[Path], str]:
    """为 S0 选择输入新闻文件。

    默认优先消费 data/cleaned 下的清洗结果；
    若没有匹配文件，则回退到 data/sources 下的原始新闻。
    为兼容旧目录结构，当 data/sources 不存在时，再回退到 data 根目录。
    """

    root = Path(data_dir)
    cleaned_dir = root / "cleaned"
    if cleaned_dir.exists():
        cleaned_files = list_raw_news_files(cleaned_dir, scope)
        if cleaned_files:
            return cleaned_files, "cleaned"

    raw_dir = resolve_raw_news_dir(root)
    return list_raw_news_files(raw_dir, scope), (
        "sources" if raw_dir.name == "sources" else "raw"
    )


def normalize_raw_news_text(text: str) -> str:
    """做与业务无关的基础文本归一化。"""

    normalized = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u3000", " ")
    normalized = re.sub(r"[\t ]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def split_news_sentences(text: str) -> list[str]:
    """把新闻正文拆成句子列表，供 news_process 提示词使用。"""

    sentences: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for sentence in split_sentences(stripped):
            normalized = normalize_surface_text(sentence)
            if normalized:
                sentences.append(normalized)
    return sentences


def parse_raw_news_file(file_path: str | Path) -> RawNewsDocument:
    """把原始 txt 新闻解析为 S0 阶段输入对象。"""

    source_path = Path(file_path)
    raw_text = source_path.read_text(encoding="utf-8")
    normalized_text = normalize_raw_news_text(raw_text)
    return RawNewsDocument(
        doc_id=build_doc_id_from_path(source_path),
        title=source_path.stem or None,
        source_path=source_path,
        raw_text=raw_text,
        normalized_text=normalized_text,
        sentences=split_news_sentences(normalized_text),
    )


def render_processed_news_text(timeline: list[dict]) -> str:
    """把 news_process 的结构化输出渲染为下游可直接消费的文本。"""

    texts: list[str] = []
    sorted_timeline = sorted(
        timeline,
        key=lambda item: (
            int(item.get("event_id", 0)),
            normalize_surface_text(str(item.get("text", ""))),
        ),
    )
    for item in sorted_timeline:
        text = normalize_surface_text(str(item.get("text", "")))
        if not text or text in texts:
            continue
        texts.append(text)
    return "\n".join(texts).strip()
