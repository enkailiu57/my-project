from __future__ import annotations

import json

from prompts.common import (
    PROMPTS_ROOT,
    parse_json_object,
    require_dict_item,
    require_list_field,
    require_string_field,
)

ALLOWED_SENTENCE_TYPES = {"事件句", "混合句", "不抽取句"}
ALLOWED_UNIT_TYPES = {"事实事件", "重要表态事件", "计划事件"}
SENTENCE_TYPE_ALIASES = {
    "事实事件": "事件句",
    "重要表态事件": "事件句",
    "计划事件": "事件句",
    "事实句": "事件句",
    "重要表态句": "事件句",
    "计划句": "事件句",
    "非事件句": "不抽取句",
    "无事件句": "不抽取句",
    "无需抽取句": "不抽取句",
}
ALLOWED_ACTIONS = {
    "单独抽取",
    "按重要表态抽取，提炼核心内容和对外信号",
    "按计划抽取，不作已发生事实",
}
DOCUMENT_SENTENCE_PLACEHOLDER = "{{待分析篇章分句列表}}"


def _normalize_sentence_type(value: str) -> str:
    normalized = value.strip()
    if normalized in ALLOWED_SENTENCE_TYPES:
        return normalized

    mapped = SENTENCE_TYPE_ALIASES.get(normalized, normalized)
    if mapped in ALLOWED_SENTENCE_TYPES:
        return mapped

    if "混合" in normalized:
        return "混合句"
    if any(
        token in normalized
        for token in ("不抽取", "无需抽取", "无需提取", "非事件", "无事件")
    ):
        return "不抽取句"
    if any(token in normalized for token in ("事件", "表态", "计划", "事实")):
        return "事件句"

    raise ValueError(f"提示词 extract_plan 输出了未知句子类型: {value}")


def build_prompt(context: dict) -> list[dict]:
    system_path = PROMPTS_ROOT / "md" / "extract_plan.md"
    if not system_path.exists():
        raise FileNotFoundError(f"提示词模板不存在: {system_path}")

    document_sentences = context.get("document_sentences")
    if not isinstance(document_sentences, list):
        raise ValueError("提示词 extract_plan 的 document_sentences 必须是列表。")

    system_template = system_path.read_text(encoding="utf-8")
    if DOCUMENT_SENTENCE_PLACEHOLDER not in system_template:
        raise ValueError("extract_plan.md 缺少待分析篇章分句列表占位符。")

    system_content = system_template.replace(
        DOCUMENT_SENTENCE_PLACEHOLDER,
        json.dumps(document_sentences, ensure_ascii=False, indent=2),
    )
    user_payload = {
        "句子编号": str(context["sentence_id"]),
        "当前请求句": str(context["sentence_text"]),
    }
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def parse_response(text: str) -> dict:
    payload = parse_json_object(text, "extract_plan")
    sentence_type = _normalize_sentence_type(
        require_string_field(payload, "句子类型", "extract_plan")
    )

    parsed_units: list[dict] = []
    for item in require_list_field(payload, "事件单元", "extract_plan"):
        row = require_dict_item(item, "extract_plan", "事件单元")
        unit_type = require_string_field(row, "单元类型", "extract_plan").strip()
        if unit_type not in ALLOWED_UNIT_TYPES:
            raise ValueError(f"提示词 extract_plan 输出了未知单元类型: {unit_type}")
        action = require_string_field(row, "抽取动作建议", "extract_plan").strip()
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"提示词 extract_plan 输出了未知抽取动作建议: {action}")
        parsed_units.append(
            {
                "编号": require_string_field(row, "编号", "extract_plan"),
                "文本片段": require_string_field(row, "文本片段", "extract_plan"),
                "单元类型": unit_type,
                "分析方法建议": require_string_field(
                    row, "分析方法建议", "extract_plan"
                ),
                "抽取动作建议": action,
            }
        )

    return {
        "句子编号": require_string_field(payload, "句子编号", "extract_plan"),
        "原句": require_string_field(payload, "原句", "extract_plan"),
        "句子类型": sentence_type,
        "事件分析": require_string_field(payload, "事件分析", "extract_plan"),
        "事件单元": parsed_units,
    }
