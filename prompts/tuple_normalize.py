from __future__ import annotations

import json
from typing import Any

from prompts.common import PROMPTS_ROOT, parse_json_object
from prompts.tuple_extract import (
    DOCUMENT_CONTEXT_PLACEHOLDER,
)

ACTION_ALIASES = {
    "replace": "replace",
    "reuse": "replace",
    "select": "replace",
    "替换": "replace",
    "复用": "replace",
    "选择": "replace",
    "keep": "keep",
    "retain": "keep",
    "new": "keep",
    "create": "keep",
    "保留": "keep",
    "新建": "keep",
    "创建": "keep",
}


def build_prompt(context: dict) -> list[dict]:
    system_path = PROMPTS_ROOT / "md" / "tuple_normalize.md"
    if not system_path.exists():
        raise FileNotFoundError(f"提示词模板不存在: {system_path}")

    document_context = context.get("document_context")
    if not isinstance(document_context, dict):
        raise ValueError("提示词 tuple_normalize 的 document_context 必须是对象。")

    system_template = system_path.read_text(encoding="utf-8")
    if DOCUMENT_CONTEXT_PLACEHOLDER not in system_template:
        raise ValueError("tuple_normalize.md 缺少篇章上下文占位符。")

    system_content = system_template.replace(
        DOCUMENT_CONTEXT_PLACEHOLDER,
        json.dumps(document_context, ensure_ascii=False, indent=2),
    )
    concept = context.get("concept")
    if not isinstance(concept, dict):
        raise ValueError("提示词 tuple_normalize 的 concept 必须是对象。")

    user_payload = {
        "当前请求句": str(context["sentence_text"]),
        "待规范化概念": concept,
        "语义近邻候选": context.get("similar_candidates", []),
    }
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def _first_present(payload: dict[str, Any], *field_names: str) -> Any:
    for field_name in field_names:
        if field_name in payload:
            return payload[field_name]
    return None


def _parse_action(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("提示词 tuple_normalize 的 替换指示符 必须是字符串。")
    normalized = ACTION_ALIASES.get(value.strip().lower()) or ACTION_ALIASES.get(
        value.strip()
    )
    if normalized not in {"replace", "keep"}:
        raise ValueError(
            "提示词 tuple_normalize 的 替换指示符 只能是 replace 或 keep。"
        )
    return normalized


def _parse_concept(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ValueError("提示词 tuple_normalize 的 规范化概念 必须是对象。")
    concept = _first_present(value, "概念", "concept")
    description = _first_present(value, "描述", "description")
    if not isinstance(concept, str) or not concept.strip():
        raise ValueError("提示词 tuple_normalize 的 规范化概念.概念 必须是非空字符串。")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("提示词 tuple_normalize 的 规范化概念.描述 必须是非空字符串。")
    return {"concept": concept.strip(), "description": description.strip()}


def parse_response(text: str) -> dict:
    payload = parse_json_object(text, "tuple_normalize")
    action = _parse_action(_first_present(payload, "替换指示符", "decision", "action"))
    concept = _parse_concept(
        _first_present(payload, "规范化概念", "selected_concept", "concept")
    )
    return {"action": action, "concept": concept}
