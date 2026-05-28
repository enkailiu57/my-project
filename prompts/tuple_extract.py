from __future__ import annotations

import json
from typing import Any

from prompts.common import PROMPTS_ROOT, parse_json_object, require_list_field

DOCUMENT_CONTEXT_PLACEHOLDER = "{{篇章上下文}}"


def build_prompt(context: dict) -> list[dict]:
    system_path = PROMPTS_ROOT / "md" / "tuple_extract.md"
    if not system_path.exists():
        raise FileNotFoundError(f"提示词模板不存在: {system_path}")

    document_context = context.get("document_context")
    if not isinstance(document_context, dict):
        raise ValueError("提示词 tuple_extract 的 document_context 必须是对象。")

    system_template = system_path.read_text(encoding="utf-8")
    if DOCUMENT_CONTEXT_PLACEHOLDER not in system_template:
        raise ValueError("tuple_extract.md 缺少篇章上下文占位符。")

    system_content = system_template.replace(
        DOCUMENT_CONTEXT_PLACEHOLDER,
        json.dumps(document_context, ensure_ascii=False, indent=2),
    )
    extraction_plan = context["extraction_plan"]
    user_payload = {
        "当前请求句": str(extraction_plan.get("原句", "")),
        "抽取规划": extraction_plan,
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


def _parse_concept(
    value: Any, field_name: str, allow_empty: bool = False
) -> dict | None:
    if value is None:
        if allow_empty:
            return None
        raise ValueError(f"提示词 tuple_extract 的字段 {field_name} 不能为空。")

    if not isinstance(value, dict):
        if allow_empty and value == "":
            return None
        raise ValueError(
            f"提示词 tuple_extract 的字段 {field_name} 必须是包含 概念 和 描述 的对象。"
        )

    concept = _first_present(value, "概念", "concept")
    if not isinstance(concept, str):
        raise ValueError(
            f"提示词 tuple_extract 的字段 {field_name}.概念 必须是字符串。"
        )
    concept = concept.strip()
    if not concept and allow_empty:
        return None
    if not concept:
        raise ValueError(f"提示词 tuple_extract 的字段 {field_name}.概念 不能为空。")

    description = _first_present(value, "描述", "description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            f"提示词 tuple_extract 的字段 {field_name}.描述 必须是非空字符串。"
        )

    return {"concept": concept, "description": description.strip()}


def _parse_concept_or_list(value: Any, field_name: str) -> dict | list[dict]:
    if isinstance(value, list):
        if not value:
            raise ValueError(f"提示词 tuple_extract 的字段 {field_name} 对象数组不能为空。")
        return [
            _parse_concept(item, f"{field_name}[{index}]")
            for index, item in enumerate(value, start=1)
        ]
    return _parse_concept(value, field_name)


def _parse_time(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("提示词 tuple_extract 的字段 时间 必须是非空字符串。")
    return value.strip()


def _parse_confidence(value: Any) -> float:
    if value is None:
        return 0.8
    if not isinstance(value, (float, int)):
        raise ValueError("提示词 tuple_extract 的字段 置信度 必须是数值。")
    confidence = float(value)
    if confidence < 0 or confidence > 1:
        raise ValueError("提示词 tuple_extract 的字段 置信度 必须位于 0 到 1 之间。")
    return confidence


def parse_response(text: str) -> dict:
    payload = parse_json_object(text, "tuple_extract")
    tuple_items = _first_present(payload, "tuples", "五元组", "事件五元组")
    if tuple_items is None:
        tuple_items = require_list_field(payload, "tuples", "tuple_extract")
    if not isinstance(tuple_items, list):
        raise ValueError("提示词 tuple_extract 的 tuples 必须是列表。")

    parsed: list[dict] = []
    for index, item in enumerate(tuple_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"提示词 tuple_extract 的第 {index} 个 tuple 必须是对象。")

        subject = _parse_concept_or_list(
            _first_present(item, "主体", "subject"),
            "主体",
        )
        relation = _parse_concept(_first_present(item, "关系", "relation"), "关系")
        object_ = _parse_concept_or_list(
            _first_present(item, "客体", "object"),
            "客体",
        )
        location = _parse_concept(
            _first_present(item, "地点", "location"),
            "地点",
            allow_empty=True,
        )
        parsed.append(
            {
                "subject": subject,
                "relation": relation,
                "object": object_,
                "time": _parse_time(_first_present(item, "时间", "time")),
                "location": location,
                "confidence": _parse_confidence(
                    _first_present(item, "置信度", "confidence")
                ),
            }
        )

    return {"tuples": parsed}
