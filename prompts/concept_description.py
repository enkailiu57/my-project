from __future__ import annotations

from typing import Any

from prompts.common import (
    build_prompt_from_md,
    parse_json_object,
)

PROMPT_NAME = "concept_description"
SCHEMA_VERSION = "structured_fields_v2"

PROMPT_NAMES_BY_TYPE = {
    "entity": "concept_description_entity",
    "relation": "concept_description_relation",
    "location": "concept_description_location",
}

FIELD_SCHEMAS: dict[str, tuple[str, ...]] = {
    "entity": ("主体类别", "核心角色"),
    "relation": ("关系类别", "核心议题", "战略作用"),
    "location": ("空间类型", "区位范围", "议题作用"),
}

FIELD_WEIGHTS: dict[str, dict[str, float]] = {
    "entity": {"主体类别": 0.45, "核心角色": 0.55},
    "relation": {"关系类别": 0.40, "核心议题": 0.40, "战略作用": 0.20},
    "location": {"空间类型": 0.35, "区位范围": 0.40, "议题作用": 0.25},
}


def build_prompt(context: dict) -> list[dict]:
    concept_type = str(context.get("concept_type", "")).strip()
    prompt_name = PROMPT_NAMES_BY_TYPE.get(concept_type)
    if prompt_name is None:
        raise ValueError(f"未知概念类型: {concept_type}")
    return build_prompt_from_md(prompt_name, context)


def _normalize_fields(payload: dict[str, Any]) -> tuple[str, dict[str, str]]:
    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, dict):
        raise ValueError(f"提示词 {PROMPT_NAME} 必须返回 fields 对象。")

    fields: dict[str, str] = {}
    for key, value in raw_fields.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, str):
            raise ValueError(f"提示词 {PROMPT_NAME} 的字段 {key} 必须是字符串。")
        normalized_key = key.strip()
        normalized_value = value.strip()
        if normalized_key and normalized_value:
            fields[normalized_key] = normalized_value

    matched_types = [
        concept_type
        for concept_type, field_names in FIELD_SCHEMAS.items()
        if all(field_name in fields for field_name in field_names)
    ]
    if not matched_types:
        expected = " / ".join(
            "、".join(field_names) for field_names in FIELD_SCHEMAS.values()
        )
        raise ValueError(f"提示词 {PROMPT_NAME} 的 fields 缺少必填字段: {expected}")
    if len(matched_types) > 1:
        raise ValueError(f"提示词 {PROMPT_NAME} 的 fields 混用了多个概念类型字段。")

    concept_type = matched_types[0]
    return concept_type, {
        field_name: fields[field_name] for field_name in FIELD_SCHEMAS[concept_type]
    }


def parse_response(text: str) -> dict[str, Any]:
    payload = parse_json_object(text, PROMPT_NAME)
    schema_type, fields = _normalize_fields(payload)
    return {
        "fields": fields,
        "description_schema": schema_type,
        "description_schema_version": SCHEMA_VERSION,
    }
