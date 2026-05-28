from __future__ import annotations

import json
from typing import Any

from prompts.common import (
    build_prompt_from_md,
    parse_json_object,
    require_list_field,
)

PROMPT_NAME = "concept_merge"
PROMPT_NAMES_BY_TYPE = {
    "entity": "concept_merge_entity",
    "relation": "concept_merge_relation",
    "location": "concept_merge_location",
}


def build_prompt(context: dict) -> list[dict]:
    concept_type = str(context.get("concept_type") or "").strip()
    prompt_name = PROMPT_NAMES_BY_TYPE.get(concept_type)
    if not prompt_name:
        raise ValueError(f"未知概念类型，无法选择归并提示词: {concept_type}")
    return build_prompt_from_md(prompt_name, context)


def normalize_reason_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value).strip()


def parse_response(text: str) -> dict[str, Any]:
    payload = parse_json_object(text, PROMPT_NAME)
    merge_groups: list[dict[str, Any]] = []
    for item in require_list_field(payload, "merge_groups", PROMPT_NAME):
        if not isinstance(item, dict):
            raise ValueError(
                "提示词 concept_merge 的 merge_groups 每一项都必须是对象。"
            )
        raw_source_concepts = item.get("source_concepts", [])
        if not isinstance(raw_source_concepts, list):
            raise ValueError(
                "提示词 concept_merge 的 merge_groups.source_concepts 必须是列表。"
            )
        seen_sources: set[str] = set()
        source_concepts = [
            source
            for source in (str(source).strip() for source in raw_source_concepts)
            if source and source not in seen_sources and not seen_sources.add(source)
        ]
        merged_concept = str(item.get("merged_concept") or "").strip()
        merged_description = str(item.get("merged_description") or "").strip()
        if len(source_concepts) < 2 or not merged_concept or not merged_description:
            continue
        merge_groups.append(
            {
                "source_concepts": source_concepts,
                "merged_concept": merged_concept,
                "merged_description": merged_description,
                "reason": normalize_reason_text(item.get("reason")),
            }
        )

    return {
        "merge_groups": merge_groups,
        "reason": normalize_reason_text(payload.get("reason")),
    }
