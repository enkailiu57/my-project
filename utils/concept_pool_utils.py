from __future__ import annotations

import json
from pathlib import Path

from utils.text_utils import build_element_id, normalize_surface_text

CONCEPT_POOL_FILES = {
    "entity": "entity_pool.json",
    "relation": "relation_pool.json",
    "location": "location_pool.json",
}


def concept_pool_file(concept_dir: str | Path, element_type: str) -> Path:
    if element_type not in CONCEPT_POOL_FILES:
        raise ValueError(f"未知概念池类型: {element_type}")
    return Path(concept_dir) / CONCEPT_POOL_FILES[element_type]


def load_concept_pool_rows(concept_dir: str | Path, element_type: str) -> list[dict]:
    file_path = concept_pool_file(concept_dir, element_type)
    if not file_path.exists():
        raise FileNotFoundError(f"概念池文件不存在: {file_path}")

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"概念池文件必须是 JSON 对象: {file_path}")

    rows: list[dict] = []
    for concept, item in sorted(payload.items(), key=lambda pair: pair[0]):
        if not isinstance(concept, str) or not isinstance(item, dict):
            continue
        concept_text = normalize_surface_text(concept)
        if not concept_text:
            continue

        description = item.get("描述")
        source_sentences = item.get("来源句")
        contexts: list[str] = []
        seen_contexts: set[str] = set()
        if isinstance(source_sentences, list):
            for raw_context in source_sentences:
                if not isinstance(raw_context, str):
                    continue
                context = normalize_surface_text(raw_context)
                if not context or context in seen_contexts:
                    continue
                seen_contexts.add(context)
                contexts.append(context)

        rows.append(
            {
                "raw_str": concept_text,
                "type": element_type,
                "id": build_element_id(element_type, concept_text),
                "count": len(contexts),
                "contexts": contexts[:5],
                "description": (
                    normalize_surface_text(description)
                    if isinstance(description, str)
                    and normalize_surface_text(description)
                    else concept_text
                ),
            }
        )

    return rows
