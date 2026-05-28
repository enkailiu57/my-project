from __future__ import annotations

import json
from typing import Any

from prompts.common import PROMPTS_ROOT, parse_json_object

PROMPT_NAME = "tuple_discourse_sort"


def build_prompt(context: dict) -> list[dict]:
    system_path = PROMPTS_ROOT / "md" / f"{PROMPT_NAME}.md"
    if not system_path.exists():
        raise FileNotFoundError(f"提示词模板不存在: {system_path}")

    document_file = context.get("document_file")
    document_text = context.get("document_text")
    tuple_file = context.get("tuple_file")
    tuple_rows = context.get("tuple_rows")

    if not isinstance(document_file, str) or not document_file.strip():
        raise ValueError(
            "提示词 tuple_discourse_sort 的 document_file 必须是非空字符串。"
        )
    if not isinstance(document_text, str) or not document_text.strip():
        raise ValueError(
            "提示词 tuple_discourse_sort 的 document_text 必须是非空字符串。"
        )
    if not isinstance(tuple_file, str) or not tuple_file.strip():
        raise ValueError("提示词 tuple_discourse_sort 的 tuple_file 必须是非空字符串。")
    if not isinstance(tuple_rows, list):
        raise ValueError("提示词 tuple_discourse_sort 的 tuple_rows 必须是列表。")

    user_payload = {
        "篇章文件": document_file.strip(),
        "篇章全文": document_text.strip(),
        "五元组文件": tuple_file.strip(),
        "待排序五元组": tuple_rows,
    }
    return [
        {"role": "system", "content": system_path.read_text(encoding="utf-8")},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
        },
    ]


def _first_present(payload: dict[str, Any], *field_names: str) -> Any:
    for field_name in field_names:
        if field_name in payload:
            return payload[field_name]
    return None


def parse_response(text: str) -> dict[str, Any]:
    payload = parse_json_object(text, PROMPT_NAME)
    tuple_ids = _first_present(
        payload,
        "sorted_tuple_ids",
        "ordered_tuple_ids",
        "tuple_id_order",
        "tuple_ids",
        "排序后tuple_id列表",
        "tuple_id列表",
    )
    if not isinstance(tuple_ids, list):
        raise ValueError(
            "提示词 tuple_discourse_sort 的输出必须包含 tuple_id 列表字段。"
        )

    parsed_ids: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(tuple_ids, start=1):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"提示词 tuple_discourse_sort 的第 {index} 个 tuple_id 必须是非空字符串。"
            )
        tuple_id = item.strip()
        if tuple_id in seen:
            raise ValueError(
                f"提示词 tuple_discourse_sort 输出存在重复 tuple_id: {tuple_id}"
            )
        seen.add(tuple_id)
        parsed_ids.append(tuple_id)

    return {"sorted_tuple_ids": parsed_ids}
