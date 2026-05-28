from __future__ import annotations

from typing import Any

from prompts.common import (
    build_prompt_from_md,
    parse_json_object,
    require_string_field,
)

PROMPT_NAME = "concept_compliance"


def build_prompt(context: dict) -> list[dict]:
    return build_prompt_from_md(PROMPT_NAME, context)


def parse_response(text: str) -> dict[str, Any]:
    payload = parse_json_object(text, PROMPT_NAME)
    verdict = require_string_field(payload, "verdict", PROMPT_NAME)
    if verdict not in {"合规", "不合规"}:
        raise ValueError("提示词 concept_compliance 的 verdict 只能是 合规 或 不合规。")
    reason = require_string_field(payload, "reason", PROMPT_NAME, allow_empty=True)
    return {"verdict": verdict, "reason": reason}
