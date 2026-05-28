from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROMPTS_ROOT = Path(__file__).resolve().parent


def build_prompt_from_md(prompt_name: str, context: dict) -> list[dict]:
    """按统一规则构造 [system, user] 两条消息。"""

    system_path = PROMPTS_ROOT / "md" / f"{prompt_name}.md"
    if not system_path.exists():
        raise FileNotFoundError(f"提示词模板不存在: {system_path}")

    return [
        {"role": "system", "content": system_path.read_text(encoding="utf-8")},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]


def parse_json_object(text: str, prompt_name: str) -> dict[str, Any]:
    """解析模型返回的 JSON 对象。"""

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"提示词 {prompt_name} 输出必须是 JSON 对象。")
    return payload


def require_string_field(
    payload: dict[str, Any],
    field_name: str,
    prompt_name: str,
    allow_empty: bool = False,
) -> str:
    """读取必填字符串字段。"""

    value = payload.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"提示词 {prompt_name} 的字段 {field_name} 必须是字符串。")
    value = value.strip()
    if not allow_empty and not value:
        raise ValueError(f"提示词 {prompt_name} 的字段 {field_name} 不能为空。")
    return value


def require_optional_string_field(
    payload: dict[str, Any],
    field_name: str,
    prompt_name: str,
) -> str | None:
    """读取可选字符串字段。"""

    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"提示词 {prompt_name} 的字段 {field_name} 必须是字符串或 null。")
    stripped = value.strip()
    return stripped if stripped else None


def require_list_field(payload: dict[str, Any], field_name: str, prompt_name: str) -> list[Any]:
    """读取必填列表字段。"""

    value = payload.get(field_name)
    if not isinstance(value, list):
        raise ValueError(f"提示词 {prompt_name} 的字段 {field_name} 必须是列表。")
    return value


def require_dict_item(item: Any, prompt_name: str, field_name: str) -> dict[str, Any]:
    """校验列表项是否为 JSON 对象。"""

    if not isinstance(item, dict):
        raise ValueError(f"提示词 {prompt_name} 的字段 {field_name} 中每一项都必须是对象。")
    return item
