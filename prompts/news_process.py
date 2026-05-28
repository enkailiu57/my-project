from __future__ import annotations

from prompts.common import (
    build_prompt_from_md,
    parse_json_object,
    require_dict_item,
    require_list_field,
    require_string_field,
)


def build_prompt(context: dict) -> list[dict]:
    """构造新闻预处理提示词请求。"""

    return build_prompt_from_md("news_process", context)


def parse_response(text: str) -> dict:
    """解析新闻预处理提示词输出。"""

    payload = parse_json_object(text, "news_process")
    timeline = require_list_field(payload, "timeline", "news_process")
    parsed_rows: list[dict] = []
    for item in timeline:
        row = require_dict_item(item, "news_process", "timeline")
        event_id = row.get("event_id")
        if not isinstance(event_id, (int, float)):
            raise ValueError("提示词 news_process 的 event_id 必须是整数。")
        if int(event_id) != event_id or int(event_id) < 0:
            raise ValueError("提示词 news_process 的 event_id 必须是非负整数。")

        source_sentence_ids = row.get("source_sentence_ids")
        if not isinstance(source_sentence_ids, list):
            raise ValueError(
                "提示词 news_process 的 source_sentence_ids 必须是整数列表。"
            )

        parsed_source_ids: list[int] = []
        for source_id in source_sentence_ids:
            if not isinstance(source_id, (int, float)):
                raise ValueError(
                    "提示词 news_process 的 source_sentence_ids 必须是整数列表。"
                )
            if int(source_id) != source_id or int(source_id) < 0:
                raise ValueError(
                    "提示词 news_process 的 source_sentence_ids 中存在非法值。"
                )
            parsed_source_ids.append(int(source_id))

        parsed_rows.append(
            {
                "event_id": int(event_id),
                "text": require_string_field(row, "text", "news_process"),
                "source_sentence_ids": parsed_source_ids,
            }
        )
    return {"timeline": parsed_rows}
