from prompts.common import build_prompt_from_md, parse_json_object, require_list_field


def build_prompt(context: dict) -> list[dict]:
    return build_prompt_from_md("dedup_check", context)


def parse_response(text: str) -> dict:
    payload = parse_json_object(text, "dedup_check")
    duplicates = [str(item).strip() for item in require_list_field(payload, "duplicates", "dedup_check") if str(item).strip()]
    return {"duplicates": duplicates}
