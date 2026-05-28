from prompts.common import build_prompt_from_md, parse_json_object, require_string_field


def build_prompt(context: dict) -> list[dict]:
    return build_prompt_from_md("canonical_select", context)


def parse_response(text: str) -> dict:
    payload = parse_json_object(text, "canonical_select")
    return {"canonical_str": require_string_field(payload, "canonical_str", "canonical_select")}
