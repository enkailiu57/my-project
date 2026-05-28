from prompts.common import build_prompt_from_md, parse_json_object, require_string_field


def build_prompt(context: dict) -> list[dict]:
    return build_prompt_from_md("describe_entity", context)


def parse_response(text: str) -> dict:
    payload = parse_json_object(text, "describe_entity")
    return {"description": require_string_field(payload, "description", "describe_entity")}
