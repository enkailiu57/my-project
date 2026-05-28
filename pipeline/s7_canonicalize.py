from __future__ import annotations

from datetime import datetime, timezone
from typing import cast

from core.types import ElementType, StageRuntime, UnresolvedAliasRecord
from utils.io_utils import read_json, read_jsonl, write_json, write_jsonl
from utils.text_utils import (
    build_element_id,
    normalize_alias_key,
    normalize_surface_text,
)


def _build_fallback_canonical(raw_str: str, element_type: str) -> dict:
    """为 alias_map 未命中的字符串构造兜底 canonical。"""

    canonical_str = normalize_surface_text(raw_str)
    alias_key = normalize_alias_key(canonical_str)
    if alias_key is None:
        raise RuntimeError(
            f"S7 无法为 {element_type} 构造 fallback canonical: {raw_str!r}"
        )

    label_map = {
        "entity": "实体",
        "relation": "关系",
        "location": "地点",
    }
    return {
        "cid": build_element_id(element_type, canonical_str),
        "canonical_str": canonical_str,
        "type": element_type,
        "description": f"{label_map[element_type]}“{canonical_str}”的兜底规范项，由 S7 在 alias_map 未命中时自动补写。",
        "aliases": [alias_key],
    }


def _merge_aliases(row: dict, alias_key: str) -> None:
    """把 alias 键合并进 canonical 记录。"""

    aliases = set(row.get("aliases", []))
    aliases.add(alias_key)
    row["aliases"] = sorted(aliases)


def _dedup_ratio(rows: list[dict]) -> float:
    """按 S6 的规则重新计算去重率。"""

    raw_total = sum(len(row.get("aliases", [])) for row in rows)
    if raw_total == 0:
        return 0.0
    return round(1.0 - (len(rows) / raw_total), 6)


def _refresh_schema_if_present(output_dir, canonical_rows: list[dict]) -> None:
    """当 S7 回写了 fallback canonical 后，同步刷新已有的 schema 快照。"""

    schema_path = output_dir / "s6_schema.json"
    if not schema_path.exists():
        return

    entities = [row for row in canonical_rows if row["type"] == "entity"]
    relations = [row for row in canonical_rows if row["type"] == "relation"]
    locations = [row for row in canonical_rows if row["type"] == "location"]
    write_json(
        schema_path,
        {
            "entities": entities,
            "relations": relations,
            "locations": locations,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stats": {
                "total_entities": len(entities),
                "total_relations": len(relations),
                "total_locations": len(locations),
                "entity_dedup_ratio": _dedup_ratio(entities),
                "relation_dedup_ratio": _dedup_ratio(relations),
                "location_dedup_ratio": _dedup_ratio(locations),
            },
        },
    )


def _resolve_with_fallback(
    raw_id: str,
    raw_str: str | None,
    element_type: str,
    alias_map: dict,
    canonical_map: dict[str, dict],
    unresolved: list[UnresolvedAliasRecord],
    strict: bool,
) -> tuple[str | None, bool]:
    """解析 alias_map，未命中时记录诊断信息。"""

    if raw_str is None:
        return None, False
    alias_key = normalize_alias_key(raw_str)
    if alias_key is None:
        return None, False
    found = alias_map.get(element_type, {}).get(alias_key)
    if found:
        return found, False
    unresolved.append(
        UnresolvedAliasRecord(
            raw_id=raw_id,
            element_type=cast(ElementType, element_type),
            raw_str=raw_str.strip(),
        )
    )
    if strict:
        return None, False

    fallback_row = _build_fallback_canonical(raw_str.strip(), element_type)
    alias_key = normalize_alias_key(raw_str)
    if alias_key is None:
        return None, False

    changed = False
    type_alias_map = alias_map.setdefault(element_type, {})
    if type_alias_map.get(alias_key) != fallback_row["cid"]:
        type_alias_map[alias_key] = fallback_row["cid"]
        changed = True

    existing_row = canonical_map.get(fallback_row["cid"])
    if existing_row is None:
        canonical_map[fallback_row["cid"]] = fallback_row
        changed = True
    else:
        before_aliases = set(existing_row.get("aliases", []))
        _merge_aliases(existing_row, alias_key)
        if set(existing_row.get("aliases", [])) != before_aliases:
            changed = True
    return fallback_row["cid"], changed


def run_stage(runtime: StageRuntime) -> None:
    """执行 S7 三元组规范化。"""

    alias_map = read_json(runtime.output_dir / "s5_alias_map.json")
    canonical_rows = list(read_jsonl(runtime.output_dir / "s5_canonical.jsonl"))
    canonical_map: dict[str, dict] = {
        str(row["cid"]): row.copy() for row in canonical_rows
    }
    rows: list[dict] = []
    unresolved: list[UnresolvedAliasRecord] = []
    wrote_fallback = False

    for row in read_jsonl(runtime.output_dir / "s2_raw_tuples.jsonl"):
        subject_cid, subject_changed = _resolve_with_fallback(
            row["tuple_id"],
            row["subject"],
            "entity",
            alias_map,
            canonical_map,
            unresolved,
            runtime.config.strict_canonical_resolution,
        )
        relation_cid, relation_changed = _resolve_with_fallback(
            row["tuple_id"],
            row["relation"],
            "relation",
            alias_map,
            canonical_map,
            unresolved,
            runtime.config.strict_canonical_resolution,
        )
        object_cid, object_changed = _resolve_with_fallback(
            row["tuple_id"],
            row["object"],
            "entity",
            alias_map,
            canonical_map,
            unresolved,
            runtime.config.strict_canonical_resolution,
        )
        location_cid, location_changed = _resolve_with_fallback(
            row["tuple_id"],
            row.get("location"),
            "location",
            alias_map,
            canonical_map,
            unresolved,
            runtime.config.strict_canonical_resolution,
        )
        rows.append(
            {
                "raw_id": row["tuple_id"],
                "doc_id": row["doc_id"],
                "subject_cid": subject_cid,
                "relation_cid": relation_cid,
                "object_cid": object_cid,
                "location_cid": location_cid,
                "time": row.get("time"),
                "source_sent_id": row["source_sent_id"],
                "confidence": row["confidence"],
            }
        )
        wrote_fallback = wrote_fallback or any(
            (subject_changed, relation_changed, object_changed, location_changed)
        )

    report_payload = {
        "unresolved": [
            {
                "raw_id": item.raw_id,
                "element_type": item.element_type,
                "raw_str": item.raw_str,
            }
            for item in unresolved
        ]
    }
    write_json(runtime.config.reports_dir / "s7_unresolved.json", report_payload)

    if unresolved and runtime.config.strict_canonical_resolution:
        raise RuntimeError(
            f"S7 发现 {len(unresolved)} 个未命中的 canonical 映射，请检查 output/reports/s7_unresolved.json"
        )

    if wrote_fallback:
        updated_rows = sorted(
            canonical_map.values(), key=lambda item: (item["type"], item["cid"])
        )
        write_jsonl(
            runtime.output_dir / "s5_canonical.jsonl", updated_rows, append_done=True
        )
        write_json(runtime.output_dir / "s5_alias_map.json", alias_map)
        _refresh_schema_if_present(runtime.output_dir, updated_rows)

    write_jsonl(
        runtime.output_dir / "s7_canonical_tuples.jsonl", rows, append_done=True
    )
