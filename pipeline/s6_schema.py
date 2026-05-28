from __future__ import annotations

from datetime import datetime, timezone

from core.types import StageRuntime
from utils.io_utils import read_jsonl, write_json


def _dedup_ratio(rows: list[dict]) -> float:
    """根据别名总数和 canonical 数量计算去重率。"""

    raw_total = sum(len(row.get("aliases", [])) for row in rows)
    if raw_total == 0:
        return 0.0
    return round(1.0 - (len(rows) / raw_total), 6)


def run_stage(runtime: StageRuntime) -> None:
    """执行 S6 Schema 汇总。"""

    rows = list(read_jsonl(runtime.output_dir / "s5_canonical.jsonl"))
    entities = [row for row in rows if row["type"] == "entity"]
    relations = [row for row in rows if row["type"] == "relation"]
    locations = [row for row in rows if row["type"] == "location"]

    schema = {
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
    }

    write_json(runtime.output_dir / "s6_schema.json", schema)
