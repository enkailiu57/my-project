from __future__ import annotations

from core.types import StageName, StageSpec
from pipeline.s0_news_process import run_stage as run_s0
from pipeline.s1_extract_plan import run_stage as run_s1
from pipeline.s2_open_ie import run_stage as run_s2
from pipeline.s3_describe import run_stage as run_s3
from pipeline.s4_embed import run_stage as run_s4
from pipeline.s5_dedup import run_stage as run_s5
from pipeline.s6_schema import run_stage as run_s6
from pipeline.s7_canonicalize import run_stage as run_s7

STAGE_ORDER: tuple[StageName, ...] = (
    "s0",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "s7",
)


STAGE_REGISTRY: dict[StageName, StageSpec] = {
    "s0": StageSpec(
        "s0", "新闻预处理", tuple(), ("s0_processed_manifest.jsonl",), run_s0
    ),
    "s1": StageSpec(
        "s1",
        "事件抽取规划",
        tuple(),
        ("s1_extract_plan_manifest.jsonl", "s1_sentence_index.jsonl"),
        run_s1,
    ),
    "s2": StageSpec(
        "s2",
        "开放五元组抽取与概念池构建",
        ("s1_extract_plan_manifest.jsonl", "s1_sentence_index.jsonl"),
        (
            "s2_tuple_extract_manifest.jsonl",
            "s2_raw_tuples.jsonl",
            "data/concept/entity_pool.json",
            "data/concept/relation_pool.json",
            "data/concept/location_pool.json",
        ),
        run_s2,
    ),
    "s3": StageSpec(
        "s3",
        "描述生成",
        (
            "data/concept/entity_pool.json",
            "data/concept/relation_pool.json",
            "data/concept/location_pool.json",
        ),
        ("s3_entity_desc.jsonl", "s3_relation_desc.jsonl", "s3_location_desc.jsonl"),
        run_s3,
    ),
    "s4": StageSpec(
        "s4",
        "向量嵌入",
        ("s3_entity_desc.jsonl", "s3_relation_desc.jsonl", "s3_location_desc.jsonl"),
        ("s4_embeddings.npz",),
        run_s4,
    ),
    "s5": StageSpec(
        "s5",
        "聚类去重",
        (
            "data/concept/entity_pool.json",
            "data/concept/relation_pool.json",
            "data/concept/location_pool.json",
            "s3_entity_desc.jsonl",
            "s3_relation_desc.jsonl",
            "s3_location_desc.jsonl",
            "s4_embeddings.npz",
        ),
        ("s5_canonical.jsonl", "s5_alias_map.json"),
        run_s5,
    ),
    "s6": StageSpec(
        "s6",
        "Schema 汇总",
        ("s5_canonical.jsonl", "s5_alias_map.json"),
        ("s6_schema.json",),
        run_s6,
    ),
    "s7": StageSpec(
        "s7",
        "三元组规范化",
        ("s2_raw_tuples.jsonl", "s5_canonical.jsonl", "s5_alias_map.json"),
        ("s7_canonical_tuples.jsonl",),
        run_s7,
    ),
}


def resolve_stage_sequence(
    run_all: bool, from_stage: StageName | None, only: list[StageName] | None
) -> list[StageName]:
    """根据 CLI 参数解析待执行的阶段列表。"""

    if only:
        return only

    if run_all:
        return list(STAGE_ORDER)

    if from_stage:
        start_index = STAGE_ORDER.index(from_stage)
        return list(STAGE_ORDER[start_index:])

    raise ValueError("必须指定 --run-all、--from 或 --only 之一。")
