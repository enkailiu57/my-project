from __future__ import annotations

from dataclasses import dataclass

from core.cache import LocalCache
from core.llm_client import LLMClient
from core.types import StageRuntime
from prompts import canonical_select as canonical_select_prompt
from prompts import dedup_check as dedup_check_prompt
from prompts import describe_entity, describe_location, describe_relation
from utils.concept_pool_utils import load_concept_pool_rows
from utils.dedup_utils import (
    build_cache_key,
    cluster_vectors_by_kmeans,
    mixed_topk_scores,
)
from utils.io_utils import load_npz, read_jsonl, write_json, write_jsonl
from utils.text_utils import (
    build_element_id,
    normalize_alias_key,
    normalize_surface_text,
)
from utils.vector_index import VectorIndex


@dataclass(slots=True)
class DedupElement:
    """S5 去重阶段使用的元素对象。"""

    id: str
    raw_str: str
    type: str
    description: str
    count: int
    contexts: list[str]
    alias_key: str
    vector: list[float]


def _stage_files(element_type: str) -> tuple[str, str]:
    if element_type == "entity":
        return "s3_entity_desc.jsonl", "s3_entity_desc.jsonl"
    if element_type == "relation":
        return "s3_relation_desc.jsonl", "s3_relation_desc.jsonl"
    return "s3_location_desc.jsonl", "s3_location_desc.jsonl"


def _describe_prompt_module(element_type: str):
    if element_type == "entity":
        return describe_entity
    if element_type == "relation":
        return describe_relation
    return describe_location


def _load_elements_for_type(
    runtime: StageRuntime, element_type: str
) -> list[DedupElement]:
    _, desc_file = _stage_files(element_type)
    pool_by_id = {
        row["id"]: row
        for row in load_concept_pool_rows(runtime.data_dir / "concept", element_type)
    }
    embedding_ids, embedding_vectors = load_npz(
        runtime.output_dir / "s4_embeddings.npz"
    )
    vector_by_id = {
        item_id: vector
        for item_id, vector in zip(embedding_ids, embedding_vectors, strict=True)
    }

    elements: list[DedupElement] = []
    for row in read_jsonl(runtime.output_dir / desc_file):
        pool_row = pool_by_id[row["id"]]
        alias_key = normalize_alias_key(row["raw_str"])
        if alias_key is None:
            raise RuntimeError(f"S5 无法为元素 {row['id']} 生成 alias_key")
        if row["id"] not in vector_by_id:
            raise RuntimeError(f"S5 缺少元素 {row['id']} 的向量")
        elements.append(
            DedupElement(
                id=row["id"],
                raw_str=row["raw_str"],
                type=row["type"],
                description=row["description"],
                count=int(pool_row["count"]),
                contexts=list(pool_row.get("contexts", [])),
                alias_key=alias_key,
                vector=vector_by_id[row["id"]],
            )
        )
    return elements


def _merge_contexts(elements: list[DedupElement], indices: list[int]) -> list[str]:
    contexts: list[str] = []
    for index in indices:
        for context in elements[index].contexts:
            if context not in contexts:
                contexts.append(context)
            if len(contexts) >= 5:
                return contexts
    return contexts


def _candidate_indices(
    target_index: int,
    remaining: set[int],
    elements: list[DedupElement],
    vector_index: VectorIndex,
    top_k: int,
    bm25_weight: float,
) -> list[int]:
    others = [index for index in remaining if index != target_index]
    if not others:
        return []

    query_text = (
        f"{elements[target_index].raw_str} {elements[target_index].description}"
    )
    candidate_texts = [
        f"{elements[index].raw_str} {elements[index].description}" for index in others
    ]
    vector_scores = [
        float(vector_index.matrix[index] @ vector_index.matrix[target_index])
        for index in others
    ]
    mixed_scores = mixed_topk_scores(
        query_text, vector_scores, candidate_texts, bm25_weight
    )
    ranked = sorted(
        zip(others, mixed_scores, vector_scores, strict=True),
        key=lambda item: (item[1], item[2]),
        reverse=True,
    )
    return [index for index, _, _ in ranked[:top_k]]


def _cached_dedup_check(
    client: LLMClient,
    cache: LocalCache,
    runtime: StageRuntime,
    target: DedupElement,
    candidates: list[DedupElement],
) -> list[str]:
    payload = {
        "target_id": target.id,
        "candidate_ids": [candidate.id for candidate in candidates],
    }
    cache_key = build_cache_key("dedup_check", payload)
    cached = cache.get(cache_key)
    if cached is not None:
        return list(cached.get("duplicates", []))

    result = client.call(
        dedup_check_prompt,
        {
            "target": {"raw_str": target.raw_str, "description": target.description},
            "candidates": [
                {"raw_str": candidate.raw_str, "description": candidate.description}
                for candidate in candidates
            ],
        },
        temperature=runtime.config.temperature_extract,
        max_tokens=runtime.config.max_tokens,
    )
    cache.set(cache_key, result)
    return list(result.get("duplicates", []))


def _cached_canonical_select(
    client: LLMClient,
    cache: LocalCache,
    runtime: StageRuntime,
    candidates: list[DedupElement],
) -> str:
    payload = {
        "candidate_ids": [candidate.id for candidate in candidates],
        "candidate_counts": {candidate.id: candidate.count for candidate in candidates},
    }
    cache_key = build_cache_key("canonical_select", payload)
    cached = cache.get(cache_key)
    if cached is not None:
        return str(cached["canonical_str"])

    result = client.call(
        canonical_select_prompt,
        {
            "candidates": [
                {"raw_str": candidate.raw_str, "count": candidate.count}
                for candidate in candidates
            ]
        },
        temperature=runtime.config.temperature_extract,
        max_tokens=runtime.config.max_tokens,
    )
    cache.set(cache_key, result)
    return str(result["canonical_str"])


def _select_target_index(remaining: set[int], elements: list[DedupElement]) -> int:
    return max(
        remaining, key=lambda index: (elements[index].count, elements[index].raw_str)
    )


def _merge_canonical_row(target: dict, incoming: dict) -> dict:
    aliases = sorted(set(target.get("aliases", [])) | set(incoming.get("aliases", [])))
    target["aliases"] = aliases
    return target


def _build_canonical_row(
    runtime: StageRuntime,
    client: LLMClient,
    cache: LocalCache,
    elements: list[DedupElement],
    group_indices: list[int],
) -> dict:
    group = [elements[index] for index in group_indices]
    if len(group) == 1:
        item = group[0]
        return {
            "cid": item.id,
            "canonical_str": item.raw_str,
            "type": item.type,
            "description": item.description,
            "aliases": [item.alias_key],
        }

    canonical_str = _cached_canonical_select(client, cache, runtime, group)
    canonical_key = normalize_alias_key(canonical_str)
    matched = next((item for item in group if item.alias_key == canonical_key), None)
    if matched is not None:
        cid = matched.id
        description = matched.description
        canonical_surface = matched.raw_str
    else:
        canonical_surface = normalize_surface_text(canonical_str)
        cid = build_element_id(group[0].type, canonical_surface)
        describe_prompt = _describe_prompt_module(group[0].type)
        describe_result = client.call(
            describe_prompt,
            {
                "raw_str": canonical_surface,
                "contexts": _merge_contexts(elements, group_indices),
            },
            temperature=runtime.config.temperature_describe,
            max_tokens=runtime.config.max_tokens,
        )
        description = describe_result["description"]

    aliases = sorted({item.alias_key for item in group})
    return {
        "cid": cid,
        "canonical_str": canonical_surface,
        "type": group[0].type,
        "description": description,
        "aliases": aliases,
    }


def _deduplicate_cluster(
    runtime: StageRuntime,
    client: LLMClient,
    cache: LocalCache,
    elements: list[DedupElement],
    cluster_indices: list[int],
    vector_index: VectorIndex,
) -> tuple[list[dict], dict[str, str]]:
    canonical_rows: list[dict] = []
    alias_map: dict[str, str] = {}
    remaining = set(cluster_indices)
    step_limit = max(1, len(cluster_indices) * runtime.config.dedup_max_iterations)
    steps = 0

    while remaining:
        steps += 1
        if steps > step_limit:
            raise RuntimeError("S5 去重循环超过安全步数，请检查聚类或提示词输出。")

        target_index = _select_target_index(remaining, elements)
        top_candidates = _candidate_indices(
            target_index=target_index,
            remaining=remaining,
            elements=elements,
            vector_index=vector_index,
            top_k=runtime.config.dedup_top_k,
            bm25_weight=runtime.config.dedup_bm25_weight,
        )
        candidate_elements = [elements[index] for index in top_candidates]
        duplicate_names = _cached_dedup_check(
            client=client,
            cache=cache,
            runtime=runtime,
            target=elements[target_index],
            candidates=candidate_elements,
        )
        duplicate_keys = {
            normalize_alias_key(name)
            for name in duplicate_names
            if normalize_alias_key(name)
        }
        duplicate_indices = [
            index
            for index in top_candidates
            if elements[index].alias_key in duplicate_keys
        ]
        group_indices = sorted({target_index, *duplicate_indices})
        canonical_row = _build_canonical_row(
            runtime=runtime,
            client=client,
            cache=cache,
            elements=elements,
            group_indices=group_indices,
        )
        canonical_rows.append(canonical_row)
        for index in group_indices:
            alias_map[elements[index].alias_key] = canonical_row["cid"]
        remaining -= set(group_indices)

    return canonical_rows, alias_map


def run_stage(runtime: StageRuntime) -> None:
    """执行 S5 聚类去重。"""

    client = LLMClient(runtime.config)
    canonical_map: dict[str, dict] = {}
    alias_map = {"entity": {}, "relation": {}, "location": {}}

    with LocalCache(runtime.config.cache_db_path) as cache:
        for element_type in ("entity", "relation", "location"):
            elements = _load_elements_for_type(runtime, element_type)
            if not elements:
                continue
            vectors = [item.vector for item in elements]
            clusters = cluster_vectors_by_kmeans(
                vectors=vectors,
                target_cluster_size=runtime.config.dedup_cluster_size,
                random_seed=runtime.config.random_seed,
            )
            vector_index = VectorIndex([item.id for item in elements], vectors)
            for cluster_indices in clusters:
                cluster_rows, cluster_alias_map = _deduplicate_cluster(
                    runtime=runtime,
                    client=client,
                    cache=cache,
                    elements=elements,
                    cluster_indices=cluster_indices,
                    vector_index=vector_index,
                )
                alias_map[element_type].update(cluster_alias_map)
                for row in cluster_rows:
                    if row["cid"] in canonical_map:
                        canonical_map[row["cid"]] = _merge_canonical_row(
                            canonical_map[row["cid"]], row
                        )
                    else:
                        canonical_map[row["cid"]] = row

    canonical_rows = sorted(
        canonical_map.values(), key=lambda row: (row["type"], row["cid"])
    )
    write_jsonl(
        runtime.output_dir / "s5_canonical.jsonl", canonical_rows, append_done=True
    )
    write_json(runtime.output_dir / "s5_alias_map.json", alias_map)
