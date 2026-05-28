from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from core.task_executor import EmbeddingTask, PromptTask, resolve_task_executor
from core.types import StageRuntime
from prompts import tuple_extract as tuple_extract_prompt
from prompts import tuple_normalize as tuple_normalize_prompt
from utils.io_utils import (
    ensure_parent_dir,
    load_npz,
    read_jsonl,
    save_npz,
    write_jsonl,
)
from utils.text_utils import normalize_surface_text

CONCEPT_FILES = {
    "entity": "entity_pool.json",
    "relation": "relation_pool.json",
    "location": "location_pool.json",
}
CONCEPT_VECTOR_FILES = {
    "entity": "entity_vectors.npz",
    "relation": "relation_vectors.npz",
    "location": "location_vectors.npz",
}
CONCEPT_VECTOR_META_FILES = {
    "entity": "entity_vector_meta.json",
    "relation": "relation_vector_meta.json",
    "location": "location_vector_meta.json",
}
CONCEPT_FIELD_TYPES = {
    "subject": "entity",
    "object": "entity",
    "relation": "relation",
    "location": "location",
}
NORMALIZATION_LOG_FILENAME = "s2_concept_normalization_log.jsonl"


@dataclass(slots=True)
class ConceptVectorStore:
    """S2 概念池的本地描述向量索引。"""

    vectors: dict[str, list[float]]
    descriptions: dict[str, str]


def _normalize_file_scope_value(value: str) -> str:
    return normalize_surface_text(value).lower()


def _filter_manifest_rows_by_file_scope(
    manifest_rows: list[dict], file_scope: str
) -> list[dict]:
    if not file_scope.strip() or file_scope.strip().lower() == "all":
        return manifest_rows

    scope_values = {
        _normalize_file_scope_value(item)
        for item in file_scope.split(",")
        if item.strip()
    }
    if not scope_values:
        return manifest_rows

    selected_rows: list[dict] = []
    for row in manifest_rows:
        source_file = str(row.get("source_file") or "")
        extract_file = str(row.get("extract_file") or "")
        doc_id = str(row.get("doc_id") or "")
        candidates = {
            _normalize_file_scope_value(source_file),
            _normalize_file_scope_value(extract_file),
            _normalize_file_scope_value(Path(source_file).stem),
            _normalize_file_scope_value(Path(extract_file).stem),
            _normalize_file_scope_value(doc_id),
        }
        if scope_values & {item for item in candidates if item}:
            selected_rows.append(row)
    return selected_rows


def _read_json_array(file_path: Path) -> list[dict]:
    if not file_path.exists():
        return []
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"JSON 文件必须是列表: {file_path}")
    return [row for row in payload if isinstance(row, dict)]


def _write_json_array(file_path: Path, rows: list[dict]) -> None:
    path = ensure_parent_dir(file_path)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _read_concept_pool(file_path: Path) -> dict[str, dict]:
    if not file_path.exists():
        return {}
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"概念池必须是 JSON 对象: {file_path}")

    pool: dict[str, dict] = {}
    for concept, row in payload.items():
        if not isinstance(concept, str) or not isinstance(row, dict):
            continue
        description = row.get("描述")
        sources = row.get("来源句")
        pool[concept] = {
            "描述": description.strip() if isinstance(description, str) else concept,
            "来源句": (
                [item for item in sources if isinstance(item, str)]
                if isinstance(sources, list)
                else []
            ),
        }
    return pool


def _write_concept_pools(concept_dir: Path, pools: dict[str, dict[str, dict]]) -> None:
    concept_dir.mkdir(parents=True, exist_ok=True)
    for element_type, filename in CONCEPT_FILES.items():
        path = concept_dir / filename
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(pools[element_type], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)


def _read_vector_meta(file_path: Path) -> dict[str, str]:
    if not file_path.exists():
        return {}
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    descriptions = payload.get("descriptions")
    if not isinstance(descriptions, dict):
        return {}
    return {
        str(concept): str(description)
        for concept, description in descriptions.items()
        if isinstance(concept, str) and isinstance(description, str)
    }


def _write_vector_meta(file_path: Path, descriptions: dict[str, str]) -> None:
    path = ensure_parent_dir(file_path)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps({"descriptions": descriptions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _load_vector_store(concept_dir: Path, element_type: str) -> ConceptVectorStore:
    vector_path = concept_dir / CONCEPT_VECTOR_FILES[element_type]
    meta_path = concept_dir / CONCEPT_VECTOR_META_FILES[element_type]
    if vector_path.exists():
        ids, vectors = load_npz(vector_path)
        vector_map = {
            item_id: vector for item_id, vector in zip(ids, vectors, strict=True)
        }
    else:
        vector_map = {}
    return ConceptVectorStore(
        vectors=vector_map,
        descriptions=_read_vector_meta(meta_path),
    )


def _load_vector_stores(concept_dir: Path) -> dict[str, ConceptVectorStore]:
    return {
        element_type: _load_vector_store(concept_dir, element_type)
        for element_type in CONCEPT_FILES
    }


def _concept_description(row: dict, concept: str) -> str:
    description = normalize_surface_text(str(row.get("描述", "")))
    return description or concept


def _write_vector_store(
    concept_dir: Path,
    element_type: str,
    store: ConceptVectorStore,
) -> None:
    ids = sorted(store.vectors)
    save_npz(
        concept_dir / CONCEPT_VECTOR_FILES[element_type],
        ids,
        [store.vectors[item_id] for item_id in ids],
    )
    _write_vector_meta(
        concept_dir / CONCEPT_VECTOR_META_FILES[element_type],
        {item_id: store.descriptions[item_id] for item_id in ids},
    )


def _sync_vector_store_for_type(
    runtime: StageRuntime,
    executor,
    concept_dir: Path,
    element_type: str,
    pool: dict[str, dict],
    store: ConceptVectorStore,
) -> None:
    desired_descriptions = {
        concept: _concept_description(row, concept) for concept, row in pool.items()
    }

    for stale_concept in set(store.vectors) - set(desired_descriptions):
        store.vectors.pop(stale_concept, None)
        store.descriptions.pop(stale_concept, None)

    concepts_to_embed = [
        concept
        for concept, description in desired_descriptions.items()
        if concept not in store.vectors
        or store.descriptions.get(concept) != description
    ]
    if concepts_to_embed:
        tasks = [
            EmbeddingTask(
                custom_id=f"{element_type}_{index}",
                text=desired_descriptions[concept],
            )
            for index, concept in enumerate(concepts_to_embed, start=1)
        ]
        vectors = executor.run_embedding_tasks(
            stage_name=runtime.stage_name,
            tasks=tasks,
            mode=runtime.mode,
        )
        for task, concept in zip(tasks, concepts_to_embed, strict=True):
            store.vectors[concept] = vectors[task.custom_id]
            store.descriptions[concept] = desired_descriptions[concept]

    _write_vector_store(concept_dir, element_type, store)


def _sync_vector_stores(
    runtime: StageRuntime,
    executor,
    concept_dir: Path,
    pools: dict[str, dict[str, dict]],
    stores: dict[str, ConceptVectorStore],
) -> None:
    for element_type in CONCEPT_FILES:
        _sync_vector_store_for_type(
            runtime,
            executor,
            concept_dir,
            element_type,
            pools[element_type],
            stores[element_type],
        )


def _update_pool(
    pool: dict[str, dict],
    concept_item: dict | None,
    source_sentence: str,
    replace_description: bool = True,
) -> None:
    if concept_item is None:
        return
    concept = normalize_surface_text(str(concept_item.get("concept", "")))
    if not concept:
        return
    description = normalize_surface_text(str(concept_item.get("description", "")))
    if not description:
        description = concept

    if concept not in pool:
        pool[concept] = {"描述": description, "来源句": []}
    elif replace_description and description:
        pool[concept]["描述"] = description

    sources = pool[concept].setdefault("来源句", [])
    if source_sentence and source_sentence not in sources:
        sources.append(source_sentence)


def _load_concept_pools(concept_dir: Path) -> dict[str, dict[str, dict]]:
    return {
        element_type: _read_concept_pool(concept_dir / filename)
        for element_type, filename in CONCEPT_FILES.items()
    }


def _build_sentence_groups(sentence_rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in sentence_rows:
        grouped[row["doc_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item.get("sentence_index", 0)))
    return grouped


def _document_context(title: str | None, sentence_rows: list[dict]) -> dict:
    return {
        "标题": title or "",
        "篇章分句列表": [
            {"句子编号": row["sent_id"], "句子内容": row["text"]}
            for row in sentence_rows
        ],
    }


def _planned_sentence_rows(extract_path: Path) -> list[dict]:
    rows = _read_json_array(extract_path)
    return [row for row in rows if row.get("事件单元")]


def _build_tuple_id(sent_id: str, index: int) -> str:
    return f"{sent_id}_t{index:03d}"


def _concept_items(value: dict | list | None) -> list[dict]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _normalize_tuple_rows(
    doc_id: str,
    sent_id: str,
    sentence_text: str,
    result: dict,
) -> list[dict]:
    rows: list[dict] = []
    for item in result.get("tuples", []):
        subjects = _concept_items(item["subject"])
        relation = item["relation"]
        objects = _concept_items(item["object"])
        location = item.get("location")
        for subject in subjects:
            for object_ in objects:
                rows.append(
                    {
                        "tuple_id": _build_tuple_id(sent_id, len(rows) + 1),
                        "doc_id": doc_id,
                        "source_sent_id": sent_id,
                        "source_sentence": sentence_text,
                        "subject": subject["concept"],
                        "relation": relation["concept"],
                        "object": object_["concept"],
                        "time": item["time"],
                        "location": location["concept"] if location else None,
                        "confidence": float(item.get("confidence", 0.8)),
                    }
                )
    return rows


def _build_pool_update_rows(source_sentence: str, result: dict) -> list[dict]:
    rows: list[dict] = []
    for item in result.get("tuples", []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "source_sentence": source_sentence,
                "concepts": {
                    "subject": item.get("subject"),
                    "relation": item.get("relation"),
                    "object": item.get("object"),
                    "location": item.get("location"),
                },
            }
        )
    return rows


def _iter_extracted_concepts(result: dict) -> list[dict]:
    extracted: list[dict] = []
    for tuple_index, item in enumerate(result.get("tuples", []), start=1):
        if not isinstance(item, dict):
            continue
        for field_name, element_type in CONCEPT_FIELD_TYPES.items():
            concept_items = _concept_items(item.get(field_name))
            for concept_index, concept_item in enumerate(concept_items, start=1):
                concept = normalize_surface_text(str(concept_item.get("concept", "")))
                description = normalize_surface_text(
                    str(concept_item.get("description", ""))
                )
                if not concept or not description:
                    continue
                extracted.append(
                    {
                        "candidate_id": f"{tuple_index}_{field_name}_{concept_index}",
                        "tuple_index": tuple_index,
                        "field": field_name,
                        "concept_index": concept_index,
                        "element_type": element_type,
                        "concept": concept,
                        "description": description,
                    }
                )
    return extracted


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _search_similar_concepts(
    pool: dict[str, dict],
    store: ConceptVectorStore,
    query_vector: list[float],
    top_k: int,
) -> list[dict]:
    scored: list[tuple[str, float]] = []
    for concept, vector in store.vectors.items():
        if concept not in pool:
            continue
        scored.append((concept, _cosine_similarity(query_vector, vector)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [
        {
            "概念": concept,
            "描述": _concept_description(pool[concept], concept),
            "相似度": round(score, 6),
        }
        for concept, score in scored[:top_k]
    ]


def _concept_key(element_type: str, concept: str) -> tuple[str, str]:
    return element_type, normalize_surface_text(concept)


def _unique_extracted_concepts(result: dict) -> list[dict]:
    unique: dict[tuple[str, str], dict] = {}
    for occurrence in _iter_extracted_concepts(result):
        key = _concept_key(occurrence["element_type"], occurrence["concept"])
        if key not in unique:
            unique[key] = {
                "candidate_id": f"{occurrence['element_type']}_{len(unique) + 1}",
                "element_type": occurrence["element_type"],
                "concept": occurrence["concept"],
                "description": occurrence["description"],
            }
    return list(unique.values())


def _build_unique_similar_candidates(
    runtime: StageRuntime,
    executor,
    concept_pools: dict[str, dict[str, dict]],
    vector_stores: dict[str, ConceptVectorStore],
    concept_row: dict,
) -> list[dict]:
    element_type = concept_row["element_type"]
    if not concept_pools[element_type] or not vector_stores[element_type].vectors:
        return []
    task = EmbeddingTask(
        custom_id=concept_row["candidate_id"],
        text=concept_row["description"],
    )
    vector_map = executor.run_embedding_tasks(
        stage_name=runtime.stage_name,
        tasks=[task],
        mode=runtime.mode,
    )
    return _search_similar_concepts(
        concept_pools[element_type],
        vector_stores[element_type],
        vector_map[task.custom_id],
        max(1, runtime.config.tuple_concept_top_k),
    )


def _candidate_by_concept(candidates: list[dict]) -> dict[str, dict]:
    return {
        normalize_surface_text(str(candidate.get("概念", ""))): candidate
        for candidate in candidates
        if isinstance(candidate.get("概念"), str)
    }


def _fallback_keep_decision(concept_row: dict) -> dict:
    return {
        "action": "keep",
        "concept": {
            "concept": concept_row["concept"],
            "description": concept_row["description"],
        },
    }


def _request_concept_normalization(
    runtime: StageRuntime,
    executor,
    sent_id: str,
    concept_index: int,
    context: dict,
    sentence_text: str,
    concept_row: dict,
    similar_candidates: list[dict],
) -> dict:
    task_name = (
        "tuple_normalize"
        if runtime.mode == "sync"
        else f"tuple_normalize_{sent_id}_{concept_index:03d}"
    )
    custom_id = f"{sent_id}_{concept_row['candidate_id']}"
    result = executor.run_prompt_tasks(
        stage_name=runtime.stage_name,
        task_name=task_name,
        prompt_module=tuple_normalize_prompt,
        tasks=[
            PromptTask(
                custom_id=custom_id,
                context={
                    "document_context": context,
                    "sentence_text": sentence_text,
                    "concept": {
                        "概念": concept_row["concept"],
                        "描述": concept_row["description"],
                    },
                    "similar_candidates": similar_candidates,
                },
            )
        ],
        temperature=runtime.config.temperature_extract,
        max_tokens=runtime.config.max_tokens,
        mode=runtime.mode,
    )
    return result[custom_id]


def _resolve_concept_decision(
    concept_row: dict,
    raw_decision: dict,
    similar_candidates: list[dict],
    concept_pools: dict[str, dict[str, dict]],
    llm_called: bool,
) -> dict:
    raw_concept = concept_row["concept"]
    raw_description = concept_row["description"]
    decision_action = str(raw_decision.get("action") or "keep")
    decision_concept = raw_decision.get("concept")
    if not isinstance(decision_concept, dict):
        decision_concept = {"concept": raw_concept, "description": raw_description}

    candidates_by_concept = _candidate_by_concept(similar_candidates)
    selected_concept = normalize_surface_text(str(decision_concept.get("concept", "")))
    selected_candidate = candidates_by_concept.get(selected_concept)
    invalid_replacement = False

    if decision_action == "replace" and selected_candidate is not None:
        normalized_concept = str(selected_candidate["概念"])
        normalized_description = normalize_surface_text(
            str(decision_concept.get("description", ""))
        ) or _concept_description(
            {"描述": selected_candidate.get("描述")},
            normalized_concept,
        )
    else:
        invalid_replacement = (
            decision_action == "replace" and selected_candidate is None
        )
        decision_action = "keep"
        normalized_concept = raw_concept
        normalized_description = (
            normalize_surface_text(str(decision_concept.get("description", "")))
            or raw_description
        )

    matched_candidate = candidates_by_concept.get(
        normalize_surface_text(normalized_concept)
    )

    element_type = concept_row["element_type"]
    existed_before = normalized_concept in concept_pools[element_type]
    pool_action = "reused_existing_concept" if existed_before else "new_concept"
    return {
        "key": _concept_key(element_type, raw_concept),
        "element_type": element_type,
        "raw_concept": raw_concept,
        "raw_description": raw_description,
        "normalized_item": {
            "concept": normalized_concept,
            "description": normalized_description,
        },
        "decision": decision_action,
        "action": pool_action,
        "pool_action": pool_action,
        "matched_candidate": matched_candidate,
        "candidate_concepts": similar_candidates,
        "llm_called": llm_called,
        "invalid_replacement": invalid_replacement,
        "concept_changed": raw_concept != normalized_concept,
        "description_changed": raw_description != normalized_description,
    }


def _apply_concept_decisions(
    result: dict, decisions: dict[tuple[str, str], dict]
) -> dict:
    def _replace_one(element_type: str, concept_item: dict) -> dict:
        concept = normalize_surface_text(str(concept_item.get("concept", "")))
        normalized = decisions.get(_concept_key(element_type, concept))
        if normalized is None:
            return dict(concept_item)
        return dict(normalized)

    normalized_tuples: list[dict] = []
    for item in result.get("tuples", []):
        if not isinstance(item, dict):
            continue
        normalized_item = dict(item)
        for field_name, element_type in CONCEPT_FIELD_TYPES.items():
            value = item.get(field_name)
            if isinstance(value, list):
                normalized_item[field_name] = [
                    _replace_one(element_type, concept_item)
                    for concept_item in value
                    if isinstance(concept_item, dict)
                ]
            elif isinstance(value, dict):
                normalized_item[field_name] = _replace_one(element_type, value)
            else:
                normalized_item[field_name] = value
        normalized_tuples.append(normalized_item)
    return {"tuples": normalized_tuples}


def _build_unique_candidate_row(
    concept_row: dict, similar_candidates: list[dict]
) -> dict:
    return {
        "element_type": concept_row["element_type"],
        "抽取概念": {
            "概念": concept_row["concept"],
            "描述": concept_row["description"],
        },
        "候选概念": similar_candidates,
    }


def _decision_to_log_row(
    doc_id: str,
    sent_id: str,
    sentence_text: str,
    decision: dict,
) -> dict:
    normalized_item = decision["normalized_item"]
    return {
        "doc_id": doc_id,
        "source_sent_id": sent_id,
        "source_sentence": sentence_text,
        "element_type": decision["element_type"],
        "raw_concept": decision["raw_concept"],
        "raw_description": decision["raw_description"],
        "normalized_concept": normalized_item["concept"],
        "normalized_description": normalized_item["description"],
        "decision": decision["decision"],
        "action": decision["action"],
        "pool_action": decision["pool_action"],
        "matched_candidate": decision["matched_candidate"],
        "candidate_concepts": decision["candidate_concepts"],
        "llm_called": decision["llm_called"],
        "invalid_replacement": decision["invalid_replacement"],
        "concept_changed": decision["concept_changed"],
        "description_changed": decision["description_changed"],
    }


def _normalize_tuple_concepts(
    runtime: StageRuntime,
    executor,
    doc_id: str,
    sent_id: str,
    context: dict,
    sentence_text: str,
    parsed_result: dict,
    concept_pools: dict[str, dict[str, dict]],
    vector_stores: dict[str, ConceptVectorStore],
) -> tuple[dict, list[dict], list[dict]]:
    if not parsed_result.get("tuples"):
        return parsed_result, [], []

    decisions: dict[tuple[str, str], dict] = {}
    candidate_rows: list[dict] = []
    log_rows: list[dict] = []
    for concept_index, concept_row in enumerate(
        _unique_extracted_concepts(parsed_result),
        start=1,
    ):
        similar_candidates = _build_unique_similar_candidates(
            runtime,
            executor,
            concept_pools,
            vector_stores,
            concept_row,
        )
        candidate_rows.append(
            _build_unique_candidate_row(concept_row, similar_candidates)
        )
        if similar_candidates:
            runtime.logger.info(
                "S2 概念规范化请求：句子 %s，%s 概念「%s」。",
                sent_id,
                concept_row["element_type"],
                concept_row["concept"],
            )
            raw_decision = _request_concept_normalization(
                runtime,
                executor,
                sent_id,
                concept_index,
                context,
                sentence_text,
                concept_row,
                similar_candidates,
            )
            llm_called = True
        else:
            runtime.logger.info(
                "S2 概念规范化跳过：句子 %s，%s 概念「%s」暂无近邻，按新概念保留。",
                sent_id,
                concept_row["element_type"],
                concept_row["concept"],
            )
            raw_decision = _fallback_keep_decision(concept_row)
            llm_called = False

        decision = _resolve_concept_decision(
            concept_row,
            raw_decision,
            similar_candidates,
            concept_pools,
            llm_called,
        )
        decisions[decision["key"]] = decision["normalized_item"]
        log_rows.append(_decision_to_log_row(doc_id, sent_id, sentence_text, decision))

    return _apply_concept_decisions(parsed_result, decisions), candidate_rows, log_rows


def _concepts_from_tuple_row(row: dict) -> dict[str, dict | list | None]:
    raw_concepts = row.get("concepts")
    if isinstance(raw_concepts, dict):
        return raw_concepts

    def _fallback_item(field_name: str) -> dict | None:
        raw_value = row.get(field_name)
        if raw_value is None:
            return None
        concept = normalize_surface_text(str(raw_value))
        if not concept:
            return None
        return {"concept": concept, "description": concept}

    return {
        "subject": _fallback_item("subject"),
        "relation": _fallback_item("relation"),
        "object": _fallback_item("object"),
        "location": _fallback_item("location"),
    }


def _update_pools_from_tuples(
    pools: dict[str, dict[str, dict]],
    tuple_rows: list[dict],
    replace_description: bool = True,
) -> None:
    for row in tuple_rows:
        source_sentence = row.get("source_sentence") or ""
        concepts = _concepts_from_tuple_row(row)
        for concept_item in _concept_items(concepts.get("subject")):
            _update_pool(
                pools["entity"],
                concept_item,
                source_sentence,
                replace_description,
            )
        for concept_item in _concept_items(concepts.get("object")):
            _update_pool(
                pools["entity"],
                concept_item,
                source_sentence,
                replace_description,
            )
        for concept_item in _concept_items(concepts.get("relation")):
            _update_pool(
                pools["relation"],
                concept_item,
                source_sentence,
                replace_description,
            )
        for concept_item in _concept_items(concepts.get("location")):
            _update_pool(
                pools["location"],
                concept_item,
                source_sentence,
                replace_description,
            )


def _load_existing_flat_tuples(tuple_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not tuple_dir.exists():
        return rows
    for file_path in sorted(tuple_dir.glob("*.json")):
        for sentence_row in _read_json_array(file_path):
            rows.extend(sentence_row.get("tuples", []))
    return rows


def _compat_tuple_rows(tuple_rows: list[dict]) -> list[dict]:
    return [
        {
            "tuple_id": row["tuple_id"],
            "doc_id": row["doc_id"],
            "sent_id": row["tuple_id"],
            "subject": row["subject"],
            "relation": row["relation"],
            "object": row["object"],
            "location": row.get("location"),
            "time": row.get("time"),
            "source_sent_id": row["source_sent_id"],
            "confidence": float(row.get("confidence", 0.8)),
        }
        for row in tuple_rows
    ]


def _write_compat_outputs(
    runtime: StageRuntime,
    tuple_rows: list[dict],
    complete: bool,
) -> None:
    write_jsonl(
        runtime.output_dir / "s2_raw_tuples.jsonl",
        _compat_tuple_rows(tuple_rows),
        append_done=complete,
    )


def _completed_sentence_ids(doc_tuple_rows: list[dict]) -> set[str]:
    return {str(row.get("句子编号", "")).strip() for row in doc_tuple_rows}


def _build_error_result(
    doc_id: str, sent_id: str, sentence_text: str, exc: Exception
) -> dict:
    return {
        "doc_id": doc_id,
        "句子编号": sent_id,
        "原句": sentence_text,
        "tuples": [],
        "error": str(exc),
    }


def run_stage(runtime: StageRuntime) -> None:
    """执行 S2 开放五元组抽取与初始概念池构建。"""

    executor = resolve_task_executor(runtime)
    tuple_dir = runtime.data_dir / "tuple_extrct"
    concept_dir = runtime.data_dir / "concept"
    tuple_dir.mkdir(parents=True, exist_ok=True)
    concept_dir.mkdir(parents=True, exist_ok=True)

    sentence_rows = list(read_jsonl(runtime.output_dir / "s1_sentence_index.jsonl"))
    sentence_by_id = {row["sent_id"]: row for row in sentence_rows}
    grouped_sentences = _build_sentence_groups(sentence_rows)
    manifest_rows = list(
        read_jsonl(runtime.output_dir / "s1_extract_plan_manifest.jsonl")
    )
    manifest_rows = _filter_manifest_rows_by_file_scope(
        manifest_rows, runtime.file_scope
    )
    if not manifest_rows:
        raise FileNotFoundError(
            f"未找到满足 file_scope={runtime.file_scope!r} 的 S1 抽取规划文档，无法执行 S2。"
        )

    concept_pools = _load_concept_pools(concept_dir)
    all_tuple_rows = _load_existing_flat_tuples(tuple_dir)
    _update_pools_from_tuples(concept_pools, all_tuple_rows, replace_description=False)
    _write_concept_pools(concept_dir, concept_pools)
    vector_stores = _load_vector_stores(concept_dir)
    _sync_vector_stores(runtime, executor, concept_dir, concept_pools, vector_stores)

    manifest_output_rows: list[dict] = []
    normalization_log_path = runtime.output_dir / NORMALIZATION_LOG_FILENAME
    normalization_log_rows = (
        list(read_jsonl(normalization_log_path))
        if normalization_log_path.exists()
        else []
    )

    for manifest_row in manifest_rows:
        doc_id = manifest_row["doc_id"]
        extract_file = manifest_row["extract_file"]
        tuple_path = tuple_dir / extract_file
        doc_tuple_rows = _read_json_array(tuple_path)
        completed_sent_ids = _completed_sentence_ids(doc_tuple_rows)
        context = _document_context(
            manifest_row.get("title") or manifest_row.get("source_file"),
            grouped_sentences.get(doc_id, []),
        )
        planned_rows = _planned_sentence_rows(
            runtime.data_dir / "extract" / extract_file
        )

        for plan_row in planned_rows:
            sent_id = str(plan_row.get("句子编号", "")).strip()
            if not sent_id or sent_id in completed_sent_ids:
                continue
            if sent_id not in sentence_by_id:
                raise RuntimeError(f"S2 在句子索引中找不到规划句: {sent_id}")

            sentence_text = sentence_by_id[sent_id]["text"]
            task_name = (
                "tuple_extract"
                if runtime.mode == "sync"
                else f"tuple_extract_{sent_id}"
            )
            try:
                runtime.logger.info(
                    "S2 初抽取请求：文档 %s，句子 %s。",
                    extract_file,
                    sent_id,
                )
                result = executor.run_prompt_tasks(
                    stage_name=runtime.stage_name,
                    task_name=task_name,
                    prompt_module=tuple_extract_prompt,
                    tasks=[
                        PromptTask(
                            custom_id=sent_id,
                            context={
                                "document_context": context,
                                "extraction_plan": plan_row,
                            },
                        )
                    ],
                    temperature=runtime.config.temperature_extract,
                    max_tokens=runtime.config.max_tokens,
                    mode=runtime.mode,
                )
                initial_result = result[sent_id]
                parsed_result, similar_candidates, sentence_log_rows = (
                    _normalize_tuple_concepts(
                        runtime,
                        executor,
                        doc_id,
                        sent_id,
                        context,
                        sentence_text,
                        initial_result,
                        concept_pools,
                        vector_stores,
                    )
                )
                tuple_rows = _normalize_tuple_rows(
                    doc_id=doc_id,
                    sent_id=sent_id,
                    sentence_text=sentence_text,
                    result=parsed_result,
                )
                pool_update_rows = _build_pool_update_rows(sentence_text, parsed_result)
                sentence_result = {
                    "doc_id": doc_id,
                    "句子编号": sent_id,
                    "原句": sentence_text,
                    "抽取规划": plan_row,
                    "初抽取五元组": initial_result.get("tuples", []),
                    "规范化五元组": parsed_result.get("tuples", []),
                    "相似概念候选": similar_candidates,
                    "规范化日志": sentence_log_rows,
                    "tuples": tuple_rows,
                }
            except Exception as exc:  # noqa: BLE001
                runtime.logger.warning(
                    "S2 句子 %s 五元组抽取失败，已写入空结果并继续。错误: %s",
                    sent_id,
                    exc,
                )
                tuple_rows = []
                pool_update_rows = []
                sentence_log_rows = []
                sentence_result = _build_error_result(
                    doc_id, sent_id, sentence_text, exc
                )

            doc_tuple_rows.append(sentence_result)
            completed_sent_ids.add(sent_id)
            all_tuple_rows.extend(tuple_rows)
            normalization_log_rows.extend(sentence_log_rows)
            _update_pools_from_tuples(
                concept_pools, pool_update_rows, replace_description=True
            )
            _sync_vector_stores(
                runtime, executor, concept_dir, concept_pools, vector_stores
            )
            _write_json_array(tuple_path, doc_tuple_rows)
            _write_concept_pools(concept_dir, concept_pools)
            _write_compat_outputs(runtime, all_tuple_rows, complete=False)
            write_jsonl(
                normalization_log_path, normalization_log_rows, append_done=False
            )

        doc_tuple_count = sum(len(row.get("tuples", [])) for row in doc_tuple_rows)
        manifest_output_rows.append(
            {
                "doc_id": doc_id,
                "source_file": manifest_row.get("source_file"),
                "extract_file": extract_file,
                "tuple_file": extract_file,
                "requested_sentence_count": len(planned_rows),
                "completed_sentence_count": len(doc_tuple_rows),
                "tuple_count": doc_tuple_count,
            }
        )
        write_jsonl(
            runtime.output_dir / "s2_tuple_extract_manifest.jsonl",
            manifest_output_rows,
            append_done=False,
        )

    _write_concept_pools(concept_dir, concept_pools)
    _write_compat_outputs(runtime, all_tuple_rows, complete=True)
    write_jsonl(normalization_log_path, normalization_log_rows, append_done=True)
    write_jsonl(
        runtime.output_dir / "s2_tuple_extract_manifest.jsonl",
        manifest_output_rows,
        append_done=True,
    )
