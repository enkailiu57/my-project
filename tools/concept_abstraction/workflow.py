from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from core.embed_client import EmbeddingClient
from core.task_executor import PromptTask
from prompts import concept_compliance, concept_description, concept_merge
from utils.io_utils import chunked, ensure_parent_dir, load_npz, save_npz, stable_hash8
from utils.text_utils import build_element_id, normalize_surface_text

ELEMENT_TYPES = ("entity", "relation", "location")
POOL_FILES = {
    "entity": "entity_pool.json",
    "relation": "relation_pool.json",
    "location": "location_pool.json",
}
TUPLE_FIELDS_BY_TYPE = {
    "entity": ("subject", "object"),
    "relation": ("relation",),
    "location": ("location",),
}
MERGE_ALGORITHM = "cluster_partition_v1"
CLUSTER_METHOD_AGGLOMERATIVE = "agglomerative"
CLUSTER_METHOD_LEIDEN_BALANCED = "leiden_balanced"


def build_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger(name)


def parse_element_types(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(ELEMENT_TYPES)
    element_types = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in element_types if item not in ELEMENT_TYPES]
    if unknown:
        raise ValueError(f"未知概念类型: {unknown}")
    return element_types


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def as_project_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def compact_filename_token(value: Any) -> str:
    if isinstance(value, float):
        normalized = f"{value:.6f}".rstrip("0").rstrip(".")
        normalized = normalized if normalized not in {"", "-0"} else "0"
    else:
        normalized = str(value).strip() or "0"
    return normalized.replace("-", "n").replace(".", "p")


def build_field_weight_token(field_weights: Any) -> str:
    if not isinstance(field_weights, dict) or not field_weights:
        return "fw0"
    values = [
        compact_filename_token(value)
        for _field_name, value in field_weights.items()
        if isinstance(value, (int, float))
    ]
    return "fw" + ("-".join(values) if values else "0")


def build_leiden_cluster_filename(
    element_type: str,
    parameters: dict[str, Any],
    cluster_count: int,
    multi_member_cluster_count: int,
) -> str:
    filename_payload = {
        "element_type": element_type,
        "parameters": parameters,
        "cluster_count": cluster_count,
        "multi_member_cluster_count": multi_member_cluster_count,
    }
    digest = stable_hash8(
        json.dumps(filename_payload, ensure_ascii=False, sort_keys=True)
    )
    parts = [
        element_type,
        "leiden_balanced",
        f"t{compact_filename_token(parameters.get('target_cluster_size', 0))}",
        f"mn{compact_filename_token(parameters.get('min_cluster_size', 0))}",
        f"mx{compact_filename_token(parameters.get('max_cluster_size', 0))}",
        f"k{compact_filename_token(parameters.get('knn_k', 0))}",
        f"ir{compact_filename_token(parameters.get('initial_resolution', 0))}",
        f"me{compact_filename_token(parameters.get('min_edge_similarity', 0))}",
        f"sa{compact_filename_token(parameters.get('small_cluster_absorb_threshold', 0))}",
        build_field_weight_token(parameters.get("field_weights")),
        f"sr{compact_filename_token(parameters.get('selected_resolution', 0))}",
        f"c{compact_filename_token(cluster_count)}",
        f"mm{compact_filename_token(multi_member_cluster_count)}",
        f"h{digest}",
    ]
    return "_".join(parts) + ".json"


def build_merge_artifact_filename(
    element_type: str,
    artifact_type: str,
    cluster_file: Path,
    extension: str,
) -> str:
    return f"{element_type}_{artifact_type}__from_{cluster_file.stem}{extension}"


def resolve_mapping_file(mapping_dir: Path, element_type: str) -> Path | None:
    cluster_specific = sorted(
        mapping_dir.glob(f"{element_type}_mapping__from_*.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    if cluster_specific:
        return cluster_specific[0]

    legacy_file = mapping_dir / f"{element_type}_mapping.json"
    if legacy_file.exists():
        return legacy_file
    return None


def read_json_object(file_path: Path) -> dict[str, Any]:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 文件必须是对象: {file_path}")
    return payload


def read_json_array(file_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"JSON 文件必须是数组: {file_path}")
    return [item for item in payload if isinstance(item, dict)]


def write_json_object(file_path: Path, payload: dict[str, Any]) -> None:
    path = ensure_parent_dir(file_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_json_array(file_path: Path, rows: list[dict[str, Any]]) -> None:
    path = ensure_parent_dir(file_path)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(file_path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = ensure_parent_dir(file_path)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(file_path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = ensure_parent_dir(file_path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl_if_exists(file_path: Path) -> list[dict[str, Any]]:
    if not file_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_pool(concept_dir: Path, element_type: str) -> dict[str, dict[str, Any]]:
    file_path = concept_dir / POOL_FILES[element_type]
    if not file_path.exists():
        raise FileNotFoundError(f"概念池文件不存在: {file_path}")
    payload = read_json_object(file_path)
    return {
        concept: item
        for concept, item in payload.items()
        if isinstance(concept, str) and isinstance(item, dict)
    }


def write_pool(
    output_dir: Path, element_type: str, pool: dict[str, dict[str, Any]]
) -> None:
    ordered_pool = {concept: pool[concept] for concept in sorted(pool)}
    write_json_object(output_dir / POOL_FILES[element_type], ordered_pool)


def concept_custom_id(element_type: str, concept: str) -> str:
    return f"{element_type}_{stable_hash8(element_type + ':' + concept)}"


def iter_pool_records(
    concept_dir: Path, element_types: list[str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for element_type in element_types:
        for concept, item in sorted(load_pool(concept_dir, element_type).items()):
            clean_concept = normalize_surface_text(concept)
            if not clean_concept:
                continue
            description = item.get("描述")
            source_sentences = item.get("来源句")
            sources = (
                [
                    normalize_surface_text(sentence)
                    for sentence in source_sentences
                    if isinstance(sentence, str) and normalize_surface_text(sentence)
                ]
                if isinstance(source_sentences, list)
                else []
            )
            records.append(
                {
                    "custom_id": concept_custom_id(element_type, clean_concept),
                    "id": build_element_id(element_type, clean_concept),
                    "concept_type": element_type,
                    "concept": clean_concept,
                    "description": (
                        normalize_surface_text(description)
                        if isinstance(description, str)
                        and normalize_surface_text(description)
                        else clean_concept
                    ),
                    "source_sentences": sources,
                    "source_count": len(sources),
                    "pool_item": item,
                }
            )
    return records


def description_field_names(element_type: str) -> tuple[str, ...]:
    field_names = concept_description.FIELD_SCHEMAS.get(element_type)
    if field_names is None:
        raise ValueError(f"未知概念类型: {element_type}")
    return field_names


def description_field_weights(element_type: str) -> dict[str, float]:
    field_names = description_field_names(element_type)
    raw_weights = concept_description.FIELD_WEIGHTS.get(element_type, {})
    weights = {
        field_name: float(raw_weights.get(field_name, 1.0))
        for field_name in field_names
    }
    total = sum(weights.values())
    if total <= 0:
        return {field_name: 1.0 / len(field_names) for field_name in field_names}
    return {field_name: weight / total for field_name, weight in weights.items()}


def normalize_description_fields(
    element_type: str,
    raw_fields: Any,
) -> dict[str, str]:
    field_names = description_field_names(element_type)
    if not isinstance(raw_fields, dict):
        raise ValueError(f"{element_type} 描述缺少 fields 对象。")

    fields: dict[str, str] = {}
    missing: list[str] = []
    for field_name in field_names:
        value = raw_fields.get(field_name)
        if isinstance(value, str) and normalize_surface_text(value):
            fields[field_name] = normalize_surface_text(value)
        else:
            missing.append(field_name)
    if missing:
        raise ValueError(f"{element_type} 描述缺少字段: {missing}")
    return fields


def validate_description_result(
    element_type: str, result: dict[str, Any]
) -> dict[str, Any]:
    fields = normalize_description_fields(element_type, result.get("fields"))
    return {
        "fields": fields,
        "description_schema": element_type,
        "description_schema_version": concept_description.SCHEMA_VERSION,
    }


def is_current_description_row(row: dict[str, Any]) -> bool:
    if row.get("description_schema_version") != concept_description.SCHEMA_VERSION:
        return False
    element_type = row.get("concept_type")
    if element_type not in concept_description.FIELD_SCHEMAS:
        return False
    try:
        validate_description_result(str(element_type), row)
    except (TypeError, ValueError):
        return False
    return True


def value_contains(value: Any, concept: str) -> bool:
    if isinstance(value, list):
        return any(value_contains(item, concept) for item in value)
    return isinstance(value, str) and value == concept


def tuple_uses_concept(row: dict[str, Any], element_type: str, concept: str) -> bool:
    return any(
        value_contains(row.get(field_name), concept)
        for field_name in TUPLE_FIELDS_BY_TYPE[element_type]
    )


def load_tuple_records(tuple_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not tuple_dir.exists():
        return rows
    for file_path in sorted(tuple_dir.glob("*.json")):
        for row in read_json_array(file_path):
            rows.append({"tuple_file": file_path.name, "tuple": row})
    return rows


def run_compliance_filter(
    project_root: Path,
    concept_dir: Path,
    tuple_dir: Path,
    output_dir: Path,
    executor: Any,
    mode: str,
    element_types: list[str],
    chunk_size: int = 20,
    temperature: float = 0.0,
    max_tokens: int = 512,
    overwrite: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    active_logger = logger or build_logger("kg-concept-compliance")
    compliance_dir = output_dir / "compliance"
    compliant_pool_dir = output_dir / "compliant_pools"
    invalid_dir = output_dir / "invalid"
    result_file = compliance_dir / "concept_compliance_results.jsonl"
    if overwrite and result_file.exists():
        result_file.unlink()

    records = iter_pool_records(concept_dir, element_types)
    result_by_id = {
        row["custom_id"]: row
        for row in read_jsonl_if_exists(result_file)
        if isinstance(row.get("custom_id"), str) and is_current_description_row(row)
    }

    tasks: list[PromptTask] = []
    task_meta: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["custom_id"] in result_by_id:
            continue
        tasks.append(
            PromptTask(
                custom_id=record["custom_id"],
                context={
                    "concept_type": record["concept_type"],
                    "concept": record["concept"],
                    "description": record["description"],
                    "source_sentences": record["source_sentences"][:8],
                },
            )
        )
        task_meta[record["custom_id"]] = record

    for batch_index, task_batch in enumerate(chunked(tasks, chunk_size), start=1):
        results = executor.run_prompt_tasks(
            stage_name="concept_abstraction",
            task_name=f"compliance_{batch_index:04d}",
            prompt_module=concept_compliance,
            tasks=task_batch,
            temperature=temperature,
            max_tokens=max_tokens,
            mode=mode,
        )
        rows_to_append: list[dict[str, Any]] = []
        for task in task_batch:
            record = task_meta[task.custom_id]
            result = results[task.custom_id]
            row = {
                "custom_id": task.custom_id,
                "concept_type": record["concept_type"],
                "concept": record["concept"],
                "description": record["description"],
                "source_count": record["source_count"],
                "verdict": result["verdict"],
                "reason": result.get("reason", ""),
            }
            rows_to_append.append(row)
            result_by_id[task.custom_id] = row
        append_jsonl(result_file, rows_to_append)
        active_logger.info(
            "概念合规性判别进度：完成第 %s 批，累计 %s/%s。",
            batch_index,
            len(result_by_id),
            len(records),
        )

    source_pools = {
        element_type: load_pool(concept_dir, element_type)
        for element_type in element_types
    }
    invalid_by_type: dict[str, set[str]] = defaultdict(set)
    invalid_concepts: list[dict[str, Any]] = []
    for record in records:
        result = result_by_id.get(record["custom_id"])
        if result is None:
            raise RuntimeError(f"缺少概念合规性结果: {record['concept']}")
        if result.get("verdict") == "合规":
            continue
        invalid_by_type[record["concept_type"]].add(record["concept"])
        invalid_concepts.append(
            {
                "concept_type": record["concept_type"],
                "concept": record["concept"],
                "description": record["description"],
                "source_count": record["source_count"],
                "source_sentences": record["source_sentences"][:8],
                "reason": result.get("reason", ""),
            }
        )

    compliant_counts: dict[str, int] = {}
    for element_type in element_types:
        invalids = invalid_by_type[element_type]
        compliant_pool = {
            concept: item
            for concept, item in source_pools[element_type].items()
            if concept not in invalids
        }
        write_pool(compliant_pool_dir, element_type, compliant_pool)
        compliant_counts[element_type] = len(compliant_pool)

    tuple_records = load_tuple_records(tuple_dir)
    invalid_tuple_rows: list[dict[str, Any]] = []
    for invalid in invalid_concepts:
        for tuple_record in tuple_records:
            tuple_row = tuple_record["tuple"]
            if not tuple_uses_concept(
                tuple_row, invalid["concept_type"], invalid["concept"]
            ):
                continue
            invalid_tuple_rows.append(
                {
                    "concept_type": invalid["concept_type"],
                    "concept": invalid["concept"],
                    "tuple_file": tuple_record["tuple_file"],
                    "tuple_id": tuple_row.get("tuple_id"),
                    "tuple": tuple_row,
                }
            )

    write_json_array(invalid_dir / "invalid_concepts.json", invalid_concepts)
    write_jsonl(invalid_dir / "invalid_tuples.jsonl", invalid_tuple_rows)
    summary = {
        "concept_dir": as_project_path(concept_dir, project_root),
        "tuple_dir": as_project_path(tuple_dir, project_root),
        "output_dir": as_project_path(output_dir, project_root),
        "total_concept_count": len(records),
        "invalid_concept_count": len(invalid_concepts),
        "invalid_tuple_reference_count": len(invalid_tuple_rows),
        "compliant_counts": compliant_counts,
    }
    write_json_object(compliance_dir / "summary.json", summary)
    return summary


def run_description_generation(
    project_root: Path,
    concept_dir: Path,
    output_dir: Path,
    executor: Any,
    mode: str,
    element_types: list[str],
    chunk_size: int = 20,
    temperature: float = 0.1,
    max_tokens: int = 512,
    overwrite: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    active_logger = logger or build_logger("kg-concept-description")
    description_dir = output_dir / "descriptions"
    result_file = description_dir / "concept_description_results.jsonl"
    if overwrite and result_file.exists():
        result_file.unlink()

    records = iter_pool_records(concept_dir, element_types)
    result_by_id = {
        row["custom_id"]: row
        for row in read_jsonl_if_exists(result_file)
        if isinstance(row.get("custom_id"), str) and is_current_description_row(row)
    }
    if result_by_id:
        active_logger.info(
            "检测到 %s 条已有描述结果，将从中断处继续。",
            len(result_by_id),
        )

    tasks: list[PromptTask] = []
    task_meta: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["custom_id"] in result_by_id:
            continue
        tasks.append(
            PromptTask(
                custom_id=record["custom_id"],
                context={
                    "concept_type": record["concept_type"],
                    "concept": record["concept"],
                    "current_description": record["description"],
                    "source_sentences": record["source_sentences"][:8],
                },
            )
        )
        task_meta[record["custom_id"]] = record

    def build_result_row(task: PromptTask, result: dict[str, Any]) -> dict[str, Any]:
        record = task_meta[task.custom_id]
        validated = validate_description_result(
            record["concept_type"],
            result,
        )
        return {
            "custom_id": task.custom_id,
            "concept_type": record["concept_type"],
            "concept": record["concept"],
            "fields": validated["fields"],
            "description_schema": validated["description_schema"],
            "description_schema_version": validated["description_schema_version"],
            "source_count": record["source_count"],
        }

    for batch_index, task_batch in enumerate(chunked(tasks, chunk_size), start=1):
        if mode == "sync":
            for task_index, task in enumerate(task_batch, start=1):
                results = executor.run_prompt_tasks(
                    stage_name="concept_abstraction",
                    task_name=f"description_{batch_index:04d}_{task_index:04d}",
                    prompt_module=concept_description,
                    tasks=[task],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    mode=mode,
                )
                row = build_result_row(task, results[task.custom_id])
                append_jsonl(result_file, [row])
                result_by_id[task.custom_id] = row
                active_logger.info(
                    "概念描述生成进度：累计 %s/%s。",
                    len(result_by_id),
                    len(records),
                )
            continue

        results = executor.run_prompt_tasks(
            stage_name="concept_abstraction",
            task_name=f"description_{batch_index:04d}",
            prompt_module=concept_description,
            tasks=task_batch,
            temperature=temperature,
            max_tokens=max_tokens,
            mode=mode,
        )
        rows_to_append: list[dict[str, Any]] = []
        for task in task_batch:
            row = build_result_row(task, results[task.custom_id])
            rows_to_append.append(row)
            result_by_id[task.custom_id] = row
        append_jsonl(result_file, rows_to_append)
        active_logger.info(
            "概念描述生成进度：完成第 %s 批，累计 %s/%s。",
            batch_index,
            len(result_by_id),
            len(records),
        )

    by_type: dict[str, dict[str, dict[str, Any]]] = {
        element_type: {} for element_type in element_types
    }
    for record in records:
        result = result_by_id.get(record["custom_id"])
        if result is None:
            raise RuntimeError(f"缺少概念描述结果: {record['concept']}")
        by_type[record["concept_type"]][record["concept"]] = {
            "fields": result["fields"],
            "description_schema": result["description_schema"],
            "description_schema_version": result["description_schema_version"],
            "source_count": record["source_count"],
            "source_sentences": record["source_sentences"][:8],
        }

    for element_type, rows in by_type.items():
        write_json_object(description_dir / f"{element_type}_descriptions.json", rows)

    summary = {
        "concept_dir": as_project_path(concept_dir, project_root),
        "output_dir": as_project_path(description_dir, project_root),
        "description_count": len(result_by_id),
        "description_counts": {
            element_type: len(rows) for element_type, rows in by_type.items()
        },
    }
    write_json_object(description_dir / "summary.json", summary)
    return summary


def load_description_records(
    concept_dir: Path,
    description_dir: Path | None,
    element_type: str,
) -> list[dict[str, Any]]:
    if description_dir is None:
        raise ValueError("聚类阶段必须提供结构化描述目录。")

    description_file = description_dir / f"{element_type}_descriptions.json"
    if not description_file.exists():
        raise FileNotFoundError(f"缺少结构化描述文件: {description_file}")
    generated = read_json_object(description_file)

    records: list[dict[str, Any]] = []
    for record in iter_pool_records(concept_dir, [element_type]):
        generated_item = generated.get(record["concept"])
        if not isinstance(generated_item, dict):
            raise ValueError(f"{element_type} 缺少结构化描述结果: {record['concept']}")
        fields = normalize_description_fields(
            element_type, generated_item.get("fields")
        )
        records.append(
            {
                "id": record["id"],
                "concept_type": element_type,
                "concept": record["concept"],
                "fields": fields,
                "source_count": record["source_count"],
            }
        )
    return records


def normalized_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.size == 0:
        return matrix.reshape((0, 0))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_description_field_texts(record: dict[str, Any]) -> dict[str, str]:
    element_type = str(record["concept_type"])
    fields = normalize_description_fields(element_type, record.get("fields"))
    return {
        field_name: (
            f"概念：{record['concept']}\n"
            f"概念类型：{element_type}\n"
            f"字段：{field_name}\n"
            f"内容：{fields[field_name]}"
        )
        for field_name in description_field_names(element_type)
    }


def combine_weighted_field_vectors(
    field_vectors_by_name: dict[str, list[list[float]]],
    weights: dict[str, float],
) -> list[list[float]]:
    pieces: list[np.ndarray] = []
    for field_name, vectors in field_vectors_by_name.items():
        weight = float(weights.get(field_name, 0.0))
        piece = normalized_matrix(vectors) * float(np.sqrt(max(weight, 0.0)))
        pieces.append(piece)
    if not pieces:
        return []
    return np.concatenate(pieces, axis=1).astype(np.float32).tolist()


def cosine_similarity_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = normalized_matrix(vectors)
    if matrix.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return matrix @ matrix.T


def ensure_description_embeddings(
    records: list[dict[str, Any]],
    embedding_dir: Path,
    element_type: str,
    embedding_client: EmbeddingClient,
    overwrite: bool = False,
) -> tuple[list[str], list[list[float]]]:
    vector_path = embedding_dir / f"{element_type}_description_vectors.npz"
    field_vector_path = embedding_dir / f"{element_type}_description_field_vectors.npz"
    meta_path = embedding_dir / f"{element_type}_description_vector_meta.json"
    fields_by_id = {
        record["id"]: normalize_description_fields(element_type, record.get("fields"))
        for record in records
    }
    field_weights = description_field_weights(element_type)
    if vector_path.exists() and meta_path.exists() and not overwrite:
        meta = read_json_object(meta_path)
        if (
            meta.get("description_schema_version") == concept_description.SCHEMA_VERSION
            and meta.get("field_weights") == field_weights
            and meta.get("fields") == fields_by_id
        ):
            return load_npz(vector_path)

    ids = [record["id"] for record in records]
    if not records:
        save_npz(vector_path, ids, [])
        write_json_object(
            meta_path,
            {
                "concepts": {},
                "fields": {},
                "field_weights": field_weights,
                "description_schema_version": concept_description.SCHEMA_VERSION,
            },
        )
        return ids, []

    field_names = description_field_names(element_type)
    field_texts_by_id = {
        record["id"]: build_description_field_texts(record) for record in records
    }
    field_vector_ids: list[str] = []
    field_texts: list[str] = []
    for record in records:
        for field_name in field_names:
            field_vector_ids.append(f"{record['id']}::{field_name}")
            field_texts.append(field_texts_by_id[record["id"]][field_name])

    embedded_field_vectors = embedding_client.embed(field_texts)
    if len(embedded_field_vectors) != len(field_texts):
        raise RuntimeError(
            f"{element_type} 字段 embedding 数量不匹配: "
            f"期望 {len(field_texts)}，实际 {len(embedded_field_vectors)}"
        )
    field_vectors_by_name = {field_name: [] for field_name in field_names}
    cursor = 0
    for _record in records:
        for field_name in field_names:
            field_vectors_by_name[field_name].append(embedded_field_vectors[cursor])
            cursor += 1

    vectors = combine_weighted_field_vectors(field_vectors_by_name, field_weights)
    save_npz(vector_path, ids, vectors)
    save_npz(field_vector_path, field_vector_ids, embedded_field_vectors)
    write_json_object(
        meta_path,
        {
            "concepts": {record["id"]: record["concept"] for record in records},
            "fields": fields_by_id,
            "field_weights": field_weights,
            "field_texts": field_texts_by_id,
            "field_vector_file": field_vector_path.name,
            "description_schema_version": concept_description.SCHEMA_VERSION,
        },
    )
    return ids, vectors


def similarity_distribution(vectors: list[list[float]]) -> dict[str, Any]:
    similarities = cosine_similarity_matrix(vectors)
    count = similarities.shape[0]
    if count <= 1:
        return {"count": int(count), "pair_count": 0}

    upper = similarities[np.triu_indices(count, k=1)]
    nearest = []
    for index in range(count):
        row = similarities[index].copy()
        row[index] = -1.0
        nearest.append(float(row.max()))

    def percentiles(values: np.ndarray | list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float32)
        return {
            "min": float(array.min()),
            "mean": float(array.mean()),
            "p50": float(np.percentile(array, 50)),
            "p75": float(np.percentile(array, 75)),
            "p85": float(np.percentile(array, 85)),
            "p90": float(np.percentile(array, 90)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
            "max": float(array.max()),
        }

    return {
        "count": int(count),
        "pair_count": int(len(upper)),
        "pair_similarity": percentiles(upper),
        "nearest_neighbor_similarity": percentiles(nearest),
    }


def fit_agglomerative_labels(
    vectors: list[list[float]], cluster_count: int
) -> np.ndarray:
    if cluster_count <= 0:
        raise ValueError("cluster_count 必须大于 0。")

    matrix = normalized_matrix(vectors)
    item_count = matrix.shape[0]
    if item_count == 0:
        return np.asarray([], dtype=np.int32)
    effective_cluster_count = min(cluster_count, item_count)
    if effective_cluster_count <= 1:
        return np.zeros(item_count, dtype=np.int32)

    try:
        import importlib

        cluster_module = importlib.import_module("sklearn.cluster")
    except ImportError as exc:
        raise RuntimeError(
            "Agglomerative 聚类需要 scikit-learn，请先运行 pip install -r requirements.txt。"
        ) from exc

    model_class = getattr(cluster_module, "AgglomerativeClustering")
    model = model_class(
        n_clusters=effective_cluster_count,
        metric="cosine",
        linkage="average",
    )
    return np.asarray(model.fit_predict(matrix), dtype=np.int32)


def sort_components(components: Iterable[Iterable[int]]) -> list[list[int]]:
    normalized = [
        sorted({int(index) for index in component}) for component in components
    ]
    normalized = [component for component in normalized if component]
    return sorted(normalized, key=lambda component: (-len(component), component[0]))


def build_cluster_payload_from_components(
    records: list[dict[str, Any]],
    similarities: np.ndarray,
    components: Iterable[Iterable[int]],
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for cluster_index, component in enumerate(sort_components(components), start=1):
        representative_index = choose_representative_index(component, similarities)
        pair_scores = [
            float(similarities[left, right])
            for offset, left in enumerate(component)
            for right in component[offset + 1 :]
        ]
        clusters.append(
            {
                "cluster_id": f"c{cluster_index:04d}",
                "size": len(component),
                "representative_concept": records[representative_index]["concept"],
                "representative_fields": records[representative_index].get(
                    "fields", {}
                ),
                "avg_similarity": float(np.mean(pair_scores)) if pair_scores else 1.0,
                "min_similarity": float(np.min(pair_scores)) if pair_scores else 1.0,
                "max_similarity": float(np.max(pair_scores)) if pair_scores else 1.0,
                "concepts": [
                    {
                        "concept": records[index]["concept"],
                        "fields": records[index].get("fields", {}),
                        "source_count": records[index].get("source_count", 0),
                    }
                    for index in component
                ],
            }
        )
    return clusters


def choose_representative_index(component: list[int], similarities: np.ndarray) -> int:
    if len(component) == 1:
        return component[0]
    best_index = component[0]
    best_score = -1.0
    for candidate in component:
        peer_scores = [
            float(similarities[candidate, peer])
            for peer in component
            if peer != candidate
        ]
        avg_score = float(np.mean(peer_scores)) if peer_scores else 1.0
        if avg_score > best_score:
            best_score = avg_score
            best_index = candidate
    return best_index


def cluster_by_agglomerative_count(
    records: list[dict[str, Any]],
    vectors: list[list[float]],
    cluster_count: int,
) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        return {
            "clusters": [],
            "requested_cluster_count": cluster_count,
            "effective_cluster_count": 0,
        }
    similarities = cosine_similarity_matrix(vectors)
    labels = fit_agglomerative_labels(vectors, cluster_count)
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        by_label[int(label)].append(index)
    clusters = build_cluster_payload_from_components(
        records,
        similarities,
        by_label.values(),
    )
    return {
        "clusters": clusters,
        "requested_cluster_count": cluster_count,
        "effective_cluster_count": min(cluster_count, count),
    }


def require_leiden_modules() -> tuple[Any, Any]:
    try:
        import importlib

        igraph_module = importlib.import_module("igraph")
        leidenalg_module = importlib.import_module("leidenalg")
    except ImportError as exc:
        raise RuntimeError(
            "Leiden 图聚类需要 igraph 和 leidenalg，请先运行 pip install -r requirements.txt。"
        ) from exc
    return igraph_module, leidenalg_module


def components_from_labels(labels: np.ndarray) -> list[list[int]]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        by_label[int(label)].append(index)
    return sort_components(by_label.values())


def knn_neighbor_lists(
    similarities: np.ndarray,
    knn_k: int,
    min_edge_similarity: float,
) -> list[list[int]]:
    count = similarities.shape[0]
    if count <= 1:
        return [[] for _ in range(count)]
    effective_k = min(max(knn_k, 1), count - 1)
    masked = similarities.copy()
    np.fill_diagonal(masked, -1.0)
    neighbors: list[list[int]] = []
    for index in range(count):
        row = masked[index]
        candidate_indices = np.argpartition(row, -effective_k)[-effective_k:]
        candidate_indices = candidate_indices[
            np.argsort(row[candidate_indices], kind="stable")[::-1]
        ]
        selected = [
            int(candidate)
            for candidate in candidate_indices.tolist()
            if float(row[candidate]) >= min_edge_similarity
        ]
        neighbors.append(selected)
    return neighbors


def fit_leiden_labels_from_similarity(
    similarities: np.ndarray,
    knn_k: int,
    resolution: float,
    min_edge_similarity: float,
    random_seed: int,
) -> np.ndarray:
    count = similarities.shape[0]
    if count == 0:
        return np.asarray([], dtype=np.int32)
    if count == 1:
        return np.zeros(1, dtype=np.int32)

    igraph_module, leidenalg_module = require_leiden_modules()
    neighbor_lists = knn_neighbor_lists(similarities, knn_k, min_edge_similarity)
    neighbor_sets = [set(items) for items in neighbor_lists]
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for left, peers in enumerate(neighbor_lists):
        for right in peers:
            if right <= left:
                continue
            if left not in neighbor_sets[right]:
                continue
            weight = float(similarities[left, right])
            if weight <= 0:
                continue
            edges.append((left, right))
            weights.append(weight)

    if not edges:
        return np.arange(count, dtype=np.int32)

    graph = igraph_module.Graph(n=count, edges=edges, directed=False)
    partition = leidenalg_module.find_partition(
        graph,
        leidenalg_module.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=float(max(resolution, 1e-6)),
        seed=int(random_seed),
    )
    labels = np.empty(count, dtype=np.int32)
    for label, component in enumerate(partition):
        for index in component:
            labels[int(index)] = int(label)
    return labels


def balanced_partition_score(
    components: list[list[int]],
    target_cluster_size: int,
    min_cluster_size: int,
    max_cluster_size: int,
) -> float:
    if not components:
        return float("inf")

    sizes = [len(component) for component in components]
    multi_sizes = [size for size in sizes if size > 1]
    score = 0.0
    for size in sizes:
        if size > max_cluster_size:
            score += float(size - max_cluster_size) * 8.0
        elif size == 1:
            score += 2.5
        elif size < min_cluster_size:
            score += float(min_cluster_size - size) * 1.5
        else:
            score += abs(float(size - target_cluster_size)) * 0.15

    if multi_sizes:
        median_size = float(np.median(multi_sizes))
        mean_size = float(np.mean(multi_sizes))
        score += abs(median_size - target_cluster_size) * 0.45
        score += abs(mean_size - target_cluster_size) * 0.15
    else:
        score += float(target_cluster_size) * 4.0

    if len(components) == 1 and sizes[0] > max_cluster_size:
        score += float(sizes[0]) * 10.0
    return score


def select_best_leiden_partition(
    similarities: np.ndarray,
    knn_k: int,
    min_edge_similarity: float,
    target_cluster_size: int,
    min_cluster_size: int,
    max_cluster_size: int,
    initial_resolution: float,
    resolution_multiplier: float,
    resolution_rounds: int,
    random_seed: int,
) -> dict[str, Any]:
    count = similarities.shape[0]
    if count == 0:
        return {
            "components": [],
            "selected_resolution": float(max(initial_resolution, 1e-6)),
            "attempts": [],
        }

    if count <= 1:
        return {
            "components": [list(range(count))],
            "selected_resolution": float(max(initial_resolution, 1e-6)),
            "attempts": [],
        }

    attempts: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    resolution = float(max(initial_resolution, 1e-6))
    rounds = max(int(resolution_rounds), 1)
    multiplier = float(max(resolution_multiplier, 1.0))

    for _ in range(rounds):
        labels = fit_leiden_labels_from_similarity(
            similarities=similarities,
            knn_k=knn_k,
            resolution=resolution,
            min_edge_similarity=min_edge_similarity,
            random_seed=random_seed,
        )
        components = components_from_labels(labels)
        score = balanced_partition_score(
            components,
            target_cluster_size=target_cluster_size,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )
        largest_cluster_size = max(
            (len(component) for component in components), default=0
        )
        attempt = {
            "resolution": round(resolution, 6),
            "cluster_count": len(components),
            "largest_cluster_size": largest_cluster_size,
            "score": round(score, 6),
        }
        attempts.append(attempt)
        candidate = {
            "components": components,
            "selected_resolution": float(resolution),
            "score": score,
            "largest_cluster_size": largest_cluster_size,
        }
        if best_result is None:
            best_result = candidate
        else:
            if candidate["score"] < best_result["score"] - 1e-9:
                best_result = candidate
            elif (
                abs(candidate["score"] - best_result["score"]) <= 1e-9
                and candidate["largest_cluster_size"]
                < best_result["largest_cluster_size"]
            ):
                best_result = candidate

        if multiplier <= 1.0:
            break
        resolution *= multiplier

    assert best_result is not None
    return {
        "components": best_result["components"],
        "selected_resolution": round(best_result["selected_resolution"], 6),
        "attempts": attempts,
    }


def refine_component_by_leiden(
    similarities: np.ndarray,
    component: list[int],
    knn_k: int,
    min_edge_similarity: float,
    target_cluster_size: int,
    min_cluster_size: int,
    max_cluster_size: int,
    initial_resolution: float,
    resolution_multiplier: float,
    resolution_rounds: int,
    max_split_depth: int,
    random_seed: int,
    depth: int = 0,
) -> tuple[list[list[int]], int]:
    if len(component) <= max_cluster_size or depth >= max_split_depth:
        return [sorted(component)], 0

    local_similarity = similarities[np.ix_(component, component)]
    selection = select_best_leiden_partition(
        similarities=local_similarity,
        knn_k=knn_k,
        min_edge_similarity=min_edge_similarity,
        target_cluster_size=target_cluster_size,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        initial_resolution=initial_resolution,
        resolution_multiplier=resolution_multiplier,
        resolution_rounds=resolution_rounds,
        random_seed=random_seed,
    )
    local_components = selection["components"]
    if len(local_components) <= 1:
        return [sorted(component)], 0

    mapped_components = [
        sorted(component[index] for index in local_component)
        for local_component in local_components
    ]
    if max((len(item) for item in mapped_components), default=0) >= len(component):
        return [sorted(component)], 0

    refined: list[list[int]] = []
    split_count = 1
    next_resolution = float(max(initial_resolution, 1e-6))
    for mapped_component in mapped_components:
        child_components, child_split_count = refine_component_by_leiden(
            similarities=similarities,
            component=mapped_component,
            knn_k=knn_k,
            min_edge_similarity=min_edge_similarity,
            target_cluster_size=target_cluster_size,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            initial_resolution=next_resolution,
            resolution_multiplier=resolution_multiplier,
            resolution_rounds=resolution_rounds,
            max_split_depth=max_split_depth,
            random_seed=random_seed,
            depth=depth + 1,
        )
        refined.extend(child_components)
        split_count += child_split_count
    return sort_components(refined), split_count


def absorb_small_components(
    components: list[list[int]],
    similarities: np.ndarray,
    min_cluster_size: int,
    max_cluster_size: int,
    similarity_threshold: float,
) -> tuple[list[list[int]], int]:
    if similarity_threshold <= 0 or min_cluster_size <= 1:
        return sort_components(components), 0

    mutable = [list(component) for component in sort_components(components)]
    absorbed_count = 0
    changed = True
    while changed:
        changed = False
        order = sorted(
            range(len(mutable)),
            key=lambda index: (
                len(mutable[index]),
                mutable[index][0] if mutable[index] else 10**9,
            ),
        )
        for source_index in order:
            source = mutable[source_index]
            if not source or len(source) >= min_cluster_size:
                continue

            best_target_index: int | None = None
            best_score = float(similarity_threshold)
            for target_index, target in enumerate(mutable):
                if target_index == source_index or not target:
                    continue
                if len(target) + len(source) > max_cluster_size:
                    continue
                cross_scores = similarities[np.ix_(source, target)]
                score = float(np.mean(cross_scores)) if cross_scores.size else -1.0
                if score > best_score:
                    best_score = score
                    best_target_index = target_index

            if best_target_index is None:
                continue

            mutable[best_target_index].extend(source)
            mutable[best_target_index] = sorted(mutable[best_target_index])
            mutable[source_index] = []
            absorbed_count += 1
            changed = True
            break

    return (
        sort_components(component for component in mutable if component),
        absorbed_count,
    )


def cluster_by_leiden_balanced(
    records: list[dict[str, Any]],
    vectors: list[list[float]],
    knn_k: int = 20,
    target_cluster_size: int = 24,
    min_cluster_size: int = 20,
    max_cluster_size: int = 30,
    initial_resolution: float = 0.2,
    resolution_multiplier: float = 1.5,
    resolution_rounds: int = 8,
    max_split_depth: int = 4,
    min_edge_similarity: float = 0.0,
    small_cluster_absorb_threshold: float = 0.0,
    random_seed: int = 0,
) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        return {
            "clusters": [],
            "selected_resolution": round(float(max(initial_resolution, 1e-6)), 6),
            "knn_k": int(knn_k),
            "recursive_split_count": 0,
            "absorbed_small_cluster_count": 0,
            "top_level_attempts": [],
        }
    if target_cluster_size <= 0:
        raise ValueError("target_cluster_size 必须大于 0。")
    if min_cluster_size <= 0:
        raise ValueError("min_cluster_size 必须大于 0。")
    if max_cluster_size < min_cluster_size:
        raise ValueError("max_cluster_size 不能小于 min_cluster_size。")
    if resolution_rounds < 1:
        raise ValueError("resolution_rounds 必须大于等于 1。")
    if max_split_depth < 0:
        raise ValueError("max_split_depth 必须大于等于 0。")

    similarities = cosine_similarity_matrix(vectors)
    top_level_target_cluster_size = max(target_cluster_size, max_cluster_size * 2)
    top_level_min_cluster_size = max(
        2,
        min(min_cluster_size, max(2, top_level_target_cluster_size // 2)),
    )
    top_level_max_cluster_size = max(
        max_cluster_size * 2, top_level_target_cluster_size
    )
    top_level = select_best_leiden_partition(
        similarities=similarities,
        knn_k=knn_k,
        min_edge_similarity=min_edge_similarity,
        target_cluster_size=top_level_target_cluster_size,
        min_cluster_size=top_level_min_cluster_size,
        max_cluster_size=top_level_max_cluster_size,
        initial_resolution=initial_resolution,
        resolution_multiplier=resolution_multiplier,
        resolution_rounds=resolution_rounds,
        random_seed=random_seed,
    )

    refined_components: list[list[int]] = []
    recursive_split_count = 0
    next_resolution = float(max(initial_resolution, 1e-6))
    for component in top_level["components"]:
        child_components, child_split_count = refine_component_by_leiden(
            similarities=similarities,
            component=component,
            knn_k=knn_k,
            min_edge_similarity=min_edge_similarity,
            target_cluster_size=target_cluster_size,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            initial_resolution=next_resolution,
            resolution_multiplier=resolution_multiplier,
            resolution_rounds=resolution_rounds,
            max_split_depth=max_split_depth,
            random_seed=random_seed,
        )
        refined_components.extend(child_components)
        recursive_split_count += child_split_count

    final_components, absorbed_small_cluster_count = absorb_small_components(
        components=refined_components,
        similarities=similarities,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        similarity_threshold=small_cluster_absorb_threshold,
    )

    return {
        "clusters": build_cluster_payload_from_components(
            records,
            similarities,
            final_components,
        ),
        "selected_resolution": float(top_level["selected_resolution"]),
        "knn_k": int(min(max(knn_k, 1), max(count - 1, 1))),
        "recursive_split_count": recursive_split_count,
        "absorbed_small_cluster_count": absorbed_small_cluster_count,
        "top_level_attempts": top_level["attempts"],
    }


def run_clustering(
    project_root: Path,
    concept_dir: Path,
    description_dir: Path | None,
    embedding_dir: Path,
    output_dir: Path,
    embedding_client: EmbeddingClient,
    element_types: list[str],
    cluster_counts_by_type: dict[str, list[int]] | None = None,
    cluster_method: str = CLUSTER_METHOD_LEIDEN_BALANCED,
    leiden_config: dict[str, Any] | None = None,
    overwrite_embeddings: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    active_logger = logger or build_logger("kg-concept-cluster")
    cluster_dir = output_dir / "clusters"
    reports_dir = output_dir / "reports"
    runs: list[dict[str, Any]] = []
    leiden_params = {
        "knn_k": 20,
        "target_cluster_size": 24,
        "min_cluster_size": 20,
        "max_cluster_size": 30,
        "initial_resolution": 0.2,
        "resolution_multiplier": 1.5,
        "resolution_rounds": 8,
        "max_split_depth": 4,
        "min_edge_similarity": 0.0,
        "small_cluster_absorb_threshold": 0.0,
        "random_seed": 0,
    }
    if leiden_config:
        leiden_params.update(leiden_config)

    for element_type in element_types:
        records = load_description_records(concept_dir, description_dir, element_type)
        ids, vectors = ensure_description_embeddings(
            records,
            embedding_dir,
            element_type,
            embedding_client,
            overwrite=overwrite_embeddings,
        )
        expected_ids = [record["id"] for record in records]
        if ids != expected_ids:
            vector_by_id = {
                item_id: vector for item_id, vector in zip(ids, vectors, strict=True)
            }
            vectors = [vector_by_id[item_id] for item_id in expected_ids]

        distribution = similarity_distribution(vectors)
        write_json_object(
            reports_dir / f"{element_type}_similarity_distribution.json", distribution
        )
        if cluster_method == CLUSTER_METHOD_AGGLOMERATIVE:
            cluster_counts = (cluster_counts_by_type or {}).get(element_type)
            if not cluster_counts:
                raise ValueError(f"缺少 {element_type} 的 cluster_count 配置。")
            for cluster_count in cluster_counts:
                cluster_payload = cluster_by_agglomerative_count(
                    records,
                    vectors,
                    cluster_count=cluster_count,
                )
                output_file = (
                    cluster_dir
                    / element_type
                    / f"{element_type}_agglomerative_k{cluster_count}.json"
                )
                payload = {
                    "concept_type": element_type,
                    "method": "agglomerative_weighted_field_cosine_average_linkage",
                    "parameters": {
                        "requested_cluster_count": cluster_count,
                        "effective_cluster_count": cluster_payload[
                            "effective_cluster_count"
                        ],
                        "metric": "weighted_field_cosine",
                        "linkage": "average",
                        "fields": list(description_field_names(element_type)),
                        "field_weights": description_field_weights(element_type),
                        "description_schema_version": concept_description.SCHEMA_VERSION,
                    },
                    "distribution_report": as_project_path(
                        reports_dir / f"{element_type}_similarity_distribution.json",
                        project_root,
                    ),
                    "cluster_count": len(cluster_payload["clusters"]),
                    "multi_member_cluster_count": sum(
                        1
                        for cluster in cluster_payload["clusters"]
                        if cluster["size"] > 1
                    ),
                    "clusters": cluster_payload["clusters"],
                }
                write_json_object(output_file, payload)
                runs.append(
                    {
                        "concept_type": element_type,
                        "cluster_method": cluster_method,
                        "cluster_file": as_project_path(output_file, project_root),
                        "requested_cluster_count": cluster_count,
                        "effective_cluster_count": payload["parameters"][
                            "effective_cluster_count"
                        ],
                        "cluster_count": payload["cluster_count"],
                        "multi_member_cluster_count": payload[
                            "multi_member_cluster_count"
                        ],
                    }
                )
                active_logger.info(
                    "%s k=%s 生成 %s 个聚类，其中多成员聚类 %s 个。",
                    element_type,
                    cluster_count,
                    payload["cluster_count"],
                    payload["multi_member_cluster_count"],
                )
            continue

        if cluster_method != CLUSTER_METHOD_LEIDEN_BALANCED:
            raise ValueError(f"未知聚类方法: {cluster_method}")

        cluster_payload = cluster_by_leiden_balanced(
            records,
            vectors,
            knn_k=int(leiden_params["knn_k"]),
            target_cluster_size=int(leiden_params["target_cluster_size"]),
            min_cluster_size=int(leiden_params["min_cluster_size"]),
            max_cluster_size=int(leiden_params["max_cluster_size"]),
            initial_resolution=float(leiden_params["initial_resolution"]),
            resolution_multiplier=float(leiden_params["resolution_multiplier"]),
            resolution_rounds=int(leiden_params["resolution_rounds"]),
            max_split_depth=int(leiden_params["max_split_depth"]),
            min_edge_similarity=float(leiden_params["min_edge_similarity"]),
            small_cluster_absorb_threshold=float(
                leiden_params["small_cluster_absorb_threshold"]
            ),
            random_seed=int(leiden_params["random_seed"]),
        )
        cluster_count = len(cluster_payload["clusters"])
        multi_member_cluster_count = sum(
            1 for cluster in cluster_payload["clusters"] if cluster["size"] > 1
        )
        payload = {
            "concept_type": element_type,
            "method": "leiden_balanced_mutual_knn_recursive",
            "parameters": {
                "metric": "weighted_field_cosine",
                "graph_type": "mutual_knn",
                "partitioner": "leiden_rb_configuration",
                "knn_k": cluster_payload["knn_k"],
                "target_cluster_size": int(leiden_params["target_cluster_size"]),
                "min_cluster_size": int(leiden_params["min_cluster_size"]),
                "max_cluster_size": int(leiden_params["max_cluster_size"]),
                "initial_resolution": float(leiden_params["initial_resolution"]),
                "selected_resolution": cluster_payload["selected_resolution"],
                "resolution_multiplier": float(leiden_params["resolution_multiplier"]),
                "resolution_rounds": int(leiden_params["resolution_rounds"]),
                "max_split_depth": int(leiden_params["max_split_depth"]),
                "min_edge_similarity": float(leiden_params["min_edge_similarity"]),
                "small_cluster_absorb_threshold": float(
                    leiden_params["small_cluster_absorb_threshold"]
                ),
                "recursive_split_count": cluster_payload["recursive_split_count"],
                "absorbed_small_cluster_count": cluster_payload[
                    "absorbed_small_cluster_count"
                ],
                "fields": list(description_field_names(element_type)),
                "field_weights": description_field_weights(element_type),
                "description_schema_version": concept_description.SCHEMA_VERSION,
            },
            "distribution_report": as_project_path(
                reports_dir / f"{element_type}_similarity_distribution.json",
                project_root,
            ),
            "cluster_count": cluster_count,
            "multi_member_cluster_count": multi_member_cluster_count,
            "clusters": cluster_payload["clusters"],
            "top_level_resolution_attempts": cluster_payload["top_level_attempts"],
        }
        output_file = (
            cluster_dir
            / element_type
            / build_leiden_cluster_filename(
                element_type=element_type,
                parameters=payload["parameters"],
                cluster_count=cluster_count,
                multi_member_cluster_count=multi_member_cluster_count,
            )
        )
        write_json_object(output_file, payload)
        runs.append(
            {
                "concept_type": element_type,
                "cluster_method": cluster_method,
                "cluster_file": as_project_path(output_file, project_root),
                "cluster_count": payload["cluster_count"],
                "multi_member_cluster_count": payload["multi_member_cluster_count"],
            }
        )
        active_logger.info(
            "%s Leiden 生成 %s 个聚类，其中多成员聚类 %s 个，输出 %s。",
            element_type,
            payload["cluster_count"],
            payload["multi_member_cluster_count"],
            output_file.name,
        )

    summary = {
        "concept_dir": as_project_path(concept_dir, project_root),
        "description_dir": (
            as_project_path(description_dir, project_root) if description_dir else None
        ),
        "embedding_dir": as_project_path(embedding_dir, project_root),
        "output_dir": as_project_path(output_dir, project_root),
        "cluster_method": cluster_method,
        "runs": runs,
    }
    write_json_object(output_dir / "cluster_summary.json", summary)
    return summary


def load_embedding_index(
    embedding_dir: Path, element_type: str
) -> tuple[dict[str, str], dict[str, list[float]]]:
    vector_file = embedding_dir / f"{element_type}_description_vectors.npz"
    meta_file = embedding_dir / f"{element_type}_description_vector_meta.json"
    ids, vectors = load_npz(vector_file)
    meta = read_json_object(meta_file)
    concepts = meta.get("concepts")
    if not isinstance(concepts, dict):
        raise ValueError(f"缺少向量概念元数据: {meta_file}")
    vector_by_id = {
        item_id: vector for item_id, vector in zip(ids, vectors, strict=True)
    }
    return {
        str(item_id): str(concept) for item_id, concept in concepts.items()
    }, vector_by_id


def unique_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def active_item_signature(item: dict[str, Any]) -> str:
    return json.dumps(
        sorted(unique_strings(item.get("source_concepts", []))),
        ensure_ascii=False,
    )


def candidate_signature(candidate_items: list[dict[str, Any]]) -> str:
    return json.dumps(
        sorted(active_item_signature(item) for item in candidate_items),
        ensure_ascii=False,
    )


def candidate_source_set(candidate_items: list[dict[str, Any]]) -> set[str]:
    return {
        source
        for item in candidate_items
        for source in unique_strings(item.get("source_concepts", []))
    }


def mean_source_vector(
    source_concepts: list[str], concept_to_vector: dict[str, list[float]]
) -> list[float]:
    vectors = [
        concept_to_vector[concept]
        for concept in source_concepts
        if concept in concept_to_vector
    ]
    if not vectors:
        return []
    vector = normalized_matrix(vectors).mean(axis=0)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return [float(value) for value in vector.tolist()]


def build_active_item(
    concept: str,
    fields: dict[str, str],
    source_concepts: list[str],
    concept_to_vector: dict[str, list[float]],
    reason: str = "",
) -> dict[str, Any]:
    clean_sources = unique_strings(source_concepts)
    return {
        "concept": concept,
        "fields": fields,
        "source_concepts": clean_sources,
        "vector": mean_source_vector(clean_sources, concept_to_vector),
        "reason": reason,
    }


def candidate_cohesion(indices: list[int], similarities: np.ndarray) -> float:
    pair_scores = [
        float(similarities[left, right])
        for offset, left in enumerate(indices)
        for right in indices[offset + 1 :]
    ]
    return float(np.mean(pair_scores)) if pair_scores else 1.0


def select_candidate_indices(
    active_items: list[dict[str, Any]],
    candidate_size: int,
    rejected_signatures: set[str],
    recent_no_progress_source_sets: list[set[str]] | None = None,
) -> list[int] | None:
    item_count = len(active_items)
    if item_count == 0:
        return None
    sample_size = min(max(candidate_size, 1), item_count)
    if sample_size == 1:
        single_choices = [
            index
            for index, item in enumerate(active_items)
            if candidate_signature([item]) not in rejected_signatures
        ]
        if not single_choices:
            return None
        single_choices.sort(
            key=lambda index: (
                str(active_items[index]["concept"]),
                active_item_signature(active_items[index]),
            )
        )
        return [single_choices[0]]

    vectors = [item["vector"] for item in active_items]
    similarities = cosine_similarity_matrix(vectors)
    multi_choices: list[tuple[float, tuple[str, ...], list[int], set[str]]] = []
    for seed_index in range(item_count):
        peer_indices = [index for index in range(item_count) if index != seed_index]
        peer_indices.sort(
            key=lambda index: (
                -float(similarities[seed_index, index]),
                str(active_items[index]["concept"]),
                active_item_signature(active_items[index]),
            )
        )
        window_size = sample_size - 1
        max_start = max(0, len(peer_indices) - window_size)
        for start in range(max_start + 1):
            selected = [seed_index, *peer_indices[start : start + window_size]]
            if len(selected) != sample_size:
                continue
            candidate_indices = sorted(selected)
            candidate_items = [active_items[index] for index in candidate_indices]
            signature = candidate_signature(candidate_items)
            if signature in rejected_signatures:
                continue
            tie_breaker = tuple(
                f"{active_items[index]['concept']}:{active_item_signature(active_items[index])}"
                for index in candidate_indices
            )
            multi_choices.append(
                (
                    candidate_cohesion(candidate_indices, similarities),
                    tie_breaker,
                    candidate_indices,
                    candidate_source_set(candidate_items),
                )
            )
    if not multi_choices:
        return None

    recent_source_sets = [
        source_set
        for source_set in (recent_no_progress_source_sets or [])[-8:]
        if source_set
    ]
    if recent_source_sets:
        target_external_count = max(1, sample_size // 4)
        bridge_choices: list[
            tuple[int, int, int, int, float, tuple[str, ...], list[int]]
        ] = []
        for recency_rank, source_set in enumerate(reversed(recent_source_sets)):
            for (
                cohesion,
                tie_breaker,
                candidate_indices,
                candidate_sources,
            ) in multi_choices:
                overlap_count = len(candidate_sources & source_set)
                external_count = len(candidate_sources - source_set)
                if overlap_count == 0 or external_count == 0:
                    continue
                bridge_choices.append(
                    (
                        recency_rank,
                        abs(external_count - target_external_count),
                        -external_count,
                        -overlap_count,
                        -cohesion,
                        tie_breaker,
                        candidate_indices,
                    )
                )
        if bridge_choices:
            bridge_choices.sort()
            return bridge_choices[0][6]

    multi_choices.sort(key=lambda item: (-item[0], item[1]))
    return multi_choices[0][2]


def candidate_pair_similarities(
    candidate_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    similarities = cosine_similarity_matrix(
        [item["vector"] for item in candidate_items]
    )
    pairs = [
        {
            "left": candidate_items[left]["concept"],
            "right": candidate_items[right]["concept"],
            "similarity": float(similarities[left, right]),
        }
        for left in range(len(candidate_items))
        for right in range(left + 1, len(candidate_items))
    ]
    return sorted(pairs, key=lambda item: item["similarity"], reverse=True)


def reference_group_signature(group: dict[str, Any]) -> str:
    return json.dumps(
        {
            "merged_concept": str(group.get("merged_concept") or ""),
            "source_concepts": sorted(unique_strings(group.get("source_concepts", []))),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def select_reference_groups(
    merged_concepts: dict[str, dict[str, Any]],
    candidate_items: list[dict[str, Any]],
    reference_size: int,
) -> list[dict[str, Any]]:
    groups = list(merged_concepts.values())
    if not groups or reference_size == 0:
        return sorted(groups, key=lambda item: str(item["merged_concept"]))
    candidate_vectors = normalized_matrix([item["vector"] for item in candidate_items])
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for group in groups:
        group_vector = normalized_matrix([group["vector"]])[0]
        score = float(np.max(candidate_vectors @ group_vector))
        scored.append((score, str(group["merged_concept"]), group))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [group for _, _, group in scored[: max(reference_size, 0)]]


def decision_signature(
    candidate_items: list[dict[str, Any]], reference_groups: list[dict[str, Any]]
) -> str:
    return json.dumps(
        {
            "candidate": candidate_signature(candidate_items),
            "references": [
                reference_group_signature(group) for group in reference_groups
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def source_to_candidate_map(
    candidate_items: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        source: item
        for item in candidate_items
        for source in item.get("source_concepts", [])
        if isinstance(source, str)
    }


def normalize_link_and_merge_decision(
    decision: dict[str, Any],
    candidate_items: list[dict[str, Any]],
    reference_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_by_source = source_to_candidate_map(candidate_items)
    valid_targets = {str(group["merged_concept"]): group for group in reference_groups}
    used_sources: set[str] = set()
    links: list[dict[str, Any]] = []

    for link in decision.get("links", []):
        if not isinstance(link, dict):
            continue
        target = str(link.get("target_merged_concept") or "").strip()
        if target not in valid_targets:
            continue
        raw_source_concepts = link.get("source_concepts", [])
        if not isinstance(raw_source_concepts, list):
            continue
        source_concepts = [
            source
            for source in unique_strings(raw_source_concepts)
            if source in candidate_by_source and source not in used_sources
        ]
        if not source_concepts:
            continue
        used_sources.update(source_concepts)
        links.append(
            {
                "target_merged_concept": target,
                "source_concepts": source_concepts,
                "reason": str(link.get("reason") or ""),
            }
        )

    new_merge_payload = decision.get("new_merge")
    new_merge: dict[str, Any] | None = None
    if (
        isinstance(new_merge_payload, dict)
        and new_merge_payload.get("should_merge") is True
    ):
        raw_merge_concepts = new_merge_payload.get("merge_concepts", [])
        if not isinstance(raw_merge_concepts, list):
            raw_merge_concepts = []
        merge_sources = [
            source
            for source in unique_strings(raw_merge_concepts)
            if source in candidate_by_source and source not in used_sources
        ]
        merged_concept = str(new_merge_payload.get("merged_concept") or "").strip()
        merged_description = str(
            new_merge_payload.get("merged_description") or ""
        ).strip()
        if len(merge_sources) >= 2 and merged_concept and merged_description:
            used_sources.update(merge_sources)
            new_merge = {
                "merge_concepts": merge_sources,
                "merged_concept": merged_concept,
                "merged_description": merged_description,
                "reason": str(new_merge_payload.get("reason") or ""),
            }

    return {
        "links": links,
        "new_merge": new_merge,
        "used_source_concepts": sorted(used_sources),
        "reason": str(decision.get("reason") or ""),
    }


def ensure_merged_concept_group(
    merged_concepts: dict[str, dict[str, Any]],
    cluster_id: str,
    merged_concept: str,
    merged_description: str,
    source_concepts: list[str],
    concept_to_vector: dict[str, list[float]],
    reason: str,
) -> dict[str, Any]:
    group = merged_concepts.get(merged_concept)
    if group is None:
        group = {
            "cluster_id": cluster_id,
            "merged_concept": merged_concept,
            "merged_description": merged_description or merged_concept,
            "source_concepts": [],
            "reasons": [],
            "vector": [],
        }
        merged_concepts[merged_concept] = group
    elif merged_description and not group.get("merged_description"):
        group["merged_description"] = merged_description

    current_sources = unique_strings(group.get("source_concepts", []))
    added_sources = [
        source
        for source in unique_strings(source_concepts)
        if source not in current_sources
    ]
    if added_sources:
        current_sources.extend(added_sources)
        group["source_concepts"] = current_sources
        group["vector"] = mean_source_vector(current_sources, concept_to_vector)
    if reason:
        reasons = group.setdefault("reasons", [])
        if reason not in reasons:
            reasons.append(reason)
    return group


def build_output_merge_groups(
    element_type: str,
    cluster_id: str,
    merged_concepts: dict[str, dict[str, Any]],
    pending_items: list[dict[str, Any]],
    start_index: int,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    output_items: list[dict[str, Any]] = []
    output_items.extend(merged_concepts.values())
    for item in pending_items:
        concept = str(item["concept"])
        output_items.append(
            {
                "cluster_id": cluster_id,
                "merged_concept": concept,
                "merged_description": concept,
                "source_concepts": unique_strings(item.get("source_concepts", [])),
                "reasons": ["未在迭代中归并或链接，作为保留概念输出。"],
            }
        )
    output_items.sort(
        key=lambda item: (str(item["merged_concept"]), reference_group_signature(item))
    )
    for offset, item in enumerate(output_items, start=start_index):
        reasons = [
            reason
            for reason in item.get("reasons", [])
            if isinstance(reason, str) and reason
        ]
        groups.append(
            {
                "group_id": f"{element_type}_merge_{offset:04d}",
                "cluster_id": cluster_id,
                "source_concepts": sorted(
                    unique_strings(item.get("source_concepts", []))
                ),
                "merged_concept": str(item["merged_concept"]),
                "merged_description": str(
                    item.get("merged_description") or item["merged_concept"]
                ),
                "reason": "；".join(reasons),
            }
        )
    return groups


def build_grouped_mapping(
    merged_groups: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped_mapping: dict[str, dict[str, Any]] = {}
    for group in merged_groups:
        merged_concept = str(group.get("merged_concept") or "").strip()
        if not merged_concept:
            continue
        merged_description = group.get("merged_description")
        entry = grouped_mapping.setdefault(
            merged_concept,
            {
                "merged_description": (
                    str(merged_description)
                    if isinstance(merged_description, str)
                    else ""
                ),
                "source_concepts": [],
                "group_ids": [],
                "cluster_ids": [],
            },
        )
        if not entry["merged_description"] and isinstance(merged_description, str):
            entry["merged_description"] = merged_description

        for concept in group.get("source_concepts", []):
            if isinstance(concept, str) and concept not in entry["source_concepts"]:
                entry["source_concepts"].append(concept)

        group_id = group.get("group_id")
        if isinstance(group_id, str) and group_id not in entry["group_ids"]:
            entry["group_ids"].append(group_id)

        cluster_id = group.get("cluster_id")
        if isinstance(cluster_id, str) and cluster_id not in entry["cluster_ids"]:
            entry["cluster_ids"].append(cluster_id)

    return {
        merged_concept: {
            "merged_description": item["merged_description"],
            "source_concepts": sorted(item["source_concepts"]),
            "group_ids": sorted(item["group_ids"]),
            "cluster_ids": sorted(item["cluster_ids"]),
        }
        for merged_concept, item in sorted(grouped_mapping.items())
    }


def load_concept_source_context_lookup(
    concept_dir: Path | None,
    element_type: str,
    source_sentence_limit: int,
) -> dict[str, dict[str, Any]]:
    if concept_dir is None:
        return {}
    file_path = concept_dir / POOL_FILES[element_type]
    if not file_path.exists():
        return {}

    lookup: dict[str, dict[str, Any]] = {}
    for concept, item in load_pool(concept_dir, element_type).items():
        concept_key = normalize_surface_text(concept)
        if not concept_key:
            continue
        raw_source_sentences = item.get("来源句")
        source_sentences = (
            unique_strings(raw_source_sentences)
            if isinstance(raw_source_sentences, list)
            else []
        )
        lookup[concept_key] = {
            "source_sentences": source_sentences[:source_sentence_limit],
            "source_count": len(source_sentences),
        }
    return lookup


def build_cluster_merge_prompt_items(
    element_type: str,
    cluster: dict[str, Any],
    source_context_lookup: dict[str, dict[str, Any]],
    source_sentence_limit: int,
) -> list[dict[str, Any]]:
    prompt_items: list[dict[str, Any]] = []
    for row in cluster.get("concepts", []):
        if not isinstance(row, dict):
            continue
        concept = str(row.get("concept") or "").strip()
        if not concept:
            continue
        try:
            fields = normalize_description_fields(element_type, row.get("fields"))
        except ValueError as exc:
            raise ValueError(
                f"聚类文件中的 {element_type} 概念缺少结构化 fields: {concept}"
            ) from exc

        source_context = source_context_lookup.get(normalize_surface_text(concept), {})
        row_source_sentences = row.get("source_sentences")
        source_sentences = (
            unique_strings(row_source_sentences)
            if isinstance(row_source_sentences, list)
            else []
        )
        if not source_sentences:
            source_sentences = list(source_context.get("source_sentences", []))
        source_sentences = source_sentences[:source_sentence_limit]

        raw_source_count = row.get("source_count")
        source_count = raw_source_count if isinstance(raw_source_count, int) else 0
        source_count = max(
            source_count,
            len(source_sentences),
            int(source_context.get("source_count") or 0),
        )

        prompt_items.append(
            {
                "concept": concept,
                "fields": fields,
                "source_sentences": source_sentences,
                "source_count": source_count,
            }
        )
    return prompt_items


def normalize_cluster_merge_decision(
    decision: dict[str, Any],
    cluster_items: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_concepts = {str(item["concept"]): item for item in cluster_items}
    used_sources: set[str] = set()
    merge_groups: list[dict[str, Any]] = []
    invalid_groups: list[dict[str, Any]] = []

    for index, group in enumerate(decision.get("merge_groups", []), start=1):
        if not isinstance(group, dict):
            invalid_groups.append({"index": index, "reason": "merge_groups 项不是对象"})
            continue

        raw_source_concepts = group.get("source_concepts", [])
        if not isinstance(raw_source_concepts, list):
            invalid_groups.append(
                {"index": index, "reason": "source_concepts 不是列表"}
            )
            continue

        source_concepts = [
            concept
            for concept in unique_strings(raw_source_concepts)
            if concept in valid_concepts and concept not in used_sources
        ]
        merged_concept = str(group.get("merged_concept") or "").strip()
        merged_description = str(group.get("merged_description") or "").strip()
        if len(source_concepts) < 2 or not merged_concept or not merged_description:
            invalid_groups.append(
                {
                    "index": index,
                    "reason": "有效 source_concepts 少于 2 个，或缺少 merged_concept / merged_description",
                }
            )
            continue

        used_sources.update(source_concepts)
        merge_groups.append(
            {
                "source_concepts": source_concepts,
                "merged_concept": merged_concept,
                "merged_description": merged_description,
                "reason": str(group.get("reason") or ""),
            }
        )

    return {
        "merge_groups": merge_groups,
        "used_source_concepts": sorted(used_sources),
        "invalid_group_count": len(invalid_groups),
        "invalid_groups": invalid_groups,
        "reason": str(decision.get("reason") or ""),
    }


def build_partition_output_merge_groups(
    element_type: str,
    cluster_id: str,
    cluster_items: list[dict[str, Any]],
    accepted_merge_groups: list[dict[str, Any]],
    start_index: int,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    used_sources = {
        source
        for group in accepted_merge_groups
        for source in unique_strings(group.get("source_concepts", []))
    }

    output_groups = [
        {
            "cluster_id": cluster_id,
            "source_concepts": unique_strings(group.get("source_concepts", [])),
            "merged_concept": str(group.get("merged_concept") or "").strip(),
            "merged_description": str(
                group.get("merged_description") or group.get("merged_concept") or ""
            ).strip(),
            "reason": str(group.get("reason") or ""),
        }
        for group in accepted_merge_groups
    ]
    for item in cluster_items:
        concept = str(item["concept"])
        if concept in used_sources:
            continue
        output_groups.append(
            {
                "cluster_id": cluster_id,
                "source_concepts": [concept],
                "merged_concept": concept,
                "merged_description": concept,
                "reason": "未被纳入任何 merge_group，保留原概念。",
            }
        )

    for offset, group in enumerate(output_groups, start=start_index):
        groups.append(
            {
                "group_id": f"{element_type}_merge_{offset:04d}",
                "cluster_id": cluster_id,
                "source_concepts": unique_strings(group.get("source_concepts", [])),
                "merged_concept": str(group.get("merged_concept") or "").strip(),
                "merged_description": str(
                    group.get("merged_description") or group.get("merged_concept") or ""
                ).strip(),
                "reason": str(group.get("reason") or ""),
            }
        )
    return groups


def run_merge_mapping(
    project_root: Path,
    cluster_file: Path,
    embedding_dir: Path,
    concept_dir: Path | None,
    output_dir: Path,
    executor: Any,
    mode: str,
    candidate_size: int = 16,
    reference_size: int = 24,
    max_iterations_per_cluster: int = 500,
    source_sentence_limit: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    overwrite: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    del embedding_dir, candidate_size, reference_size, max_iterations_per_cluster
    if source_sentence_limit < 0:
        raise ValueError("source_sentence_limit 必须大于等于 0。")

    active_logger = logger or build_logger("kg-concept-merge")
    cluster_payload = read_json_object(cluster_file)
    element_type = str(cluster_payload["concept_type"])
    source_context_lookup = load_concept_source_context_lookup(
        concept_dir,
        element_type,
        source_sentence_limit=source_sentence_limit,
    )

    decision_file = output_dir / build_merge_artifact_filename(
        element_type=element_type,
        artifact_type="merge_decisions",
        cluster_file=cluster_file,
        extension=".jsonl",
    )
    if overwrite and decision_file.exists():
        decision_file.unlink()
    decision_by_id = {
        row["custom_id"]: row["decision"]
        for row in read_jsonl_if_exists(decision_file)
        if row.get("algorithm") == MERGE_ALGORITHM
        if isinstance(row.get("custom_id"), str)
        and isinstance(row.get("decision"), dict)
    }

    merged_groups: list[dict[str, Any]] = []
    cluster_reports: list[dict[str, Any]] = []

    for cluster in cluster_payload.get("clusters", []):
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id"))
        cluster_items = build_cluster_merge_prompt_items(
            element_type=element_type,
            cluster=cluster,
            source_context_lookup=source_context_lookup,
            source_sentence_limit=source_sentence_limit,
        )
        initial_concept_count = len(cluster_items)
        if not cluster_items:
            cluster_reports.append(
                {
                    "cluster_id": cluster_id,
                    "input_concept_count": 0,
                    "accepted_merge_group_count": 0,
                    "preserved_singleton_count": 0,
                    "invalid_group_count": 0,
                    "final_group_count": 0,
                    "llm_called": False,
                    "exit_reason": "cluster_empty",
                }
            )
            continue

        decision: dict[str, Any] = {"merge_groups": [], "reason": ""}
        llm_called = len(cluster_items) >= 2
        exit_reason = (
            "single_pass_skipped_singleton"
            if len(cluster_items) == 1
            else "single_pass_completed"
        )

        if llm_called:
            signature_payload = {
                "cluster_id": cluster_id,
                "concept_type": element_type,
                "concepts": cluster_items,
                "source_sentence_limit": source_sentence_limit,
            }
            signature_text = json.dumps(
                signature_payload,
                ensure_ascii=False,
                sort_keys=True,
            )
            custom_id = f"{element_type}_{cluster_id}_{stable_hash8(MERGE_ALGORITHM + ':' + signature_text)}"
            if custom_id in decision_by_id:
                decision = decision_by_id[custom_id]
            else:
                result = executor.run_prompt_tasks(
                    stage_name="concept_abstraction",
                    task_name=f"merge_mapping_{custom_id}",
                    prompt_module=concept_merge,
                    tasks=[
                        PromptTask(
                            custom_id=custom_id,
                            context={
                                "concept_type": element_type,
                                "cluster_id": cluster_id,
                                "concepts": cluster_items,
                            },
                        )
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    mode=mode,
                )
                decision = result[custom_id]
                decision_by_id[custom_id] = decision
                append_jsonl(
                    decision_file,
                    [
                        {
                            "algorithm": MERGE_ALGORITHM,
                            "custom_id": custom_id,
                            "cluster_id": cluster_id,
                            "cluster_signature": stable_hash8(signature_text),
                            "concepts": [
                                str(item["concept"]) for item in cluster_items
                            ],
                            "source_sentence_limit": source_sentence_limit,
                            "decision": decision,
                        }
                    ],
                )

        normalized_decision = normalize_cluster_merge_decision(
            decision,
            cluster_items=cluster_items,
        )
        final_groups = build_partition_output_merge_groups(
            element_type=element_type,
            cluster_id=cluster_id,
            cluster_items=cluster_items,
            accepted_merge_groups=normalized_decision["merge_groups"],
            start_index=len(merged_groups) + 1,
        )
        merged_groups.extend(final_groups)
        preserved_singleton_count = sum(
            1 for group in final_groups if len(group.get("source_concepts", [])) == 1
        )
        cluster_reports.append(
            {
                "cluster_id": cluster_id,
                "input_concept_count": initial_concept_count,
                "accepted_merge_group_count": len(normalized_decision["merge_groups"]),
                "preserved_singleton_count": preserved_singleton_count,
                "invalid_group_count": normalized_decision["invalid_group_count"],
                "final_group_count": len(final_groups),
                "llm_called": llm_called,
                "exit_reason": exit_reason,
            }
        )
        active_logger.info(
            "完成聚类 %s 的整簇概念归并：最终概念 %s 个，接受归并组 %s 个，保留单概念 %s 个。",
            cluster_id,
            len(final_groups),
            len(normalized_decision["merge_groups"]),
            preserved_singleton_count,
        )

    grouped_mapping = build_grouped_mapping(merged_groups)
    created_merge_count = sum(
        1 for group in merged_groups if len(group.get("source_concepts", [])) >= 2
    )
    preserved_singleton_count = sum(
        1 for group in merged_groups if len(group.get("source_concepts", [])) == 1
    )
    payload = {
        "concept_type": element_type,
        "algorithm": MERGE_ALGORITHM,
        "cluster_file": as_project_path(cluster_file, project_root),
        "decision_file": as_project_path(decision_file, project_root),
        "concept_dir": (
            as_project_path(concept_dir, project_root)
            if concept_dir is not None
            else ""
        ),
        "parameters": {
            "source_sentence_limit": source_sentence_limit,
            "cluster_call_strategy": "single_llm_call_per_cluster",
            "singleton_strategy": "auto_preserve_if_not_in_merge_groups",
        },
        "mapping_count": sum(
            len(group.get("source_concepts", [])) for group in merged_groups
        ),
        "merged_concept_count": len(grouped_mapping),
        "merged_group_count": len(merged_groups),
        "created_merge_count": created_merge_count,
        "preserved_singleton_count": preserved_singleton_count,
        "mapping": grouped_mapping,
        "merged_groups": merged_groups,
        "cluster_reports": cluster_reports,
    }
    mapping_file = output_dir / build_merge_artifact_filename(
        element_type=element_type,
        artifact_type="mapping",
        cluster_file=cluster_file,
        extension=".json",
    )
    payload["mapping_file"] = as_project_path(mapping_file, project_root)
    write_json_object(mapping_file, payload)
    return payload


def load_mapping(mapping_dir: Path, element_type: str) -> dict[str, dict[str, Any]]:
    mapping_file = resolve_mapping_file(mapping_dir, element_type)
    if mapping_file is None or not mapping_file.exists():
        return {}
    payload = read_json_object(mapping_file)
    mapping = payload.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError(f"映射表格式错误: {mapping_file}")

    flattened: dict[str, dict[str, Any]] = {}
    for mapping_key, item in mapping.items():
        merged_concept = str(mapping_key)
        if isinstance(item, list):
            for source in item:
                if isinstance(source, str):
                    flattened[source] = {"merged_concept": merged_concept}
            continue

        if not isinstance(item, dict):
            continue

        source_concepts = item.get("source_concepts")
        if isinstance(source_concepts, list):
            merged_description = item.get("merged_description")
            group_ids = item.get("group_ids")
            cluster_ids = item.get("cluster_ids")
            base_item = {
                "merged_concept": merged_concept,
                "merged_description": (
                    str(merged_description)
                    if isinstance(merged_description, str)
                    else ""
                ),
                "group_ids": (
                    [group_id for group_id in group_ids if isinstance(group_id, str)]
                    if isinstance(group_ids, list)
                    else []
                ),
                "cluster_ids": (
                    [
                        cluster_id
                        for cluster_id in cluster_ids
                        if isinstance(cluster_id, str)
                    ]
                    if isinstance(cluster_ids, list)
                    else []
                ),
            }
            for source in source_concepts:
                if isinstance(source, str):
                    flattened[source] = dict(base_item)
            continue

        flattened[merged_concept] = item

    if mapping and not flattened:
        raise ValueError(f"映射表格式错误: {mapping_file}")
    return flattened


def map_value(value: Any, mapping: dict[str, dict[str, Any]]) -> Any:
    if isinstance(value, list):
        mapped_values: list[Any] = []
        seen: set[str] = set()
        for item in value:
            mapped = map_value(item, mapping)
            values = mapped if isinstance(mapped, list) else [mapped]
            for inner in values:
                marker = json.dumps(inner, ensure_ascii=False, sort_keys=True)
                if marker in seen:
                    continue
                seen.add(marker)
                mapped_values.append(inner)
        return mapped_values[0] if len(mapped_values) == 1 else mapped_values
    if isinstance(value, str) and value in mapping:
        merged_concept = mapping[value].get("merged_concept")
        return merged_concept if isinstance(merged_concept, str) else value
    return value


def tuple_signature(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "subject": row.get("subject"),
            "relation": row.get("relation"),
            "object": row.get("object"),
            "location": row.get("location"),
            "time": row.get("time"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def run_apply_mapping(
    project_root: Path,
    tuple_dir: Path,
    mapping_dir: Path,
    output_dir: Path,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    active_logger = logger or build_logger("kg-concept-apply-mapping")
    mappings = {
        element_type: load_mapping(mapping_dir, element_type)
        for element_type in ELEMENT_TYPES
    }

    report_rows: list[dict[str, Any]] = []
    total_before = 0
    total_after = 0
    for tuple_file in sorted(tuple_dir.glob("*.json")):
        rows = read_json_array(tuple_file)
        total_before += len(rows)
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        duplicate_count = 0
        replacement_count = 0
        for row in rows:
            new_row = dict(row)
            old_projection = tuple_signature(new_row)
            new_row["subject"] = map_value(new_row.get("subject"), mappings["entity"])
            new_row["object"] = map_value(new_row.get("object"), mappings["entity"])
            new_row["relation"] = map_value(
                new_row.get("relation"), mappings["relation"]
            )
            new_row["location"] = map_value(
                new_row.get("location"), mappings["location"]
            )
            if tuple_signature(new_row) != old_projection:
                replacement_count += 1
            signature = tuple_signature(new_row)
            if signature in seen:
                duplicate_count += 1
                continue
            seen.add(signature)
            deduped.append(new_row)
        total_after += len(deduped)
        write_json_array(output_dir / tuple_file.name, deduped)
        report_rows.append(
            {
                "file": tuple_file.name,
                "before_count": len(rows),
                "after_count": len(deduped),
                "replacement_count": replacement_count,
                "duplicate_count": duplicate_count,
            }
        )
        active_logger.info("完成映射替换: %s", tuple_file.name)

    report = {
        "tuple_dir": as_project_path(tuple_dir, project_root),
        "mapping_dir": as_project_path(mapping_dir, project_root),
        "output_dir": as_project_path(output_dir, project_root),
        "before_tuple_count": total_before,
        "after_tuple_count": total_after,
        "removed_duplicate_count": total_before - total_after,
        "files": report_rows,
    }
    write_json_object(output_dir / "apply_mapping_report.json", report)
    return report
