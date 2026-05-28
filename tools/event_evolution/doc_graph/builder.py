from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from core.config import AppConfig
from core.embed_client import EmbeddingClient
from tools.event_evolution.manifest_utils import extract_node_id, split_title_stem
from tools.event_evolution.time_utils import build_boundary_date, parse_title_time_info
from utils.io_utils import ensure_parent_dir, stable_hash8, write_json, write_jsonl
from utils.text_utils import split_sentences

TIMELINE_SUFFIX_PATTERN = re.compile(r"＜(?P<label>[^＜＞]+)＞\s*$")
DOC_ID_PREFIX_PATTERN = re.compile(r"^(?P<prefix>[A-Za-z]+)\d{3}")


class SentenceEmbeddingProvider(Protocol):
    """句嵌入提供器协议。"""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """把句子列表编码成向量列表。"""
        ...


class SiliconFlowSentenceEmbedder:
    """基于当前工程配置的句嵌入模型封装。"""

    def __init__(
        self,
        project_root: str | Path,
        *,
        embed_dimensions: int | None = None,
        embed_batch_size: int | None = None,
    ):
        config = AppConfig.load(project_root)
        if embed_dimensions is not None:
            config.embed_dimensions = int(embed_dimensions)
        if embed_batch_size is not None:
            config.embed_batch_size = int(embed_batch_size)
        if not config.api_key:
            raise RuntimeError(
                "当前未配置 SiliconFlow API Key，无法按要求调用句嵌入模型。"
            )
        self.client = EmbeddingClient(config)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(texts)


@dataclass(slots=True)
class ParsedTimelineLabel:
    """timeline 文件名中结束时间标签的结构化结果。"""

    label: str
    normalized: str | None
    pattern: str
    precision: str
    year: int | None
    month: int | None
    day: int | None
    anchor_date: str | None
    ordinal: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "normalized": self.normalized,
            "pattern": self.pattern,
            "precision": self.precision,
            "year": self.year,
            "month": self.month,
            "day": self.day,
            "anchor_date": self.anchor_date,
        }


@dataclass(slots=True)
class SentenceRecord:
    """单条句子记录。"""

    sentence_id: str
    doc_id: str
    sentence_index: int
    text: str

    def to_row(self) -> dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "doc_id": self.doc_id,
            "sentence_index": self.sentence_index,
            "text": self.text,
        }


@dataclass(slots=True)
class DocumentRecord:
    """单篇文档的文本、时间和句向量集合。"""

    doc_id: str
    title: str
    short_title: str
    text_source_path: str
    timeline_source_path: str | None
    time_info: ParsedTimelineLabel
    title_malformed: bool
    sentence_records: list[SentenceRecord]
    sentence_vectors: np.ndarray
    sentence_set_path: str
    preview: str

    @property
    def category(self) -> str:
        match = DOC_ID_PREFIX_PATTERN.match(self.doc_id)
        if match:
            return match.group("prefix")
        return "UNKNOWN"

    def to_manifest_row(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "short_title": self.short_title,
            "category": self.category,
            "text_source_path": self.text_source_path,
            "timeline_source_path": self.timeline_source_path,
            "time_label": self.time_info.label,
            "time_normalized": self.time_info.normalized,
            "time_precision": self.time_info.precision,
            "time_anchor_date": self.time_info.anchor_date,
            "time_pattern": self.time_info.pattern,
            "title_malformed": self.title_malformed,
            "sentence_count": len(self.sentence_records),
            "sentence_set_path": self.sentence_set_path,
            "preview": self.preview,
        }


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _extract_timeline_label(stem: str) -> str:
    match = TIMELINE_SUFFIX_PATTERN.search(stem)
    if match:
        return match.group("label").strip() or "UNKNOWN_TIME"
    return "UNKNOWN_TIME"


def _strip_doc_prefix(title: str, doc_id: str) -> str:
    if title.startswith(doc_id):
        stripped = title[len(doc_id) :].strip()
        if stripped:
            return stripped
    return title.strip()


def parse_timeline_label(label: str) -> ParsedTimelineLabel:
    """把 timeline 文件名中的结束时间转换成可比较结构。"""

    cleaned = label.strip() or "UNKNOWN_TIME"
    if cleaned.upper() == "UNKNOWN_TIME":
        return ParsedTimelineLabel(
            label="UNKNOWN_TIME",
            normalized=None,
            pattern="unknown",
            precision="unknown",
            year=None,
            month=None,
            day=None,
            anchor_date=None,
            ordinal=None,
        )

    title_info = parse_title_time_info(cleaned)
    candidate = title_info.candidate
    if candidate is None:
        return ParsedTimelineLabel(
            label=cleaned,
            normalized=None,
            pattern=title_info.pattern,
            precision="unknown",
            year=None,
            month=None,
            day=None,
            anchor_date=None,
            ordinal=None,
        )

    anchor = build_boundary_date(
        year=candidate.year,
        month=candidate.month,
        day=candidate.day,
        is_end=True,
    )
    return ParsedTimelineLabel(
        label=cleaned,
        normalized=candidate.normalized,
        pattern=title_info.pattern,
        precision=candidate.precision,
        year=candidate.year,
        month=candidate.month,
        day=candidate.day,
        anchor_date=anchor.isoformat(),
        ordinal=anchor.toordinal(),
    )


def _collect_sentences(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n+", text) if part.strip()]
    sentences: list[str] = []
    for paragraph in paragraphs:
        parts = split_sentences(paragraph)
        if parts:
            sentences.extend(part.strip() for part in parts if part.strip())
        else:
            sentences.append(paragraph)
    if not sentences and text.strip():
        return [text.strip()]
    return sentences


def _build_timeline_index(timeline_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not timeline_dir.exists():
        return index

    for path in sorted(timeline_dir.glob("*.txt")):
        if not path.is_file():
            continue
        title, _, _ = split_title_stem(path.stem)
        doc_id = extract_node_id(title)
        index[doc_id] = path
    return index


def _save_sentence_set(
    output_dir: Path,
    doc_id: str,
    sentence_records: list[SentenceRecord],
    sentence_vectors: np.ndarray,
) -> Path:
    sentence_set_path = output_dir / "sentence_sets" / f"{doc_id}.npz"
    ensure_parent_dir(sentence_set_path)
    np.savez_compressed(
        sentence_set_path,
        sentence_ids=np.array([item.sentence_id for item in sentence_records]),
        sentence_indices=np.array([item.sentence_index for item in sentence_records]),
        sentences=np.array([item.text for item in sentence_records], dtype=object),
        vectors=sentence_vectors.astype(np.float32),
    )
    return sentence_set_path


def _load_documents(
    input_dir: Path,
    timeline_dir: Path,
    output_dir: Path,
    embedder: SentenceEmbeddingProvider,
    limit: int | None,
) -> list[DocumentRecord]:
    text_paths = sorted([path for path in input_dir.glob("*.txt") if path.is_file()])
    if limit is not None and limit > 0:
        text_paths = text_paths[:limit]
    if not text_paths:
        raise RuntimeError(f"未在目录中找到 txt 文档: {input_dir}")

    timeline_index = _build_timeline_index(timeline_dir)
    raw_documents: list[dict[str, Any]] = []
    all_sentences: list[str] = []
    for text_path in text_paths:
        title, _, malformed = split_title_stem(text_path.stem)
        doc_id = extract_node_id(title)
        short_title = _strip_doc_prefix(title, doc_id)
        timeline_path = timeline_index.get(doc_id)
        time_info = parse_timeline_label(
            _extract_timeline_label(timeline_path.stem)
            if timeline_path
            else "UNKNOWN_TIME"
        )
        text = text_path.read_text(encoding="utf-8").strip()
        sentences = _collect_sentences(text)
        if not sentences:
            sentences = [short_title or title]

        sentence_records: list[SentenceRecord] = []
        for sentence_index, sentence in enumerate(sentences):
            sentence_id = f"{doc_id}_s{sentence_index:03d}_{stable_hash8(sentence)}"
            sentence_records.append(
                SentenceRecord(
                    sentence_id=sentence_id,
                    doc_id=doc_id,
                    sentence_index=sentence_index,
                    text=sentence,
                )
            )
            all_sentences.append(sentence)

        raw_documents.append(
            {
                "doc_id": doc_id,
                "title": title,
                "short_title": short_title,
                "text_source_path": text_path.as_posix(),
                "timeline_source_path": (
                    None if timeline_path is None else timeline_path.as_posix()
                ),
                "time_info": time_info,
                "title_malformed": malformed,
                "sentence_records": sentence_records,
                "preview": " ".join(sentences[:2])[:200],
            }
        )

    vectors = np.array(embedder.embed_texts(all_sentences), dtype=np.float32)
    vectors = _normalize_rows(vectors)

    documents: list[DocumentRecord] = []
    offset = 0
    for raw_document in raw_documents:
        sentence_records = raw_document["sentence_records"]
        next_offset = offset + len(sentence_records)
        sentence_vectors = vectors[offset:next_offset]
        offset = next_offset
        sentence_set_path = _save_sentence_set(
            output_dir=output_dir,
            doc_id=raw_document["doc_id"],
            sentence_records=sentence_records,
            sentence_vectors=sentence_vectors,
        )
        documents.append(
            DocumentRecord(
                doc_id=raw_document["doc_id"],
                title=raw_document["title"],
                short_title=raw_document["short_title"],
                text_source_path=raw_document["text_source_path"],
                timeline_source_path=raw_document["timeline_source_path"],
                time_info=raw_document["time_info"],
                title_malformed=raw_document["title_malformed"],
                sentence_records=sentence_records,
                sentence_vectors=sentence_vectors,
                sentence_set_path=sentence_set_path.as_posix(),
                preview=raw_document["preview"],
            )
        )
    return documents


def _compute_directed_similarity(
    anchor_vectors: np.ndarray, candidate_vectors: np.ndarray
) -> float:
    """按用户定义计算候选历史文档到锚点文档的集合相似度。"""

    if anchor_vectors.size == 0 or candidate_vectors.size == 0:
        return float("nan")

    # 对锚点文档每个句向量，计算其与候选文档所有句向量的平均余弦相似度。
    # 再对锚点文档所有句子的平均相似度取最大值，得到候选文档到锚点文档的权重。
    candidate_mean_vector = np.mean(candidate_vectors, axis=0, dtype=np.float32)
    sentence_mean_scores = anchor_vectors @ candidate_mean_vector
    return float(np.max(sentence_mean_scores))


def _build_graph(
    documents: list[DocumentRecord],
) -> tuple[list[DocumentRecord], list[dict[str, Any]], np.ndarray, np.ndarray]:
    ordered_documents = sorted(
        documents,
        key=lambda item: (
            item.time_info.ordinal is None,
            item.time_info.ordinal or 0,
            item.doc_id,
        ),
    )
    total = len(ordered_documents)
    weight_matrix = np.full((total, total), np.nan, dtype=np.float32)
    eligible_mask = np.zeros((total, total), dtype=bool)
    edges: list[dict[str, Any]] = []

    for anchor_index, anchor_document in enumerate(ordered_documents):
        anchor_ordinal = anchor_document.time_info.ordinal
        if anchor_ordinal is None:
            continue

        for candidate_index, candidate_document in enumerate(ordered_documents):
            if candidate_index == anchor_index:
                continue
            candidate_ordinal = candidate_document.time_info.ordinal
            if candidate_ordinal is None or candidate_ordinal > anchor_ordinal:
                continue

            eligible_mask[candidate_index, anchor_index] = True
            score = _compute_directed_similarity(
                anchor_vectors=anchor_document.sentence_vectors,
                candidate_vectors=candidate_document.sentence_vectors,
            )
            weight_matrix[candidate_index, anchor_index] = score
            edges.append(
                {
                    "source": candidate_document.doc_id,
                    "target": anchor_document.doc_id,
                    "weight": round(score, 6),
                    "source_time": candidate_document.time_info.normalized,
                    "target_time": anchor_document.time_info.normalized,
                    "source_sentence_count": len(candidate_document.sentence_records),
                    "target_sentence_count": len(anchor_document.sentence_records),
                }
            )

    edges.sort(key=lambda item: item["weight"], reverse=True)
    return ordered_documents, edges, weight_matrix, eligible_mask


def _build_graph_payload(
    documents: list[DocumentRecord],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    nodes = [document.to_manifest_row() for document in documents]
    known_time_documents = sum(
        1 for document in documents if document.time_info.ordinal is not None
    )
    unknown_time_documents = len(documents) - known_time_documents
    return {
        "meta": {
            "document_count": len(documents),
            "edge_count": len(edges),
            "known_time_documents": known_time_documents,
            "unknown_time_documents": unknown_time_documents,
            "similarity_definition": {
                "name": "max_anchor_sentence_average_cosine",
                "formula": "score(candidate->anchor)=max_i mean_j cosine(anchor_sentence_i, candidate_sentence_j)",
                "space": "unit_hypersphere",
                "metric": "cosine_similarity",
            },
            "edge_direction": "candidate_history_document -> anchor_document",
            "candidate_rule": "candidate.end_time <= anchor.end_time and candidate != anchor",
        },
        "nodes": nodes,
        "edges": edges,
    }


def _save_weight_matrix(
    output_dir: Path,
    documents: list[DocumentRecord],
    weight_matrix: np.ndarray,
    eligible_mask: np.ndarray,
) -> Path:
    matrix_path = output_dir / "document_graph_matrix.npz"
    ensure_parent_dir(matrix_path)
    np.savez_compressed(
        matrix_path,
        doc_ids=np.array([document.doc_id for document in documents]),
        weights=weight_matrix.astype(np.float32),
        eligible_mask=eligible_mask,
        ordinals=np.array(
            [
                -1 if document.time_info.ordinal is None else document.time_info.ordinal
                for document in documents
            ],
            dtype=np.int64,
        ),
    )
    return matrix_path


def build_document_graph_artifacts(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    timeline_dir: str | Path,
    project_root: str | Path,
    embed_dimensions: int | None = None,
    embed_batch_size: int | None = None,
    limit: int | None = None,
    embedder: SentenceEmbeddingProvider | None = None,
) -> dict[str, Any]:
    """按句向量集合构建历史文档到锚点文档的有向图。"""

    input_root = Path(input_dir)
    output_root = Path(output_dir)
    timeline_root = Path(timeline_dir)
    if not input_root.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_root}")
    if not timeline_root.exists():
        raise FileNotFoundError(f"时间目录不存在: {timeline_root}")

    active_embedder = embedder or SiliconFlowSentenceEmbedder(
        project_root=project_root,
        embed_dimensions=embed_dimensions,
        embed_batch_size=embed_batch_size,
    )
    documents = _load_documents(
        input_dir=input_root,
        timeline_dir=timeline_root,
        output_dir=output_root,
        embedder=active_embedder,
        limit=limit,
    )
    ordered_documents, edges, weight_matrix, eligible_mask = _build_graph(documents)
    graph_payload = _build_graph_payload(ordered_documents, edges)

    sentence_rows = [
        record.to_row()
        for document in ordered_documents
        for record in document.sentence_records
    ]
    document_rows = [document.to_manifest_row() for document in ordered_documents]

    write_jsonl(output_root / "sentence_manifest.jsonl", sentence_rows)
    write_jsonl(output_root / "document_manifest.jsonl", document_rows)
    matrix_path = _save_weight_matrix(
        output_dir=output_root,
        documents=ordered_documents,
        weight_matrix=weight_matrix,
        eligible_mask=eligible_mask,
    )
    write_json(output_root / "document_graph.json", graph_payload)

    summary = {
        "input_dir": input_root.as_posix(),
        "timeline_dir": timeline_root.as_posix(),
        "output_dir": output_root.as_posix(),
        "document_count": len(ordered_documents),
        "sentence_count": len(sentence_rows),
        "eligible_edge_count": int(np.count_nonzero(eligible_mask)),
        "known_time_documents": graph_payload["meta"]["known_time_documents"],
        "unknown_time_documents": graph_payload["meta"]["unknown_time_documents"],
        "sentence_set_dir": (output_root / "sentence_sets").as_posix(),
        "weight_matrix_path": matrix_path.as_posix(),
        "graph_json_path": (output_root / "document_graph.json").as_posix(),
    }
    write_json(output_root / "build_summary.json", summary)
    return summary
