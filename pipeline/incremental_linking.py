from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.config import AppConfig
from core.embed_client import EmbeddingClient
from core.llm_client import LLMClient
from prompts import describe_entity, describe_location, describe_relation
from utils.io_utils import load_npz, read_json, read_jsonl
from utils.text_utils import build_element_id, normalize_alias_key


@dataclass(slots=True)
class IncrementalCandidate:
    """增量概念链接候选项。"""

    cid: str
    canonical_str: str
    score: float


class IncrementalLinker:
    """独立的增量概念链接器。"""

    def __init__(self, config: AppConfig, top_k: int = 3):
        self.config = config
        self.top_k = top_k
        self.llm_client = LLMClient(config)
        self.embedding_client = EmbeddingClient(config)

    def build_new_element(
        self, element_type: str, raw_str: str, contexts: list[str]
    ) -> dict:
        """构造新增元素对象。"""

        return {
            "id": build_element_id(element_type, raw_str.strip()),
            "raw_str": raw_str.strip(),
            "type": element_type,
            "alias_key": normalize_alias_key(raw_str),
            "contexts": contexts,
        }

    def _describe_prompt(self, element_type: str):
        if element_type == "entity":
            return describe_entity
        if element_type == "relation":
            return describe_relation
        return describe_location

    def _description_file(self, element_type: str) -> str:
        if element_type == "entity":
            return "s3_entity_desc.jsonl"
        if element_type == "relation":
            return "s3_relation_desc.jsonl"
        return "s3_location_desc.jsonl"

    def _build_description(
        self, element_type: str, raw_str: str, contexts: list[str]
    ) -> str:
        prompt_module = self._describe_prompt(element_type)
        result = self.llm_client.call(
            prompt_module,
            {"raw_str": raw_str, "contexts": contexts},
            temperature=self.config.temperature_describe,
            max_tokens=self.config.max_tokens,
        )
        return str(result["description"])

    def search(
        self,
        output_dir: str | Path,
        element_type: str,
        raw_str: str,
        contexts: list[str],
    ) -> list[IncrementalCandidate]:
        """将新增元素链接到最相近的 canonical。"""

        output_path = Path(output_dir)
        alias_map = read_json(output_path / "s5_alias_map.json")
        canonical_rows = list(read_jsonl(output_path / "s5_canonical.jsonl"))
        description_rows = list(
            read_jsonl(output_path / self._description_file(element_type))
        )
        embedding_ids, embedding_vectors = load_npz(output_path / "s4_embeddings.npz")
        vector_by_id = {
            item_id: np.array(vector, dtype=np.float32)
            for item_id, vector in zip(embedding_ids, embedding_vectors, strict=True)
        }

        description = self._build_description(element_type, raw_str, contexts)
        query_vector = np.array(
            self.embedding_client.embed([description])[0], dtype=np.float32
        )
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            query_norm = 1.0
        query_vector = query_vector / query_norm

        best_scores: dict[str, float] = {}
        for row in description_rows:
            alias_key = normalize_alias_key(row["raw_str"])
            if alias_key is None:
                continue
            cid = alias_map.get(element_type, {}).get(alias_key)
            if cid is None or row["id"] not in vector_by_id:
                continue
            vector = vector_by_id[row["id"]]
            vector_norm = np.linalg.norm(vector)
            if vector_norm == 0:
                continue
            score = float(query_vector @ (vector / vector_norm))
            best_scores[cid] = max(best_scores.get(cid, -1.0), score)

        canonical_map = {
            row["cid"]: row["canonical_str"]
            for row in canonical_rows
            if row["type"] == element_type
        }
        ranked = sorted(best_scores.items(), key=lambda item: item[1], reverse=True)[
            : self.top_k
        ]
        return [
            IncrementalCandidate(
                cid=cid,
                canonical_str=canonical_map.get(cid, cid),
                score=score,
            )
            for cid, score in ranked
        ]
