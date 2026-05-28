from __future__ import annotations

import numpy as np


class VectorIndex:
    """轻量向量索引。"""

    def __init__(self, ids: list[str], vectors: list[list[float]]):
        if len(ids) != len(vectors):
            raise ValueError("ids 与 vectors 数量不一致")
        self.ids = ids
        matrix = np.array(vectors, dtype=np.float32)
        if matrix.size == 0:
            self.matrix = matrix
        else:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.matrix = matrix / norms
        self.id_to_index = {item_id: index for index, item_id in enumerate(ids)}

    def get_vector(self, item_id: str) -> list[float]:
        index = self.id_to_index[item_id]
        return self.matrix[index].tolist()

    def topk_similar(
        self,
        query_index: int,
        candidate_indices: list[int],
        k: int,
    ) -> list[tuple[int, float]]:
        """返回候选集合中与查询向量最相近的 top-k。"""

        if not candidate_indices or k <= 0:
            return []
        query = self.matrix[query_index]
        candidate_matrix = self.matrix[candidate_indices]
        scores = candidate_matrix @ query
        sorted_pairs = sorted(
            zip(candidate_indices, scores.tolist(), strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        return sorted_pairs[:k]
