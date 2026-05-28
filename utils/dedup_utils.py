from __future__ import annotations

import hashlib
import json
from math import ceil, log

import numpy as np

from utils.text_utils import lexical_tokenize, normalize_score_list


def build_cache_key(namespace: str, payload: dict) -> str:
    """为去重相关调用生成稳定缓存键。"""

    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


def cluster_vectors_by_kmeans(
    vectors: list[list[float]],
    target_cluster_size: int,
    random_seed: int,
    max_iter: int = 50,
) -> list[list[int]]:
    """用轻量 k-means 把向量划分为若干簇。"""

    if not vectors:
        return []

    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    total = len(vectors)
    if total <= target_cluster_size:
        return [list(range(total))]

    n_clusters = max(1, ceil(total / target_cluster_size))
    rng = np.random.default_rng(random_seed)
    centroid_indices = rng.choice(total, size=n_clusters, replace=False)
    centroids = matrix[centroid_indices]

    assignments = np.full(total, -1, dtype=np.int32)
    for _ in range(max_iter):
        similarities = matrix @ centroids.T
        new_assignments = np.argmax(similarities, axis=1)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments
        for cluster_index in range(n_clusters):
            members = matrix[assignments == cluster_index]
            if len(members) == 0:
                fallback = int(rng.integers(0, total))
                centroids[cluster_index] = matrix[fallback]
                continue
            centroid = np.mean(members, axis=0)
            norm = np.linalg.norm(centroid)
            if norm == 0:
                norm = 1.0
            centroids[cluster_index] = centroid / norm

    clusters: list[list[int]] = []
    for cluster_index in range(n_clusters):
        members = np.where(assignments == cluster_index)[0].tolist()
        if members:
            clusters.append(sorted(members))
    return clusters


def bm25_scores(query_text: str, corpus_texts: list[str]) -> list[float]:
    """基于轻量词法切分计算 BM25 风格分数。"""

    if not corpus_texts:
        return []

    corpus_tokens = [lexical_tokenize(text) for text in corpus_texts]
    query_tokens = lexical_tokenize(query_text)
    if not query_tokens:
        return [0.0 for _ in corpus_texts]

    doc_freq: dict[str, int] = {}
    doc_lengths = [len(tokens) for tokens in corpus_tokens]
    avg_doc_len = sum(doc_lengths) / max(1, len(doc_lengths))
    for tokens in corpus_tokens:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    total_docs = len(corpus_tokens)
    for tokens in corpus_tokens:
        token_counts: dict[str, int] = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        score = 0.0
        doc_len = len(tokens) or 1
        for token in query_tokens:
            if token not in token_counts:
                continue
            df = doc_freq.get(token, 0)
            idf = log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
            freq = token_counts[token]
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * doc_len / max(1.0, avg_doc_len))
            score += idf * numerator / denominator
        scores.append(score)
    return scores


def mixed_topk_scores(
    query_text: str,
    vector_scores: list[float],
    corpus_texts: list[str],
    bm25_weight: float,
) -> list[float]:
    """融合向量相似度与词法 BM25 分数。"""

    weight = min(max(bm25_weight, 0.0), 1.0)
    lexical_scores = bm25_scores(query_text, corpus_texts)
    normalized_lexical = normalize_score_list(lexical_scores)
    normalized_vector = normalize_score_list(vector_scores)
    return [
        (1.0 - weight) * vector_score + weight * lexical_score
        for vector_score, lexical_score in zip(normalized_vector, normalized_lexical, strict=True)
    ]
