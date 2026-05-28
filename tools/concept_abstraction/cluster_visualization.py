from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from utils.io_utils import ensure_parent_dir, load_npz, read_json


def build_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger(name)


def read_cluster_payload(cluster_file: Path) -> dict[str, Any]:
    payload = read_json(cluster_file)
    if not isinstance(payload, dict):
        raise ValueError(f"聚类文件必须是对象: {cluster_file}")
    return payload


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def normalize_weight_mapping(
    field_names: list[str], raw_field_weights: Any
) -> dict[str, float]:
    if not isinstance(raw_field_weights, dict):
        raise ValueError("聚类文件 parameters.field_weights 必须是对象。")
    weights = {
        field_name: float(raw_field_weights.get(field_name, 0.0))
        for field_name in field_names
    }
    total = sum(weight for weight in weights.values() if weight > 0)
    if total <= 0:
        raise ValueError("聚类文件 parameters.field_weights 必须至少包含一个正权重。")
    return {
        field_name: max(weight, 0.0) / total for field_name, weight in weights.items()
    }


def load_field_embedding_bundle(
    embedding_dir: Path,
    element_type: str,
) -> tuple[dict[str, str], dict[str, dict[str, np.ndarray]]]:
    meta_file = embedding_dir / f"{element_type}_description_vector_meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"缺少 embedding 元数据文件: {meta_file}")
    meta = read_json(meta_file)
    concepts = meta.get("concepts")
    if not isinstance(concepts, dict):
        raise ValueError(f"embedding 元数据缺少 concepts 映射: {meta_file}")

    field_vector_file_name = meta.get("field_vector_file")
    if (
        not isinstance(field_vector_file_name, str)
        or not field_vector_file_name.strip()
    ):
        field_vector_file_name = f"{element_type}_description_field_vectors.npz"
    field_vector_file = embedding_dir / field_vector_file_name
    if not field_vector_file.exists():
        raise FileNotFoundError(f"缺少字段 embedding 文件: {field_vector_file}")

    concept_to_record_id: dict[str, str] = {}
    for record_id, concept in concepts.items():
        if not isinstance(record_id, str) or not isinstance(concept, str):
            continue
        if concept in concept_to_record_id:
            raise ValueError(
                f"embedding 元数据中 concept 重复，无法唯一映射: {concept}"
            )
        concept_to_record_id[concept] = record_id

    field_vector_ids, field_vectors = load_npz(field_vector_file)
    field_vectors_by_record: dict[str, dict[str, np.ndarray]] = {}
    for field_vector_id, vector in zip(field_vector_ids, field_vectors, strict=True):
        record_id, separator, field_name = str(field_vector_id).partition("::")
        if not separator or not record_id or not field_name:
            raise ValueError(f"字段向量 ID 格式错误: {field_vector_id}")
        field_vectors_by_record.setdefault(record_id, {})[field_name] = np.asarray(
            vector,
            dtype=np.float32,
        )
    return concept_to_record_id, field_vectors_by_record


def build_cluster_concept_rows(cluster_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster_index, cluster in enumerate(cluster_payload.get("clusters", [])):
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or f"c{cluster_index + 1:04d}")
        cluster_number = cluster_index + 1
        for concept_index, item in enumerate(cluster.get("concepts", [])):
            if not isinstance(item, dict):
                continue
            concept = str(item.get("concept") or "").strip()
            if not concept:
                continue
            rows.append(
                {
                    "concept": concept,
                    "cluster_id": cluster_id,
                    "cluster_number": cluster_number,
                    "cluster_index": cluster_index,
                    "concept_index": concept_index,
                }
            )
    return rows


def build_weighted_concept_vectors(
    cluster_payload: dict[str, Any],
    concept_rows: list[dict[str, Any]],
    concept_to_record_id: dict[str, str],
    field_vectors_by_record: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    parameters = cluster_payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("聚类文件缺少 parameters。")
    raw_field_names = parameters.get("fields")
    if not isinstance(raw_field_names, list) or not raw_field_names:
        raise ValueError("聚类文件 parameters.fields 必须是非空列表。")
    field_names = [str(field_name) for field_name in raw_field_names if str(field_name)]
    field_weights = normalize_weight_mapping(
        field_names,
        parameters.get("field_weights"),
    )

    points: list[dict[str, Any]] = []
    for row in concept_rows:
        concept = str(row["concept"])
        record_id = concept_to_record_id.get(concept)
        if record_id is None:
            raise KeyError(f"embedding 元数据中找不到概念: {concept}")
        field_vectors = field_vectors_by_record.get(record_id, {})
        aggregated_vector: np.ndarray | None = None
        for field_name in field_names:
            vector = field_vectors.get(field_name)
            if vector is None:
                raise KeyError(f"概念 {concept} 缺少字段向量: {field_name}")
            weighted_vector = l2_normalize(vector) * float(field_weights[field_name])
            aggregated_vector = (
                weighted_vector
                if aggregated_vector is None
                else aggregated_vector + weighted_vector
            )
        assert aggregated_vector is not None
        points.append(
            {
                **row,
                "vector": l2_normalize(aggregated_vector),
            }
        )
    return points


def reduce_embeddings(
    vectors: np.ndarray,
    method: str,
    seed: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    tsne_perplexity: float,
) -> np.ndarray:
    count = int(vectors.shape[0])
    if count == 0:
        raise ValueError("没有可视化点。")
    if count == 1:
        return np.zeros((1, 2), dtype=np.float32)
    if count == 2:
        return np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    if method == "umap":
        try:
            import importlib

            umap = importlib.import_module("umap")
        except ImportError as exc:
            raise RuntimeError(
                "UMAP 可视化需要 umap-learn，请先运行 pip install -r requirements.txt。"
            ) from exc
        reducer = umap.UMAP(
            n_components=2,
            metric="cosine",
            n_neighbors=min(max(2, int(umap_n_neighbors)), count - 1),
            min_dist=float(max(0.0, umap_min_dist)),
            random_state=int(seed),
        )
        return np.asarray(reducer.fit_transform(vectors), dtype=np.float32)

    if method == "tsne":
        from sklearn.manifold import TSNE

        effective_perplexity = min(
            float(tsne_perplexity),
            max(1.0, float(count - 1) / 3.0),
        )
        reducer = TSNE(
            n_components=2,
            metric="cosine",
            init="pca",
            learning_rate="auto",
            perplexity=max(1.0, effective_perplexity),
            random_state=int(seed),
        )
        return np.asarray(reducer.fit_transform(vectors), dtype=np.float32)

    raise ValueError(f"未知降维方法: {method}")


def build_cluster_color_map(
    cluster_ids: list[str],
) -> dict[str, tuple[float, float, float]]:
    from matplotlib.colors import hsv_to_rgb

    unique_cluster_ids = list(dict.fromkeys(cluster_ids))
    color_by_cluster: dict[str, tuple[float, float, float]] = {}
    for index, cluster_id in enumerate(unique_cluster_ids):
        hue = float((index * 0.61803398875) % 1.0)
        saturation = 0.72 + 0.12 * float(index % 2)
        value = 0.78 + 0.12 * float((index // 2) % 2)
        color = hsv_to_rgb((hue, min(saturation, 0.92), min(value, 0.94)))
        color_by_cluster[cluster_id] = (
            float(color[0]),
            float(color[1]),
            float(color[2]),
        )
    return color_by_cluster


def render_cluster_scatter(
    points: list[dict[str, Any]],
    coordinates: np.ndarray,
    output_file: Path,
    method: str,
    point_size: float,
    alpha: float,
    cluster_number_fontsize: float,
    figsize: tuple[float, float],
    dpi: int,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "绘制聚类散点图需要 matplotlib，请先运行 pip install -r requirements.txt。"
        ) from exc

    cluster_ids = [str(point["cluster_id"]) for point in points]
    color_by_cluster = build_cluster_color_map(cluster_ids)
    colors = [color_by_cluster[cluster_id] for cluster_id in cluster_ids]

    figure, axis = plt.subplots(figsize=figsize, dpi=dpi)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#fbfbfd")
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=colors,
        s=float(max(point_size, 1.0)),
        alpha=float(min(max(alpha, 0.0), 1.0)),
        linewidths=0.0,
    )
    text_alpha = min(max(alpha + 0.1, 0.25), 1.0)
    for point, (x_coord, y_coord), color in zip(
        points,
        coordinates.tolist(),
        colors,
        strict=True,
    ):
        axis.annotate(
            str(point["cluster_number"]),
            xy=(float(x_coord), float(y_coord)),
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=float(max(cluster_number_fontsize, 1.0)),
            color=color,
            alpha=text_alpha,
            ha="left",
            va="bottom",
            annotation_clip=True,
        )
    axis.set_title(title, fontsize=12)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    ensure_parent_dir(output_file)
    figure.savefig(output_file, bbox_inches="tight")
    plt.close(figure)


def derive_output_file(output_dir: Path, cluster_file: Path, method: str) -> Path:
    return output_dir / f"{cluster_file.stem}_{method}.png"


def parse_figsize(value: str) -> tuple[float, float]:
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if len(pieces) != 2:
        raise ValueError("figsize 必须是形如 宽,高 的两个数字。")
    width, height = (float(piece) for piece in pieces)
    if width <= 0 or height <= 0:
        raise ValueError("figsize 的宽和高都必须大于 0。")
    return width, height


def run_cluster_visualization(
    project_root: Path,
    cluster_file: Path,
    embedding_dir: Path,
    output_dir: Path,
    method: str = "umap",
    seed: int = 42,
    umap_n_neighbors: int = 20,
    umap_min_dist: float = 0.1,
    tsne_perplexity: float = 30.0,
    point_size: float = 18.0,
    alpha: float = 0.85,
    cluster_number_fontsize: float = 4.5,
    figsize: tuple[float, float] = (12.0, 10.0),
    dpi: int = 220,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    active_logger = logger or build_logger("kg-concept-cluster-visualize")
    payload = read_cluster_payload(cluster_file)
    element_type = str(payload.get("concept_type") or "").strip()
    if not element_type:
        raise ValueError("聚类文件缺少 concept_type。")

    concept_rows = build_cluster_concept_rows(payload)
    concept_to_record_id, field_vectors_by_record = load_field_embedding_bundle(
        embedding_dir,
        element_type,
    )
    points = build_weighted_concept_vectors(
        payload,
        concept_rows,
        concept_to_record_id,
        field_vectors_by_record,
    )
    vectors = np.asarray([point["vector"] for point in points], dtype=np.float32)
    coordinates = reduce_embeddings(
        vectors=vectors,
        method=method,
        seed=seed,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        tsne_perplexity=tsne_perplexity,
    )
    output_file = derive_output_file(output_dir, cluster_file, method)
    render_cluster_scatter(
        points=points,
        coordinates=coordinates,
        output_file=output_file,
        method=method,
        point_size=point_size,
        alpha=alpha,
        cluster_number_fontsize=cluster_number_fontsize,
        figsize=figsize,
        dpi=dpi,
        title=(
            f"{element_type} | {len(points)} concepts | "
            f"{int(payload.get('cluster_count', 0))} clusters | {method.upper()}"
        ),
    )
    summary = {
        "concept_type": element_type,
        "cluster_file": cluster_file.resolve().as_posix(),
        "embedding_dir": embedding_dir.resolve().as_posix(),
        "output_file": output_file.resolve().as_posix(),
        "point_count": len(points),
        "cluster_count": int(payload.get("cluster_count", 0)),
        "method": method,
        "parameters": {
            "umap_n_neighbors": umap_n_neighbors,
            "umap_min_dist": umap_min_dist,
            "tsne_perplexity": tsne_perplexity,
            "point_size": point_size,
            "alpha": alpha,
            "cluster_number_fontsize": cluster_number_fontsize,
            "figsize": [figsize[0], figsize[1]],
            "dpi": dpi,
        },
    }
    active_logger.info(
        "聚类可视化完成：%s 个点、%s 个簇，输出 %s。",
        summary["point_count"],
        summary["cluster_count"],
        output_file.name,
    )
    return summary
