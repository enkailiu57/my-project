from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.config import AppConfig
from core.embed_client import EmbeddingClient
from tools.concept_abstraction.workflow import (
    build_logger,
    parse_element_types,
    parse_int_list,
    run_clustering,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于概念结构化描述生成字段加权 embedding，并输出聚类结果；默认使用 mutual-kNN + Leiden + 递归切分。"
    )
    parser.add_argument("--root", default=".", help="项目根目录，默认当前目录。")
    parser.add_argument(
        "--concept-dir",
        default="output/concept_abstraction/compliant_pools",
        help="待聚类概念池目录。",
    )
    parser.add_argument(
        "--description-dir",
        default="output/concept_abstraction/descriptions",
        help="概念抽象描述目录。",
    )
    parser.add_argument(
        "--embedding-dir",
        default="output/concept_abstraction/embeddings",
        help="描述向量缓存目录。",
    )
    parser.add_argument(
        "--output-dir",
        default="output/concept_abstraction",
        help="概念抽象流程输出目录。",
    )
    parser.add_argument(
        "--element-types",
        default="all",
        help="概念类型：all 或 entity,relation,location 的逗号组合。",
    )
    parser.add_argument(
        "--method",
        choices=["leiden_balanced", "agglomerative"],
        default="leiden_balanced",
        help="聚类方法；默认使用无需预设簇数的 Leiden balanced。",
    )
    parser.add_argument(
        "--cluster-counts",
        default="80,120,160",
        help="默认聚类数量列表，仅 agglomerative 方法使用。",
    )
    parser.add_argument(
        "--entity-cluster-counts",
        default="",
        help="实体概念聚类数量列表，仅 agglomerative 方法使用；为空时使用 --cluster-counts。",
    )
    parser.add_argument(
        "--relation-cluster-counts",
        default="",
        help="关系概念聚类数量列表，仅 agglomerative 方法使用；为空时使用 --cluster-counts。",
    )
    parser.add_argument(
        "--location-cluster-counts",
        default="",
        help="地点概念聚类数量列表，仅 agglomerative 方法使用；为空时使用 --cluster-counts。",
    )
    parser.add_argument(
        "--target-cluster-size",
        type=int,
        default=24,
        help="Leiden balanced 期望的目标簇大小。",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=20,
        help="Leiden balanced 希望大多数簇不低于的大小。",
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=30,
        help="Leiden balanced 递归切分时尽量压到的最大簇大小。",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=20,
        help="Leiden 图构建时每个节点的 kNN 邻居数。",
    )
    parser.add_argument(
        "--initial-resolution",
        type=float,
        default=0.2,
        help="Leiden 初始 resolution。",
    )
    parser.add_argument(
        "--resolution-multiplier",
        type=float,
        default=1.5,
        help="Leiden 递增搜索与递归切分时的 resolution 放大倍数。",
    )
    parser.add_argument(
        "--resolution-rounds",
        type=int,
        default=8,
        help="Leiden 每层尝试的 resolution 候选数。",
    )
    parser.add_argument(
        "--max-split-depth",
        type=int,
        default=4,
        help="Leiden 递归切分超大簇的最大深度。",
    )
    parser.add_argument(
        "--min-edge-similarity",
        type=float,
        default=0.0,
        help="构图时保留边的最小 cosine 相似度。",
    )
    parser.add_argument(
        "--small-cluster-absorb-threshold",
        type=float,
        default=0.0,
        help="将过小簇并回邻近簇所需的平均相似度阈值；0 表示关闭。",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Leiden 随机种子。",
    )
    parser.add_argument(
        "--overwrite-embeddings", action="store_true", help="重新生成描述向量。"
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def build_cluster_counts_by_type(
    element_types: list[str],
    default_counts: str,
    entity_counts: str,
    relation_counts: str,
    location_counts: str,
) -> dict[str, list[int]]:
    default_values = parse_int_list(default_counts)
    raw_by_type = {
        "entity": entity_counts,
        "relation": relation_counts,
        "location": location_counts,
    }
    counts_by_type: dict[str, list[int]] = {}
    for element_type in element_types:
        raw_value = raw_by_type[element_type].strip()
        values = parse_int_list(raw_value) if raw_value else default_values
        if not values:
            raise ValueError(f"{element_type} 至少需要一个聚类数量。")
        if any(value <= 0 for value in values):
            raise ValueError(f"{element_type} 的聚类数量必须全部大于 0: {values}")
        counts_by_type[element_type] = values
    return counts_by_type


def main() -> int:
    args = parse_args()
    project_root = Path(args.root).resolve()
    logger = build_logger("kg-concept-cluster")
    config = AppConfig.load(project_root)
    config.ensure_runtime_dirs()
    if not config.api_key:
        raise RuntimeError(
            "概念聚类需要生成 embedding，请配置 config.yaml 的 api_key 或 SILICONFLOW_API_KEY。"
        )
    element_types = parse_element_types(args.element_types)
    cluster_counts_by_type = (
        build_cluster_counts_by_type(
            element_types=element_types,
            default_counts=args.cluster_counts,
            entity_counts=args.entity_cluster_counts,
            relation_counts=args.relation_cluster_counts,
            location_counts=args.location_cluster_counts,
        )
        if args.method == "agglomerative"
        else None
    )
    summary = run_clustering(
        project_root=project_root,
        concept_dir=resolve_path(project_root, args.concept_dir),
        description_dir=resolve_path(project_root, args.description_dir),
        embedding_dir=resolve_path(project_root, args.embedding_dir),
        output_dir=resolve_path(project_root, args.output_dir),
        embedding_client=EmbeddingClient(config),
        element_types=element_types,
        cluster_counts_by_type=cluster_counts_by_type,
        cluster_method=args.method,
        leiden_config={
            "knn_k": args.knn_k,
            "target_cluster_size": args.target_cluster_size,
            "min_cluster_size": args.min_cluster_size,
            "max_cluster_size": args.max_cluster_size,
            "initial_resolution": args.initial_resolution,
            "resolution_multiplier": args.resolution_multiplier,
            "resolution_rounds": args.resolution_rounds,
            "max_split_depth": args.max_split_depth,
            "min_edge_similarity": args.min_edge_similarity,
            "small_cluster_absorb_threshold": args.small_cluster_absorb_threshold,
            "random_seed": args.random_seed,
        },
        overwrite_embeddings=args.overwrite_embeddings,
        logger=logger,
    )
    logger.info("概念聚类完成：%s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
