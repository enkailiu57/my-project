from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.concept_abstraction.cluster_visualization import (
    build_logger,
    parse_figsize,
    run_cluster_visualization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取概念聚类文件与字段 embedding，按 field_weights 加权聚合后降维为 2D 散点图。"
    )
    parser.add_argument("--root", default=".", help="项目根目录，默认当前目录。")
    parser.add_argument(
        "--cluster-file",
        required=True,
        help="output/concept_abstraction/clusters 下的某个聚类结果文件。",
    )
    parser.add_argument(
        "--embedding-dir",
        default="output/concept_abstraction/embeddings",
        help="描述 embedding 缓存目录。",
    )
    parser.add_argument(
        "--output-dir",
        default="output/concept_abstraction/visualizations",
        help="聚类散点图输出目录。",
    )
    parser.add_argument(
        "--method",
        choices=["umap", "tsne"],
        default="umap",
        help="降维方法；默认使用更适合 embedding 聚类可视化的 UMAP。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="降维随机种子。",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=20,
        help="UMAP 的邻居数。",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP 的最小距离参数。",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE 的 perplexity。",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=18.0,
        help="散点大小。",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="散点透明度。",
    )
    parser.add_argument(
        "--cluster-number-fontsize",
        type=float,
        default=4.5,
        help="点旁聚类数字序号的字号。",
    )
    parser.add_argument(
        "--figsize",
        default="12,10",
        help="图像尺寸，格式为 宽,高。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="输出图像 DPI。",
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> int:
    args = parse_args()
    project_root = Path(args.root).resolve()
    logger = build_logger("kg-concept-cluster-visualize")
    summary = run_cluster_visualization(
        project_root=project_root,
        cluster_file=resolve_path(project_root, args.cluster_file),
        embedding_dir=resolve_path(project_root, args.embedding_dir),
        output_dir=resolve_path(project_root, args.output_dir),
        method=args.method,
        seed=args.seed,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        tsne_perplexity=args.tsne_perplexity,
        point_size=args.point_size,
        alpha=args.alpha,
        cluster_number_fontsize=args.cluster_number_fontsize,
        figsize=parse_figsize(args.figsize),
        dpi=args.dpi,
        logger=logger,
    )
    logger.info("概念聚类可视化完成：%s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
