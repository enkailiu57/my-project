from __future__ import annotations

import argparse
from pathlib import Path

from tools.event_evolution.doc_graph.builder import build_document_graph_artifacts
from tools.event_evolution.doc_graph.visualizer import (
    render_document_graph_threshold_html,
)


def build_argument_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="基于句嵌入集合与时间约束候选召回构建事件演化文档图"
    )
    parser.add_argument(
        "--input-dir",
        default=str(project_root / "data" / "processed"),
        help="正文输入目录，默认使用 data/processed",
    )
    parser.add_argument(
        "--timeline-dir",
        default=str(project_root / "data" / "processed_timeline"),
        help="结束时间来源目录，默认使用 data/processed_timeline",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "output" / "event_evolution" / "doc_graph"),
        help="输出目录，默认写入 output/event_evolution/doc_graph",
    )
    parser.add_argument(
        "--embed-dimensions",
        type=int,
        default=512,
        help="句嵌入维度，默认 512",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="句嵌入批大小，默认 64",
    )
    parser.add_argument(
        "--html-threshold",
        type=float,
        default=0.2,
        help="生成 HTML 时默认显示的边阈值，默认 0.2",
    )
    parser.add_argument(
        "--center-doc-id",
        default="",
        help="可视化初始中心文档 ID，默认空表示全局视图",
    )
    parser.add_argument(
        "--center-hops",
        type=int,
        default=1,
        help="可视化初始中心文档的邻域跳数，默认 1",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只处理前 N 篇文档，0 表示处理全部",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    summary = build_document_graph_artifacts(
        input_dir=args.input_dir,
        timeline_dir=args.timeline_dir,
        output_dir=args.output_dir,
        project_root=project_root,
        embed_dimensions=args.embed_dimensions,
        embed_batch_size=args.embed_batch_size,
        limit=args.limit or None,
    )
    html_path = render_document_graph_threshold_html(
        graph_json_path=Path(args.output_dir) / "document_graph.json",
        output_html_path=Path(args.output_dir) / "document_graph_threshold_view.html",
        default_threshold=args.html_threshold,
        default_center_doc=args.center_doc_id,
        default_center_hops=max(1, args.center_hops),
    )
    print(
        "已完成事件演化文档图构建："
        f"文档 {summary['document_count']} 篇，"
        f"句子 {summary['sentence_count']} 条，"
        f"候选有向边 {summary['eligible_edge_count']} 条。"
    )
    print(f"输出目录: {summary['output_dir']}")
    print(f"阈值可视化: {html_path.as_posix()}")


if __name__ == "__main__":
    main()
