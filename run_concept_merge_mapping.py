from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.config import AppConfig
from core.task_executor import TaskExecutor
from tools.concept_abstraction.workflow import build_logger, run_merge_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对一个聚类结果文件执行整簇单次 LLM 归并，并生成概念映射表。"
    )
    parser.add_argument("--root", default=".", help="项目根目录，默认当前目录。")
    parser.add_argument(
        "--cluster-file",
        required=True,
        help="run_concept_cluster.py 生成的某个聚类 JSON 文件。",
    )
    parser.add_argument(
        "--embedding-dir",
        default="output/concept_abstraction/embeddings",
        help="已废弃；整簇归并不再依赖描述向量缓存，保留该参数仅为兼容旧命令。",
    )
    parser.add_argument(
        "--concept-dir",
        default="output/concept_abstraction/compliant_pools",
        help="合规概念池目录；用于为归并输入补充来源句。",
    )
    parser.add_argument(
        "--output-dir",
        default="output/concept_abstraction/mappings",
        help="概念映射表输出目录。",
    )
    parser.add_argument(
        "--mode", choices=["sync", "batch"], help="覆盖 config.yaml 的 execution_mode。"
    )
    parser.add_argument(
        "--candidate-size",
        type=int,
        default=16,
        help="已废弃；整簇归并不再按候选窗口迭代采样。",
    )
    parser.add_argument(
        "--reference-size",
        type=int,
        default=24,
        help="已废弃；整簇归并不再维护扩充参考集。",
    )
    parser.add_argument(
        "--max-iterations-per-cluster",
        type=int,
        default=500,
        help="已废弃；整簇归并每个 cluster 只调用一次 LLM。",
    )
    parser.add_argument(
        "--source-sentence-limit",
        type=int,
        default=2,
        help="每个原始概念注入给归并提示词的来源句上限。",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM 温度。")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="LLM 最大输出 token；截断时客户端会自动扩容重试。",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="清空已有合并决策缓存后重跑。"
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> int:
    args = parse_args()
    project_root = Path(args.root).resolve()
    logger = build_logger("kg-concept-merge")
    config = AppConfig.load(project_root)
    config.ensure_runtime_dirs()
    mode = args.mode or config.execution_mode
    if not config.api_key:
        raise RuntimeError(
            "概念合并映射需要 LLM API Key，请配置 config.yaml 的 api_key 或 SILICONFLOW_API_KEY。"
        )
    executor = TaskExecutor(config, logger)
    summary = run_merge_mapping(
        project_root=project_root,
        cluster_file=resolve_path(project_root, args.cluster_file),
        embedding_dir=resolve_path(project_root, args.embedding_dir),
        concept_dir=resolve_path(project_root, args.concept_dir),
        output_dir=resolve_path(project_root, args.output_dir),
        executor=executor,
        mode=mode,
        candidate_size=args.candidate_size,
        reference_size=args.reference_size,
        max_iterations_per_cluster=args.max_iterations_per_cluster,
        source_sentence_limit=args.source_sentence_limit,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        overwrite=args.overwrite,
        logger=logger,
    )
    logger.info("概念合并映射完成：%s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
