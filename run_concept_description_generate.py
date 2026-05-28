from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.config import AppConfig
from core.task_executor import TaskExecutor
from tools.concept_abstraction.workflow import (
    build_logger,
    parse_element_types,
    run_description_generation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为合规概念池生成分类型结构化抽象描述，供后续字段加权 embedding 与聚类使用。"
    )
    parser.add_argument("--root", default=".", help="项目根目录，默认当前目录。")
    parser.add_argument(
        "--concept-dir",
        default="output/concept_abstraction/compliant_pools",
        help="合规概念池目录，默认使用合规性检验输出。",
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
        "--mode", choices=["sync", "batch"], help="覆盖 config.yaml 的 execution_mode。"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=20, help="每批 LLM 任务数量。"
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM 温度。")
    parser.add_argument(
        "--max-tokens", type=int, default=768, help="LLM 最大输出 token。"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="清空已有描述缓存后重跑。"
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> int:
    args = parse_args()
    project_root = Path(args.root).resolve()
    logger = build_logger("kg-concept-description")
    config = AppConfig.load(project_root)
    config.ensure_runtime_dirs()
    mode = args.mode or config.execution_mode
    if not config.api_key:
        raise RuntimeError(
            "概念描述生成需要 LLM API Key，请配置 config.yaml 的 api_key 或 SILICONFLOW_API_KEY。"
        )
    executor = TaskExecutor(config, logger)
    summary = run_description_generation(
        project_root=project_root,
        concept_dir=resolve_path(project_root, args.concept_dir),
        output_dir=resolve_path(project_root, args.output_dir),
        executor=executor,
        mode=mode,
        element_types=parse_element_types(args.element_types),
        chunk_size=args.chunk_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        overwrite=args.overwrite,
        logger=logger,
    )
    logger.info("概念描述生成完成：%s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
