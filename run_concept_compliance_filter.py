from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.config import AppConfig
from core.task_executor import TaskExecutor
from tools.concept_abstraction.workflow import (
    build_logger,
    parse_element_types,
    run_compliance_filter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 data/concept 三类概念池做 LLM 合规性检验，并输出合规池与不合规引用记录。"
    )
    parser.add_argument("--root", default=".", help="项目根目录，默认当前目录。")
    parser.add_argument(
        "--concept-dir", default="data/concept", help="原始概念池目录。"
    )
    parser.add_argument(
        "--tuple-dir",
        default="data/tuple_pocessed&sorted",
        help="用于追溯不合规概念引用的五元组目录。",
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
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM 温度。")
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="LLM 最大输出 token。"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="清空已有判别缓存后重跑。"
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> int:
    args = parse_args()
    project_root = Path(args.root).resolve()
    logger = build_logger("kg-concept-compliance")
    config = AppConfig.load(project_root)
    config.ensure_runtime_dirs()
    mode = args.mode or config.execution_mode
    if not config.api_key:
        raise RuntimeError(
            "概念合规性检验需要 LLM API Key，请配置 config.yaml 的 api_key 或 SILICONFLOW_API_KEY。"
        )
    executor = TaskExecutor(config, logger)
    summary = run_compliance_filter(
        project_root=project_root,
        concept_dir=resolve_path(project_root, args.concept_dir),
        tuple_dir=resolve_path(project_root, args.tuple_dir),
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
    logger.info("概念合规性检验完成：%s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
