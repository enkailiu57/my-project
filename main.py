from __future__ import annotations

import argparse
import logging
import sys
from typing import cast

from core.config import AppConfig
from core.types import ExecutionMode, StageName, StageRuntime
from pipeline.registry import STAGE_ORDER, STAGE_REGISTRY, resolve_stage_sequence
from utils.stage_utils import (
    collect_missing_deps,
    describe_stage_artifact,
    should_skip_stage,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="事件知识图谱构建流水线入口")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-all", action="store_true", help="按顺序运行全部阶段")
    group.add_argument(
        "--from", dest="from_stage", choices=STAGE_ORDER, help="从指定阶段运行到最后"
    )
    group.add_argument("--only", nargs="+", choices=STAGE_ORDER, help="只运行指定阶段")
    parser.add_argument(
        "--mode", choices=["sync", "batch"], help="覆盖配置中的执行模式"
    )
    parser.add_argument(
        "--news-scope",
        default="all",
        help="新闻文件处理范围，对 s0 新闻预处理和 s1 事件抽取规划生效。支持 all、D001、E001-E014、D001,E001-E014。",
    )
    parser.add_argument(
        "--file-scope",
        default="all",
        help=(
            "篇章文件过滤范围，仅对 s2 开放五元组抽取生效。"
            "支持按 source_file、extract_file、文件 stem 或 doc_id 精确匹配，多个值用逗号分隔。"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只做依赖检查和执行计划展示，不真正执行"
    )
    return parser.parse_args()


def build_logger() -> logging.Logger:
    """创建统一日志器。"""

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger("kg-pipeline")


def validate_dependencies(stage_names: list[StageName], config: AppConfig) -> None:
    """在真正执行前统一检查依赖文件。

    该函数严格遵循设计文档的要求：
    只要有一个阶段缺少“本次执行计划之外”的前置依赖，就立即报错退出。

    注意：
    如果某个依赖会在本次运行中由更早的阶段产出，则不应在启动时判定为缺失，
    否则 fresh run-all 会被错误拦截。
    """

    planned_outputs: set[str] = set()
    for stage_name in stage_names:
        spec = STAGE_REGISTRY[stage_name]
        if not spec.deps:
            planned_outputs.update(spec.outputs)
            continue
        deps_to_check = [dep for dep in spec.deps if dep not in planned_outputs]
        missing = collect_missing_deps(
            config.output_dir,
            deps_to_check,
            project_root=config.project_root,
        )
        if missing:
            missing_text = ", ".join(describe_stage_artifact(name) for name in missing)
            raise RuntimeError(
                f"无法运行阶段 {stage_name}：缺少依赖文件 {missing_text}。"
                f"请先补齐前置阶段输出后再执行。"
            )
        planned_outputs.update(spec.outputs)


def main() -> int:
    """程序主入口。"""

    args = parse_args()
    logger = build_logger()
    config = AppConfig.load(project_root=".")
    raw_mode = args.mode or config.execution_mode
    if raw_mode not in {"sync", "batch"}:
        raise ValueError(f"不支持的执行模式: {raw_mode}")
    mode = cast(ExecutionMode, raw_mode)

    stage_names = resolve_stage_sequence(args.run_all, args.from_stage, args.only)
    config.validate_for_stage_names(stage_names, dry_run=args.dry_run)
    if config.strict_dependency_check:
        validate_dependencies(stage_names, config)

    for stage_name in stage_names:
        spec = STAGE_REGISTRY[stage_name]
        force = bool(args.only) or (
            stage_name == "s2" and args.file_scope.strip().lower() != "all"
        )
        if should_skip_stage(
            spec,
            config.output_dir,
            force=force,
            project_root=config.project_root,
        ):
            logger.info(
                "跳过阶段 %s（%s），原因：输出已完整存在。", spec.name, spec.title
            )
            continue

        logger.info("开始执行阶段 %s（%s）", spec.name, spec.title)
        runtime = StageRuntime(
            config=config,
            stage_name=stage_name,
            mode=mode,
            logger=logger,
            project_root=config.project_root,
            data_dir=config.data_dir,
            output_dir=config.output_dir,
            news_scope=args.news_scope,
            file_scope=args.file_scope,
            force=force,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            logger.info("dry-run 模式：仅展示计划，不执行阶段 %s。", spec.name)
            continue

        spec.runner(runtime)
        logger.info("完成阶段 %s（%s）", spec.name, spec.title)

    return 0


if __name__ == "__main__":
    sys.exit(main())
