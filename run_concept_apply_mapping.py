from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.concept_abstraction.workflow import build_logger, run_apply_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把人工审核后的 entity/relation/location 概念映射表应用到排序五元组，并合并替换后完全相同的五元组。"
    )
    parser.add_argument("--root", default=".", help="项目根目录，默认当前目录。")
    parser.add_argument(
        "--tuple-dir",
        default="data/tuple_pocessed&sorted",
        help="待替换概念的五元组目录。",
    )
    parser.add_argument(
        "--mapping-dir",
        default="output/concept_abstraction/mappings",
        help="人工审核后的概念映射表目录。",
    )
    parser.add_argument(
        "--output-dir",
        default="data/tuple_pocessed&sorted&abstracted",
        help="替换概念后的五元组输出目录。",
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> int:
    args = parse_args()
    project_root = Path(args.root).resolve()
    logger = build_logger("kg-concept-apply-mapping")
    summary = run_apply_mapping(
        project_root=project_root,
        tuple_dir=resolve_path(project_root, args.tuple_dir),
        mapping_dir=resolve_path(project_root, args.mapping_dir),
        output_dir=resolve_path(project_root, args.output_dir),
        logger=logger,
    )
    logger.info("概念映射应用完成：%s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
