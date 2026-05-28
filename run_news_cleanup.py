from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from utils.io_utils import write_jsonl, write_text
from utils.news_preprocess_utils import (
    clean_news_text,
    list_raw_news_files,
)


def parse_args() -> argparse.Namespace:
    """解析清理脚本参数。"""

    parser = argparse.ArgumentParser(description="新闻非法字符/词汇清理脚本")
    parser.add_argument(
        "--scope",
        default="all",
        help="处理范围。支持 all、D001、E001-E014、D001,E001-E014。",
    )
    parser.add_argument(
        "--input-dir",
        default="data/sources",
        help="原始新闻目录，默认 data/sources",
    )
    parser.add_argument(
        "--output-dir",
        default="data/cleaned",
        help="清理结果输出目录，默认 data/cleaned",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只展示将处理的文件，不写出结果"
    )
    return parser.parse_args()


def build_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger("kg-news-cleanup")


def main() -> int:
    args = parse_args()
    logger = build_logger()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = list_raw_news_files(input_dir, args.scope)
    if not files:
        raise FileNotFoundError(
            f"未找到满足范围 {args.scope!r} 的原始 txt 新闻文件。"
            "请检查 data/sources 或你通过 --input-dir 指定的目录。"
        )

    logger.info("匹配到 %s 篇新闻，范围=%s", len(files), args.scope)
    for file_path in files:
        logger.info("待处理: %s", file_path.name)

    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    for file_path in files:
        raw_text = file_path.read_text(encoding="utf-8")
        cleaned, cleanup_stats = clean_news_text(raw_text)
        target_path = output_dir / file_path.name
        write_text(target_path, cleaned + ("\n" if cleaned else ""))
        logger.info(
            "已清理 %s: %s -> %s 字符",
            file_path.name,
            len(raw_text),
            len(cleaned),
        )
        if cleanup_stats:
            for item in cleanup_stats:
                logger.info("  %s x%s", item.description, item.count)
        else:
            logger.info("  未命中任何清理规则")
        manifest_rows.append(
            {
                "source_file": file_path.name,
                "output_file": target_path.name,
                "input_char_count": len(raw_text),
                "output_char_count": len(cleaned),
                "cleanup_rules": [
                    {"description": item.description, "count": item.count}
                    for item in cleanup_stats
                ],
            }
        )

    write_jsonl(output_dir / "cleanup_manifest.jsonl", manifest_rows, append_done=True)
    logger.info("清理完成，结果已写入 %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
