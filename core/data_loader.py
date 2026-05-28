from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from core.types import DocumentRecord
from utils.news_preprocess_utils import (
    build_doc_id_from_path,
    list_raw_news_files,
    resolve_raw_news_dir,
)


def _load_documents_jsonl(input_path: Path) -> list[DocumentRecord]:
    """按 documents.jsonl 协议读取统一文档对象。"""

    records: list[DocumentRecord] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            doc_id = str(payload["doc_id"])
            text = str(payload["text"])
            meta = dict(payload.get("meta", {}))
            records.append(DocumentRecord(doc_id=doc_id, text=text, meta=meta))
    return records


class BaseDataLoader(ABC):
    """原始文档加载器抽象基类。

    当前项目尚未确定 data 目录下文档的正式格式，
    因此这里只定义统一接口，不绑定具体数据源实现。
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    @abstractmethod
    def load(self) -> list[DocumentRecord]:
        """读取原始文档并转换为统一对象列表。"""


class PlaceholderDataLoader(BaseDataLoader):
    """占位数据加载器。

    该实现只服务于开发期和测试期：
    - 若 data 目录下存在 documents.jsonl，则按最小协议读取。
    - 否则抛出明确异常，提示后续接入真实格式。
    """

    def load(self) -> list[DocumentRecord]:
        input_path = self.data_dir / "documents.jsonl"
        if not input_path.exists():
            raise NotImplementedError(
                "当前尚未接入正式数据格式。请先在 data/documents.jsonl 中提供测试数据，"
                "或实现新的 DataLoader。"
            )

        return _load_documents_jsonl(input_path)


class NewsProcessedTextLoader(BaseDataLoader):
    """新闻文本加载器。

    加载优先级如下：
    1. 若存在 data/documents.jsonl，则按统一协议读取，兼容测试与最小样例。
    2. 若存在 data/processed/*.txt，则读取新闻预处理后的文本文件。
    3. 若仅存在 data/sources/*.txt 原始新闻，则提示用户先执行 s0 预处理阶段。
       若项目仍是旧结构，也兼容检测 data/*.txt。
    """

    def load(self) -> list[DocumentRecord]:
        input_path = self.data_dir / "documents.jsonl"
        if input_path.exists():
            return _load_documents_jsonl(input_path)

        processed_dir = self.data_dir / "processed"
        processed_files = (
            sorted(processed_dir.glob("*.txt")) if processed_dir.exists() else []
        )
        if processed_files:
            records: list[DocumentRecord] = []
            for file_path in processed_files:
                text = file_path.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                records.append(
                    DocumentRecord(
                        doc_id=build_doc_id_from_path(file_path),
                        text=text,
                        meta={
                            "title": file_path.stem,
                            "source_path": str(file_path),
                            "is_processed": True,
                        },
                    )
                )
            if records:
                return records
            raise RuntimeError(
                "data/processed 中存在 txt 文件，但内容为空，无法继续执行。"
            )

        raw_dir = resolve_raw_news_dir(self.data_dir)
        raw_files = list_raw_news_files(raw_dir)
        if raw_files:
            raise RuntimeError(
                f"检测到 {raw_dir.as_posix()}/ 下存在原始 txt 新闻，但未发现 data/processed 预处理结果。"
                "请先执行 s0 新闻预处理阶段。"
            )

        raise NotImplementedError(
            "当前既没有 data/documents.jsonl，也没有 data/processed/*.txt。"
            "请先提供测试数据或执行新闻预处理阶段。"
        )


def build_data_loader(loader_name: str, data_dir: str | Path) -> BaseDataLoader:
    """根据配置创建数据加载器实例。"""

    if loader_name == "placeholder":
        return PlaceholderDataLoader(data_dir)
    if loader_name == "news_txt":
        return NewsProcessedTextLoader(data_dir)

    raise ValueError(f"未知的数据加载器类型: {loader_name}")
