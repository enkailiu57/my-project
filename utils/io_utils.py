from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TypeVar

DONE_SENTINEL = {"__done__": True}
T = TypeVar("T")


def ensure_parent_dir(file_path: str | Path) -> Path:
    """确保目标文件的父目录存在。"""

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(file_path: str | Path, skip_done: bool = True) -> Iterator[dict[str, Any]]:
    """逐行读取 JSONL 文件。

    默认会跳过阶段完成哨兵行，避免业务逻辑将其当成真实数据处理。
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL 文件不存在: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if skip_done and payload == DONE_SENTINEL:
                continue
            yield payload


def write_jsonl(file_path: str | Path, rows: Iterable[dict[str, Any]], append_done: bool = False) -> None:
    """写入 JSONL 文件。

    当 append_done 为真时，函数会在末尾额外写入一行阶段完成哨兵。
    """

    path = ensure_parent_dir(file_path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if append_done:
            handle.write(json.dumps(DONE_SENTINEL, ensure_ascii=False) + "\n")


def append_done_sentinel(file_path: str | Path) -> None:
    """在 JSONL 文件末尾追加完成哨兵。"""

    path = ensure_parent_dir(file_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(DONE_SENTINEL, ensure_ascii=False) + "\n")


def is_jsonl_done(file_path: str | Path) -> bool:
    """检查 JSONL 文件最后一条记录是否为完成哨兵。"""

    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        return False

    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        step = min(4096, size)
        handle.seek(-step, 2)
        tail = handle.read().decode("utf-8", errors="ignore")

    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if not lines:
        return False

    try:
        return json.loads(lines[-1]) == DONE_SENTINEL
    except json.JSONDecodeError:
        return False


def save_npz(file_path: str | Path, ids: list[str], vectors: list[list[float]]) -> None:
    """保存向量矩阵到 NPZ 文件。

    这里延迟导入 numpy，避免未安装依赖时影响不需要向量功能的模块导入。
    """

    import numpy as np

    path = ensure_parent_dir(file_path)
    np.savez(
        path,
        ids=np.array(ids),
        vectors=np.array(vectors, dtype=np.float32),
    )


def load_npz(file_path: str | Path) -> tuple[list[str], list[list[float]]]:
    """从 NPZ 文件读取向量数据。"""

    import numpy as np

    data = np.load(Path(file_path), allow_pickle=True)
    ids = [str(item) for item in data["ids"]]
    vectors = data["vectors"].tolist()
    return ids, vectors


def read_json(file_path: str | Path) -> dict[str, Any]:
    """读取普通 JSON 文件。"""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(file_path: str | Path, payload: dict[str, Any]) -> None:
    """写入普通 JSON 文件。"""

    path = ensure_parent_dir(file_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(file_path: str | Path, content: str) -> None:
    """写入纯文本文件。"""

    path = ensure_parent_dir(file_path)
    path.write_text(content, encoding="utf-8")


def chunked(items: Sequence[T], size: int) -> Iterator[list[T]]:
    """把序列按固定大小分块。"""

    if size <= 0:
        raise ValueError("chunk size 必须大于 0")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def stable_hash8(value: str) -> str:
    """生成稳定的 8 位哈希，用于元素 ID。"""

    return hashlib.md5(value.encode("utf-8")).hexdigest()[:8]
