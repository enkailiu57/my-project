from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class LocalCache:
    """基于 SQLite 的本地缓存。

    首期主要给 S5 的实时去重调用复用，后续如果其他阶段也需要，
    可以直接复用这一实现。
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_key TEXT PRIMARY KEY,
                cache_value TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def get(self, cache_key: str) -> dict | None:
        """读取缓存值。"""

        row = self.conn.execute(
            "SELECT cache_value FROM cache_entries WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, cache_key: str, cache_value: dict) -> None:
        """写入缓存值。"""

        self.conn.execute(
            "INSERT OR REPLACE INTO cache_entries(cache_key, cache_value) VALUES(?, ?)",
            (cache_key, json.dumps(cache_value, ensure_ascii=False)),
        )
        self.conn.commit()

    def close(self) -> None:
        """关闭数据库连接。"""

        self.conn.close()

    def __enter__(self) -> "LocalCache":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
