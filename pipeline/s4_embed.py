from __future__ import annotations

from core.task_executor import EmbeddingTask, resolve_task_executor
from core.types import StageRuntime
from utils.io_utils import read_jsonl, save_npz


def run_stage(runtime: StageRuntime) -> None:
    """执行 S4 向量嵌入。"""

    executor = resolve_task_executor(runtime)
    rows = []
    for filename in (
        "s3_entity_desc.jsonl",
        "s3_relation_desc.jsonl",
        "s3_location_desc.jsonl",
    ):
        rows.extend(read_jsonl(runtime.output_dir / filename))

    ids = [row["id"] for row in rows]
    descriptions = [row["description"] for row in rows]
    tasks = [
        EmbeddingTask(custom_id=item_id, text=text)
        for item_id, text in zip(ids, descriptions, strict=True)
    ]
    vector_map = executor.run_embedding_tasks(
        stage_name=runtime.stage_name,
        tasks=tasks,
        mode=runtime.mode,
    )
    vectors = [vector_map[item_id] for item_id in ids]

    save_npz(runtime.output_dir / "s4_embeddings.npz", ids, vectors)
