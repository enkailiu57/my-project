from __future__ import annotations

from core.task_executor import PromptTask, resolve_task_executor
from core.types import StageRuntime
from prompts import describe_entity, describe_location, describe_relation
from utils.concept_pool_utils import load_concept_pool_rows
from utils.io_utils import write_jsonl


def _describe_file(
    runtime: StageRuntime,
    element_type: str,
    output_name: str,
    prompt_module,
) -> None:
    executor = resolve_task_executor(runtime)
    input_rows = load_concept_pool_rows(runtime.data_dir / "concept", element_type)
    if not input_rows:
        write_jsonl(runtime.output_dir / output_name, [], append_done=True)
        return
    tasks = [
        PromptTask(
            custom_id=row["id"],
            context={"raw_str": row["raw_str"], "contexts": row.get("contexts", [])},
        )
        for row in input_rows
    ]
    results = executor.run_prompt_tasks(
        stage_name=runtime.stage_name,
        task_name=output_name.replace(".jsonl", ""),
        prompt_module=prompt_module,
        tasks=tasks,
        temperature=runtime.config.temperature_describe,
        max_tokens=runtime.config.max_tokens,
        mode=runtime.mode,
    )
    rows: list[dict] = []

    for row in input_rows:
        rows.append(
            {
                "id": row["id"],
                "raw_str": row["raw_str"],
                "type": row["type"],
                "description": results[row["id"]]["description"],
            }
        )

    write_jsonl(runtime.output_dir / output_name, rows, append_done=True)


def run_stage(runtime: StageRuntime) -> None:
    """执行 S3 描述生成。"""

    _describe_file(
        runtime,
        "entity",
        "s3_entity_desc.jsonl",
        describe_entity,
    )
    _describe_file(
        runtime,
        "relation",
        "s3_relation_desc.jsonl",
        describe_relation,
    )
    _describe_file(
        runtime,
        "location",
        "s3_location_desc.jsonl",
        describe_location,
    )
