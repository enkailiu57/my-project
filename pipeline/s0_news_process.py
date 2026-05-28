from __future__ import annotations

from core.task_executor import PromptTask, resolve_task_executor
from core.types import StageRuntime
from prompts import news_process as news_process_prompt
from utils.io_utils import write_jsonl, write_text
from utils.news_preprocess_utils import (
    parse_raw_news_file,
    render_processed_news_text,
    resolve_news_preprocess_input_files,
)


def run_stage(runtime: StageRuntime) -> None:
    """执行 S0 新闻预处理。

    该阶段默认优先读取 data/cleaned 下的清洗后 txt 新闻；
    若没有匹配文件，则回退到 data/sources 下的原始 txt 新闻。
    为兼容旧目录结构，当 data/sources 不存在时，再回退到 data 根目录。
    随后先做基础归一化与分句，
    再通过 news_process 提示词压缩为主事件链文本，写入 data/processed 目录。
    """

    raw_files, input_source = resolve_news_preprocess_input_files(
        runtime.data_dir, runtime.news_scope
    )
    if not raw_files:
        raise FileNotFoundError(
            f"未找到满足范围 {runtime.news_scope!r} 的新闻 txt 文件，"
            "已检查 data/cleaned、data/sources 和兼容旧结构的 data/ 目录，无法执行 S0。"
        )

    runtime.logger.info(
        "S0 输入来源: %s",
        (
            "data/cleaned"
            if input_source == "cleaned"
            else ("data/sources" if input_source == "sources" else "data")
        ),
    )

    processed_dir = runtime.data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_documents = [parse_raw_news_file(file_path) for file_path in raw_files]
    manifest_rows: list[dict] = []
    tasks: list[PromptTask] = []
    for document in raw_documents:
        if not document.sentences:
            continue
        tasks.append(
            PromptTask(
                custom_id=document.doc_id,
                context={
                    "title": document.title,
                    "sentences": [
                        {"sentence_id": index, "text": sentence}
                        for index, sentence in enumerate(document.sentences)
                    ],
                },
            )
        )

    executor = resolve_task_executor(runtime)
    results = executor.run_prompt_tasks(
        stage_name=runtime.stage_name,
        task_name="news_process",
        prompt_module=news_process_prompt,
        tasks=tasks,
        temperature=runtime.config.temperature_extract,
        max_tokens=runtime.config.max_tokens,
        mode=runtime.mode,
    )

    for document in raw_documents:
        timeline = results.get(document.doc_id, {}).get("timeline", [])
        processed_text = render_processed_news_text(timeline)
        target_path = processed_dir / document.source_path.name
        write_text(target_path, processed_text + ("\n" if processed_text else ""))
        manifest_rows.append(
            {
                "doc_id": document.doc_id,
                "title": document.title,
                "source_file": document.source_path.name,
                "input_source": input_source,
                "processed_file": target_path.name,
                "input_sentence_count": len(document.sentences),
                "output_event_count": len(timeline),
                "output_char_count": len(processed_text),
            }
        )

    write_jsonl(
        runtime.output_dir / "s0_processed_manifest.jsonl",
        manifest_rows,
        append_done=True,
    )
