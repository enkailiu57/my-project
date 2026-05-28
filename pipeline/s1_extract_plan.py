from __future__ import annotations

import json
import re
from pathlib import Path

from core.task_executor import PromptTask, resolve_task_executor
from core.types import StageRuntime
from prompts import extract_plan as extract_plan_prompt
from utils.io_utils import ensure_parent_dir, write_jsonl
from utils.news_preprocess_utils import (
    build_doc_id_from_path,
    extract_news_code,
    parse_raw_news_file,
    select_news_files,
)
from utils.text_utils import normalize_surface_text

TRAILING_QUOTE_PATTERN = re.compile(r"[\"'“”‘’＂」』）》〉】]+$")
TRAILING_PUNCT_PATTERN = re.compile(r"[，。！？；,.!?;、]+$")


def _is_recoverable_extract_plan_error(exc: Exception) -> bool:
    if isinstance(exc, RuntimeError) and str(exc).startswith("LLM 调用失败:"):
        return False
    return isinstance(exc, (ValueError, json.JSONDecodeError))


def _build_degraded_plan_row(sent_id: str, sentence: str, exc: Exception) -> dict:
    error_text = str(exc).replace("\n", " ").strip()
    return {
        "句子编号": sent_id,
        "原句": sentence,
        "句子类型": "不抽取句",
        "事件分析": (
            "【句法结构】模型输出未通过结构化校验，程序已自动降级跳过。\n"
            "【公共框架】未生成可用的结构化公共框架。\n"
            "【事件动作单元】本句未保留事件单元。\n"
            "【主客体获得方式】未获得稳定可用的主客体结构。\n"
            "【单元关系】无。\n"
            f"【抽取注意事项】本句因模型输出异常被自动标记为不抽取句；错误：{error_text}"
        ),
        "事件单元": [],
    }


def _normalize_checkpoint_sentence(text: str) -> str:
    normalized = normalize_surface_text(text).strip("\"'“”‘’＂")
    normalized = TRAILING_QUOTE_PATTERN.sub("", normalized)
    normalized = TRAILING_PUNCT_PATTERN.sub("", normalized)
    normalized = TRAILING_QUOTE_PATTERN.sub("", normalized)
    return normalized


def _sentences_equivalent(expected_sentence: str, actual_sentence: str) -> bool:
    if actual_sentence == expected_sentence:
        return True
    return _normalize_checkpoint_sentence(
        actual_sentence
    ) == _normalize_checkpoint_sentence(expected_sentence)


def _finalize_plan_row(sent_id: str, sentence: str, plan_row: dict) -> dict:
    finalized = dict(plan_row)
    finalized["句子编号"] = sent_id
    finalized["原句"] = sentence
    return finalized


def _build_doc_id(source_path: Path) -> str:
    code = extract_news_code(source_path)
    if code is not None:
        prefix, number = code
        return f"{prefix}{number:03d}"
    return build_doc_id_from_path(source_path)


def _resolve_input_files(runtime: StageRuntime) -> list[Path]:
    timeline_dir = runtime.data_dir / "processed_timeline"
    if not timeline_dir.exists():
        raise FileNotFoundError(
            "未找到 data/processed_timeline 目录，无法执行 S1 事件抽取规划。"
            "请先准备时间线校正后的篇章文本。"
        )

    candidate_files = [path for path in timeline_dir.glob("*.txt") if path.is_file()]
    selected_files = select_news_files(candidate_files, runtime.news_scope)
    if not selected_files:
        raise FileNotFoundError(
            f"未找到满足范围 {runtime.news_scope!r} 的 processed_timeline 文档，无法执行 S1。"
        )
    return selected_files


def _build_document_sentences(
    doc_id: str, sentences: list[str]
) -> list[dict[str, str]]:
    return [
        {"句子编号": f"{doc_id}_s{index:03d}", "句子内容": sentence}
        for index, sentence in enumerate(sentences, start=1)
    ]


def _build_sentence_rows(
    doc_id: str,
    source_file: str,
    extract_file: str,
    document_sentences: list[dict[str, str]],
) -> list[dict]:
    rows: list[dict] = []
    for index, item in enumerate(document_sentences, start=1):
        rows.append(
            {
                "sent_id": item["句子编号"],
                "doc_id": doc_id,
                "sentence_index": index,
                "text": item["句子内容"],
                "source_file": source_file,
                "extract_file": extract_file,
            }
        )
    return rows


def _load_existing_plan_rows(
    extract_path: Path,
    document_sentences: list[dict[str, str]],
) -> tuple[list[dict], str | None, bool]:
    if not extract_path.exists():
        return [], None, False

    raw_text = extract_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return [], None, False

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return [], f"S1 断点文件损坏，已忽略并从头重跑: {extract_path}（{exc}）", False

    if not isinstance(payload, list):
        return [], f"S1 断点文件格式错误，已忽略并从头重跑: {extract_path}", False
    if len(payload) > len(document_sentences):
        return (
            [],
            (
                "S1 断点文件句子数超过当前篇章句子数，已忽略并从头重跑: "
                f"{extract_path}"
            ),
            False,
        )

    valid_prefix: list[dict] = []
    checkpoint_dirty = False
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            return (
                valid_prefix,
                (
                    f"S1 断点文件第 {index + 1} 项不是对象，已保留前缀 {len(valid_prefix)} 句并继续: {extract_path}"
                ),
                checkpoint_dirty,
            )

        expected = document_sentences[index]
        if row.get("句子编号") != expected["句子编号"]:
            return (
                valid_prefix,
                (
                    "S1 断点文件与当前篇章句子编号不一致，"
                    f"已保留前缀 {len(valid_prefix)} 句并从第 {index + 1} 句重跑: {extract_path}"
                ),
                checkpoint_dirty,
            )

        actual_sentence = str(row.get("原句", "")).strip()
        if actual_sentence and not _sentences_equivalent(
            expected["句子内容"], actual_sentence
        ):
            return (
                valid_prefix,
                (
                    "S1 断点文件与当前篇章原句不一致，"
                    f"已保留前缀 {len(valid_prefix)} 句并从第 {index + 1} 句重跑: {extract_path}"
                ),
                checkpoint_dirty,
            )

        finalized_row = _finalize_plan_row(
            expected["句子编号"],
            expected["句子内容"],
            row,
        )
        if finalized_row != row:
            checkpoint_dirty = True
        valid_prefix.append(finalized_row)

    return valid_prefix, None, checkpoint_dirty


def _write_plan_rows(extract_path: Path, plan_rows: list[dict]) -> None:
    path = ensure_parent_dir(extract_path)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(plan_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def run_stage(runtime: StageRuntime) -> None:
    """执行 S1 事件抽取规划。"""

    executor = resolve_task_executor(runtime)
    extract_dir = runtime.data_dir / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    sentence_rows: list[dict] = []

    for file_path in _resolve_input_files(runtime):
        document = parse_raw_news_file(file_path)
        doc_id = _build_doc_id(file_path)
        extract_file = f"{file_path.stem}.json"
        extract_path = extract_dir / extract_file

        document_sentences = _build_document_sentences(doc_id, document.sentences)
        sentence_rows.extend(
            _build_sentence_rows(
                doc_id=doc_id,
                source_file=file_path.name,
                extract_file=extract_file,
                document_sentences=document_sentences,
            )
        )

        plan_rows, resume_warning, checkpoint_dirty = _load_existing_plan_rows(
            extract_path, document_sentences
        )
        if resume_warning:
            runtime.logger.warning(resume_warning)
        processed_count = len(plan_rows)
        if processed_count:
            if processed_count == len(document_sentences):
                runtime.logger.info(
                    "S1 文档 %s 已存在完整规划结果，跳过 LLM 调用。",
                    file_path.name,
                )
            else:
                runtime.logger.info(
                    "S1 文档 %s 检测到断点，已完成 %s/%s 句，将从下一句继续。",
                    file_path.name,
                    processed_count,
                    len(document_sentences),
                )
        if checkpoint_dirty:
            _write_plan_rows(extract_path, plan_rows)

        for item in document_sentences[processed_count:]:
            sent_id = item["句子编号"]
            sentence = item["句子内容"]
            task_name = (
                "extract_plan" if runtime.mode == "sync" else f"extract_plan_{sent_id}"
            )
            try:
                result = executor.run_prompt_tasks(
                    stage_name=runtime.stage_name,
                    task_name=task_name,
                    prompt_module=extract_plan_prompt,
                    tasks=[
                        PromptTask(
                            custom_id=sent_id,
                            context={
                                "document_sentences": document_sentences,
                                "sentence_id": sent_id,
                                "sentence_text": sentence,
                            },
                        )
                    ],
                    temperature=runtime.config.temperature_extract,
                    max_tokens=runtime.config.max_tokens,
                    mode=runtime.mode,
                )
                plan_row = result[sent_id]
            except Exception as exc:  # noqa: BLE001
                if not _is_recoverable_extract_plan_error(exc):
                    raise
                runtime.logger.warning(
                    "S1 句子 %s 结构化解析失败，已自动降级跳过。错误: %s",
                    sent_id,
                    exc,
                )
                plan_row = _build_degraded_plan_row(sent_id, sentence, exc)

            plan_rows.append(_finalize_plan_row(sent_id, sentence, plan_row))
            _write_plan_rows(extract_path, plan_rows)

        manifest_rows.append(
            {
                "doc_id": doc_id,
                "title": document.title,
                "source_file": file_path.name,
                "extract_file": extract_file,
                "sentence_count": len(document.sentences),
                "planned_sentence_count": sum(
                    1 for item in plan_rows if item.get("事件单元")
                ),
                "planned_unit_count": sum(
                    len(item.get("事件单元", [])) for item in plan_rows
                ),
            }
        )

    write_jsonl(
        runtime.output_dir / "s1_extract_plan_manifest.jsonl",
        manifest_rows,
        append_done=True,
    )
    write_jsonl(
        runtime.output_dir / "s1_sentence_index.jsonl",
        sentence_rows,
        append_done=True,
    )
