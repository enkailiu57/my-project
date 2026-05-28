from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import AppConfig
from prompts import news_process as news_process_prompt
from utils.news_preprocess_utils import (
    list_raw_news_files,
    parse_raw_news_file,
    resolve_news_preprocess_input_files,
    resolve_raw_news_dir,
)


@dataclass(slots=True)
class QualitySummary:
    """模型输出质量的轻量摘要。"""

    parse_ok: bool
    parse_error: str | None
    event_count: int
    unique_event_count: int
    covered_sentence_count: int
    covered_sentence_ratio: float
    preview_events: list[str]


@dataclass(slots=True)
class ProbeRoundResult:
    """单轮探测结果。"""

    round_index: int
    elapsed_seconds: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_hit: bool
    cache_ratio: float
    finish_reason: str | None
    reasoning_chars: int
    content_chars: int
    quality: QualitySummary
    request_path: str
    response_path: str


class APIRequestError(RuntimeError):
    """包装 HTTP/网络错误，便于把失败结果结构化落盘。"""

    def __init__(
        self,
        *,
        status_code: int | None,
        response_payload: dict[str, Any] | None,
        response_text: str,
    ) -> None:
        self.status_code = status_code
        self.response_payload = response_payload
        self.response_text = response_text

        error_code = None
        error_message = None
        if response_payload:
            error_block = response_payload.get("error") or {}
            error_code = error_block.get("code")
            error_message = error_block.get("message")

        parts: list[str] = []
        if status_code is not None:
            parts.append(f"HTTP {status_code}")
        if error_code is not None:
            parts.append(f"error.code={error_code}")
        if error_message:
            parts.append(str(error_message))
        super().__init__(" | ".join(parts) if parts else "API 请求失败")


def parse_args() -> argparse.Namespace:
    """解析脚本参数。"""

    parser = argparse.ArgumentParser(description="S0 新闻预处理缓存/质量探针")
    parser.add_argument(
        "--doc-code",
        default="M038",
        help="测试文档编号，例如 M038。默认 M038。",
    )
    parser.add_argument(
        "--input-source",
        choices=("auto", "cleaned", "sources"),
        default="auto",
        help="输入来源。auto=与 S0 一致自动选择；cleaned=强制 data/cleaned；sources=强制 data/sources。",
    )
    parser.add_argument(
        "--model",
        default="",
        help="模型名称。默认读取 config.yaml 中的 llm_model。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="采样温度，默认 0.1。",
    )
    parser.add_argument(
        "--thinking",
        choices=("enabled", "disabled", "omit"),
        default="disabled",
        help="深度思考参数。omit=不传思考字段；SiliconFlow 会映射为 enable_thinking。",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="重复请求次数。默认 2，用于观察 cached_tokens 是否在第二轮命中。",
    )
    parser.add_argument(
        "--sentence-limit",
        type=int,
        default=0,
        help="仅取文档前 N 句送检。0 表示使用全文；当全文触发风控时可用该参数做 M038 子样本探测。",
    )
    parser.add_argument(
        "--cache-probe-mode",
        choices=("exact", "user-nonce"),
        default="exact",
        help="缓存探测模式。exact=完全相同请求；user-nonce=给 user JSON 加轮次标记。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=15000,
        help="输出 token 上限，默认 15000。",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="单次请求超时秒数，默认 300。",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="轮次之间的等待秒数，默认 0。",
    )
    parser.add_argument(
        "--preview-events",
        type=int,
        default=5,
        help="输出质量摘要中展示多少条事件文本，默认 5。",
    )
    parser.add_argument(
        "--output-dir",
        default="output/debug/s0_news_process_probe",
        help="探针输出目录，默认 output/debug/s0_news_process_probe。",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="项目根目录，默认当前目录。",
    )
    return parser.parse_args()


def build_chat_url(base_url: str) -> str:
    """构造 OpenAI 兼容 chat/completions 接口地址。"""

    return f"{base_url.rstrip('/')}/chat/completions"


def resolve_probe_file(
    config: AppConfig, doc_code: str, input_source: str
) -> tuple[Path, str]:
    """解析本次探测要使用的新闻文件。"""

    if input_source == "auto":
        files, source = resolve_news_preprocess_input_files(config.data_dir, doc_code)
    elif input_source == "cleaned":
        files = list_raw_news_files(config.data_dir / "cleaned", doc_code)
        source = "cleaned"
    else:
        sources_dir = config.data_dir / "sources"
        target_dir = (
            sources_dir
            if sources_dir.exists()
            else resolve_raw_news_dir(config.data_dir)
        )
        files = list_raw_news_files(target_dir, doc_code)
        source = "sources" if target_dir.name == "sources" else "raw"

    if not files:
        raise FileNotFoundError(
            f"未找到编号 {doc_code!r} 的测试文档。已检查 input_source={input_source!r} 对应目录。"
        )
    if len(files) > 1:
        raise RuntimeError(
            f"编号 {doc_code!r} 匹配到多个文件，请先收敛唯一文件: {[item.name for item in files]}"
        )
    return files[0], source


def build_probe_context(document, sentence_limit: int) -> dict[str, Any]:
    """按 S0 生产路径构造 news_process 上下文。"""

    selected_sentences = list(document.sentences)
    if sentence_limit > 0:
        selected_sentences = selected_sentences[:sentence_limit]
    if not selected_sentences:
        raise ValueError("送检句子为空，请检查 --sentence-limit 参数。")

    return {
        "title": document.title,
        "sentences": [
            {"sentence_id": index, "text": sentence}
            for index, sentence in enumerate(selected_sentences)
        ],
    }


def mutate_context_for_probe(
    base_context: dict[str, Any], cache_probe_mode: str, round_index: int
) -> dict[str, Any]:
    """为缓存探测模式构造 user 上下文。"""

    context = copy.deepcopy(base_context)
    if cache_probe_mode == "user-nonce":
        # 该字段不会影响生产逻辑，仅用于观察“非完全相同请求”下的缓存命中情况。
        context["_probe_round"] = round_index
    return context


def build_request_body(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    thinking: str,
) -> dict[str, Any]:
    """构造 chat/completions 请求体。"""

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if thinking != "omit":
        body["enable_thinking"] = thinking == "enabled"
    return body


def post_chat_completion(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    """直接调用 OpenAI 兼容 chat/completions，保留完整响应 JSON。"""

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        response_text = exc.read().decode("utf-8", errors="replace")
        response_payload = None
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict):
                response_payload = parsed
        except json.JSONDecodeError:
            response_payload = None
        raise APIRequestError(
            status_code=exc.code,
            response_payload=response_payload,
            response_text=response_text,
        ) from exc
    except urllib.error.URLError as exc:
        raise APIRequestError(
            status_code=None,
            response_payload=None,
            response_text=str(exc),
        ) from exc


def summarize_quality(
    response_content: str,
    sentence_count: int,
    preview_events: int,
) -> QualitySummary:
    """校验 news_process 输出，并给出质量摘要。"""

    try:
        parsed = news_process_prompt.parse_response(response_content)
    except Exception as exc:  # noqa: BLE001
        return QualitySummary(
            parse_ok=False,
            parse_error=str(exc),
            event_count=0,
            unique_event_count=0,
            covered_sentence_count=0,
            covered_sentence_ratio=0.0,
            preview_events=[],
        )

    timeline = parsed.get("timeline", [])
    event_texts = [item["text"].strip() for item in timeline if item["text"].strip()]
    covered_sentence_ids: set[int] = set()
    for item in timeline:
        for sentence_id in item.get("source_sentence_ids", []):
            covered_sentence_ids.add(int(sentence_id))

    covered_sentence_count = len(covered_sentence_ids)
    covered_sentence_ratio = (
        covered_sentence_count / sentence_count if sentence_count else 0.0
    )
    return QualitySummary(
        parse_ok=True,
        parse_error=None,
        event_count=len(timeline),
        unique_event_count=len(set(event_texts)),
        covered_sentence_count=covered_sentence_count,
        covered_sentence_ratio=covered_sentence_ratio,
        preview_events=event_texts[:preview_events],
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """把探针数据写成 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_summary_payload(
    *,
    args: argparse.Namespace,
    probe_file: Path,
    source: str,
    base_url: str,
    model: str,
    base_messages: list[dict[str, Any]],
    original_sentence_count: int,
    selected_sentence_count: int,
    round_results: list[ProbeRoundResult],
    error_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成本次探针的汇总 JSON。"""

    payload: dict[str, Any] = {
        "doc_code": args.doc_code,
        "input_file": str(probe_file),
        "input_source": source,
        "api_base_url": base_url,
        "model": model,
        "temperature": args.temperature,
        "thinking": args.thinking,
        "max_tokens": args.max_tokens,
        "repeat": args.repeat,
        "sentence_limit": args.sentence_limit,
        "cache_probe_mode": args.cache_probe_mode,
        "system_prompt_chars": len(base_messages[0]["content"]),
        "user_prompt_chars": len(base_messages[1]["content"]),
        "original_sentence_count": original_sentence_count,
        "selected_sentence_count": selected_sentence_count,
        "rounds": [asdict(item) for item in round_results],
    }
    if error_info is not None:
        payload["error"] = error_info
    return payload


def print_round_summary(result: ProbeRoundResult) -> None:
    """打印单轮探测摘要。"""

    quality = result.quality
    print(
        f"[round {result.round_index}] elapsed={result.elapsed_seconds:.2f}s "
        f"prompt={result.prompt_tokens} cached={result.cached_tokens} "
        f"completion={result.completion_tokens} total={result.total_tokens} "
        f"cache_ratio={result.cache_ratio:.2%}"
    )
    print(
        f"  finish_reason={result.finish_reason or 'unknown'} "
        f"reasoning_chars={result.reasoning_chars} content_chars={result.content_chars}"
    )
    print(
        f"  parse_ok={quality.parse_ok} events={quality.event_count} "
        f"unique={quality.unique_event_count} "
        f"coverage={quality.covered_sentence_count} ({quality.covered_sentence_ratio:.2%})"
    )
    if quality.parse_error:
        print(f"  parse_error={quality.parse_error}")
    for index, event_text in enumerate(quality.preview_events, start=1):
        print(f"  preview_{index}: {event_text}")


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    config = AppConfig.load(project_root)
    config.ensure_runtime_dirs()

    if not config.api_key:
        raise RuntimeError("未检测到 SiliconFlow API Key，无法执行探针脚本。")

    model = args.model or config.llm_model
    probe_file, source = resolve_probe_file(config, args.doc_code, args.input_source)
    document = parse_raw_news_file(probe_file)
    base_context = build_probe_context(document, args.sentence_limit)
    base_messages = news_process_prompt.build_prompt(base_context)
    selected_sentence_count = len(base_context["sentences"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = (
        project_root / args.output_dir / f"{args.doc_code}_{timestamp}"
    ).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    summary_path = session_dir / "summary.json"

    print("=== S0 新闻预处理探针 ===")
    print(f"文档编号: {args.doc_code}")
    print(f"输入文件: {probe_file}")
    print(f"输入来源: {source}")
    print(f"原始句子数: {len(document.sentences)}")
    print(f"实际送检句子数: {selected_sentence_count}")
    print(f"system_prompt_chars: {len(base_messages[0]['content'])}")
    print(f"user_prompt_chars: {len(base_messages[1]['content'])}")
    print(f"api_base_url: {config.base_url}")
    print(f"model: {model}")
    print(f"temperature: {args.temperature}")
    print(f"thinking: {args.thinking}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"repeat: {args.repeat}")
    print(f"sentence_limit: {args.sentence_limit}")
    print(f"cache_probe_mode: {args.cache_probe_mode}")
    print(f"输出目录: {session_dir}")

    chat_url = build_chat_url(config.base_url)
    round_results: list[ProbeRoundResult] = []

    for round_index in range(1, args.repeat + 1):
        context = mutate_context_for_probe(
            base_context, args.cache_probe_mode, round_index
        )
        messages = news_process_prompt.build_prompt(context)
        request_body = build_request_body(
            model=model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            thinking=args.thinking,
        )

        request_path = session_dir / f"round_{round_index:02d}_request.json"
        response_path = session_dir / f"round_{round_index:02d}_response.json"
        error_path = session_dir / f"round_{round_index:02d}_error.json"
        save_json(request_path, request_body)

        start_time = time.perf_counter()
        try:
            response_json = post_chat_completion(
                url=chat_url,
                api_key=config.api_key,
                payload=request_body,
                timeout=args.timeout,
            )
        except APIRequestError as exc:
            elapsed_seconds = time.perf_counter() - start_time
            response_payload = exc.response_payload or {}
            error_block = response_payload.get("error") or {}
            error_info = {
                "round_index": round_index,
                "elapsed_seconds": elapsed_seconds,
                "request_path": str(request_path),
                "error_path": str(error_path),
                "status_code": exc.status_code,
                "error_code": error_block.get("code"),
                "error_message": error_block.get("message") or str(exc),
                "content_filter": response_payload.get("contentFilter"),
                "response_json": response_payload or None,
                "response_text": exc.response_text,
            }
            save_json(error_path, error_info)
            save_json(
                summary_path,
                build_summary_payload(
                    args=args,
                    probe_file=probe_file,
                    source=source,
                    base_url=config.base_url,
                    model=model,
                    base_messages=base_messages,
                    original_sentence_count=len(document.sentences),
                    selected_sentence_count=selected_sentence_count,
                    round_results=round_results,
                    error_info=error_info,
                ),
            )
            print(f"[round {round_index}] 请求失败: {exc}")
            print(f"  error_log: {error_path}")
            print(f"summary: {summary_path}")
            return 1

        elapsed_seconds = time.perf_counter() - start_time
        save_json(response_path, response_json)

        usage = response_json.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or 0)
        prompt_tokens_details = usage.get("prompt_tokens_details") or {}
        cached_tokens = int(prompt_tokens_details.get("cached_tokens") or 0)

        choice = (response_json.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "")
        reasoning_content = str(message.get("reasoning_content") or "")
        quality = summarize_quality(
            response_content=content,
            sentence_count=selected_sentence_count,
            preview_events=args.preview_events,
        )
        result = ProbeRoundResult(
            round_index=round_index,
            elapsed_seconds=elapsed_seconds,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cache_hit=cached_tokens > 0,
            cache_ratio=(cached_tokens / prompt_tokens if prompt_tokens else 0.0),
            finish_reason=choice.get("finish_reason"),
            reasoning_chars=len(reasoning_content),
            content_chars=len(content),
            quality=quality,
            request_path=str(request_path),
            response_path=str(response_path),
        )
        round_results.append(result)
        print_round_summary(result)

        if args.sleep_seconds > 0 and round_index < args.repeat:
            time.sleep(args.sleep_seconds)

    save_json(
        summary_path,
        build_summary_payload(
            args=args,
            probe_file=probe_file,
            source=source,
            base_url=config.base_url,
            model=model,
            base_messages=base_messages,
            original_sentence_count=len(document.sentences),
            selected_sentence_count=selected_sentence_count,
            round_results=round_results,
        ),
    )

    cache_hits = [item.cached_tokens for item in round_results]
    print("=== 总结 ===")
    print(f"summary: {summary_path}")
    print(f"cached_tokens_by_round: {cache_hits}")
    print(
        "说明: cached_tokens 反映的是整个 prompt 中被平台复用的输入 token。"
        "在本脚本里 system prompt 固定不变，因此第二轮及之后若 cached_tokens > 0，"
        "可视为已至少激活了 system prompt 对应的上下文缓存。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
