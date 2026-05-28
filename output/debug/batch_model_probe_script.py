import json
from pathlib import Path
from openai import OpenAI
from core.config import AppConfig

cfg = AppConfig.load('.')
client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.request_timeout_seconds)
candidates = [
    'Qwen/Qwen2.5-7B-Instruct',
    'Pro/Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen3-14B',
    'deepseek-ai/DeepSeek-V3',
    'stepfun-ai/Step-3.5-Flash',
    'Pro/zai-org/GLM-4.7',
]
base = Path('output/debug/batch_model_probe')
base.mkdir(parents=True, exist_ok=True)
for index, model in enumerate(candidates, start=1):
    path = base / f'probe_{index:02d}.jsonl'
    request = {
        'custom_id': f'probe_{index}',
        'method': 'POST',
        'url': '/v1/chat/completions',
        'body': {
            'model': model,
            'messages': [
                {'role': 'system', 'content': '你是一个助手'},
                {'role': 'user', 'content': '你好'}
            ],
            'max_tokens': 32,
        },
    }
    path.write_text(json.dumps(request, ensure_ascii=False) + '\n', encoding='utf-8')
    with path.open('rb') as handle:
        file_obj = client.files.create(file=handle, purpose='batch')
    payload = file_obj.model_dump()
    file_id = payload.get('id') or ((payload.get('data') or {}).get('id'))
    try:
        batch = client.batches.create(
            input_file_id=file_id,
            endpoint='/v1/chat/completions',
            completion_window=cfg.batch_completion_window,
        )
        batch_payload = batch.model_dump()
        batch_id = batch_payload.get('id')
        print({'model': model, 'ok': True, 'batch_id': batch_id, 'status': batch_payload.get('status')})
        if batch_id:
            cancelled = client.batches.cancel(batch_id)
            print({'model': model, 'cancelled_status': cancelled.model_dump().get('status')})
        break
    except Exception as exc:
        print({'model': model, 'ok': False, 'error': str(exc)})
