from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from statistics import median
from pathlib import Path

from dataset import read_jsonl, write_jsonl
from prepare import INSTRUCTION


FIELDS = ('queue', 'priority', 'order_id', 'needs_review')


def parse_answer(text: str) -> dict | None:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        return None
    if value['queue'] not in ('delivery', 'billing', 'technical', 'manual'):
        return None
    if value['priority'] not in ('critical', 'normal', 'low'):
        return None
    if value['order_id'] is not None and (not isinstance(value['order_id'], str) or not re.fullmatch(r'ORD-\d{5}', value['order_id'])):
        return None
    if type(value['needs_review']) is not bool:
        return None
    return value


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    valid = [row for row in rows if row['prediction'] is not None]
    def has_json_object(row: dict) -> bool:
        if 'raw' not in row:
            return row['prediction'] is not None
        try:
            return isinstance(json.loads(row['raw'].strip()), dict)
        except (json.JSONDecodeError, TypeError):
            return False
    return {
        'cases': n,
        'json_validity': round(sum(has_json_object(row) for row in rows) / n, 4) if n else None,
        'schema_validity': round(len(valid) / n, 4) if n else None,
        'exact_match': round(sum(row['prediction'] == row['target'] for row in rows) / n, 4) if n else None,
        'field_accuracy': {field: round(sum(row['prediction'] is not None and row['prediction'][field] == row['target'][field] for row in rows) / n, 4) if n else None for field in FIELDS},
        'latency_mean_ms': round(sum(row['elapsed_ms'] for row in rows) / n, 2) if n else None,
        'latency_median_ms': round(median(row['elapsed_ms'] for row in rows), 2) if n else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3.5-0.8B')
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--out', type=Path, default=Path('results/base-zero-shot.jsonl'))
    parser.add_argument('--few-shot', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    import torch

    model_name = str(args.adapter) if args.adapter else args.model
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_name, max_seq_length=1024, load_in_4bit=False)
    FastLanguageModel.for_inference(model)
    examples = []
    if args.few_shot:
        seen = set()
        for row in read_jsonl(args.data / 'train.jsonl'):
            if row['target']['queue'] not in seen:
                seen.add(row['target']['queue'])
                examples.extend([{'role': 'user', 'content': [{'type': 'text', 'text': row['text']}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': json.dumps(row['target'], ensure_ascii=False)}]}])
    cases = read_jsonl(args.cases or args.data / 'test.jsonl')
    if args.limit is not None:
        cases = cases[:args.limit]
    results = []
    for index, row in enumerate(cases, 1):
        messages = [{'role': 'system', 'content': [{'type': 'text', 'text': INSTRUCTION}]}, *examples, {'role': 'user', 'content': [{'type': 'text', 'text': row['text']}]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text=prompt, return_tensors='pt').to(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        raw = tokenizer.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        results.append({'id': row['id'], 'family_id': row['family_id'], 'text_sha256': hashlib.sha256(row['text'].encode('utf-8')).hexdigest(), 'target': row['target'],
                        'prediction': parse_answer(raw), 'raw': raw, 'elapsed_ms': elapsed_ms,
                        'model': model_name, 'few_shot': args.few_shot})
        print(f'{index}/{len(cases)}', flush=True)
    write_jsonl(args.out, results)
    print(json.dumps(summarize(results), indent=2))


if __name__ == '__main__':
    main()
