from __future__ import annotations



import argparse

import hashlib

import json
import re

import urllib.request

from datetime import datetime, timezone

from pathlib import Path



from dataset import read_jsonl, write_jsonl





def request_variant(row: dict, model: str, timeout: int) -> str:

    anchors = row['anchors']

    instruction = (
        'Escreva um novo email curto e natural em portugues para este caso ficticio. '
        'Mude a abertura e a ordem das frases, sem acrescentar fatos. '
        'Os quatro trechos abaixo devem aparecer literalmente uma unica vez no novo email. '
        'Responda apenas o texto do email, sem JSON ou explicacao.\n'
        + 'Trechos obrigatorios:\n' + '\n'.join(f'- {anchor}' for anchor in anchors)
        + '\nMensagem original:\n' + row['text']
    )
    body = json.dumps({

        'model': model, 'stream': False, 'think': False,

        'options': {'temperature': 0.4, 'num_predict': 128},

        'messages': [{'role': 'user', 'content': instruction}],

    }).encode()

    request = urllib.request.Request('http://127.0.0.1:11434/api/chat', data=body, headers={'Content-Type': 'application/json'})

    with urllib.request.urlopen(request, timeout=timeout) as response:

        payload = json.load(response)

    content = payload['message']['content'].strip()

    if content.startswith('{'):

        parsed = json.loads(content)

        if isinstance(parsed, dict):

            content = parsed.get('text', '')

    return content.strip().strip('\"')





def validate(row: dict, text: str) -> str | None:

    if not isinstance(text, str) or len(text) < 40 or len(text) > 600:

        return 'length'

    if text == row['text']:

        return 'unchanged'

    if any(anchor not in text for anchor in row['anchors']):

        return 'anchor_missing'

    if row['target']['order_id'] and text.count(row['target']['order_id']) != 1:

        return 'order_id_changed'

    return None





def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=Path, default=Path('data'))

    parser.add_argument('--model', default='qwen3:1.7b')

    parser.add_argument('--limit', type=int, default=24)

    parser.add_argument('--timeout', type=int, default=180)

    args = parser.parse_args()

    rows = read_jsonl(args.data / 'train.jsonl')[:args.limit]

    accepted, rejected = [], []

    for row in rows:

        try:

            text = request_variant(row, args.model, args.timeout)

            reason = validate(row, text)

            if reason:

                rejected.append({'id': row['id'], 'reason': reason})

                continue

            accepted.append({

                **row, 'id': row['id'].replace('-base', '-aug1'), 'text': text,

                'origin': {

                    'kind': 'ollama_paraphrase', 'model': args.model,

                    'prompt_version': 'augment-v1', 'generated_at': datetime.now(timezone.utc).isoformat(),

                    'source_sha256': hashlib.sha256(row['text'].encode()).hexdigest(),

                },

            })

        except (OSError, ValueError, KeyError, TimeoutError) as error:

            rejected.append({'id': row['id'], 'reason': type(error).__name__})

        print(f"{len(accepted) + len(rejected)}/{len(rows)} accepted={len(accepted)} rejected={len(rejected)}", flush=True)

    write_jsonl(args.data / 'train_augmented.jsonl', accepted)

    write_jsonl(args.data / 'augmentation_rejections.jsonl', rejected)





if __name__ == '__main__':

    main()

