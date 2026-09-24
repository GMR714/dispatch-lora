from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset import read_jsonl, write_jsonl

INSTRUCTION = (
    'Classifique o chamado fictício. Responda somente JSON com queue '
    '(delivery, billing, technical ou manual), priority (critical, normal ou low), '
    'order_id (string ou null) e needs_review (boolean). Não invente um pedido ausente.'
)


def format_row(row: dict) -> dict:
    return {
        'prompt': [
            {'role': 'system', 'content': [{'type': 'text', 'text': INSTRUCTION}]},
            {'role': 'user', 'content': [{'type': 'text', 'text': row['text']}]},
        ],
        'completion': [
            {'role': 'assistant', 'content': [{'type': 'text', 'text': json.dumps(row['target'], ensure_ascii=False, sort_keys=True)}]},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--no-augmentation', action='store_true')
    args = parser.parse_args()
    out = args.out or args.data / 'sft'
    for split in ('train', 'validation'):
        rows = read_jsonl(args.data / f'{split}.jsonl')
        if split == 'train' and not args.no_augmentation and (args.data / 'train_augmented.jsonl').exists():
            rows += read_jsonl(args.data / 'train_augmented.jsonl')
        write_jsonl(out / f'{split}.jsonl', [format_row(row) for row in rows])
    print(f'Prepared prompt-completion SFT data in {out}')


if __name__ == '__main__':
    main()
