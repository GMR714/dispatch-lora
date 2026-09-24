from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from dataset import read_jsonl


def audit(data: Path) -> dict:
    splits = {name: read_jsonl(data / f'{name}.jsonl') for name in ('train', 'validation', 'test')}
    augmented = read_jsonl(data / 'train_augmented.jsonl')
    source = {row['id']: row for row in splits['train']}
    families = {name: {row['family_id'] for row in rows} for name, rows in splits.items()}
    for left in splits:
        for right in splits:
            if left != right and families[left] & families[right]:
                raise ValueError(f'Family leakage: {left}/{right}')
    ids = [row['id'] for rows in splits.values() for row in rows] + [row['id'] for row in augmented]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate IDs')
    for row in augmented:
        base_id = row['id'].replace('-aug1', '-base')
        if base_id not in source or row['split'] != 'train':
            raise ValueError(f'Unknown augmentation source: {row["id"]}')
        parent = source[base_id]
        if row['target'] != parent['target'] or row['anchors'] != parent['anchors']:
            raise ValueError(f'Label or anchor changed: {row["id"]}')
        if row['text'] == parent['text'] or any(anchor not in row['text'] for anchor in row['anchors']):
            raise ValueError(f'Invalid paraphrase: {row["id"]}')
        expected_ids = [row['target']['order_id']] if row['target']['order_id'] else []
        if re.findall(r'ORD-\d{5}', row['text']) != expected_ids:
            raise ValueError(f'Unexpected order reference: {row["id"]}')
    all_texts = [row['text'] for rows in splits.values() for row in rows] + [row['text'] for row in augmented]
    if len(all_texts) != len(set(all_texts)):
        raise ValueError('Duplicate message text')
    return {
        'splits': {name: len(rows) for name, rows in splits.items()},
        'augmented': len(augmented),
        'augmented_share_of_training': round(len(augmented) / (len(splits['train']) + len(augmented)), 4),
        'train_labels': dict(Counter(row['target']['queue'] for row in splits['train'])),
        'test_labels': dict(Counter(row['target']['queue'] for row in splits['test'])),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data'))
    args = parser.parse_args()
    print(json.dumps(audit(args.data), indent=2))
