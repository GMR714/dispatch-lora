from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset import read_jsonl
from evaluate import summarize


def read_run(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate case IDs in {path}')
    return {row['id']: row for row in rows}


def split_summary(rows: list[dict]) -> dict:
    result = summarize(rows)
    review_cases = [row for row in rows if row['target']['needs_review']]
    result['review_recall'] = round(
        sum(row['prediction'] is not None and row['prediction']['needs_review'] for row in review_cases) / len(review_cases), 4
    ) if review_cases else None
    result['errors'] = [
        {'id': row['id'], 'target': row['target'], 'prediction': row['prediction']}
        for row in rows if row['prediction'] != row['target']
    ][:5]
    return result


def paired(left: dict[str, dict], right: dict[str, dict], ids: list[str]) -> dict:
    wins = losses = 0
    for case_id in ids:
        left_ok = left[case_id]['prediction'] == left[case_id]['target']
        right_ok = right[case_id]['prediction'] == right[case_id]['target']
        wins += right_ok and not left_ok
        losses += left_ok and not right_ok
    return {'right_only_correct': wins, 'left_only_correct': losses}


def compare(paths: dict[str, Path]) -> dict:
    runs = {name: read_run(path) for name, path in paths.items()}
    first = next(iter(runs.values()))
    ids = list(first)
    if len(ids) != 74:
        raise ValueError(f'Expected 74 frozen evaluation cases, found {len(ids)}')
    for name, rows in runs.items():
        if set(rows) != set(ids):
            raise ValueError(f'Case set mismatch: {name}')
        for case_id in ids:
            if rows[case_id]['target'] != first[case_id]['target']:
                raise ValueError(f'Target mismatch for {case_id}: {name}')
            if rows[case_id]['text_sha256'] != first[case_id]['text_sha256']:
                raise ValueError(f'Input mismatch for {case_id}: {name}')
    splits = {
        'template_test': [case_id for case_id in ids if not case_id.startswith('challenge-')],
        'challenge': [case_id for case_id in ids if case_id.startswith('challenge-')],
    }
    if len(splits['template_test']) != 54 or len(splits['challenge']) != 20:
        raise ValueError('Unexpected frozen split sizes')
    return {
        split: {
            'runs': {name: split_summary([rows[case_id] for case_id in split_ids]) for name, rows in runs.items()},
            'augmented_vs_unaugmented': paired(runs['lora_base'], runs['lora_augmented'], split_ids),
        }
        for split, split_ids in splits.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ('base_zero', 'base_few', 'lora_base', 'lora_augmented'):
        parser.add_argument(f'--{name.replace("_", "-")}', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('results/comparison.json'))
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ('base_zero', 'base_few', 'lora_base', 'lora_augmented')}
    report = compare(paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({split: {name: row['exact_match'] for name, row in data['runs'].items()} for split, data in report.items()}, indent=2))


if __name__ == '__main__':
    main()
