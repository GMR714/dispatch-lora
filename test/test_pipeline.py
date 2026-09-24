from __future__ import annotations

import sys
from collections import Counter
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from audit import audit
from augment import validate
from dataset import read_jsonl
from evaluate import parse_answer, summarize


class PipelineTest(unittest.TestCase):
    def test_dataset_audit_and_challenge_separation(self) -> None:
        result = audit(ROOT / 'data')
        self.assertEqual(result['splits']['test'], 54)
        self.assertEqual(result['augmented'], 31)
        training_texts = {row['text'] for row in read_jsonl(ROOT / 'data/train.jsonl')}
        challenge = read_jsonl(ROOT / 'data/challenge.jsonl')
        self.assertEqual(len(challenge), 20)
        self.assertFalse(training_texts & {row['text'] for row in challenge})
        self.assertFalse({row['family_id'] for row in read_jsonl(ROOT / 'data/train.jsonl')} & {row['family_id'] for row in challenge})
        self.assertEqual(dict(Counter(row['target']['queue'] for row in challenge)), {'delivery': 5, 'billing': 5, 'technical': 5, 'manual': 5})

    def test_augmentation_rejects_changed_order(self) -> None:
        row = read_jsonl(ROOT / 'data/train.jsonl')[0]
        text = row['text'].replace(row['target']['order_id'], 'ORD-99999')
        self.assertEqual(validate(row, text), 'anchor_missing')

    def test_invalid_json_is_counted_as_failure(self) -> None:
        self.assertIsNone(parse_answer('I cannot decide'))
        self.assertIsNone(parse_answer('Result: {"queue": "billing"}'))
        target = {'queue': 'billing', 'priority': 'normal', 'order_id': None, 'needs_review': True}
        rows = [
            {'prediction': target, 'target': target, 'elapsed_ms': 10},
            {'prediction': None, 'target': target, 'elapsed_ms': 20},
        ]
        result = summarize(rows)
        self.assertEqual(result['json_validity'], 0.5)
        self.assertEqual(result['exact_match'], 0.5)
        self.assertEqual(result['field_accuracy']['queue'], 0.5)


if __name__ == '__main__':
    unittest.main()
