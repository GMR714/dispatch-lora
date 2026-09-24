from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

COMPANIES = [
    'Brisa Norte', 'Cedro Azul', 'Faro Sete', 'Jardim Leste', 'Ponte Clara',
    'Vento Sul', 'Cais Novo', 'Monte Firme', 'Linha Verde', 'Lagoa Alta',
]
ISSUES = {
    'delivery': ['a entrega está atrasada', 'a remessa não chegou', 'o envio passou do prazo'],
    'billing': ['houve uma cobrança duplicada', 'a fatura veio em dobro', 'o pagamento foi cobrado duas vezes'],
    'technical': ['o painel não abre', 'a integração retorna erro', 'a tela fica indisponível'],
    'unknown': ['não consigo identificar se o problema é entrega ou cobrança', 'o assunto ainda não está claro'],
}
IMPACTS = {
    'critical': 'A operação está parada hoje.',
    'normal': 'Precisamos de uma atualização quando possível.',
    'low': 'Sem urgência; podemos aguardar a próxima semana.',
}
OPENERS = [
    'Bom dia, equipe.', 'Olá, preciso de ajuda.', 'Escrevo em nome da empresa.',
    'Por favor, verifiquem o caso.', 'Estamos acompanhando este atendimento.',
]


def split_for(family_id: str) -> str:
    value = int(hashlib.sha256(family_id.encode()).hexdigest()[:8], 16) % 100
    return 'train' if value < 70 else 'validation' if value < 85 else 'test'


def build(seed: int, count: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        family_id = f'case-{index:04d}'
        queue = rng.choice(tuple(ISSUES))
        priority = rng.choice(tuple(IMPACTS))
        company = rng.choice(COMPANIES)
        issue = rng.choice(ISSUES[queue])
        impact = IMPACTS[priority]
        order_id = None if rng.random() < 0.18 else f'ORD-{10000 + index}'
        opener = rng.choice(OPENERS)
        reference = f'Referência {order_id}.' if order_id else 'Ainda não tenho o número do pedido.'
        text = f'{opener} Na {company}, {issue}. {reference} {impact}'
        target = {
            'queue': queue if queue != 'unknown' else 'manual',
            'priority': priority,
            'order_id': order_id,
            'needs_review': queue == 'unknown' or order_id is None,
        }
        rows.append({
            'id': f'{family_id}-base', 'family_id': family_id, 'split': split_for(family_id),
            'text': text, 'target': target, 'anchors': [company, issue, reference, impact],
            'origin': {'kind': 'template', 'seed': seed},
        })
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows), encoding='utf-8')


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=714)
    parser.add_argument('--count', type=int, default=360)
    parser.add_argument('--out', type=Path, default=Path('data'))
    args = parser.parse_args()
    rows = build(args.seed, args.count)
    for split in ('train', 'validation', 'test'):
        write_jsonl(args.out / f'{split}.jsonl', [row for row in rows if row['split'] == split])
    print({split: sum(row['split'] == split for row in rows) for split in ('train', 'validation', 'test')})


if __name__ == '__main__':
    main()
