"""Build the static, case-level report from frozen evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "base_zero": "base-zero",
    "base_few": "base-few",
    "lora_base": "lora-base",
    "lora_augmented": "lora-augmented",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build() -> str:
    cases = read_jsonl(ROOT / "data/eval_all.jsonl")
    comparison = json.loads((ROOT / "results/comparison.json").read_text(encoding="utf-8"))
    training = json.loads((ROOT / "results/training.json").read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in cases}
    if len(cases) != 74 or len(by_id) != len(cases):
        raise ValueError("The frozen evaluation must contain 74 unique cases")

    runs = {}
    for arm, filename in ARMS.items():
        rows = read_jsonl(ROOT / f"results/{filename}.jsonl")
        indexed = {row["id"]: row for row in rows}
        if len(rows) != len(indexed) or set(indexed) != set(by_id):
            raise ValueError(f"{arm}: case IDs do not match the frozen set")
        for case_id, row in indexed.items():
            case = by_id[case_id]
            digest = hashlib.sha256(case["text"].encode("utf-8")).hexdigest()
            if row["target"] != case["target"] or row["text_sha256"] != digest:
                raise ValueError(f"{arm}: target or input hash changed for {case_id}")
        runs[arm] = indexed

    accepted = training["runs"]["lora-augmented"]["training_examples"] - training["runs"]["lora-base"]["training_examples"]
    rejected = len(read_jsonl(ROOT / "data/augmentation_rejections.jsonl"))
    if accepted + rejected != 100:
        raise ValueError("Augmentation acceptance and rejection counts changed")

    report = {
        "schema_version": 1,
        "augmentation": {"attempted": accepted + rejected, "accepted": accepted, "rejected": rejected},
        "model": training["recipe"]["model"],
        "comparison": comparison,
        "training": {
            "recipe": training["recipe"],
            "runs": training["runs"],
            "evaluation_sha256": training["evaluation_sha256"],
        },
        "cases": [
            {
                "id": case["id"],
                "split": "challenge" if case["id"].startswith("challenge-") else "template_test",
                "text": case["text"],
                "target": case["target"],
                "predictions": {
                    arm: {
                        "value": runs[arm][case["id"]]["prediction"],
                        "elapsed_ms": runs[arm][case["id"]]["elapsed_ms"],
                    }
                    for arm in ARMS
                },
            }
            for case in cases
        ],
    }
    for split, count in (("template_test", 54), ("challenge", 20)):
        if sum(case["split"] == split for case in report["cases"]) != count:
            raise ValueError(f"Unexpected {split} count")
        for arm in ARMS:
            measured = sum(
                case["target"] == case["predictions"][arm]["value"]
                for case in report["cases"]
                if case["split"] == split
            )
            reported = round(comparison[split]["runs"][arm]["exact_match"] * count)
            if measured != reported:
                raise ValueError(f"{split}/{arm}: report count differs from frozen predictions")

    return "window.dispatchReport = " + json.dumps(report, ensure_ascii=False, separators=(",", ":")) + ";\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if docs/report-data.js is stale")
    args = parser.parse_args()
    output = ROOT / "docs/report-data.js"
    content = build()
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != content:
            raise SystemExit("docs/report-data.js is stale; run python src/dashboard_data.py")
        print("Dashboard data matches frozen evaluation artifacts")
    else:
        output.parent.mkdir(exist_ok=True)
        output.write_text(content, encoding="utf-8")
        print(f"Wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
