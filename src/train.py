from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/sft'))
    parser.add_argument('--output', type=Path, default=Path('outputs/dispatch-lora'))
    parser.add_argument('--model', default='Qwen/Qwen3.5-0.8B')
    parser.add_argument('--steps', type=int, default=120)
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for this training recipe')
    torch.manual_seed(714)
    started = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=1024, load_in_4bit=False,
        load_in_16bit=True, full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        use_gradient_checkpointing='unsloth', random_state=714, max_seq_length=1024,
    )
    dataset = load_dataset('json', data_files={
        'train': str(args.data / 'train.jsonl'),
        'validation': str(args.data / 'validation.jsonl'),
    })
    interval = min(30, args.steps)
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=dataset['train'], eval_dataset=dataset['validation'],
        args=SFTConfig(
            output_dir=str(args.output), max_length=1024, completion_only_loss=True, assistant_only_loss=False,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=8, max_steps=args.steps,
            learning_rate=2e-4, warmup_steps=max(1, args.steps // 20), optim='adamw_8bit',
            bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
            eval_strategy='steps', eval_steps=interval, save_steps=interval,
            save_total_limit=1, logging_steps=max(1, min(10, args.steps)),
            seed=714, dataset_num_proc=1, report_to='none',
        ),
    )
    trainer.train()
    adapter = args.output / 'adapter'
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    metadata = {
        'model': args.model, 'seed': 714, 'steps': args.steps,
        'train_sha256': sha256(args.data / 'train.jsonl'),
        'validation_sha256': sha256(args.data / 'validation.jsonl'),
        'gpu': torch.cuda.get_device_name(),
        'peak_vram_bytes': torch.cuda.max_memory_allocated(),
        'elapsed_seconds': round(time.perf_counter() - started, 2),
        'versions': {name: version(name) for name in ('torch', 'transformers', 'trl', 'unsloth')},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
