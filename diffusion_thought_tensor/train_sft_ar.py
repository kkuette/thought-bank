"""
Supervised Fine-Tuning (SFT) script for ARThoughtModel using TRL's SFTTrainer.

Usage example:
    # Generic
    python diffusion_thought_tensor/train_sft_ar.py \
        --dataset wikitext --dataset_config wikitext-2-raw-v1 --text_field text \
        --output_dir outputs/ar_sft --max_seq_len 256 --batch_size 8 --epochs 1 \
        --tokenizer_name Qwen/Qwen2.5-0.5B

    # Coding presets
    # CodeParrot Clean (field: content, no validation split)
    python diffusion_thought_tensor/train_sft_ar.py \
        --dataset_preset codeparrot-clean --output_dir outputs/ar_sft_cp \
        --max_seq_len 2048 --batch_size 4 --epochs 1 --tokenizer_name Qwen/Qwen2.5-0.5B

    # The Stack Smol (field: content, no validation split)
    python diffusion_thought_tensor/train_sft_ar.py \
        --dataset_preset the-stack-smol --output_dir outputs/ar_sft_stack \
        --max_seq_len 2048 --batch_size 4 --epochs 1 --tokenizer_name Qwen/Qwen2.5-0.5B

    # CodeSearchNet Python (field: func_code_string)
    python diffusion_thought_tensor/train_sft_ar.py \
        --dataset_preset codesearchnet-python --output_dir outputs/ar_sft_csn \
        --max_seq_len 1024 --batch_size 8 --epochs 1 --tokenizer_name Qwen/Qwen2.5-0.5B

Notes:
- The wrapper computes AR next-token loss by unrolling steps with teacher forcing.
- Tokenizer defaults to gpt2; pad token is set to eos.
"""

from __future__ import annotations

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from dataclasses import dataclass
from typing import Optional

import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from diffusion_thought_tensor.utils.callbacks import (
    PerplexityLoggerCallback,
    ThinkingEvalCallback,
)

from diffusion_thought_tensor.model.trl_wrapper import ARHFConfig, ARForCausalLM
from diffusion_thought_tensor.utils.tokenizer_utils import ensure_think_token


@dataclass
class SFTConfig:
    dataset: str = "wikitext"
    dataset_config: Optional[str] = "wikitext-2-raw-v1"
    text_field: str = "text"
    output_dir: str = "outputs/ar_sft"
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"
    think_token_text: str = "<think>"
    dataset_preset: Optional[str] = (
        None  # e.g., codeparrot-clean, the-stack-smol, codesearchnet-python
    )
    d_model: int = 448
    n_layers: int = 12
    n_heads: int = 7
    d_ff: int = 2048
    thought_dim: int = 64
    max_seq_len: int = 256
    max_thoughts: int = 16
    batch_size: int = 4
    lr: float = 5e-5
    epochs: int = 1
    gradient_accumulation_steps: int = 2
    logging_steps: int = 1
    save_steps: int = 1000
    eval_steps: int = 0  # set >0 to enable eval
    gradient_checkpointing: bool = False


def build_model_and_tokenizer(cfg: SFTConfig):
    """Create model and tokenizer and return the think token id as well.

    Returns:
        model, tokenizer, think_token_id
    """
    # Use a modern tokenizer (e.g., Qwen-family). trust_remote_code allows custom tokenizers.
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name,
        trust_remote_code=True,
        use_fast=True,
        model_max_length=cfg.max_seq_len,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Ensure think token is present before model init so vocab_size matches
    tokenizer, think_id, _ = ensure_think_token(tokenizer, cfg.think_token_text)
    hf_cfg = ARHFConfig(
        vocab_size=len(tokenizer),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        thought_dim=cfg.thought_dim,
        max_seq_len=cfg.max_seq_len,
        max_thoughts=cfg.max_thoughts,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )
    model = ARForCausalLM(hf_cfg)
    # If we added tokens post hoc (shouldn't happen since we add before), resize anyway for safety
    if len(model.get_input_embeddings().weight) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer, think_id


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_yaml_to_args(args, y: dict) -> None:
    """Override args in-place from YAML keys if provided.

    Simple 1:1 mapping: YAML keys must match argument names.
    """
    for k, v in (y or {}).items():
        # Nested dicts: flatten one level for common sections like training/generation
        if isinstance(v, dict):
            for kk, vv in v.items():
                setattr(args, kk, vv)
        else:
            setattr(args, k, v)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument("--dataset", type=str, default=SFTConfig.dataset)
    parser.add_argument("--dataset_config", type=str, default=SFTConfig.dataset_config)
    parser.add_argument("--text_field", type=str, default=SFTConfig.text_field)
    parser.add_argument(
        "--dataset_preset",
        type=str,
        choices=["codeparrot-clean", "the-stack-smol", "codesearchnet-python"],
        default=SFTConfig.dataset_preset,
    )
    parser.add_argument("--output_dir", type=str, default=SFTConfig.output_dir)
    parser.add_argument("--tokenizer_name", type=str, default=SFTConfig.tokenizer_name)
    parser.add_argument("--d_model", type=int, default=SFTConfig.d_model)
    parser.add_argument(
        "--think_token_text", type=str, default=SFTConfig.think_token_text
    )
    parser.add_argument("--n_layers", type=int, default=SFTConfig.n_layers)
    parser.add_argument("--n_heads", type=int, default=SFTConfig.n_heads)
    parser.add_argument("--d_ff", type=int, default=SFTConfig.d_ff)
    parser.add_argument("--thought_dim", type=int, default=SFTConfig.thought_dim)
    parser.add_argument("--max_seq_len", type=int, default=SFTConfig.max_seq_len)
    parser.add_argument("--max_thoughts", type=int, default=SFTConfig.max_thoughts)
    parser.add_argument("--batch_size", type=int, default=SFTConfig.batch_size)
    parser.add_argument("--lr", type=float, default=SFTConfig.lr)
    parser.add_argument("--epochs", type=int, default=SFTConfig.epochs)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=SFTConfig.gradient_accumulation_steps,
    )
    parser.add_argument("--logging_steps", type=int, default=SFTConfig.logging_steps)
    parser.add_argument("--save_steps", type=int, default=SFTConfig.save_steps)
    parser.add_argument("--eval_steps", type=int, default=SFTConfig.eval_steps)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    # Generation metrics settings
    parser.add_argument("--gen_samples", type=int, default=8)
    parser.add_argument("--gen_max_new_tokens", type=int, default=128)
    parser.add_argument("--gen_max_new_thoughts", type=int, default=64)
    parser.add_argument("--gen_temperature", type=float, default=1.0)
    parser.add_argument(
        "--gen_top_k", type=int, default=0, help="0 disables top-k; >0 enables"
    )
    parser.add_argument("--log_thinking_on_eval", action="store_true")
    parser.add_argument(
        "--log_thinking_on_train_every",
        type=int,
        default=0,
        help="If >0, also log thinking metrics to TensorBoard during training every N steps (uses eval samples)",
    )
    parser.add_argument("--log_thought_trace_on_eval", action="store_true")
    parser.add_argument("--log_thought_trace_on_train", action="store_true")
    # Control usage of <think> token during generation
    parser.add_argument(
        "--think_logit_bias",
        type=float,
        default=0.0,
        help="Additive bias to the <think> token logit during generation",
    )
    parser.add_argument(
        "--min_thoughts_first",
        type=int,
        default=0,
        help="Force the first N generation steps to be <think> tokens",
    )
    args = parser.parse_args()

    # If YAML config is provided, override args from the YAML
    if args.config:
        y = _load_yaml(args.config)
        _apply_yaml_to_args(args, y)

    cfg = SFTConfig(
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        text_field=args.text_field,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        think_token_text=args.think_token_text,
        dataset_preset=args.dataset_preset,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        thought_dim=args.thought_dim,
        max_seq_len=args.max_seq_len,
        max_thoughts=args.max_thoughts,
        batch_size=args.batch_size,
        lr=float(args.lr),
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Apply dataset preset mapping if provided
    if cfg.dataset_preset:
        if cfg.dataset_preset == "codeparrot-clean":
            cfg.dataset = "codeparrot/codeparrot-clean"
            cfg.dataset_config = None
            cfg.text_field = "content"
        elif cfg.dataset_preset == "the-stack-smol":
            cfg.dataset = "bigcode/the-stack-smol"
            cfg.dataset_config = None
            cfg.text_field = "content"
        elif cfg.dataset_preset == "codesearchnet-python":
            cfg.dataset = "code_search_net"
            cfg.dataset_config = "python"
            cfg.text_field = "func_code_string"
        else:
            raise ValueError(f"Unknown dataset preset: {cfg.dataset_preset}")

    # Data
    dataset = load_dataset(cfg.dataset, cfg.dataset_config)

    # Model and tokenizer
    model, tokenizer, think_id = build_model_and_tokenizer(cfg)

    # Report parameter count (~100M target)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params/1e6:.2f}M")

    # Training args
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        save_steps=cfg.save_steps,
        eval_steps=(cfg.eval_steps if cfg.eval_steps and cfg.eval_steps > 0 else None),
        report_to=["tensorboard"],
        run_name=(
            os.path.basename(cfg.output_dir.rstrip("/")) if cfg.output_dir else None
        ),
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=cfg.gradient_checkpointing,
        remove_unused_columns=True,
        # Explicitly use PyTorch .bin saving to avoid safetensors aliasing issues
        save_safetensors=False,
        # LR and optimizer controls (read from YAML via args if provided)
        weight_decay=getattr(args, "weight_decay", 0.0),
        warmup_steps=getattr(args, "warmup_steps", 0),
        warmup_ratio=getattr(args, "warmup_ratio", None),
        lr_scheduler_type=getattr(args, "lr_scheduler_type", "cosine"),
        optim=getattr(args, "optim", "adamw_torch"),
        adam_beta1=getattr(args, "adam_beta1", 0.9),
        adam_beta2=getattr(args, "adam_beta2", 0.95),
        max_grad_norm=getattr(args, "max_grad_norm", 1.0),
        max_steps=getattr(args, "max_steps", -1),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=(
            dataset.get("validation") if cfg.eval_steps and cfg.eval_steps > 0 else None
        ),
        dataset_text_field=cfg.text_field,
        max_seq_length=cfg.max_seq_len,
        packing=False,  # keep causal order straightforward for the AR loop
        args=training_args,
    )

    # Wire perplexity computation into eval/train logs
    trainer.add_callback(PerplexityLoggerCallback())
    # Optionally log thinking metrics at each evaluation step
    if (
        (
            args.log_thinking_on_eval
            or args.log_thinking_on_train_every > 0
            or args.log_thought_trace_on_eval
            or args.log_thought_trace_on_train
        )
        and "validation" in dataset
        and dataset["validation"] is not None
    ):
        trainer.add_callback(
            ThinkingEvalCallback(
                tokenizer=tokenizer,
                eval_dataset=dataset["validation"],
                text_field=cfg.text_field,
                think_token_id=think_id,
                sample_size=args.gen_samples,
                gen_max_new_tokens=args.gen_max_new_tokens,
                gen_max_new_thoughts=args.gen_max_new_thoughts,
                temperature=args.gen_temperature,
                top_k=(
                    args.gen_top_k if args.gen_top_k and args.gen_top_k > 0 else None
                ),
                train_log_interval=args.log_thinking_on_train_every,
                model=model,
                log_trace_eval=args.log_thought_trace_on_eval,
                log_trace_train=args.log_thought_trace_on_train,
                think_logit_bias=args.think_logit_bias,
                min_thoughts_first=args.min_thoughts_first,
            )
        )

    trainer.train()

    # Final evaluation and explicit perplexity logging if eval set is provided
    if cfg.eval_steps and cfg.eval_steps > 0 and "validation" in dataset:
        eval_metrics = trainer.evaluate(dataset["validation"])
        # Perplexity will be added by callback during evaluation, but ensure present here too
        import math

        if "eval_loss" in eval_metrics:
            eval_metrics["eval_ppl"] = float(math.exp(eval_metrics["eval_loss"]))
        print("Final eval metrics:", eval_metrics)

        # Also run a one-off thinking metrics computation after training if not enabled during eval
        if not args.log_thinking_on_eval:
            try:
                tm_cb = ThinkingEvalCallback(
                    tokenizer=tokenizer,
                    eval_dataset=dataset["validation"],
                    text_field=cfg.text_field,
                    think_token_id=think_id,
                    sample_size=args.gen_samples,
                    gen_max_new_tokens=args.gen_max_new_tokens,
                    gen_max_new_thoughts=args.gen_max_new_thoughts,
                    temperature=args.gen_temperature,
                    top_k=(
                        args.gen_top_k
                        if args.gen_top_k and args.gen_top_k > 0
                        else None
                    ),
                )

                # emulate a call
                class _S:
                    global_step = 0

                _dummy_state = _S()
                tm_cb.on_evaluate(
                    training_args, _dummy_state, None, model=model, metrics=eval_metrics
                )
                print(
                    "Thinking eval metrics:",
                    {
                        k: v
                        for k, v in eval_metrics.items()
                        if k.startswith("eval_think_")
                    },
                )
            except Exception as e:
                print("Thinking metrics computation failed:", e)

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
