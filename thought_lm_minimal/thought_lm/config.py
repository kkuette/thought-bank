from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class RunConfig:
    device: str = "auto"
    precision: str = "bf16"
    log_interval: int = 10
    save_every: int = 0
    save_dir: str = "checkpoints"
    enable_tb: bool = True
    tb_log_dir: str = "runs"
    run_name: str = "default"
    tb_heatmap_max_rows: int = 8
    tb_heatmap_row_scale: int = 20
    tb_heatmap_col_scale: int = 8
    metrics_file: str = "metrics.jsonl"


@dataclass
class HFDataConfig:
    name: Optional[str] = None  # e.g., "bigcode/the-stack-smol"
    split: str = "train"
    text_field: str = "content"
    languages: Optional[list[str]] = None  # e.g., ["python"]
    streaming: bool = True
    shuffle_buffer: int = 10000
    max_samples: Optional[int] = None
    filter_long: bool = True  # drop samples whose tokenized length > seq_len+1


@dataclass
class DataConfig:
    train_dir: Optional[str] = None
    seq_len: int = 256
    batch_size: int = 16
    num_workers: int = 2
    synthetic_samples: int = 20000
    hf_dataset: Optional[HFDataConfig] = None


@dataclass
class ModelConfig:
    vocab_size: int = 260
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.0
    mem_dim: int = 256
    max_mem: int = 32


@dataclass
class TokenizerConfig:
    kind: str = "hf"  # "hf" or "byte"
    hf_name: str = "bigcode/starcoderbase-1b"
    trust_remote_code: bool = False
    add_pad_if_missing: bool = True
    # Default to role tokens only (no context separator)
    special_tokens: list[str] = field(default_factory=lambda: ["<|user|>", "<|assistant|>"])


@dataclass
class LossConfig:
    r_target: float = 0.2
    lambda_budget: float = 0.1
    lambda_pred: float = 0.1
    lambda_margin: float = 0.1
    k_future: int = 32
    margin_delta: float = 0.05


@dataclass
class OptimConfig:
    lr: float = 3.0e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    steps: int = 200
    warmup_steps: int = 20


@dataclass
class EvalConfig:
    enabled: bool = True
    every: int = 100  # evaluate every N steps
    max_batches: int = 50  # number of eval batches
    hf_split: Optional[str] = None  # override split for HF dataset; if null, reuse train split
    mc_sweep_delays: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    mc_sweep_contexts: list[int] = field(default_factory=lambda: [3, 4, 5, 6])


@dataclass
class Config:
    seed: int = 42
    run: RunConfig = RunConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    optim: OptimConfig = OptimConfig()
    eval: EvalConfig = EvalConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()

    @staticmethod
    def from_yaml(path: str | Path) -> "Config":
        """Load a config from a YAML file.

        Parameters
        ----------
        path: str | Path
            YAML path.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        # Minimal manual parsing to dataclasses
        def _get(section: str, cls: Any) -> Any:
            data = raw.get(section, {}) if raw else {}
            return cls(**data)

        data_raw: dict[str, Any] = raw.get("data", {}) if raw else {}
        hf_raw = data_raw.get("hf_dataset")
        hf_cfg = HFDataConfig(**hf_raw) if isinstance(hf_raw, dict) else None
        data_cfg = DataConfig(
            train_dir=data_raw.get("train_dir"),
            seq_len=int(data_raw.get("seq_len", 256)),
            batch_size=int(data_raw.get("batch_size", 16)),
            num_workers=int(data_raw.get("num_workers", 2)),
            synthetic_samples=int(data_raw.get("synthetic_samples", 20000)),
            hf_dataset=hf_cfg,
        )

        tok_cfg = TokenizerConfig(**(raw.get("tokenizer", {}) or {}))

        return Config(
            seed=int(raw.get("seed", 42)),
            run=_get("run", RunConfig),
            data=data_cfg,
            model=_get("model", ModelConfig),
            loss=_get("loss", LossConfig),
            optim=_get("optim", OptimConfig),
            eval=_get("eval", EvalConfig),
            tokenizer=tok_cfg,
        )

