from __future__ import annotations

import sys
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from .config import Config
from .model import ThoughtLM
from .tokenization import build_tokenizer, HFBuildArgs
from .memory_bench import build_mc_dataloader, MCGenConfig, evaluate_multi_context
from .train import FutureProjector, compute_losses, JSONLLogger


# AMP scaler import consistent with train.py
try:  # PyTorch >= 2.1
    from torch.amp import GradScaler as AmpGradScaler  # type: ignore
except Exception:  # pragma: no cover
    AmpGradScaler = None  # type: ignore
try:
    from torch.cuda.amp import GradScaler as CudaGradScaler  # type: ignore
except Exception:  # pragma: no cover
    CudaGradScaler = None  # type: ignore


def _select_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_config() -> Config:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/default.yaml")
    return Config.from_yaml(cfg_path)


def _make_optim(model: nn.Module, cfg: Config) -> optim.Optimizer:
    return optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, betas=cfg.optim.betas, weight_decay=cfg.optim.weight_decay
    )


def main() -> None:
    cfg = _load_config()
    _set_seed(cfg.seed)

    device = _select_device(cfg.run.device)

    # TensorBoard writer
    writer: Optional[SummaryWriter] = None
    if cfg.run.enable_tb:
        log_dir = Path(cfg.run.tb_log_dir) / f"{cfg.run.run_name}_mc"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))

    # Tokenizer
    tok_cfg = cfg.tokenizer
    tok = build_tokenizer(
        kind=tok_cfg.kind,
        hf_args=(
            HFBuildArgs(
                name=tok_cfg.hf_name,
                trust_remote_code=tok_cfg.trust_remote_code,
                add_pad_if_missing=tok_cfg.add_pad_if_missing,
            )
            if tok_cfg.kind == "hf"
            else None
        ),
        extra_special_tokens=tok_cfg.special_tokens,
    )

    # Data: multi-context episodes for training and evaluation (same generator, different size)
    dl_train = build_mc_dataloader(
        tokenizer=tok,
        seq_len=cfg.data.seq_len,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        size=cfg.data.synthetic_samples,
        gen_cfg=MCGenConfig(),
    )
    dl_eval = build_mc_dataloader(
        tokenizer=tok,
        seq_len=cfg.data.seq_len,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        size=max(256, cfg.data.batch_size * cfg.eval.max_batches),
        gen_cfg=MCGenConfig(),
    )

    # Model
    model = ThoughtLM(
        vocab_size=tok.vocab_size,
        dim=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        mem_dim=cfg.model.mem_dim,
        max_mem=cfg.model.max_mem,
    ).to(device)

    proj_fut = FutureProjector(cfg.model.d_model, cfg.model.mem_dim).to(device)

    # Optimizer & LR schedule
    opt = _make_optim(model, cfg)
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: min(1.0, (s + 1) / max(1, cfg.optim.warmup_steps)))

    # AMP scaler
    use_fp16 = (device.type == "cuda" and cfg.run.precision == "fp16")
    if AmpGradScaler is not None:
        scaler = AmpGradScaler(enabled=use_fp16)  # type: ignore
    elif CudaGradScaler is not None:
        scaler = CudaGradScaler(enabled=use_fp16)  # type: ignore
    else:
        class _NoScaler:
            def scale(self, x):
                return x
            def step(self, opt):
                opt.step()
            def update(self):
                return None
            def unscale_(self, opt):
                return None
        scaler = _NoScaler()  # type: ignore

    # Metrics file for this run
    metrics_path = Path(cfg.run.metrics_file)
    if metrics_path.name == "metrics.jsonl":
        metrics_path = metrics_path.with_name("metrics_mc.jsonl")
    metrics = JSONLLogger(metrics_path)

    model.train()
    step = 0
    window_tokens = 0
    window_time = 0.0

    while step < cfg.optim.steps:
        for batch in dl_train:
            step += 1
            t0 = time.perf_counter()
            x = batch.input_ids.to(device)
            y = batch.labels.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=((torch.bfloat16 if cfg.run.precision == "bf16" else torch.float16) if device.type == "cuda" else torch.float32),
            ):
                out = model(x)
                loss, logs = compute_losses(
                    out,
                    y,
                    proj_fut,
                    r_target=cfg.loss.r_target,
                    k_future=cfg.loss.k_future,
                    margin_delta=cfg.loss.margin_delta,
                    lambda_budget=cfg.loss.lambda_budget,
                    lambda_pred=cfg.loss.lambda_pred,
                    lambda_margin=cfg.loss.lambda_margin,
                )

            scaler.scale(loss).backward()
            if hasattr(scaler, "unscale_"):
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()

            # Throughput accounting
            t1 = time.perf_counter()
            window_time += (t1 - t0)
            window_tokens += int(y[:, 1:].numel())

            # Logging
            if step % cfg.run.log_interval == 0 or step == 1:
                ppl_mem = float(math.exp(logs["ce_mem"]))
                ppl_nomem = float(math.exp(logs["ce_nomem"]))
                tok_s = float(window_tokens / window_time) if window_time > 0 else float("nan")
                print(
                    f"mc step={step} loss={logs['loss']:.4f} ce_mem={logs['ce_mem']:.4f} ce_nomem={logs['ce_nomem']:.4f} "
                    f"ppl_mem={ppl_mem:.2f} ppl_nomem={ppl_nomem:.2f} r_hat={logs['r_hat']:.3f} tok/s={tok_s:.0f}",
                    flush=True,
                )
                metrics.log({
                    "phase": "train_mc",
                    "step": int(step),
                    "lr": float(opt.param_groups[0]["lr"]),
                    **{k: float(v) for k, v in logs.items()},
                    "ppl_mem": ppl_mem,
                    "ppl_nomem": ppl_nomem,
                    "tokens_per_s": tok_s,
                })
                if writer is not None:
                    writer.add_scalar("train_mc/loss", logs["loss"], step)
                    writer.add_scalar("train_mc/ce_mem", logs["ce_mem"], step)
                    writer.add_scalar("train_mc/ce_nomem", logs["ce_nomem"], step)
                    writer.add_scalar("train_mc/ppl_mem", ppl_mem, step)
                    writer.add_scalar("train_mc/ppl_nomem", ppl_nomem, step)
                    writer.add_scalar("train_mc/r_hat", logs["r_hat"], step)
                    writer.add_scalar("train_mc/loss_budget", logs["loss_budget"], step)
                    writer.add_scalar("train_mc/loss_pred", logs["loss_pred"], step)
                    writer.add_scalar("train_mc/loss_margin", logs["loss_margin"], step)
                    writer.add_scalar("train_mc/tokens_per_s", tok_s, step)
                    writer.add_scalar("train_mc/lr", float(opt.param_groups[0]["lr"]), step)
                    # Thought metrics and p_gates histogram
                    pg_gpu = out["p_gates"].detach().float()
                    pg = pg_gpu.cpu()
                    writer.add_histogram("train_mc/p_gates", pg, step)
                    writer.add_scalar("thought_mc/write_rate_hard", logs["write_rate_hard"], step)
                    writer.add_scalar("thought_mc/gate_entropy", logs["gate_entropy"], step)
                    writer.add_scalar("thought_mc/mem_norm", logs["mem_norm"], step)
                    writer.add_scalar("thought_mc/mem_nonzero_frac", logs["mem_nonzero_frac"], step)
                    writer.add_scalar("thought_mc/mem_effect", logs["mem_effect"], step)
                window_time = 0.0
                window_tokens = 0

            # Periodic evaluation on multi-context episodes
            if cfg.eval.enabled and (step % cfg.eval.every == 0):
                stats = evaluate_multi_context(model, dl_eval, device)
                print(
                    f"MC EVAL step={step} ce_mem={stats.ce_mem:.4f} ce_nomem={stats.ce_nomem:.4f} "
                    f"ppl_mem={stats.ppl_mem:.2f} ppl_nomem={stats.ppl_nomem:.2f} acc={stats.token_acc:.3f} "
                    f"r_hat={stats.r_hat:.3f} write_rate_hard={stats.write_rate_hard:.3f}",
                    flush=True,
                )
                metrics.log({
                    "phase": "eval_mc",
                    "step": int(step),
                    "ce_mem": float(stats.ce_mem),
                    "ce_nomem": float(stats.ce_nomem),
                    "ppl_mem": float(stats.ppl_mem),
                    "ppl_nomem": float(stats.ppl_nomem),
                    "token_acc": float(stats.token_acc),
                    "r_hat": float(stats.r_hat),
                    "write_rate_hard": float(stats.write_rate_hard),
                })
                if writer is not None:
                    writer.add_scalar("eval_mc/ce_mem", stats.ce_mem, step)
                    writer.add_scalar("eval_mc/ce_nomem", stats.ce_nomem, step)
                    writer.add_scalar("eval_mc/ppl_mem", stats.ppl_mem, step)
                    writer.add_scalar("eval_mc/ppl_nomem", stats.ppl_nomem, step)
                    writer.add_scalar("eval_mc/token_acc", stats.token_acc, step)
                    writer.add_scalar("eval_mc/r_hat", stats.r_hat, step)
                    writer.add_scalar("eval_mc/write_rate_hard", stats.write_rate_hard, step)

                # Per-delay and per-context sweeps (fixed small bins by default)
                delays = getattr(cfg.eval, "mc_sweep_delays", [1, 2, 3, 4])
                ctx_bins = getattr(cfg.eval, "mc_sweep_contexts", [3, 4, 5, 6])
                sweep_size = max(256, cfg.data.batch_size * cfg.eval.max_batches)

                for d in delays:
                    dl_d = build_mc_dataloader(
                        tokenizer=tok,
                        seq_len=cfg.data.seq_len,
                        batch_size=cfg.data.batch_size,
                        num_workers=cfg.data.num_workers,
                        size=sweep_size,
                        gen_cfg=MCGenConfig(min_delay=d, max_delay=d, min_facts=1, max_facts=1),
                    )
                    st_d = evaluate_multi_context(model, dl_d, device)
                    metrics.log({
                        "phase": "eval_mc_by_delay",
                        "step": int(step),
                        "bin": int(d),
                        "ce_mem": float(st_d.ce_mem),
                        "ce_nomem": float(st_d.ce_nomem),
                        "ppl_mem": float(st_d.ppl_mem),
                        "ppl_nomem": float(st_d.ppl_nomem),
                        "token_acc": float(st_d.token_acc),
                        "r_hat": float(st_d.r_hat),
                        "write_rate_hard": float(st_d.write_rate_hard),
                    })
                    if writer is not None:
                        prefix = f"eval_mc_by_delay/d_{d}"
                        writer.add_scalar(f"{prefix}/ce_mem", st_d.ce_mem, step)
                        writer.add_scalar(f"{prefix}/ce_nomem", st_d.ce_nomem, step)
                        writer.add_scalar(f"{prefix}/ppl_mem", st_d.ppl_mem, step)
                        writer.add_scalar(f"{prefix}/ppl_nomem", st_d.ppl_nomem, step)
                        writer.add_scalar(f"{prefix}/token_acc", st_d.token_acc, step)
                        writer.add_scalar(f"{prefix}/r_hat", st_d.r_hat, step)
                        writer.add_scalar(f"{prefix}/write_rate_hard", st_d.write_rate_hard, step)

                for c in ctx_bins:
                    dl_c = build_mc_dataloader(
                        tokenizer=tok,
                        seq_len=cfg.data.seq_len,
                        batch_size=cfg.data.batch_size,
                        num_workers=cfg.data.num_workers,
                        size=sweep_size,
                        gen_cfg=MCGenConfig(min_contexts=c, max_contexts=c, min_facts=1, max_facts=1),
                    )
                    st_c = evaluate_multi_context(model, dl_c, device)
                    metrics.log({
                        "phase": "eval_mc_by_contexts",
                        "step": int(step),
                        "bin": int(c),
                        "ce_mem": float(st_c.ce_mem),
                        "ce_nomem": float(st_c.ce_nomem),
                        "ppl_mem": float(st_c.ppl_mem),
                        "ppl_nomem": float(st_c.ppl_nomem),
                        "token_acc": float(st_c.token_acc),
                        "r_hat": float(st_c.r_hat),
                        "write_rate_hard": float(st_c.write_rate_hard),
                    })
                    if writer is not None:
                        prefix = f"eval_mc_by_contexts/c_{c}"
                        writer.add_scalar(f"{prefix}/ce_mem", st_c.ce_mem, step)
                        writer.add_scalar(f"{prefix}/ce_nomem", st_c.ce_nomem, step)
                        writer.add_scalar(f"{prefix}/ppl_mem", st_c.ppl_mem, step)
                        writer.add_scalar(f"{prefix}/ppl_nomem", st_c.ppl_nomem, step)
                        writer.add_scalar(f"{prefix}/token_acc", st_c.token_acc, step)
                        writer.add_scalar(f"{prefix}/r_hat", st_c.r_hat, step)
                        writer.add_scalar(f"{prefix}/write_rate_hard", st_c.write_rate_hard, step)

            if step >= cfg.optim.steps:
                break

    metrics.close()
    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()

