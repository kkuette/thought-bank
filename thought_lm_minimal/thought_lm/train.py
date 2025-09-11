from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, Any
import json
import math
import time

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# Robust GradScaler import across PyTorch versions
try:  # PyTorch >= 2.1
    from torch.amp import GradScaler as AmpGradScaler  # type: ignore
except Exception:  # pragma: no cover
    AmpGradScaler = None  # type: ignore
try:
    from torch.cuda.amp import GradScaler as CudaGradScaler  # type: ignore
except Exception:  # pragma: no cover
    CudaGradScaler = None  # type: ignore

from .config import Config
from .data import CodeTextDataset, build_dataloader, build_hf_stream_dataloader
from .model import ThoughtLM
from .tokenization import build_tokenizer, HFBuildArgs, TokenizerAdapter


def select_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FutureProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


def load_config_from_argv() -> Config:
    # Default path if not provided
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/default.yaml")
    return Config.from_yaml(cfg_path)


def build_eval_loader(cfg: Config, tok: ByteTokenizer) -> DataLoader:
    if cfg.data.hf_dataset and cfg.data.hf_dataset.name:
        hf = cfg.data.hf_dataset
        split = hf.split if cfg.eval.hf_split is None else cfg.eval.hf_split
        max_samples = cfg.eval.max_batches * cfg.data.batch_size
        return build_hf_stream_dataloader(
            name=hf.name,
            split=split,
            text_field=hf.text_field,
            tokenizer=tok,
            seq_len=cfg.data.seq_len,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            languages=hf.languages,
            streaming=hf.streaming,
            shuffle_buffer=hf.shuffle_buffer,
            max_samples=max_samples,
            filter_long=hf.filter_long,
        )
    else:
        ds = CodeTextDataset(
            tokenizer=tok,
            seq_len=cfg.data.seq_len,
            train_dir=cfg.data.train_dir,
            synthetic_samples=cfg.data.synthetic_samples,
            seed=cfg.seed + 1,
        )
        return build_dataloader(ds, cfg.data.batch_size, cfg.data.num_workers)


def run_evaluation(
    model: ThoughtLM,
    dl_eval: DataLoader,
    device: torch.device,
    max_batches: int,
    writer: Optional[SummaryWriter],
    step: int,
    metrics_logger: Optional[JSONLLogger],
) -> None:
    model_was_training = model.training
    model.eval()
    ce_mem_sum = 0.0
    ce_nomem_sum = 0.0
    r_hat_sum = 0.0
    mem_effect_sum = 0.0
    write_rate_hard_sum = 0.0
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(dl_eval):
            if i >= max_batches:
                break
            x = batch.input_ids.to(device)
            y = batch.labels.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=(torch.bfloat16 if device.type == "cuda" else torch.float32),
            ):
                out = model(x)
            logits_mem = out["logits_mem"]
            logits_nomem = out["logits_nomem"]
            p_gates = out["p_gates"]
            ce_mem = F.cross_entropy(logits_mem[:, :-1, :].transpose(1, 2), y[:, 1:])
            ce_nomem = F.cross_entropy(logits_nomem[:, :-1, :].transpose(1, 2), y[:, 1:])
            ce_mem_sum += float(ce_mem.detach().cpu())
            ce_nomem_sum += float(ce_nomem.detach().cpu())
            r_hat_sum += float(p_gates.mean().detach().cpu())
            write_rate_hard_sum += float(((p_gates > 0.5).float().mean()).detach().cpu())
            if "mem_delta" in out:
                mem_effect_sum += float(out["mem_delta"].mean().detach().cpu())
            count += 1

    if count > 0:
        ce_mem_avg = ce_mem_sum / count
        ce_nomem_avg = ce_nomem_sum / count
        r_hat_avg = r_hat_sum / count
        write_rate_hard_avg = write_rate_hard_sum / count
        mem_effect_avg = mem_effect_sum / max(1, count)
        ppl_mem_avg = float(math.exp(ce_mem_avg))
        ppl_nomem_avg = float(math.exp(ce_nomem_avg))
        tqdm.write(
            f"EVAL step={step} ce_mem={ce_mem_avg:.4f} ce_nomem={ce_nomem_avg:.4f} r_hat={r_hat_avg:.3f} "
            f"write_rate_hard={write_rate_hard_avg:.3f} mem_effect={mem_effect_avg:.4f} "
            f"ppl_mem={ppl_mem_avg:.2f} ppl_nomem={ppl_nomem_avg:.2f}"
        )
        if writer is not None:
            writer.add_scalar("eval/ce_mem", ce_mem_avg, step)
            writer.add_scalar("eval/ce_nomem", ce_nomem_avg, step)
            writer.add_scalar("eval/ppl_mem", ppl_mem_avg, step)
            writer.add_scalar("eval/ppl_nomem", ppl_nomem_avg, step)
            writer.add_scalar("eval/r_hat", r_hat_avg, step)
            writer.add_scalar("eval/write_rate_hard", write_rate_hard_avg, step)
            writer.add_scalar("eval/mem_effect", mem_effect_avg, step)
        if metrics_logger is not None:
            metrics_logger.log({
                "phase": "eval",
                "step": int(step),
                "ce_mem": float(ce_mem_avg),
                "ce_nomem": float(ce_nomem_avg),
                "ppl_mem": float(ppl_mem_avg),
                "ppl_nomem": float(ppl_nomem_avg),
                "r_hat": float(r_hat_avg),
                "write_rate_hard": float(write_rate_hard_avg),
                "mem_effect": float(mem_effect_avg),
            })

    if model_was_training:
        model.train()


def save_checkpoint(path: Path, model: nn.Module, opt: optim.Optimizer, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model": model.state_dict(), "optimizer": opt.state_dict()}, path)


def make_optim(model: nn.Module, cfg: Config) -> optim.Optimizer:
    return optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        weight_decay=cfg.optim.weight_decay,
    )


def compute_losses(
    out: dict[str, torch.Tensor],
    targets: torch.LongTensor,
    proj_fut: nn.Module,
    r_target: float,
    k_future: int,
    margin_delta: float,
    lambda_budget: float,
    lambda_pred: float,
    lambda_margin: float,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """Compute LM, budget, predictive, and margin losses and derive thought metrics."""
    logits_mem = out["logits_mem"]
    logits_nomem = out["logits_nomem"]
    h_base = out["h_base"]
    p_gates = out["p_gates"]
    m_used = out["m_used"]
    mem_delta = out.get("mem_delta", None)

    # Cross-entropy with memory vs without
    ce_mem = F.cross_entropy(logits_mem[:, :-1, :].transpose(1, 2), targets[:, 1:])
    ce_nomem = F.cross_entropy(logits_nomem[:, :-1, :].transpose(1, 2), targets[:, 1:])

    # Budget: mean gate prob toward target rate
    r_hat = p_gates.mean()
    loss_budget = (r_hat - r_target) ** 2

    # Predictive: align each written vector to average of next k hidden states
    B, T, H = h_base.shape
    k = min(k_future, T - 1)
    if k > 0:
        fut_means = []
        for t in range(T - k):
            fut = h_base[:, t + 1 : t + 1 + k, :].mean(dim=1)
            fut_means.append(fut)
        fut_means = torch.stack(fut_means, dim=1)  # [B, T-k, H]
        m_trunc = m_used[:, : T - k, :]
        loss_pred = F.mse_loss(m_trunc, proj_fut(fut_means).detach())
    else:
        loss_pred = torch.zeros((), device=h_base.device)

    # Margin: ensure memory helps by delta
    loss_margin = F.relu(margin_delta - (ce_nomem - ce_mem))

    loss = (
        ce_mem + loss_budget * lambda_budget + loss_pred * lambda_pred + loss_margin * lambda_margin
    )
    # Thought metrics
    with torch.no_grad():
        write_rate_hard = (p_gates > 0.5).float().mean()
        eps = 1e-6
        pe = p_gates.clamp(eps, 1 - eps)
        gate_entropy = -(pe * pe.log() + (1 - pe) * (1 - pe).log()).mean()
        mem_norm = torch.norm(m_used, dim=-1).mean()
        mem_nonzero_frac = (torch.norm(m_used, dim=-1) > 1e-6).float().mean()
        mem_effect = (
            mem_delta.mean() if mem_delta is not None else torch.zeros((), device=h_base.device)
        )

    logs = {
        "loss": float(loss.detach().cpu()),
        "ce_mem": float(ce_mem.detach().cpu()),
        "ce_nomem": float(ce_nomem.detach().cpu()),
        "r_hat": float(r_hat.detach().cpu()),
        "loss_budget": float(loss_budget.detach().cpu()),
        "loss_pred": float(loss_pred.detach().cpu()),
        "loss_margin": float(loss_margin.detach().cpu()),
        "write_rate_hard": float(write_rate_hard.detach().cpu()),
        "gate_entropy": float(gate_entropy.detach().cpu()),
        "mem_norm": float(mem_norm.detach().cpu()),
        "mem_nonzero_frac": float(mem_nonzero_frac.detach().cpu()),
        "mem_effect": float(mem_effect.detach().cpu()),
    }
    return loss, logs


class JSONLLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")
    def log(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def main() -> None:
    cfg = load_config_from_argv()
    set_seed(cfg.seed)

    device = select_device(cfg.run.device)

    # TensorBoard writer
    writer: Optional[SummaryWriter] = None
    if cfg.run.enable_tb:
        log_dir = Path(cfg.run.tb_log_dir) / cfg.run.run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))

    # Metrics file logger (overwrite each run)
    metrics_logger = JSONLLogger(Path(cfg.run.metrics_file))

    # Data / Tokenizer
    tok_cfg = cfg.tokenizer
    tok = build_tokenizer(
        kind=tok_cfg.kind,
        hf_args=(HFBuildArgs(name=tok_cfg.hf_name, trust_remote_code=tok_cfg.trust_remote_code, add_pad_if_missing=tok_cfg.add_pad_if_missing) if tok_cfg.kind == "hf" else None),
        extra_special_tokens=tok_cfg.special_tokens,
    )
    # Prefer HF streaming if configured
    if cfg.data.hf_dataset and cfg.data.hf_dataset.name:
        hf = cfg.data.hf_dataset
        dl = build_hf_stream_dataloader(
            name=hf.name,
            split=hf.split,
            text_field=hf.text_field,
            tokenizer=tok,
            seq_len=cfg.data.seq_len,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            languages=hf.languages,
            streaming=hf.streaming,
            shuffle_buffer=hf.shuffle_buffer,
            max_samples=hf.max_samples,
            filter_long=hf.filter_long,
        )
    else:
        ds = CodeTextDataset(
            tokenizer=tok,
            seq_len=cfg.data.seq_len,
            train_dir=cfg.data.train_dir,
            synthetic_samples=cfg.data.synthetic_samples,
            seed=cfg.seed,
        )
        dl = build_dataloader(ds, cfg.data.batch_size, cfg.data.num_workers)

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

    # Optimizer
    opt = make_optim(model, cfg)
    sched = optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=lambda step: min(1.0, (step + 1) / max(1, cfg.optim.warmup_steps)),
    )

    model.train()

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
        scaler = _NoScaler()  # type: ignore

    step = 0
    window_tokens = 0
    window_time = 0.0
    pbar: Any = tqdm(total=cfg.optim.steps, initial=step, desc="Training", dynamic_ncols=True)
    while step < cfg.optim.steps:
        for batch in dl:
            t0 = time.perf_counter()
            step += 1
            x = batch.input_ids.to(device)
            y = batch.labels.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=(
                    (torch.bfloat16 if cfg.run.precision == "bf16" else torch.float16)
                    if device.type == "cuda"
                    else torch.float32
                ),
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
            grad_norm_pre = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip))
            total_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.float().norm(2).item()
                    total_sq += param_norm * param_norm
            grad_norm_post = float(math.sqrt(total_sq)) if total_sq > 0.0 else 0.0
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            t1 = time.perf_counter()
            step_time = t1 - t0
            num_pred_tokens = int(y[:, 1:].numel())
            window_time += step_time
            window_tokens += num_pred_tokens
            # Update progress bar
            curr_ppl_mem = float(math.exp(logs["ce_mem"]))
            curr_tokens_per_s = float(window_tokens / window_time) if window_time > 0 else float("nan")
            pbar.set_postfix(
                loss=f"{logs['loss']:.4f}",
                ppl_mem=f"{curr_ppl_mem:.2f}",
                tok_s=f"{curr_tokens_per_s:.0f}",
            )
            pbar.update(1)

            if step % cfg.run.log_interval == 0 or step == 1:
                ppl_mem = float(math.exp(logs["ce_mem"]))
                ppl_nomem = float(math.exp(logs["ce_nomem"]))
                tokens_per_s = float(window_tokens / window_time) if window_time > 0 else float("nan")
                tqdm.write(
                    f"step={step} loss={logs['loss']:.4f} ce_mem={logs['ce_mem']:.4f} ce_nomem={logs['ce_nomem']:.4f} "
                    f"r_hat={logs['r_hat']:.3f} budget={logs['loss_budget']:.4f} pred={logs['loss_pred']:.4f} margin={logs['loss_margin']:.4f} "
                    f"ppl_mem={ppl_mem:.2f} ppl_nomem={ppl_nomem:.2f} tok/s={tokens_per_s:.0f} grad={grad_norm_pre:.2f}->{grad_norm_post:.2f}"
                )
                # Write metrics to JSONL
                pg = out["p_gates"].detach().float().cpu()
                metrics_logger.log({
                    "phase": "train",
                    "step": int(step),
                    "lr": float(opt.param_groups[0]["lr"]),
                    **{k: float(v) for k, v in logs.items()},
                    "ppl_mem": ppl_mem,
                    "ppl_nomem": ppl_nomem,
                    "tokens_per_s": tokens_per_s,
                    "grad_norm_pre": grad_norm_pre,
                    "grad_norm_post": grad_norm_post,
                    "pg_mean": float(pg.mean().item()),
                    "pg_std": float(pg.std(unbiased=False).item()),
                })
                # reset throughput window after logging
                window_tokens = 0
                window_time = 0.0

                if writer is not None:
                    writer.add_scalar("train/loss", logs["loss"], step)
                    writer.add_scalar("train/ce_mem", logs["ce_mem"], step)
                    writer.add_scalar("train/ce_nomem", logs["ce_nomem"], step)
                    writer.add_scalar("train/ppl_mem", ppl_mem, step)
                    writer.add_scalar("train/ppl_nomem", ppl_nomem, step)
                    writer.add_scalar("train/tokens_per_s", tokens_per_s, step)
                    writer.add_scalar("train/grad_norm_pre", grad_norm_pre, step)
                    writer.add_scalar("train/grad_norm_post", grad_norm_post, step)
                    writer.add_scalar("train/r_hat", logs["r_hat"], step)
                    writer.add_scalar("train/loss_budget", logs["loss_budget"], step)
                    writer.add_scalar("train/loss_pred", logs["loss_pred"], step)
                    writer.add_scalar("train/loss_margin", logs["loss_margin"], step)
                    writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
                    # Histogram of gate probabilities (on a small sample to avoid huge logs)
                    pg_gpu = out["p_gates"].detach().float()
                    pg = pg_gpu.cpu()
                    writer.add_histogram("train/p_gates", pg, step)
                    writer.add_scalar("thought/write_rate_hard", logs["write_rate_hard"], step)
                    writer.add_scalar("thought/gate_entropy", logs["gate_entropy"], step)
                    writer.add_scalar("thought/mem_norm", logs["mem_norm"], step)
                    writer.add_scalar("thought/mem_nonzero_frac", logs["mem_nonzero_frac"], step)
                    writer.add_scalar("thought/mem_effect", logs["mem_effect"], step)
                    # Structural-token gate means (newline, colon, '(')
                    ids = x.detach()
                    if tok.kind == "byte":
                        def _avg_gate_for_byte(tok_id: int) -> float:
                            m = ids == tok_id
                            if m.any():
                                return float(pg_gpu[m].mean().detach().cpu())
                            return float("nan")
                        newline_id = 4 + 10
                        colon_id = 4 + 58
                        lparen_id = 4 + 40
                        writer.add_scalar("thought/gate_newline", _avg_gate_for_byte(newline_id), step)
                        writer.add_scalar("thought/gate_colon", _avg_gate_for_byte(colon_id), step)
                        writer.add_scalar("thought/gate_lparen", _avg_gate_for_byte(lparen_id), step)
                    else:
                        def _avg_gate_for_char(ch: str) -> float:
                            vals = []
                            for b in range(ids.shape[0]):
                                for t in range(ids.shape[1]):
                                    piece = tok.decode([int(ids[b, t].item())])
                                    if ch in piece:
                                        vals.append(float(pg_gpu[b, t].item()))
                            if vals:
                                return float(sum(vals) / len(vals))
                            return float("nan")
                        writer.add_scalar("thought/gate_newline", _avg_gate_for_char("\n"), step)
                        writer.add_scalar("thought/gate_colon", _avg_gate_for_char(":"), step)
                        writer.add_scalar("thought/gate_lparen", _avg_gate_for_char("("), step)
                    # Heatmap of p_gates (upsampled)
                    rows = min(pg.shape[0], int(cfg.run.tb_heatmap_max_rows))
                    heat = pg[:rows, :]  # [rows, T]
                    img = heat.unsqueeze(0).unsqueeze(0)  # [1,1,rows,T]
                    img_up = F.interpolate(
                        img,
                        scale_factor=(int(cfg.run.tb_heatmap_row_scale), int(cfg.run.tb_heatmap_col_scale)),
                        mode="nearest",
                    )  # [1,1,H,W]
                    img_chw = img_up.squeeze(0).repeat(3, 1, 1)  # [3,H,W]
                    writer.add_image("thought/heatmap_p_gates", img_chw, step)

                    # Heatmap of memory effect (mem_delta), min-max normalized per image
                    if "mem_delta" in out:
                        md = out["mem_delta"].detach().float().cpu()  # [B,T]
                        md = md[:rows, :]
                        if md.numel() > 0:
                            md_min = md.min()
                            md_max = md.max()
                            md_norm = (md - md_min) / (md_max - md_min + 1e-6)
                            md_img = md_norm.unsqueeze(0).unsqueeze(0)
                            md_up = F.interpolate(
                                md_img,
                                scale_factor=(int(cfg.run.tb_heatmap_row_scale), int(cfg.run.tb_heatmap_col_scale)),
                                mode="nearest",
                            )
                            md_chw = md_up.squeeze(0).repeat(3, 1, 1)
                            writer.add_image("thought/heatmap_mem_effect", md_chw, step)

                    # Histogram of memory vector norms
                    mem_norm_values = torch.norm(out["m_used"].detach().float(), dim=-1).flatten().cpu()
                    writer.add_histogram("thought/mem_norm_values", mem_norm_values, step)

            if cfg.run.save_every and step % cfg.run.save_every == 0:
                ckpt_path = Path(cfg.run.save_dir) / f"step_{step}.pt"
                save_checkpoint(ckpt_path, model, opt, step)

            # Periodic evaluation
            if cfg.eval.enabled and (step % cfg.eval.every == 0):
                run_evaluation(
                    model,
                    build_eval_loader(cfg, tok),
                    device,
                    max_batches=cfg.eval.max_batches,
                    writer=writer,
                    step=step,
                    metrics_logger=metrics_logger,
                )

            if step >= cfg.optim.steps:
                break

    pbar.close()
    if writer is not None:
        writer.flush()
        writer.close()
    if metrics_logger is not None:
        metrics_logger.close()


if __name__ == "__main__":
    main()
