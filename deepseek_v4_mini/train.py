"""
Training script for DeepSeekV4Mini.

Usage:
    python -m deepseek_v4_mini.train configs/tiny.yaml
    python -m deepseek_v4_mini.train configs/small.yaml

Trains with:
  - CE loss on next-token prediction
  - MoE balance auxiliary loss (weighted by balance_loss_weight)
  - Optional thought-memory margin loss (ensures memory augmentation helps)
  - Muon (2-D weights) + AdamW (1-D / embeddings) + linear warmup + cosine decay
  - HuggingFace streaming dataset support
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .config import DeepSeekV4MiniConfig
from .model import DeepSeekV4Mini, DualModalDeepSeekV4Mini


# ── Muon optimiser ────────────────────────────────────────────────────────────

def _zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """
    Hybrid Newton-Schulz orthogonalisation (DeepSeek-V4 §2.4 eq. 28).

    Two-stage schedule:
      - First (steps-2) iterations: (a,b,c) = (3.4445, -4.7750, 2.0315)
        drives singular values rapidly toward 1.
      - Final 2 iterations: (a,b,c) = (2, -1.5, 0.5)
        stabilises singular values precisely at 1.
    """
    assert G.ndim == 2
    X = G / (G.norm() + 1e-7)
    if G.size(0) > G.size(1):
        X = X.T
    fast_steps = max(steps - 2, 0)
    for i in range(steps):
        a, b, c = (3.4445, -4.7750, 2.0315) if i < fast_steps else (2.0, -1.5, 0.5)
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(optim.Optimizer):
    """
    Muon — Momentum Orthogonalised by Newton-Schulz.

    Applies Nesterov momentum then orthogonalises the update via Newton-Schulz,
    targeting 2-D weight matrices.  All other parameters (biases, norms,
    embeddings) are handled by a bundled AdamW group.

    Usage:
        muon_params, adam_params = _split_muon_params(model)
        opt = Muon(muon_params, lr=0.02, wd=0.01,
                   adam_params=adam_params, adam_lr=3e-4)

    Reference: Jordan et al., 2024.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        wd: float = 0.0,
        adam_params=None,
        adam_lr: float = 3e-4,
        adam_betas: tuple = (0.9, 0.95),
        adam_wd: float = 0.1,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, wd=wd)
        super().__init__(params, defaults)
        # Internal AdamW for non-matrix params
        if adam_params is not None:
            self._adam = optim.AdamW(
                adam_params, lr=adam_lr, betas=adam_betas, weight_decay=adam_wd
            )
        else:
            self._adam = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr       = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd       = group["wd"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p)

                buf = state["buf"]
                buf.mul_(momentum).add_(g)

                # Nesterov: g + momentum * buf  (one-step look-ahead)
                update = g.add(buf, alpha=momentum) if nesterov else buf.clone()

                # Orthogonalise 2-D updates via Newton-Schulz
                if update.ndim == 2:
                    update = _zeropower_via_newtonschulz(update.float(), ns_steps)
                    # Scale to match RMS of a standard normal (Jordan et al.)
                    update = update * (update.size(1) ** 0.5)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)

        if self._adam is not None:
            self._adam.step()

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        if self._adam is not None:
            self._adam.zero_grad(set_to_none=set_to_none)


def _split_muon_params(model: nn.Module):
    """
    Split model parameters into:
      - muon_params : 2-D weight matrices that benefit from orthogonalisation
      - adam_params : everything else

    Per DeepSeek-V4 §2.4: AdamW is used for embedding, prediction head, RMSNorm
    weights, AND the static biases (S_pre, S_res, S_post) and gating scalars
    (alpha_pre, alpha_res, alpha_post) of mHC modules.  These are 1-D or scalar
    parameters and fall into adam_params naturally via the ndim != 2 check.
    """
    muon, adam = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 2-D matrices go to Muon, except lookup tables and mHC dynamic generators
        # are excluded by name when they are embedding-like.
        is_matrix = p.ndim == 2
        is_embed  = "embed" in name          # nn.Embedding weight
        is_mhc_static = any(k in name for k in ("S_pre", "S_res", "S_post",
                                                  "alpha_pre", "alpha_res", "alpha_post"))
        if is_matrix and not is_embed and not is_mhc_static:
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam


# ── Utilities ─────────────────────────────────────────────────────────────────

def _device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Data helpers (HF streaming) ───────────────────────────────────────────────

def _build_dataloader(cfg_dict: dict, tokenizer, split: str = "train"):
    """Thin wrapper around HF streaming dataset → fixed-length batches."""
    from datasets import load_dataset
    from torch.utils.data import IterableDataset, DataLoader

    hf     = cfg_dict["data"]
    seq_len = hf["seq_len"]
    bs      = hf["batch_size"]

    class StreamDS(IterableDataset):
        def __iter__(self):
            ds = load_dataset(hf["name"], split=split, streaming=True)
            buf = []
            for ex in ds:
                text = ex.get(hf.get("text_field", "text"), "") or ""
                ids  = tokenizer.encode(text)
                buf.extend(ids)
                while len(buf) >= seq_len + 1:
                    chunk = buf[:seq_len + 1]
                    buf   = buf[seq_len + 1:]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:],  dtype=torch.long)
                    yield x, y

    return DataLoader(StreamDS(), batch_size=bs, num_workers=0)


# ── Loss ──────────────────────────────────────────────────────────────────────

def compute_loss(
    out: dict,
    targets: torch.LongTensor,
    balance_weight: float,
) -> tuple[torch.Tensor, dict]:
    logits       = out["logits"]          # [B, T, V]
    balance_loss = out["balance_loss"]
    p_gates      = out.get("p_gates")    # [B, T] or None (legacy model only)

    ce = F.cross_entropy(
        logits[:, :-1, :].transpose(1, 2),
        targets[:, 1:],
    )

    loss = ce + balance_weight * balance_loss

    logs: dict = {
        "ce":      float(ce.detach()),
        "ppl":     float(math.exp(float(ce.detach()))),
        "balance": float(balance_loss.detach()),
    }

    if p_gates is not None:
        logs["r_hat"] = float(p_gates.mean().detach())

    return loss, logs


# ── Checkpointing ─────────────────────────────────────────────────────────────

def _save(path: Path, model: nn.Module, opt: "Muon", step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer_muon": opt.state_dict(),
        "optimizer_adam": opt._adam.state_dict() if opt._adam else None,
    }, path)
    tqdm.write(f"Saved checkpoint → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("deepseek_v4_mini/configs/tiny.yaml")
    import yaml
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    model_cfg = DeepSeekV4MiniConfig.from_yaml(cfg_path)
    train_cfg = raw.get("training", {})
    data_cfg  = raw.get("data", {})

    device = _device(train_cfg.get("device", "auto"))
    _set_seed(train_cfg.get("seed", 42))

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_name = train_cfg.get("tokenizer", "gpt2")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_cfg.vocab_size = len(tokenizer)

    # ── Model ─────────────────────────────────────────────────────────────────
    if model_cfg.use_dual_stream:
        model = DualModalDeepSeekV4Mini(model_cfg).to(device)
        tqdm.write("Architecture: DualModalDeepSeekV4Mini (with memory bank)")
    else:
        model = DeepSeekV4Mini(model_cfg).to(device)
        tqdm.write("Architecture: DeepSeekV4Mini (no memory bank)")
    tqdm.write(f"Model: {model.num_params():,} parameters")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    muon_lr   = float(train_cfg.get("muon_lr", 0.02))
    adam_lr   = float(train_cfg.get("lr", 3e-4))
    wd        = float(train_cfg.get("weight_decay", 0.1))

    muon_params, adam_params = _split_muon_params(model)
    tqdm.write(
        f"Muon params: {sum(p.numel() for p in muon_params):,}  "
        f"Adam params: {sum(p.numel() for p in adam_params):,}"
    )
    opt = Muon(
        muon_params, lr=muon_lr, momentum=0.95, nesterov=True, ns_steps=10, wd=wd,
        adam_params=adam_params, adam_lr=adam_lr, adam_betas=(0.9, 0.95), adam_wd=wd,
    )

    total_steps   = int(train_cfg.get("steps", 10_000))
    warmup_steps  = int(train_cfg.get("warmup_steps", 200))

    def _lr_fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    # Schedule both Muon LR and Adam LR together
    sched_muon = optim.lr_scheduler.LambdaLR(opt, _lr_fn)
    sched_adam = optim.lr_scheduler.LambdaLR(opt._adam, _lr_fn)

    def sched_step():
        sched_muon.step()
        sched_adam.step()

    # ── Data ──────────────────────────────────────────────────────────────────
    raw["data"] = {**data_cfg, "batch_size": data_cfg.get("batch_size", 4)}
    raw["data"].setdefault("seq_len", model_cfg.max_seq_len)
    dl = _build_dataloader(raw, tokenizer)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    run_name = train_cfg.get("run_name", "dsv4mini")
    writer: Optional[SummaryWriter] = None
    if train_cfg.get("tensorboard", True):
        tb_dir = Path(train_cfg.get("tb_dir", "runs")) / run_name
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))

    # ── Metrics JSONL ─────────────────────────────────────────────────────────
    metrics_path = Path(train_cfg.get("metrics_file", f"runs/{run_name}/metrics.jsonl"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_fh = metrics_path.open("w")

    # ── AMP scaler ────────────────────────────────────────────────────────────
    use_amp   = train_cfg.get("precision", "bf16") in ("bf16", "fp16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if train_cfg.get("precision", "bf16") == "bf16" else torch.float16
    scaler    = torch.cuda.amp.GradScaler(enabled=(train_cfg.get("precision") == "fp16"))

    grad_clip   = float(train_cfg.get("grad_clip", 1.0))
    log_every   = int(train_cfg.get("log_every", 50))
    save_every  = int(train_cfg.get("save_every", 1000))
    save_dir    = Path(train_cfg.get("save_dir", f"checkpoints/{run_name}"))
    balance_w   = float(model_cfg.balance_loss_weight)

    model.train()
    step = 0
    pbar = tqdm(total=total_steps, desc="Training")

    for x, y in dl:
        t0 = time.perf_counter()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(x)
            loss, logs = compute_loss(out, y, balance_w)

        scaler.scale(loss).backward()
        if hasattr(scaler, "unscale_"):
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched_step()

        step += 1
        dt    = time.perf_counter() - t0
        tok_s = x.numel() / dt

        pbar.set_postfix(loss=f"{logs['ce']:.3f}", ppl=f"{logs['ppl']:.1f}", tok_s=f"{tok_s:.0f}")
        pbar.update(1)

        if step % log_every == 0 or step == 1:
            tqdm.write(
                f"step={step:>6}  ce={logs['ce']:.4f}  ppl={logs['ppl']:.2f}"
                f"  balance={logs['balance']:.4f}"
                + (f"  r_hat={logs.get('r_hat', 0):.3f}" if "r_hat" in logs else "")
                + f"  lr={opt.param_groups[0]['lr']:.2e}  tok/s={tok_s:.0f}"
            )
            rec = {"step": step, "lr": opt._adam.param_groups[0]["lr"], **logs}
            metrics_fh.write(json.dumps(rec) + "\n")
            metrics_fh.flush()
            if writer:
                for k, v in logs.items():
                    writer.add_scalar(f"train/{k}", v, step)
                writer.add_scalar("train/lr", opt._adam.param_groups[0]["lr"], step)

        if save_every and step % save_every == 0:
            _save(save_dir / f"step_{step}.pt", model, opt, step)

        if step >= total_steps:
            break

    pbar.close()
    _save(save_dir / "final.pt", model, opt, step)
    metrics_fh.close()
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
