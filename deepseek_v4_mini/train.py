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
from torch.utils.checkpoint import checkpoint
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

def _build_synthetic_recall(cfg_dict: dict):
    """Procedural associative-recall task that *requires* cross-segment memory.

    Layout per sequence (small synthetic vocab):
        [BOS] k1 v1 k2 v2 ... kM vM [SEP] kq vq kq vq ...
    The M key→value pairs (keys distinct) sit in the first segments; the query
    section repeats already-seen keys whose value must be reproduced. Because the
    text stream only attends within one `mem_segment_len` window, a query landing
    in a later segment than its pair can ONLY be answered through the thought bank.
    So `ablation_gap` (CE without − CE with memory) is a near-binary verdict on
    whether the memory works.

    Vocab layout: 0=PAD 1=BOS 2=SEP, keys=[3, 3+n_keys), values=[3+n_keys, ...).
    Set the model's `vocab_size` to at least 3 + n_keys + n_values.
    """
    from torch.utils.data import IterableDataset, DataLoader

    d        = cfg_dict["data"]
    seq_len  = d["seq_len"]
    bs       = d["batch_size"]
    n_keys   = int(d.get("n_keys", 40))
    n_values = int(d.get("n_values", 16))
    n_pairs  = min(int(d.get("n_pairs", n_keys)), n_keys)   # distinct keys
    PAD, BOS, SEP, KEY_OFF = 0, 1, 2, 3
    VAL_OFF = KEY_OFF + n_keys

    class SynthDS(IterableDataset):
        def __iter__(self):
            while True:
                keys = torch.randperm(n_keys)[:n_pairs]
                vals = torch.randint(0, n_values, (n_pairs,))
                kv   = {int(k): int(v) for k, v in zip(keys.tolist(), vals.tolist())}
                klist = list(kv.keys())

                seq = [BOS]
                for k, v in zip(keys.tolist(), vals.tolist()):
                    seq.append(KEY_OFF + k)
                    seq.append(VAL_OFF + v)
                seq.append(SEP)
                while len(seq) < seq_len + 1:                  # query section
                    qk = klist[int(torch.randint(0, len(klist), (1,)))]
                    seq.append(KEY_OFF + qk)
                    if len(seq) < seq_len + 1:
                        seq.append(VAL_OFF + kv[qk])

                chunk = torch.tensor(seq[:seq_len + 1], dtype=torch.long)
                yield chunk[:-1], chunk[1:]

    return DataLoader(SynthDS(), batch_size=bs, num_workers=0)


def _build_latent_context(cfg_dict: dict):
    """Procedural 'gist memory' task: a persistent latent context beyond the
    attention window.

    Layout per sequence:
        [BOS] [CTX_c] s s s s ...        (symbols s ~ P_c, a FIXED per-context dist)
    The context token c appears once at the start (segment 0). Every later symbol
    is drawn from a distribution P_c that depends on c, so predicting symbols in
    segments >0 is far easier IF the model still 'knows' c — but attention only
    spans one `mem_segment_len` window, so after segment 0 the ONLY way to keep c
    is the thought bank. This is the gist/summary use-case (remember *broadly*
    what's going on), not exact recall.

    CE floor per symbol: ~H(P_c) with the gist carried vs ~ln(n_symbols) without,
    so ablation_gap ≈ ln(S) − H(P_c), large and *sustained* if the memory works.

    Vocab: 0=PAD 1=BOS, contexts=[2, 2+C), symbols=[2+C, 2+C+S).
    Set vocab_size >= 2 + n_contexts + n_symbols.
    """
    from torch.utils.data import IterableDataset, DataLoader

    d        = cfg_dict["data"]
    seq_len  = d["seq_len"]
    bs       = d["batch_size"]
    C        = int(d.get("n_contexts", 8))
    S        = int(d.get("n_symbols", 64))
    p_pref   = float(d.get("pref_mass", 0.9))      # prob mass on a context's block
    PAD, BOS, CTX_OFF = 0, 1, 2
    SYM_OFF  = CTX_OFF + C

    # Fixed context→distribution map (shared across all sequences, so it is
    # learnable): each context concentrates p_pref on a disjoint block of symbols.
    k = max(1, S // C)
    dists = torch.empty(C, S)
    for c in range(C):
        dists[c].fill_((1.0 - p_pref) / max(1, S - k))
        dists[c, c * k: c * k + k] = p_pref / k
    dists = dists / dists.sum(dim=1, keepdim=True)

    class CtxDS(IterableDataset):
        def __iter__(self):
            n_sym = seq_len + 1 - 2                 # minus BOS and the context token
            while True:
                c    = int(torch.randint(0, C, (1,)))
                syms = torch.multinomial(dists[c], n_sym, replacement=True)
                seq  = torch.cat([
                    torch.tensor([BOS, CTX_OFF + c], dtype=torch.long),
                    SYM_OFF + syms,
                ])
                yield seq[:-1], seq[1:]

    return DataLoader(CtxDS(), batch_size=bs, num_workers=0)


def _build_dataloader(cfg_dict: dict, tokenizer, split: str = "train"):
    """Thin wrapper around HF streaming dataset → fixed-length batches."""
    if cfg_dict["data"].get("task") == "associative_recall":
        return _build_synthetic_recall(cfg_dict)
    if cfg_dict["data"].get("task") == "latent_context":
        return _build_latent_context(cfg_dict)

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

def _ce_chunk(h_c: torch.Tensor, weight: torch.Tensor, t_c: torch.Tensor) -> torch.Tensor:
    """Cross-entropy (summed) for one chunk of flattened tokens.

    Logits [chunk, V] are produced here and consumed by cross_entropy without
    leaving the function, so under checkpointing they are never stored for the
    backward pass — they get recomputed instead.
    """
    logits = F.linear(h_c, weight)                       # [chunk, V]
    return F.cross_entropy(logits.float(), t_c, reduction="sum")


def fused_cross_entropy(
    hidden: torch.Tensor,        # [B, T, d]
    weight: torch.Tensor,        # [V, d]  (tied LM-head weight)
    targets: torch.LongTensor,   # [B, T]
    chunk_tokens: int = 1024,
) -> torch.Tensor:
    """Memory-efficient next-token cross-entropy.

    Materialising the full [B, T, V] logits (and their fp32 upcast inside
    cross_entropy) is the memory bottleneck with a ~129k vocab. Here we flatten
    the predicted positions and run cross-entropy over chunks of `chunk_tokens`,
    checkpointing each chunk so peak memory is O(chunk_tokens * V) rather than
    O(B * T * V).
    """
    # targets are ALREADY next-token shifted by the dataloader (y=chunk[1:]),
    # aligned position-for-position with `hidden`. Do NOT shift again.
    d   = hidden.size(-1)
    h   = hidden.reshape(-1, d)                          # [N, d]
    tgt = targets.reshape(-1)                            # [N]
    N   = h.size(0)
    if chunk_tokens <= 0:
        chunk_tokens = N

    total = h.new_zeros(())
    for s in range(0, N, chunk_tokens):
        h_c = h[s:s + chunk_tokens]
        t_c = tgt[s:s + chunk_tokens]
        if torch.is_grad_enabled() and h_c.requires_grad:
            loss_c = checkpoint(_ce_chunk, h_c, weight, t_c, use_reentrant=False)
        else:
            loss_c = _ce_chunk(h_c, weight, t_c)
        total = total + loss_c
    return total / max(1, N)


def compute_loss(
    out: dict,
    targets: torch.LongTensor,
    balance_weight: float,
    ce_chunk_tokens: int = 1024,
) -> tuple[torch.Tensor, dict]:
    balance_loss = out["balance_loss"]
    p_gates      = out.get("p_gates")    # [B, T] or None (legacy model only)

    if out.get("logits") is not None:
        logits = out["logits"]            # [B, T, V]
        # targets are ALREADY next-token shifted by the dataloader (y=chunk[1:]),
        # aligned position-for-position with the model input x. Do NOT shift again.
        ce = F.cross_entropy(
            logits.transpose(1, 2),       # [B, V, T]
            targets,                      # [B, T]
        )
    else:
        # Memory-efficient path: model returned hidden states + tied head weight.
        ce = fused_cross_entropy(
            out["hidden"], out["lm_head_weight"], targets, ce_chunk_tokens,
        )

    loss = ce + balance_weight * balance_loss

    ce_val = float(ce.detach())
    logs: dict = {
        "ce":      ce_val,
        "ppl":     float(math.exp(min(ce_val, 30.0))),
        "balance": float(balance_loss.detach()),
    }

    if p_gates is not None:
        logs["r_hat"] = float(p_gates.mean().detach())

    write_alpha = out.get("write_alpha")
    if write_alpha is not None:
        logs["write_alpha"] = float(write_alpha)  # mean write prob α (write/skip)

    return loss, logs


def forward_backward(
    model: nn.Module,
    x: torch.LongTensor,
    y: torch.LongTensor,
    *,
    fused_ce: bool,
    ce_chunk: int,
    balance_w: float,
    scaler,
    seg_len: int,
    grad_accum: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    bptt_window: int = 2,
) -> dict:
    """Forward + backward for one micro-batch, returning averaged logs.

    When `seg_len` is set and shorter than the sequence, the sequence is split
    into segments processed in order while the thought-memory bank is carried
    forward as `init_mem` (truncated BPTT). With `seg_len <= 0` this is a single
    pass and the memory bank never grows past one slot.

    BPTT window (`bptt_window`, W): why it matters for the WRITE path. The memory
    write is a pure *output* of a segment — the segment's own loss never depends
    on it (the write happens after the LM head). The only consumer of a written
    bank is the *next* segment's read. So with W=1 (detach every boundary, the
    old behaviour) the write head — write_ctx_q, write_gate, thought_head,
    write_decision — receives ZERO gradient and can never learn; the bank is
    filled by an untrained projection. With W>=2 the graph is kept across W-1 boundaries
    and backward runs once per window, so segment i+1's loss flows back into
    segment i's write. Memory cost is W segments of activations live at once
    (still bounded). W=2 is the minimal value that trains the write head.
    """
    T = x.size(1)
    if seg_len and 0 < seg_len < T:
        xs = x.split(seg_len, dim=1)
        ys = y.split(seg_len, dim=1)
    else:
        xs, ys = (x,), (y,)
    n_seg = len(xs)
    W = max(1, bptt_window)

    mem: Optional[torch.Tensor] = None
    agg: dict = {}
    window_loss = None          # sum of per-segment losses over the current window
    win_count = 0
    for i, (x_s, y_s) in enumerate(zip(xs, ys)):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(x_s, init_mem=mem, compute_logits=not fused_ce)
            loss, logs = compute_loss(out, y_s, balance_w, ce_chunk)
        # Scale so accumulated grads match a single large batch averaged over
        # both gradient-accumulation micro-batches and segments.
        seg_loss = scaler.scale(loss / (grad_accum * n_seg))
        window_loss = seg_loss if window_loss is None else window_loss + seg_loss
        win_count += 1
        # Carry the bank WITH its graph so the next segment's read connects back
        # to this segment's write; only detach at a window boundary (truncation).
        mem = out["mem_bank"]
        is_boundary = (win_count == W) or (i == n_seg - 1)
        if is_boundary:
            window_loss.backward()       # one backward for the whole window
            mem = mem.detach()
            window_loss = None
            win_count = 0
        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + v
    agg["mem_slots"] = float(mem.size(1)) if mem is not None else 0.0
    return {k: (v / n_seg if k != "mem_slots" else v) for k, v in agg.items()}


@torch.no_grad()
def memory_probe(
    model: nn.Module,
    x: torch.LongTensor,
    y: torch.LongTensor,
    *,
    fused_ce: bool,
    ce_chunk: int,
    balance_w: float,
    seg_len: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> dict:
    """Is the memory bank actually useful? Ablation probe.

    Runs the sequence in segments, carrying the bank as in training. For every
    segment that has a non-empty bank we measure CE twice on the *same* tokens:
      - with the carried bank injected (init_mem = bank)
      - with no memory at all       (init_mem = None)
    The gap `CE_without - CE_with` is how much the memory lowers the loss
    (positive = the bank helps prediction). We also report slot diversity (std
    across slots; ~0 means the slots collapsed to the same vector = useless).
    """
    was_training = model.training
    model.eval()

    xs = x.split(seg_len, dim=1) if (seg_len and 0 < seg_len < x.size(1)) else (x,)
    ys = y.split(seg_len, dim=1) if (seg_len and 0 < seg_len < y.size(1)) else (y,)

    mem: Optional[torch.Tensor] = None
    ce_with, ce_without, alphas = [], [], []
    for x_s, y_s in zip(xs, ys):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out_w = model(x_s, init_mem=mem, compute_logits=not fused_ce)
            _, logs_w = compute_loss(out_w, y_s, balance_w, ce_chunk)
            if out_w.get("write_alpha") is not None:
                alphas.append(float(out_w["write_alpha"]))
            if mem is not None and mem.size(1) > 0:
                out_o = model(x_s, init_mem=None, compute_logits=not fused_ce)
                _, logs_o = compute_loss(out_o, y_s, balance_w, ce_chunk)
                ce_with.append(logs_w["ce"])
                ce_without.append(logs_o["ce"])
        mem = out_w["mem_bank"].detach()

    if was_training:
        model.train()

    gap = (sum(ce_without) - sum(ce_with)) / len(ce_with) if ce_with else 0.0
    diversity = (
        float(mem.float().std(dim=1).mean()) if mem is not None and mem.size(1) > 1 else 0.0
    )
    bank_norm = float(mem.float().norm(dim=-1).mean()) if mem is not None else 0.0
    write_rate = sum(alphas) / len(alphas) if alphas else 0.0
    return {
        "mem_ablation_gap": gap,        # CE_without - CE_with  (>0 => memory helps)
        "mem_diversity":    diversity,  # std across slots (~0 => collapsed/useless)
        "mem_bank_norm":    bank_norm,
        "mem_slots_final":  float(mem.size(1)) if mem is not None else 0.0,
        "mem_write_rate":   write_rate, # mean α: how strongly the model commits writes
    }


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
    # Synthetic tasks have no text: keep vocab_size from the yaml, no tokenizer.
    if data_cfg.get("task") in ("associative_recall", "latent_context"):
        tokenizer = None
        tqdm.write(f"Synthetic task: {data_cfg['task']}  (vocab_size={model_cfg.vocab_size})")
    else:
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
    grad_accum  = max(1, int(train_cfg.get("grad_accum", 1)))
    fused_ce    = bool(train_cfg.get("fused_ce", True))
    ce_chunk    = int(train_cfg.get("ce_chunk_tokens", 1024))
    mem_seg_len = int(train_cfg.get("mem_segment_len", 0))   # 0 = single pass
    # TBPTT window: W>=2 lets gradient reach the memory write head (see
    # forward_backward). W=1 keeps the old behaviour (write head never trains).
    mem_bptt_window = max(1, int(train_cfg.get("mem_bptt_window", 2)))
    mem_probe_every = int(train_cfg.get("mem_probe_every", 0))  # 0 = off
    log_every   = int(train_cfg.get("log_every", 50))
    save_every  = int(train_cfg.get("save_every", 1000))
    save_dir    = Path(train_cfg.get("save_dir", f"checkpoints/{run_name}"))
    balance_w   = float(model_cfg.balance_loss_weight)

    model.train()
    step  = 0          # optimiser steps (one per grad_accum micro-batches)
    micro = 0          # micro-batches seen since the last optimiser step
    t0    = time.perf_counter()
    toks  = 0          # tokens accumulated across the current optimiser step
    pbar  = tqdm(total=total_steps, desc="Training")

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        toks += x.numel()

        logs = forward_backward(
            model, x, y,
            fused_ce=fused_ce, ce_chunk=ce_chunk, balance_w=balance_w,
            scaler=scaler, seg_len=mem_seg_len, grad_accum=grad_accum,
            device=device, amp_dtype=amp_dtype, use_amp=use_amp,
            bptt_window=mem_bptt_window,
        )
        micro += 1
        if micro % grad_accum != 0:
            continue

        if hasattr(scaler, "unscale_"):
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched_step()

        step += 1
        dt    = time.perf_counter() - t0
        tok_s = toks / dt
        t0    = time.perf_counter()
        toks  = 0

        pbar.set_postfix(loss=f"{logs['ce']:.3f}", ppl=f"{logs['ppl']:.1f}", tok_s=f"{tok_s:.0f}")
        pbar.update(1)

        # Memory usefulness probe (ablation: CE with vs without the bank)
        if mem_probe_every and mem_seg_len and step % mem_probe_every == 0:
            probe = memory_probe(
                model, x, y, fused_ce=fused_ce, ce_chunk=ce_chunk,
                balance_w=balance_w, seg_len=mem_seg_len, device=device,
                amp_dtype=amp_dtype, use_amp=use_amp,
            )
            logs.update(probe)
            tqdm.write(
                f"  [mem-probe] ablation_gap(CE↓)={probe['mem_ablation_gap']:+.4f}"
                f"  diversity={probe['mem_diversity']:.4f}"
                f"  bank_norm={probe['mem_bank_norm']:.3f}"
                f"  write_rate(α)={probe['mem_write_rate']:.3f}"
                f"  slots={probe['mem_slots_final']:.0f}"
            )

        if step % log_every == 0 or step == 1:
            tqdm.write(
                f"step={step:>6}  ce={logs['ce']:.4f}  ppl={logs['ppl']:.2f}"
                f"  balance={logs['balance']:.4f}"
                + (f"  r_hat={logs.get('r_hat', 0):.3f}" if "r_hat" in logs else "")
                + (f"  mem={logs['mem_slots']:.0f}" if "mem_slots" in logs else "")
                + (f"  α={logs['write_alpha']:.3f}" if "write_alpha" in logs else "")
                + (f"  gap={logs['mem_ablation_gap']:+.3f}" if "mem_ablation_gap" in logs else "")
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
