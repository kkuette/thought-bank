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


def _build_persistent_file_loader(cfg_dict: dict, tokenizer, split: str = "train"):
    """Ordered loader for cross-sequence memory persistence.

    Unlike the default loader (independent random chunks), each of the B batch
    lanes streams ONE source file at a time, emitting its consecutive seq_len
    chunks IN ORDER. The training loop carries the thought bank across steps, so
    within a file the bank accumulates state ("what was defined earlier"); at a
    file boundary that lane is reset. Yields (x, y, reset) where reset[b] is True
    on the first chunk of a new file in lane b.

    The persistence signal comes from long files (> ~2*seq_len tokens give a
    continuation chunk); short files reset immediately and behave like the
    baseline. The codeparrot stream is not repo-contiguous, so the file is the
    natural unit of coherence here.
    """
    from datasets import load_dataset

    hf      = cfg_dict["data"]
    seq_len = hf["seq_len"]
    B       = hf["batch_size"]
    field   = hf.get("text_field", "text")
    name    = hf["name"]

    class PersistDS:
        def __iter__(self):
            it = iter(load_dataset(name, split=split, streaming=True))
            bufs  = [[] for _ in range(B)]   # remaining tokens of each lane's file
            fresh = [True] * B               # is the next chunk the start of a file?

            def next_file(b):
                ex = next(it)
                bufs[b] = tokenizer.encode(ex.get(field, "") or "")
                fresh[b] = True

            while True:
                xs, ys, rs = [], [], []
                for b in range(B):
                    while len(bufs[b]) < seq_len + 1:   # current file exhausted
                        next_file(b)                    # -> reset this lane
                    chunk   = bufs[b][:seq_len + 1]
                    bufs[b] = bufs[b][seq_len + 1:]
                    xs.append(chunk[:-1]); ys.append(chunk[1:]); rs.append(fresh[b])
                    fresh[b] = False                    # later chunks continue the file
                yield (torch.tensor(xs), torch.tensor(ys), torch.tensor(rs, dtype=torch.bool))

    return PersistDS()


def _encode_conversation(ex, tokenizer, u_mark, a_mark, mfield, max_len):
    """Encode one chat example to (ids, turn_starts) or None if too short.

    ids is the full token stream with role markers; turn_starts are EXCHANGE
    boundaries — the index before each user turn except the first. Splitting there
    makes each segment a full exchange [user_k, assistant_k]: the model sees the
    current question in-window and the bank carries only OLDER exchanges (the real
    'remember earlier turns' case), instead of the bank having to carry even the
    immediate question (which an assistant-boundary split forced)."""
    msgs = ex.get(mfield) or []
    ids, turn_starts = [], []
    for m in msgs:
        role    = m.get("role", "")
        content = m.get("content", "") or ""
        mark    = a_mark if role == "assistant" else u_mark
        if role == "user" and ids:          # exchange boundary (not before the 1st turn)
            turn_starts.append(len(ids))
        ids.extend(mark)
        ids.extend(tokenizer.encode(content, add_special_tokens=False))
        if len(ids) >= max_len + 1:
            break
    ids = ids[:max_len + 1]
    turn_starts = [t for t in turn_starts if 0 < t < len(ids) - 1]
    return (ids, turn_starts) if (len(ids) >= 8 and turn_starts) else None


def _build_multiturn_loader(cfg_dict: dict, tokenizer, split: str = "train_sft"):
    """Turn-aligned BATCHED multi-turn loader for turn-cadenced memory writes.

    Writes fire once per TURN (a semantic boundary) instead of every N tokens.
    To keep the GPU busy despite per-turn forwards, B lanes are processed in
    lock-step by TURN INDEX: each yield is turn-slot t of all B lanes, padded to a
    common length. Each lane independently streams conversations (refilling when a
    conversation runs out, like the persistent file loader), so every lane is
    always active — no all-pad rows. The bank [B, M, d] persists across turn-slots
    within a conversation and is reset per-lane at a conversation boundary.

    Yields (x [B,L], y [B,L], loss_mask [B,L] bool, reset [B] bool) where reset[b]
    marks the first turn-segment of a new conversation in lane b. Right-padding +
    causal attention means real tokens never attend to pads; loss_mask zeroes pad
    targets and the write pool ignores pad positions.
    """
    from datasets import load_dataset

    hf      = cfg_dict["data"]
    name    = hf["name"]
    max_len = hf["seq_len"]
    B       = max(1, int(hf.get("batch_size", 1)))
    mfield  = hf.get("messages_field", "messages")
    pad_id  = tokenizer.pad_token_id or 0
    u_mark  = tokenizer.encode("\n<|user|>\n", add_special_tokens=False)
    a_mark  = tokenizer.encode("\n<|assistant|>\n", add_special_tokens=False)

    def conv_to_turns(ids, turn_starts):
        """Split a conversation into (x_seg, y_seg) per turn at turn_starts."""
        x, y = ids[:-1], ids[1:]
        bounds = [0] + turn_starts + [len(x)]
        return [(x[a:b], y[a:b]) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]

    class MultiTurnBatchedDS:
        def __iter__(self):
            it = iter(load_dataset(name, split=split, streaming=True))
            queues  = [[] for _ in range(B)]   # remaining turn-segments per lane
            pending = [False] * B              # next pop starts a new conversation

            def refill(b):
                while not queues[b]:
                    enc = _encode_conversation(next(it), tokenizer, u_mark, a_mark, mfield, max_len)
                    if enc is None:
                        continue
                    queues[b] = conv_to_turns(*enc)
                    pending[b] = True

            while True:
                segs, resets = [], []
                for b in range(B):
                    if not queues[b]:
                        refill(b)
                    segs.append(queues[b].pop(0))
                    resets.append(pending[b])
                    pending[b] = False
                L = max(len(xs) for xs, _ in segs)
                x = torch.full((B, L), pad_id, dtype=torch.long)
                y = torch.full((B, L), pad_id, dtype=torch.long)
                mask = torch.zeros((B, L), dtype=torch.bool)
                for b, (xs, ys) in enumerate(segs):
                    n = len(xs)
                    x[b, :n] = torch.tensor(xs, dtype=torch.long)
                    y[b, :n] = torch.tensor(ys, dtype=torch.long)
                    mask[b, :n] = True
                yield x, y, mask, torch.tensor(resets, dtype=torch.bool)

    return MultiTurnBatchedDS()


def _build_dataloader(cfg_dict: dict, tokenizer, split: str = "train"):
    """Thin wrapper around HF streaming dataset → fixed-length batches."""
    if cfg_dict["data"].get("task") == "associative_recall":
        return _build_synthetic_recall(cfg_dict)
    if cfg_dict["data"].get("task") == "latent_context":
        return _build_latent_context(cfg_dict)
    if cfg_dict["data"].get("task") == "multiturn":
        return _build_multiturn_loader(cfg_dict, tokenizer, cfg_dict["data"].get("split", "train_sft"))
    if cfg_dict["data"].get("persist"):
        return _build_persistent_file_loader(cfg_dict, tokenizer, split)

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
    # _ce_chunk sums and skips ignore_index (-100), so divide by REAL tokens only.
    valid = int((tgt != -100).sum())
    return total / max(1, valid)


def compute_loss(
    out: dict,
    targets: torch.LongTensor,
    balance_weight: float,
    ce_chunk_tokens: int = 1024,
    write_cost: float = 0.0,
    write_diversity: float = 0.0,
    loss_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    balance_loss = out["balance_loss"]
    p_gates      = out.get("p_gates")    # [B, T] or None (legacy model only)

    # Padded multi-turn batches: ignore pad positions in the loss (set to -100,
    # which F.cross_entropy skips; the fused path divides by real-token count).
    if loss_mask is not None:
        targets = targets.masked_fill(~loss_mask.bool(), -100)

    if out.get("logits") is not None:
        logits = out["logits"]            # [B, T, V]
        # targets are ALREADY next-token shifted by the dataloader (y=chunk[1:]),
        # aligned position-for-position with the model input x. Do NOT shift again.
        ce = F.cross_entropy(
            logits.transpose(1, 2),       # [B, V, T]
            targets,                      # [B, T]
            ignore_index=-100,
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

    # Sparsity budget on the write decision: cost · E[-log(1-α)]. Gives writing an
    # opportunity cost so α stops saturating at 1 and becomes selective. Probes call
    # compute_loss with the default write_cost=0.0, so probe CE stays uncontaminated.
    write_penalty = out.get("write_penalty")
    if write_penalty is not None and write_cost > 0.0:
        pen  = write_cost * write_penalty
        loss = loss + pen
        logs["write_pen"] = float(pen.detach())

    # Novelty-gated write: penalise similarity of the new write to existing slots.
    # Pushes the head to store diverse thoughts (raises bank effective rank). The
    # raw redundancy (max cosine) is logged regardless of weight, as a diagnostic.
    write_redundancy = out.get("write_redundancy")
    if write_redundancy is not None:
        logs["write_redund"] = float(write_redundancy.detach())
        if write_diversity > 0.0:
            div  = write_diversity * write_redundancy
            loss = loss + div
            logs["write_div"] = float(div.detach())

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
    init_mem: Optional[torch.Tensor] = None,
    write_cost: float = 0.0,
    write_diversity: float = 0.0,
    seg_bounds: Optional[torch.Tensor] = None,
) -> tuple[dict, Optional[torch.Tensor]]:
    """Forward + backward for one micro-batch, returning (averaged logs, final bank).

    `init_mem` seeds the thought bank at the first segment (for cross-sequence
    persistence the caller passes the previous step's detached bank). The returned
    bank is detached and can be carried into the next step.

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
    if seg_bounds is not None and seg_bounds.numel() > 0:
        # Explicit semantic segmentation (multi-turn: split at assistant-turn starts).
        # One write fires per turn-segment; cadence is the turn, not a fixed length.
        bounds = [int(b) for b in seg_bounds.tolist() if 0 < int(b) < T]
        xs = torch.tensor_split(x, bounds, dim=1)
        ys = torch.tensor_split(y, bounds, dim=1)
    elif seg_len and 0 < seg_len < T:
        xs = x.split(seg_len, dim=1)
        ys = y.split(seg_len, dim=1)
    else:
        xs, ys = (x,), (y,)
    n_seg = len(xs)
    W = max(1, bptt_window)

    mem: Optional[torch.Tensor] = init_mem   # seeded bank (persistence) or None
    agg: dict = {}
    window_loss = None          # sum of per-segment losses over the current window
    win_count = 0
    for i, (x_s, y_s) in enumerate(zip(xs, ys)):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(x_s, init_mem=mem, compute_logits=not fused_ce)
            loss, logs = compute_loss(out, y_s, balance_w, ce_chunk, write_cost, write_diversity)
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
    logs = {k: (v / n_seg if k != "mem_slots" else v) for k, v in agg.items()}
    return logs, mem        # mem is detached (last segment is always a boundary)


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
    eff_rank = _effective_rank(mem)
    return {
        "mem_ablation_gap": gap,        # CE_without - CE_with  (>0 => memory helps)
        "mem_diversity":    diversity,  # std across slots (~0 => collapsed/useless)
        "mem_eff_rank":     eff_rank,   # entropy eff. rank of slots (~1 => duplicates)
        "mem_bank_norm":    bank_norm,
        "mem_slots_final":  float(mem.size(1)) if mem is not None else 0.0,
        "mem_write_rate":   write_rate, # mean α: how strongly the model commits writes
    }


def _effective_rank(mem: Optional[torch.Tensor]) -> float:
    """Entropy effective rank of the bank slots: exp(H(p)) with p = normalised
    squared singular values of the centred bank. ~1 => all slots ≈ one direction
    (near-duplicate writes); ~max_mem => fully diverse. The honest redundancy
    metric — unlike per-dim std (mem_diversity), it catches directional collapse."""
    if mem is None or mem.size(1) < 2:
        return 0.0
    bank = mem[0].float()
    bank = bank - bank.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(bank)
    s2 = sv ** 2
    if float(s2.sum()) <= 0:
        return 0.0
    p = s2 / s2.sum()
    return float(torch.exp(-(p * (p + 1e-12).log()).sum()))


@torch.no_grad()
def persistence_probe(
    model: nn.Module,
    tokenizer,
    cfg_dict: dict,
    *,
    n_chunks: int,
    fused_ce: bool,
    ce_chunk: int,
    balance_w: float,
    seg_len: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> dict:
    """Does carrying the bank ACROSS chunks of the same file lower later-chunk CE?

    Finds one long file, splits it into `n_chunks` consecutive chunks, and runs it
    twice: (a) carrying the bank from chunk to chunk, (b) resetting the bank every
    chunk. The gap on chunks k>0 (CE_reset − CE_carried) is the persistence value —
    the real verdict for the 'remember earlier context' use case. ~0 means
    cross-sequence memory is not helping.
    """
    from datasets import load_dataset

    hf = cfg_dict["data"]
    field, seq_len = hf.get("text_field", "text"), hf["seq_len"]
    need = (seq_len + 1) * n_chunks
    toks = None
    for ex in load_dataset(hf["name"], split=hf.get("split", "train"), streaming=True):
        t = tokenizer.encode(ex.get(field, "") or "")
        if len(t) >= need:
            toks = t[:need]
            break
    if toks is None:
        return {}

    chunks = [
        torch.tensor(toks[i * (seq_len + 1):(i + 1) * (seq_len + 1)],
                     dtype=torch.long).unsqueeze(0).to(device)
        for i in range(n_chunks)
    ]
    was_training = model.training
    model.eval()

    def run_chunk(x, init_mem):
        """One chunk in segments (carrying within-chunk), returns (mean_ce, end_mem)."""
        xs = x.split(seg_len, dim=1) if (seg_len and 0 < seg_len < x.size(1)) else (x,)
        mem = init_mem
        ces = []
        for x_s in xs:
            y_s = x_s[:, 1:]
            x_in = x_s[:, :-1]
            if x_in.size(1) == 0:
                continue
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x_in, init_mem=mem, compute_logits=not fused_ce)
                _, logs = compute_loss(out, y_s, balance_w, ce_chunk)
            mem = out["mem_bank"].detach()
            ces.append(logs["ce"])
        return (sum(ces) / max(1, len(ces))), mem

    # Disentangle CONTENT from STRUCTURE. persist_gap (carried vs reset) conflates
    # two things: the written content AND the bank structure (slot count + slot
    # positional embeddings) — a carried bank has ~max_mem slots, a reset one
    # rebuilds from empty. A zero-content carried arm (writes zeroed, slots still
    # appended → identical slot count) isolates them:
    #   content_gap   = CE_zero  - CE_real   (pure content, slot count held equal)
    #   structure_gap = CE_reset - CE_zero   (slot-count / positional structure)
    #   persist_gap   = CE_reset - CE_real   = content_gap + structure_gap
    ts = getattr(model, "thought_stream", None)
    orig_new = ts._new_thought if ts is not None else None

    def carried_run(zero_content: bool):
        if ts is not None:
            ts._new_thought = (lambda H, b=None, p=None: torch.zeros_like(orig_new(H, b, p))) if zero_content else orig_new
        mem = None
        ces = []
        for c in chunks:
            ce_c, mem = run_chunk(c, mem)
            ces.append(ce_c)
        if ts is not None:
            ts._new_thought = orig_new
        return ces

    ce_real = carried_run(zero_content=False)
    ce_zero = carried_run(zero_content=True)
    ce_reset = [run_chunk(c, None)[0] for c in chunks]

    if was_training:
        model.train()

    def avg_tail(xs):                                     # chunks k>0 only
        return sum(xs[1:]) / max(1, len(xs[1:]))
    R, Z, S = avg_tail(ce_real), avg_tail(ce_zero), avg_tail(ce_reset)
    return {
        "persist_gap":   S - R,                          # content + structure (legacy headline)
        "content_gap":   Z - R,                          # PURE content benefit (the metric to trust)
        "structure_gap": S - Z,                          # slot-count / positional component
        "persist_chunks": float(len(ce_real)),
    }


@torch.no_grad()
def multiturn_probe(
    model: nn.Module,
    tokenizer,
    cfg_dict: dict,
    *,
    fused_ce: bool,
    ce_chunk: int,
    balance_w: float,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> dict:
    """Turn-cadenced memory probe on one freshly-streamed conversation (batch=1).

    Fetches its own conversation (independent of the training batch, which is now a
    padded turn-slot), segments it by assistant-turn starts and runs three arms over
    turns k>0, measuring mean CE:
      real   : bank carried across turns, real writes
      zero   : bank carried, writes zeroed (slots kept) -> isolates content
      ablate : no bank at all (init_mem=None each turn)
    Returns ablation_gap (ablate-real), content_gap (zero-real, the metric to trust)
    and the final bank's effective rank (redundancy of stored thoughts).
    """
    from datasets import load_dataset
    hf      = cfg_dict["data"]
    max_len = hf["seq_len"]
    mfield  = hf.get("messages_field", "messages")
    min_t   = int(hf.get("min_turns", 2))
    u_mark  = tokenizer.encode("\n<|user|>\n", add_special_tokens=False)
    a_mark  = tokenizer.encode("\n<|assistant|>\n", add_special_tokens=False)
    enc = None
    for ex in load_dataset(hf["name"], split=hf.get("split", "train_sft"), streaming=True):
        e = _encode_conversation(ex, tokenizer, u_mark, a_mark, mfield, max_len)
        if e is not None and len(e[1]) >= min_t:
            enc = e
            break
    if enc is None:
        return {}
    ids, turn_starts = enc
    x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).to(device)
    y = torch.tensor(ids[1:],  dtype=torch.long).unsqueeze(0).to(device)
    bounds = [int(b) for b in turn_starts if 0 < int(b) < x.size(1)]
    if not bounds:
        return {}
    xs = torch.tensor_split(x, bounds, dim=1)
    ys = torch.tensor_split(y, bounds, dim=1)
    was_training = model.training
    model.eval()

    ts = getattr(model, "thought_stream", None)
    orig_new = ts._new_thought if ts is not None else None

    def run(zero_content: bool, ablate: bool):
        if ts is not None:
            ts._new_thought = (
                (lambda H, b=None, p=None: torch.zeros_like(orig_new(H, b, p))) if zero_content else orig_new
            )
        mem = None
        ces = []
        for i, (x_s, y_s) in enumerate(zip(xs, ys)):
            if x_s.size(1) < 2:
                continue
            init = None if ablate else mem
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x_s, init_mem=init, compute_logits=not fused_ce)
                _, lg = compute_loss(out, y_s, balance_w, ce_chunk)
            if not ablate:
                mem = out["mem_bank"].detach()
            if i > 0:
                ces.append(lg["ce"])
        if ts is not None:
            ts._new_thought = orig_new
        return (sum(ces) / max(1, len(ces)) if ces else 0.0), mem

    ce_real, mem = run(zero_content=False, ablate=False)
    ce_zero, _   = run(zero_content=True,  ablate=False)
    ce_abl,  _   = run(zero_content=False, ablate=True)

    if was_training:
        model.train()
    return {
        "mem_ablation_gap": ce_abl - ce_real,            # whole-pathway benefit
        "content_gap":      ce_zero - ce_real,           # pure content (trust this)
        "mem_eff_rank":     _effective_rank(mem),        # redundancy of stored thoughts
        "mem_slots_final":  float(mem.size(1)) if mem is not None else 0.0,
        "mem_turns":        float(len(xs)),
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
    mem_persist     = bool(data_cfg.get("persist", False))      # carry bank across steps
    mem_multiturn   = data_cfg.get("task") == "multiturn"        # segment by turn, bank/conversation
    persist_chunks  = int(train_cfg.get("persist_probe_chunks", 6))
    log_every   = int(train_cfg.get("log_every", 50))
    save_every  = int(train_cfg.get("save_every", 1000))
    save_dir    = Path(train_cfg.get("save_dir", f"checkpoints/{run_name}"))
    balance_w   = float(model_cfg.balance_loss_weight)
    write_cost  = float(getattr(model_cfg, "mem_write_cost", 0.0))       # write-sparsity budget
    write_div   = float(getattr(model_cfg, "mem_write_diversity", 0.0))  # novelty-gated write

    model.train()
    step  = 0          # optimiser steps (one per grad_accum micro-batches)
    micro = 0          # micro-batches seen since the last optimiser step
    t0    = time.perf_counter()
    toks  = 0          # tokens accumulated across the current optimiser step
    pbar  = tqdm(total=total_steps, desc="Training")
    persist_mem: Optional[torch.Tensor] = None   # bank carried across steps
    window_loss = None                            # multi-turn TBPTT window accumulator

    for batch in dl:
        if mem_multiturn:
            # One TURN-SLOT of B lanes (padded). The bank persists across slots
            # within a conversation, resets per-lane at a conversation boundary, and
            # backward runs over a TBPTT window of mem_bptt_window slots.
            x, y, lmask, reset = batch
            x, y, lmask, reset = x.to(device), y.to(device), lmask.to(device), reset.to(device)
            toks += int(lmask.sum())

            init_mem = persist_mem
            if init_mem is not None and reset.any():
                init_mem = init_mem.masked_fill(reset.view(-1, 1, 1), 0.0)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x, init_mem=init_mem, compute_logits=not fused_ce, pad_mask=lmask)
                loss, logs = compute_loss(
                    out, y, balance_w, ce_chunk, write_cost, write_div, loss_mask=lmask
                )
            persist_mem = out["mem_bank"]                  # keep graph within the window
            seg_loss = scaler.scale(loss / grad_accum)
            window_loss = seg_loss if window_loss is None else window_loss + seg_loss
            micro += 1
            at_step = (micro % grad_accum == 0)
            if micro % mem_bptt_window == 0 or at_step:     # flush window (TBPTT truncation)
                window_loss.backward()
                window_loss   = None
                persist_mem   = persist_mem.detach()
            if not at_step:
                continue
        else:
            if mem_persist:
                x, y, reset = batch
                reset = reset.to(device)
            else:
                x, y = batch
                reset = None
            x, y = x.to(device), y.to(device)
            toks += x.numel()

            init_mem = persist_mem
            if mem_persist and init_mem is not None and reset is not None and reset.any():
                # zero the slots of lanes that just started a new file (fresh context)
                init_mem = init_mem.masked_fill(reset.view(-1, 1, 1), 0.0)

            logs, mem_out = forward_backward(
                model, x, y,
                fused_ce=fused_ce, ce_chunk=ce_chunk, balance_w=balance_w,
                scaler=scaler, seg_len=mem_seg_len, grad_accum=grad_accum,
                device=device, amp_dtype=amp_dtype, use_amp=use_amp,
                bptt_window=mem_bptt_window, init_mem=init_mem,
                write_cost=write_cost, write_diversity=write_div,
            )
            if mem_persist:
                persist_mem = mem_out      # detached; carried into next step
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

        # Multi-turn probe: turn-cadenced content_gap + effective rank
        if mem_probe_every and mem_multiturn and step % mem_probe_every == 0:
            mp = multiturn_probe(
                model, tokenizer, raw, fused_ce=fused_ce, ce_chunk=ce_chunk,
                balance_w=balance_w, device=device, amp_dtype=amp_dtype, use_amp=use_amp,
            )
            logs.update(mp)
            if mp:
                tqdm.write(
                    f"  [mt-probe] ablation_gap={mp['mem_ablation_gap']:+.4f}"
                    f"  content_gap={mp['content_gap']:+.4f}"
                    f"  eff_rank={mp['mem_eff_rank']:.2f}/{mp['mem_slots_final']:.0f}"
                    f"  turns={mp['mem_turns']:.0f}"
                )

        # Memory usefulness probe (ablation: CE with vs without the bank)
        if mem_probe_every and mem_seg_len and not mem_multiturn and step % mem_probe_every == 0:
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
            # Cross-sequence persistence: does carrying the bank across chunks of
            # the same file lower later-chunk CE? (the real use-case verdict)
            if mem_persist:
                pp = persistence_probe(
                    model, tokenizer, raw, n_chunks=persist_chunks, fused_ce=fused_ce,
                    ce_chunk=ce_chunk, balance_w=balance_w, seg_len=mem_seg_len,
                    device=device, amp_dtype=amp_dtype, use_amp=use_amp,
                )
                logs.update(pp)
                if pp:
                    tqdm.write(
                        f"  [persist-probe] persist_gap={pp['persist_gap']:+.4f}"
                        f"  = content_gap={pp['content_gap']:+.4f}"
                        f"  + structure_gap={pp['structure_gap']:+.4f}"
                        f"  ({pp['persist_chunks']:.0f} chunks)"
                    )

        if step % log_every == 0 or step == 1:
            tqdm.write(
                f"step={step:>6}  ce={logs['ce']:.4f}  ppl={logs['ppl']:.2f}"
                f"  balance={logs['balance']:.4f}"
                + (f"  r_hat={logs.get('r_hat', 0):.3f}" if "r_hat" in logs else "")
                + (f"  mem={logs['mem_slots']:.0f}" if "mem_slots" in logs else "")
                + (f"  α={logs['write_alpha']:.3f}" if "write_alpha" in logs else "")
                + (f"  wpen={logs['write_pen']:.3f}" if "write_pen" in logs else "")
                + (f"  redund={logs['write_redund']:.3f}" if "write_redund" in logs else "")
                + (f"  erank={logs['mem_eff_rank']:.2f}" if "mem_eff_rank" in logs else "")
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
