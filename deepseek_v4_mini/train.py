"""
Training script for TrunkLM.

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

from .config import ThoughtBankConfig
from .model import TrunkLM, ThoughtBankLM


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
        rms_match: bool = False,
        adam_params=None,
        adam_lr: float = 3e-4,
        adam_betas: tuple = (0.9, 0.95),
        adam_wd: float = 0.1,
        adam_eps: float = 1e-8,
        adam_fused: bool = False,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, wd=wd, rms_match=rms_match)
        super().__init__(params, defaults)
        # Internal AdamW for non-matrix params. fused=True (opt-in) fuses the
        # AdamW element-wise update into one CUDA kernel — free on GPU, numerically
        # ~identical (NOT bit-identical), so gated off by default for repro.
        if adam_params is not None:
            self._adam = optim.AdamW(
                adam_params, lr=adam_lr, betas=adam_betas, weight_decay=adam_wd,
                eps=adam_eps, fused=adam_fused,
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
                    if group["rms_match"]:
                        # DeepSeek-V4/Kimi convention: rescale so update RMS = 0.2
                        # for EVERY shape — one lr serves square backbone blocks and
                        # tall hypernet maps alike. The legacy sqrt(cols) scaling
                        # gives RMS 1.0 on squares but ~0.1 on the fw_A/fw_B
                        # hypernets: the read trains ~8x slower than the backbone.
                        update = update * (0.2 * max(update.shape) ** 0.5)
                    else:
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


_GIST_DISTS_CACHE: dict = {}


def _gist_dists(C: int, S: int, p_pref: float) -> torch.Tensor:
    """Fixed context→symbol distribution map shared by the synthetic gist loader
    and its probe (so both agree on P_c). Each context puts p_pref on a disjoint
    block of k = S//C symbols and spreads the rest uniformly."""
    key = (C, S, round(p_pref, 6))
    if key not in _GIST_DISTS_CACHE:
        k = max(1, S // C)
        d = torch.full((C, S), (1.0 - p_pref) / max(1, S - k))
        for c in range(C):
            d[c, c * k: c * k + k] = p_pref / k
        _GIST_DISTS_CACHE[key] = d / d.sum(dim=1, keepdim=True)
    return _GIST_DISTS_CACHE[key]


def _build_synthetic_multiturn(cfg_dict: dict):
    """Synthetic multi-turn GIST task: a latent context set in turn 0 that later,
    NON-overlapping short turns can only get from memory.

    Each conversation (FIXED turn count, so the B lanes stay turn-aligned and reset
    together):
        turn 0 : [BOS, CTX_c]              -> establishes context c (loss-masked)
        turn t : [Q] -> predict s_t ~ P_c  -> answer depends ONLY on c
    Turn t's attention window is a single [Q] segment: no symbol and no c are in
    view, so the ONLY path to c is the thought bank written at turn 0. Predicting
    s_t from [Q] alone costs ~ln(S); with the gist carried it costs ~H(P_c). So
    content_gap ≈ ln(S) - H(P_c) is LARGE and SUSTAINED iff the bank carries the
    gist across turns — the clean pass/fail UltraChat (self-contained turns) never
    produced.

    Yields the SAME contract as the real multiturn loader, one turn-slot at a time:
        (x [B,L], y [B,L], loss_mask [B,L] bool, reset [B] bool)

    Vocab: 0=PAD 1=BOS 2=Q, contexts=[3, 3+C), symbols=[3+C, 3+C+S).
    Set vocab_size >= 3 + n_contexts + n_symbols.
    """
    d      = cfg_dict["data"]
    bs     = int(d["batch_size"])
    C      = int(d.get("n_contexts", 8))
    S      = int(d.get("n_symbols", 64))
    p_pref = float(d.get("pref_mass", 0.9))
    turns  = int(d.get("turns_per_conv", 6))     # 1 context turn + (turns-1) answers
    BOS, Q, CTX_OFF = 1, 2, 3
    SYM_OFF = CTX_OFF + C
    dists   = _gist_dists(C, S, p_pref)

    class GistDS:
        def __iter__(self):
            while True:
                cs = torch.randint(0, C, (bs,))                  # fresh context per lane
                # turn 0: [BOS, CTX_c] — loss-masked, only writes c into the bank
                x = torch.stack([torch.tensor([BOS, CTX_OFF + int(c)]) for c in cs])
                yield (x, x.clone(),
                       torch.zeros((bs, 2), dtype=torch.bool),
                       torch.ones(bs, dtype=torch.bool))
                # answer turns: predict one symbol ~ P_c from [Q] (needs the bank)
                for _ in range(turns - 1):
                    s = torch.stack([torch.multinomial(dists[int(c)], 1) for c in cs]).view(bs)
                    yield (torch.full((bs, 1), Q, dtype=torch.long),
                           (SYM_OFF + s).view(bs, 1),
                           torch.ones((bs, 1), dtype=torch.bool),
                           torch.zeros(bs, dtype=torch.bool))

    return GistDS()


def _build_synthetic_multiturn_kv(cfg_dict: dict):
    """Multi-context KEYED gist task: capacity + read stress for the thought bank.

    Generalises multiturn_gist from one context to K keyed contexts. Each context
    is written on its own establish turn (one write -> one slot), then queried by
    its key across the conversation:
        establish i (0..K-1) : [Q_i, CTX_{c_i}]   writes (key Q_i, gist c_i) (masked)
        answer turn          : [Q_i] -> s ~ P_{c_i}   read the slot keyed Q_i

    The model must (1) hold K contexts across turns (capacity: eff_rank should rise
    toward K, not stay ~1) and (2) address the right one from the query key (read).
    Values are gist distributions, not exact tokens, and K is small — so this maps
    where keyed retrieval degrades as K grows, without collapsing into the many-key
    associative-recall regime the bank already fails. K=1 ≈ multiturn_gist.

    Same yield contract (x, y, loss_mask, reset), one turn-slot at a time; fixed
    K + turns_per_conv keeps the B lanes turn-aligned and resetting together.

    Vocab: 0=PAD, keys/queries=[2, 2+K), contexts=[2+K, 2+K+C), symbols=[2+K+C, ...+S).
    Set vocab_size >= 2 + n_query_slots + n_contexts + n_symbols.
    """
    d      = cfg_dict["data"]
    bs     = int(d["batch_size"])
    K      = int(d.get("n_query_slots", 2))
    C      = int(d.get("n_contexts", 8))
    S      = int(d.get("n_symbols", 64))
    p_pref = float(d.get("pref_mass", 0.9))
    turns  = int(d.get("turns_per_conv", 8))     # answer turns after K establish turns
    Q_OFF   = 2
    CTX_OFF = Q_OFF + K
    SYM_OFF = CTX_OFF + C
    dists   = _gist_dists(C, S, p_pref)

    class KVDS:
        def __iter__(self):
            while True:
                cs = torch.randint(0, C, (bs, K))                 # per-lane context per slot
                for i in range(K):                                # establish turn per slot
                    x = torch.stack([torch.tensor([Q_OFF + i, CTX_OFF + int(cs[b, i])])
                                     for b in range(bs)])
                    yield (x, x.clone(),
                           torch.zeros((bs, 2), dtype=torch.bool),
                           torch.full((bs,), i == 0, dtype=torch.bool))   # reset on 1st establish
                for _ in range(turns):                            # answer turns
                    slot = torch.randint(0, K, (bs,))
                    s = torch.stack([torch.multinomial(dists[int(cs[b, int(slot[b])])], 1)
                                     for b in range(bs)]).view(bs)
                    yield ((Q_OFF + slot).view(bs, 1),
                           (SYM_OFF + s).view(bs, 1),
                           torch.ones((bs, 1), dtype=torch.bool),
                           torch.zeros(bs, dtype=torch.bool))

    return KVDS()


def _rule_space(d: dict):
    """Rule space for multiturn_rule: family, pools and evaluator.

    Families (data.rule_family):
      shift  (default) : y = (x + s) mod S — rules are the S-1 non-trivial shifts;
                         held-out via heldout_shifts / train_shift_max (1D circle).
      affine           : y = (a·x + s) mod S, a coprime to S — φ(S)·S rules on a
                         2D torus (16·32 = 512 for S=32). This is the task-DIVERSITY
                         squeeze: with ~450 trained rules, memorizing one read cell
                         per rule costs more hypernet capacity than learning the
                         (a, s) law (Raventós et al. 2023: past a diversity
                         threshold, ICL flips from repertoire lookup to
                         generalization). Held-out via heldout_rule_mod m: pairs
                         with (a_idx + s) % m == 0 — grid-interleaved on the torus,
                         so every held rule has trained neighbours at distance 1 in
                         BOTH directions (a and s). The identity rule (a=1, s=0) is
                         excluded from both pools.

    Rules travel as flat ids rid = a_idx * S + s (shift family: rid = s, a_idx = 0
    with units = [1] — legacy ids unchanged). Returns
    (units, n_rules, train_pool, held_pool, apply) with apply(rid, x) -> y.
    """
    S   = int(d.get("n_symbols", 32))
    fam = str(d.get("rule_family", "shift"))
    if fam == "affine":
        _cu = d.get("affine_units")          # optional unit subset (family-transfer
        if _cu:                              # cells: keep the rule count tractable
            units = [int(u) for u in _cu]
            assert all(math.gcd(u, S) == 1 for u in units), "affine_units must be coprime to S"
        else:
            units = [u for u in range(1, S) if math.gcd(u, S) == 1]
        n_rules = len(units) * S
        mod     = int(d.get("heldout_rule_mod", 8))
        # Per-unit s subsampling (capacity-matched multi-family cells): keep
        # s ≡ -a_i (mod stride) so the mod-held pattern (a_i+s)%mod==0 stays
        # inside the kept set (mod must be a multiple of stride).
        stride = int(d.get("affine_s_stride", 1))
        s_off  = int(d.get("affine_s_stride_offset", 0))   # probe-side: match the
        if stride > 1:                                     # parity a unit had in a
            assert mod % stride == 0, "heldout_rule_mod must be a multiple of affine_s_stride"  # larger unit list
        train_pool, held_pool = [], []
        for rid in range(n_rules):
            a_i, s = divmod(rid, S)
            if units[a_i] == 1 and s == 0:
                continue                              # identity: trivially solvable
            if stride > 1 and s % stride != (s_off - a_i) % stride:
                continue
            if mod and (int(d.get("heldout_rule_off", 0)) + a_i + s) % mod == 0:
                held_pool.append(rid)
            else:
                train_pool.append(rid)
    else:
        units   = [1]
        n_rules = S
        held    = sorted(int(v) for v in d.get("heldout_shifts", []))
        s_max   = int(d.get("train_shift_max", S - 1))
        train_pool = [v for v in range(1, s_max + 1) if v not in held]
        held_pool  = held if held else list(range(s_max + 1, S))

    def apply(rid: int, x: int) -> int:
        a_i, s = divmod(int(rid), S)
        return (units[a_i] * x + s) % S

    return units, n_rules, train_pool, held_pool, apply


def _fourier_codes(d: dict, dim: int) -> torch.Tensor:
    """Fixed teacher codes: Fourier features on the rule manifold (unit RMS).

    shift  : rid = s on a circle of size S → pairs cos/sin(2πks/S), k = 1..dim/2.
    affine : rid = a_i·S + s on a torus U×S → half the dims carry s-frequencies,
             half carry a-frequencies.
    """
    S   = int(d.get("n_symbols", 32))
    fam = str(d.get("rule_family", "shift"))
    if fam == "affine":
        _cu = d.get("affine_units")          # honor the same unit subset as _rule_space
        units = ([int(u) for u in _cu] if _cu
                 else [u for u in range(1, S) if math.gcd(u, S) == 1])
    else:
        units = [1]
    U, n_rules = len(units), len(units) * S if fam == "affine" else S

    kmax = int(d.get("_fourier_kmax", 0))     # injected by the caller (model cfg)

    def _feats(vals: torch.Tensor, period: int, n_pairs: int) -> torch.Tensor:
        k  = torch.arange(1, n_pairs + 1, dtype=torch.float32)          # [F]
        if kmax > 0:
            k = (k - 1) % kmax + 1            # cycle k = 1..kmax to fill the dims
        th = 2 * math.pi * vals.float().unsqueeze(1) * k / period        # [N, F]
        return torch.stack([th.cos(), th.sin()], dim=-1).flatten(1)      # [N, 2F]

    rid = torch.arange(n_rules)
    if U == 1:
        codes = _feats(rid, S, dim // 2)
    else:
        a_i, s = rid // S, rid % S
        half = (dim // 2) // 2 * 2                                       # even split
        codes = torch.cat([_feats(s, S, half // 2),
                           _feats(a_i, U, (dim - half) // 2)], dim=1)
    if codes.size(1) < dim:                                              # odd-dim padding
        codes = torch.cat([codes, torch.zeros(n_rules, dim - codes.size(1))], dim=1)
    return codes * (2.0 ** 0.5)                                          # RMS 1 per code


def _build_synthetic_rule(cfg_dict: dict):
    """Continual-learning task: a NOVEL rule presented at turn 0, APPLIED later.

    Each conversation draws a fresh modular-shift rule y = (x + s) mod S, s in
    1..S-1 — random per conversation, so it is NOT in the frozen weights. Turn 0
    shows m example pairs [x_i, (x_i+s)%S]; later turns query symbols that were NOT
    shown and must be answered by APPLYING the rule.

        turn 0 : [x_0, y_0, x_1, y_1, ...]  in-window ICL (loss on applied x-pos j>=1)
        turn t : [x_q] -> (x_q + s) mod S    x_q UNSEEN; window has only x_q

    The unseen queries are the discriminator: a lookup memory (storing the shown
    pairs) can't answer them; only a forward pass that INFERRED the rule and APPLIES
    it can. And the rule must cross the turn boundary through the bank (the answer
    window holds no examples). So this separates "the forward has learned" from "has
    a note in context" — the fast-weights hypothesis. Deterministic: CE floor 0 with
    the rule carried, ln(S) without.

    Vocab: 0=PAD, symbols=[SYM_OFF, SYM_OFF+S).  Set vocab_size >= SYM_OFF + n_symbols.
    """
    d      = cfg_dict["data"]
    bs     = int(d["batch_size"])
    S      = int(d.get("n_symbols", 32))
    m      = int(d.get("n_examples", 6))         # example pairs shown per presentation turn
    turns  = int(d.get("turns_per_conv", 8))     # answer turns (unseen queries)
    K      = int(d.get("n_contexts", 1))         # rules per conversation, each behind a key
    # Rule pools + evaluator come from the rule space (shift or affine family);
    # rules travel as flat ids, _apply(rid, x) evaluates them.
    _units, _n_rules, train_pool, _held_pool, _apply = _rule_space(d)
    # Rule switch: after `switch_at` query turns, RE-present a fresh rule s2 != s1
    # mid-conversation (bank carried, no reset) and query s2 for the remaining
    # turns. Tests retention POLICY: the model must drop s1, not just retain.
    sw     = int(d.get("switch_at", 0))
    # Structure randomization (free structural augmentation): per-conversation
    # turn count drawn from turns_range, plus PERIODIC mid-conversation
    # switches — a random key is RE-presented with a fresh rule every
    # switch_phase_min..max query turns (bank carried, no reset). The fixed
    # format (8 turns, no switch) lets a FIFO cliff install as a BEHAVIOUR
    # (horizon probe: exact cliff at turn 16 in both gate arms); randomizing
    # horizon and switch position makes rehearsal/forgetting the only viable
    # policy (vision random-crop twin). All B lanes share one structure per
    # conversation (turn alignment); rules/examples stay per-lane. Segment
    # count varies per conversation: it is published in
    # cfg_dict["_conv"]["n_seg"] and the training loop steps the optimiser at
    # conversation end instead of a fixed grad_accum.
    t_rng  = d.get("turns_range")
    t_lo, t_hi = (int(t_rng[0]), int(t_rng[1])) if t_rng else (turns, turns)
    ph_lo  = int(d.get("switch_phase_min", 0))   # 0 = no mid-conversation switches
    ph_hi  = int(d.get("switch_phase_max", ph_lo))
    sw_max = int(d.get("switch_max", 0))         # 0 = unlimited switches per conv
    # n_contexts_range: per-conversation KEY-COUNT draw K_c in [lo, hi] — the
    # concurrent-rule axis of the augmentation (routing K2..Kn, capacity,
    # retention through eviction, all in one distribution). Ceiling heuristic
    # (user): K_hi = mem_dim / x with x unknown; margin x=6 -> 32/6 = 5. The
    # K_c key tokens are a random SUBSET of the K_hi available ones, so every
    # key token sees every role (key-identity twin of the position invariance).
    k_rng  = d.get("n_contexts_range")
    k_lo, k_hi = (int(k_rng[0]), int(k_rng[1])) if k_rng else (K, K)
    if t_rng or ph_lo or k_rng:
        assert not sw, "structure knobs replace the legacy switch_at"
        assert t_lo >= 2 and t_hi >= t_lo and (not ph_lo or ph_hi >= ph_lo >= 1)
        assert k_hi >= k_lo >= 1
    # Hop (reflection cell): with prob hop_p a query turn becomes a 2-hop task
    # [key, HOP, key, x] -> f(f(x)) (hop_keys=2: [k_a, HOP, k_b, x] ->
    # f_b(f_a(x))). HOP (token 2, free) is a local composition operator (user
    # design 2026-07-07): `k HOP` = "apply k then pass the result on"; the
    # final key stays bare before x. Depth = the HOP count (n-hop reads
    # [k,HOP,k,HOP,...,k,x]), and the plain query surface [k,x] is never
    # shadowed — no collision with the pre-trained last-key parse.
    # hop_grant=true trains the THINK protocol BY LABELS, not by input format
    # (user decision 2026-07-07): the hop query's answer label is THINK; the
    # harness then grants one scratch segment whose sole input is THINK and
    # whose label is the final answer. The bank is the only bridge — the
    # scratch segment carries nothing about x, so the intermediate must have
    # been WRITTEN. hop_grant=false is the no-think control arm: same hop
    # query, direct f(f(x)) label, no granted segment (must stay at chance,
    # else the one-shot shortcut n·s invalidates the cell).
    THINK  = 1                                   # vocab token 1 is free here
    HOPTOK = 2                                   # composition operator `k HOP`
    hop_p     = float(d.get("hop_p", 0.0))
    hop_grant = bool(d.get("hop_grant", True))
    hop_keys  = int(d.get("hop_keys", 1))
    # hop_teacher: teacher targets move to the HOP segments ONLY — the hop
    # query's write is blended/distilled toward the Fourier code of the
    # INTERMEDIATE SYMBOL mid (rules and symbols live on the same mod-S
    # circle, one table serves both) and presentations yield -1 (no teacher:
    # post-anneal rule codes have drifted off the Fourier circle, re-forcing
    # them would tear the organized circuit — dsv5d collapse, 2026-07-07).
    hop_tf    = bool(d.get("hop_teacher", False))
    if hop_p:
        assert max(K, k_hi) > 1, "hop needs key tokens (n_contexts >= 2)"
        assert hop_keys in (1, 2)
    SYM_OFF = 3
    KEY_OFF = SYM_OFF + S                        # key tokens live above the symbols
    # each phase draws its own perm, so the unseen-query budget is per phase
    max_phase = max(sw, turns - sw) if sw else t_hi
    assert m + max_phase <= S, "need m examples + distinct unseen queries within S symbols"
    assert train_pool, "empty training shift pool"
    if max(K, k_hi) > 1:
        assert int(cfg_dict.get("vocab_size", 0)) >= KEY_OFF + max(K, k_hi), \
            "vocab_size must cover SYM_OFF + n_symbols + n_contexts key tokens"
    if sw:
        assert K == 1, "switch_at requires n_contexts=1 (keyless conversations)"
        assert 0 < sw < turns, "switch_at must fall inside the conversation"
        assert len(train_pool) >= 2, "switch needs two distinct shifts"

    # Turn-slot layout (K=1 keeps the exact legacy format, no key token):
    #   presentation turn k : [key_k?, x_0, y_0, ..., x_{m-1}, y_{m-1}]   rule s_k
    #   answer turn t       : [key_k?, x_q] with k = t % K               apply s_k
    # The 5th yielded element is the rule id for teacher-forcing: s_k on
    # presentation turns, -1 on answer turns (the blend must not fire there).
    # Diversity curriculum: draw rules from a growing prefix of a (seeded)
    # shuffle of the train pool. Bootstraps the circuit on few rules (fast crack,
    # validated recipe) then dilates toward the full pool — separates "build the
    # circuit" from "generalize it to N rules". The training loop drives
    # cfg_dict["_curriculum"]["frac"] from step/curriculum_full_step.
    cur_n0 = int(d.get("curriculum_rules_start", 0))       # 0 = curriculum off
    if cur_n0:
        cfg_dict.setdefault("_curriculum", {"frac": 0.0})
        assert not sw, "curriculum not wired for switch conversations"

    class RuleDS:
        def __iter__(self):
            off = 1 if max(K, k_hi) > 1 else 0   # key token prefix width
            pool_t = torch.tensor(train_pool, dtype=torch.long)
            g = torch.Generator().manual_seed(1234)
            pool_shuf = pool_t[torch.randperm(len(pool_t), generator=g)]

            def _cur_pool():
                if not cur_n0:
                    return pool_t
                cur = cfg_dict["_curriculum"]
                if "n" in cur:                       # mastery-gated: absolute pool size
                    n = int(cur["n"])
                else:                                # clock ramp: frac of the full pool
                    n = max(cur_n0, int(round(float(cur["frac"]) * len(pool_t))))
                return pool_shuf[:min(n, len(pool_t))]

            def _present(s_k, ex_k, reset: bool):
                # presentation segment: [x_0, y_0, ..., x_{m-1}, y_{m-1}] (K=1)
                L = 2 * m
                X = torch.zeros((bs, L), dtype=torch.long)
                Y = torch.zeros((bs, L), dtype=torch.long)
                Msk = torch.zeros((bs, L), dtype=torch.bool)
                for b in range(bs):
                    for j, xi in enumerate(ex_k[b]):
                        yi = _apply(int(s_k[b]), xi)
                        X[b, 2 * j] = SYM_OFF + xi
                        X[b, 2 * j + 1] = SYM_OFF + yi
                        Y[b, 2 * j] = SYM_OFF + yi
                        Msk[b, 2 * j] = (j >= 1)
                return X, Y, Msk, torch.full((bs,), reset, dtype=torch.bool), s_k.clone()

            def _query(s_k, unseen_k, t_ph):
                xq = torch.zeros((bs, 1), dtype=torch.long)
                yq = torch.zeros((bs, 1), dtype=torch.long)
                msk = torch.ones((bs, 1), dtype=torch.bool)
                for b in range(bs):
                    pool = unseen_k[b]
                    q = pool[t_ph % len(pool)]
                    xq[b, 0] = SYM_OFF + q
                    yq[b, 0] = SYM_OFF + _apply(int(s_k[b]), q)
                return xq, yq, msk, torch.zeros(bs, dtype=torch.bool), torch.full((bs,), -1, dtype=torch.long)

            if sw:
                while True:
                    # two distinct rules per lane; s2 re-presented mid-conversation
                    s1 = pool_t[torch.randint(0, len(pool_t), (bs,))]
                    s2 = pool_t[torch.randint(0, len(pool_t), (bs,))]
                    while bool((s2 == s1).any()):
                        clash = s2 == s1
                        s2[clash] = pool_t[torch.randint(0, len(pool_t), (int(clash.sum()),))]
                    exs, uns = [], []
                    for s_k in (s1, s2):
                        ex_k, un_k = [], []
                        for b in range(bs):
                            perm = torch.randperm(S).tolist()
                            ex_k.append(perm[:m]); un_k.append(perm[m:])
                        exs.append(ex_k); uns.append(un_k)
                    yield _present(s1, exs[0], reset=True)
                    for t in range(sw):
                        yield _query(s1, uns[0], t)
                    yield _present(s2, exs[1], reset=False)   # bank carried: s1 must be DROPPED
                    for t in range(turns - sw):
                        yield _query(s2, uns[1], t)
            while True:
                cp      = _cur_pool()
                turns_c = (int(torch.randint(t_lo, t_hi + 1, (1,))) if t_hi > t_lo
                           else t_lo)
                K_c     = (int(torch.randint(k_lo, k_hi + 1, (1,))) if k_hi > k_lo
                           else k_lo)
                # key tokens: random subset of the k_hi available ones (see knob doc)
                key_tok = (torch.randperm(k_hi)[:K_c].tolist() if k_rng
                           else list(range(K_c)))
                sw_at = []          # query-turn indices with a re-presentation before them
                if ph_lo:
                    nxt = int(torch.randint(ph_lo, ph_hi + 1, (1,)))
                    while nxt <= turns_c - 1 and (not sw_max or len(sw_at) < sw_max):
                        sw_at.append(nxt)            # ≥1 query after the last switch
                        nxt += int(torch.randint(ph_lo, ph_hi + 1, (1,)))
                # hop draws are fixed up front: each granted hop adds one segment
                hop_at = ([bool(torch.rand(1) < hop_p) for _ in range(turns_c)]
                          if hop_p else [False] * turns_c)
                n_hop_seg = sum(hop_at) if hop_grant else 0
                cfg_dict["_conv"] = {"n_seg": K_c + turns_c + len(sw_at) + n_hop_seg,
                                     "k": K_c}
                s      = cp[torch.randint(0, len(cp), (bs, K_c))]
                ex     = [[] for _ in range(bs)]  # per lane, per context: shown inputs
                unseen = [[] for _ in range(bs)]  # per lane, per context: query pool
                for b in range(bs):
                    for k in range(K_c):
                        perm = torch.randperm(S).tolist()
                        ex[b].append(perm[:m]); unseen[b].append(perm[m:])

                def _present_key(k: int, reset: bool):
                    L = off + 2 * m
                    X = torch.zeros((bs, L), dtype=torch.long)
                    Y = torch.zeros((bs, L), dtype=torch.long)
                    Msk = torch.zeros((bs, L), dtype=torch.bool)
                    if off:
                        X[:, 0] = KEY_OFF + key_tok[k]
                    for b in range(bs):
                        for j, xi in enumerate(ex[b][k]):
                            yi = _apply(int(s[b, k]), xi)
                            X[b, off + 2 * j] = SYM_OFF + xi
                            X[b, off + 2 * j + 1] = SYM_OFF + yi
                            Y[b, off + 2 * j] = SYM_OFF + yi
                            Msk[b, off + 2 * j] = (j >= 1)   # j=0 unlearnable (shift unknown yet)
                    return (X, Y, Msk, torch.full((bs,), reset, dtype=torch.bool),
                            (torch.full((bs,), -1, dtype=torch.long) if hop_tf
                             else s[:, k].clone()))

                for k in range(K_c):
                    yield _present_key(k, reset=(k == 0))
                q_cnt = [0] * K_c   # per-key queries since its last presentation
                sw_i  = 0
                for t in range(turns_c):
                    while sw_i < len(sw_at) and sw_at[sw_i] == t:
                        # switch: re-present ONE key with a fresh rule (bank
                        # carried, no reset — the old rule must be DROPPED)
                        k_sw  = int(torch.randint(0, K_c, (1,)))
                        new_s = cp[torch.randint(0, len(cp), (bs,))]
                        while bool((new_s == s[:, k_sw]).any()):
                            clash = new_s == s[:, k_sw]
                            new_s[clash] = cp[torch.randint(0, len(cp), (int(clash.sum()),))]
                        s[:, k_sw] = new_s
                        for b in range(bs):
                            perm = torch.randperm(S).tolist()
                            ex[b][k_sw] = perm[:m]; unseen[b][k_sw] = perm[m:]
                        q_cnt[k_sw] = 0
                        yield _present_key(k_sw, reset=False)
                        sw_i += 1
                    k = t % K_c
                    no_tf = torch.full((bs,), -1, dtype=torch.long)
                    if hop_at[t]:
                        # hop query: [key_a, HOP, key_b, x]; k_a applied first,
                        # its result passed on to k_b (bare key = final apply).
                        # hop_keys=1 doubles the same key (f∘f); hop_keys=2
                        # chains two different keys (f_b∘f_a).
                        ka = k
                        kb = k if hop_keys == 1 else int((k + 1 + int(torch.randint(0, K_c - 1, (1,)))) % K_c)
                        xq = torch.zeros((bs, 4), dtype=torch.long)
                        yq = torch.zeros((bs, 4), dtype=torch.long)
                        msk = torch.zeros((bs, 4), dtype=torch.bool)
                        xq[:, 0] = KEY_OFF + key_tok[ka]
                        xq[:, 1] = HOPTOK
                        xq[:, 2] = KEY_OFF + key_tok[kb]
                        msk[:, 3] = True
                        yfin = torch.zeros(bs, dtype=torch.long)
                        ymid = torch.zeros(bs, dtype=torch.long)
                        for b in range(bs):
                            pool = unseen[b][ka]
                            q = pool[q_cnt[ka] % len(pool)]
                            mid = _apply(int(s[b, ka]), q)
                            xq[b, 3] = SYM_OFF + q
                            ymid[b] = SYM_OFF + mid
                            yfin[b] = SYM_OFF + _apply(int(s[b, kb]), mid)
                        q_cnt[ka] += 1
                        if hop_grant:
                            # answer label = THINK -> the model learns to ASK
                            # for the scratch segment. The granted segment is a
                            # SUPERVISED CHAIN (user design, multi-output think):
                            # X = [THINK, mid] (teacher-forced), Y = [mid, final].
                            # Position 0 must DECODE the intermediate from the
                            # bank (nothing about x in-window = the causal claim);
                            # position 1 applies the rule to the in-window mid
                            # (already-trained skill). Factorizes the unsupervised
                            # bank roundtrip into two supervised skills.
                            yq[:, 3] = THINK
                            # hop_teacher: the hop forward's write must carry the
                            # intermediate — teacher-force it toward Fourier[mid]
                            yield (xq, yq, msk, torch.zeros(bs, dtype=torch.bool),
                                   (ymid - SYM_OFF).clone() if hop_tf else no_tf.clone())
                            xt = torch.full((bs, 2), THINK, dtype=torch.long)
                            xt[:, 1] = ymid
                            yt = torch.stack([ymid, yfin], dim=1)
                            mt = torch.ones((bs, 2), dtype=torch.bool)
                            yield xt, yt, mt, torch.zeros(bs, dtype=torch.bool), no_tf.clone()
                        else:
                            yq[:, 3] = yfin   # no-think control: direct 2-hop label
                            yield xq, yq, msk, torch.zeros(bs, dtype=torch.bool), no_tf.clone()
                        continue
                    xq = torch.zeros((bs, off + 1), dtype=torch.long)
                    yq = torch.zeros((bs, off + 1), dtype=torch.long)
                    msk = torch.zeros((bs, off + 1), dtype=torch.bool)
                    if off:
                        xq[:, 0] = KEY_OFF + key_tok[k]
                    msk[:, off] = True
                    for b in range(bs):
                        pool = unseen[b][k]
                        q = pool[q_cnt[k] % len(pool)]
                        xq[b, off] = SYM_OFF + q
                        yq[b, off] = SYM_OFF + _apply(int(s[b, k]), q)
                    q_cnt[k] += 1
                    yield xq, yq, msk, torch.zeros(bs, dtype=torch.bool), no_tf

    return RuleDS()


def _build_dataloader(cfg_dict: dict, tokenizer, split: str = "train"):
    """Thin wrapper around HF streaming dataset → fixed-length batches."""
    if cfg_dict["data"].get("task") == "associative_recall":
        return _build_synthetic_recall(cfg_dict)
    if cfg_dict["data"].get("task") == "latent_context":
        return _build_latent_context(cfg_dict)
    if cfg_dict["data"].get("task") == "multiturn_gist":
        return _build_synthetic_multiturn(cfg_dict)
    if cfg_dict["data"].get("task") == "multiturn_gist_kv":
        return _build_synthetic_multiturn_kv(cfg_dict)
    if cfg_dict["data"].get("task") == "multiturn_rule":
        return _build_synthetic_rule(cfg_dict)
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
    write_target: float = 0.0,
    write_target_weight: float = 0.0,
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

    # Target-rate objective: pull E[α] toward a target with a quadratic well. Curbs
    # both α→1 and α→0 (a monotone budget can only do the former, and overshoots to
    # the latter). Probes pass weight=0.0 so probe CE stays uncontaminated.
    write_alpha_mean = out.get("write_alpha_mean")
    if write_alpha_mean is not None and write_target_weight > 0.0:
        tgt  = write_target_weight * (write_alpha_mean - write_target) ** 2
        loss = loss + tgt
        logs["write_tgt"] = float(tgt.detach())

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
    write_target: float = 0.0,
    write_target_weight: float = 0.0,
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
            loss, logs = compute_loss(out, y_s, balance_w, ce_chunk, write_cost, write_diversity,
                                      write_target, write_target_weight)
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


def synthetic_multiturn_probe(
    model: nn.Module,
    cfg_dict: dict,
    *,
    fused_ce: bool,
    ce_chunk: int,
    balance_w: float,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    n_conv: int = 16,
) -> dict:
    """Gist probe for the synthetic multi-turn task: averages CE over the answer
    turns (t>0) of n_conv fresh conversations under three arms.
      real   : bank carried from turn 0, real writes
      zero   : bank carried, writes zeroed (slots kept) -> isolates content
      ablate : no bank at all (init_mem=None each turn)
    content_gap = ce_zero - ce_real is the verdict: ≈ ln(S) - H(P_c) if the gist is
    carried, ~0 if the bank can't hold one context across turns.
    """
    d      = cfg_dict["data"]
    C      = int(d.get("n_contexts", 8))
    S      = int(d.get("n_symbols", 64))
    p_pref = float(d.get("pref_mass", 0.9))
    turns  = int(d.get("turns_per_conv", 6))
    BOS, Q, CTX_OFF = 1, 2, 3
    SYM_OFF = CTX_OFF + C
    dists   = _gist_dists(C, S, p_pref)

    was_training = model.training
    model.eval()
    ts = getattr(model, "thought_stream", None)
    orig_new = ts._new_thought if ts is not None else None

    # Same conversations across the three arms for a low-variance gap.
    convs = []
    for _ in range(n_conv):
        c = int(torch.randint(0, C, (1,)))
        syms = [int(torch.multinomial(dists[c], 1)) for _ in range(turns - 1)]
        convs.append((c, syms))

    def run(zero_content: bool, ablate: bool):
        if ts is not None:
            ts._new_thought = (
                (lambda H, b=None, p=None: torch.zeros_like(orig_new(H, b, p))) if zero_content else orig_new
            )
        ces, last_mem = [], None
        for c, syms in convs:
            mem = None
            x0 = torch.tensor([[BOS, CTX_OFF + c]], device=device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x0, init_mem=None, compute_logits=not fused_ce)
            if not ablate:
                mem = out["mem_bank"].detach()
            for s in syms:
                xq = torch.tensor([[Q]], device=device)
                yq = torch.tensor([[SYM_OFF + s]], device=device)
                init = None if ablate else mem
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(xq, init_mem=init, compute_logits=not fused_ce)
                    _, lg = compute_loss(out, yq, balance_w, ce_chunk)
                if not ablate:
                    mem = out["mem_bank"].detach()
                ces.append(lg["ce"])
            last_mem = mem
        if ts is not None:
            ts._new_thought = orig_new
        return (sum(ces) / max(1, len(ces)) if ces else 0.0), last_mem

    ce_real, mem = run(zero_content=False, ablate=False)
    ce_zero, _   = run(zero_content=True,  ablate=False)
    ce_abl,  _   = run(zero_content=False, ablate=True)

    if was_training:
        model.train()
    return {
        "mem_ablation_gap": ce_abl - ce_real,
        "content_gap":      ce_zero - ce_real,
        "mem_eff_rank":     _effective_rank(mem) if mem is not None else 0.0,
        "mem_slots_final":  float(mem.size(1)) if mem is not None else 0.0,
        "mem_turns":        float(turns),
    }


def synthetic_multiturn_kv_probe(
    model: nn.Module,
    cfg_dict: dict,
    *,
    fused_ce: bool,
    ce_chunk: int,
    balance_w: float,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    n_conv: int = 48,
) -> dict:
    """Probe for the multi-context keyed gist task (see _build_synthetic_multiturn_kv).

    Establishes K keyed contexts, then averages CE over answer turns under the same
    real/zero/ablate arms. content_gap = ce_zero - ce_real is the verdict; eff_rank
    of the final bank shows whether K distinct contexts are actually stored (~K) or
    collapsed (~1). Same conversations across arms for a low-variance gap.
    """
    d      = cfg_dict["data"]
    K      = int(d.get("n_query_slots", 2))
    C      = int(d.get("n_contexts", 8))
    S      = int(d.get("n_symbols", 64))
    p_pref = float(d.get("pref_mass", 0.9))
    turns  = int(d.get("turns_per_conv", 8))
    Q_OFF   = 2
    CTX_OFF = Q_OFF + K
    SYM_OFF = CTX_OFF + C
    dists   = _gist_dists(C, S, p_pref)

    was_training = model.training
    model.eval()
    ts = getattr(model, "thought_stream", None)
    orig_new = ts._new_thought if ts is not None else None

    convs = []
    for _ in range(n_conv):
        cs = [int(torch.randint(0, C, (1,))) for _ in range(K)]
        qa = []
        for _ in range(turns):
            slot = int(torch.randint(0, K, (1,)))
            qa.append((slot, int(torch.multinomial(dists[cs[slot]], 1))))
        convs.append((cs, qa))

    def run(zero_content: bool, ablate: bool):
        if ts is not None:
            ts._new_thought = (
                (lambda H, b=None, p=None: torch.zeros_like(orig_new(H, b, p))) if zero_content else orig_new
            )
        ces, last_mem = [], None
        for cs, qa in convs:
            mem = None
            for i in range(K):
                x = torch.tensor([[Q_OFF + i, CTX_OFF + cs[i]]], device=device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(x, init_mem=(None if ablate else mem), compute_logits=not fused_ce)
                if not ablate:
                    mem = out["mem_bank"].detach()
            for slot, s in qa:
                xq = torch.tensor([[Q_OFF + slot]], device=device)
                yq = torch.tensor([[SYM_OFF + s]], device=device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(xq, init_mem=(None if ablate else mem), compute_logits=not fused_ce)
                    _, lg = compute_loss(out, yq, balance_w, ce_chunk)
                if not ablate:
                    mem = out["mem_bank"].detach()
                ces.append(lg["ce"])
            last_mem = mem
        if ts is not None:
            ts._new_thought = orig_new
        return (sum(ces) / max(1, len(ces)) if ces else 0.0), last_mem

    ce_real, mem = run(zero_content=False, ablate=False)
    ce_zero, _   = run(zero_content=True,  ablate=False)
    ce_abl,  _   = run(zero_content=False, ablate=True)

    if was_training:
        model.train()
    return {
        "mem_ablation_gap": ce_abl - ce_real,
        "content_gap":      ce_zero - ce_real,
        "mem_eff_rank":     _effective_rank(mem) if mem is not None else 0.0,
        "mem_slots_final":  float(mem.size(1)) if mem is not None else 0.0,
        "mem_turns":        float(K + turns),
    }


def synthetic_rule_probe(
    model: nn.Module,
    cfg_dict: dict,
    *,
    fused_ce: bool,
    ce_chunk: int,
    balance_w: float,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    n_conv: int = 48,
) -> dict:
    """Probe for the continual-learning rule task (see _build_synthetic_rule).

    Per conversation: show m example pairs of a fresh shift rule at turn 0, then
    query UNSEEN symbols. Averages CE and accuracy over the answer turns under the
    real/zero/ablate arms. content_gap = ce_zero - ce_real and acc_real are the
    verdict: if the bank carries an APPLICABLE rule, ce_real→0 and acc_real→1 on
    unseen queries (which a lookup memory cannot do). Uses logits for accuracy.
    """
    d      = cfg_dict["data"]
    S      = int(d.get("n_symbols", 32))
    m      = int(d.get("n_examples", 6))
    turns  = int(d.get("turns_per_conv", 8))
    K      = int(d.get("n_contexts", 1))
    SYM_OFF = 3
    KEY_OFF = SYM_OFF + S

    was_training = model.training
    model.eval()
    ts = getattr(model, "thought_stream", None)
    orig_new = ts._new_thought if ts is not None else None

    _units, _n_rules, train_pool, held_pool, _apply = _rule_space(d)
    sw    = int(d.get("switch_at", 0))           # rule switch: s2 re-presented at turn sw
    n_ctx = 2 if sw else K

    def _make_convs(pool):
        cs = []
        for _ in range(n_conv):
            ctxs = []
            for _k in range(n_ctx):
                while True:
                    s = pool[int(torch.randint(0, len(pool), (1,)))]
                    if not (sw and _k == 1 and s == ctxs[0][0] and len(pool) > 1):
                        break
                perm = torch.randperm(S).tolist()
                ctxs.append((s, perm[:m], perm[m:]))
            cs.append(ctxs)
        return cs

    convs = _make_convs(train_pool)               # same convs across arms

    @torch.no_grad()
    def run(zero_content: bool, ablate: bool, convs=convs):
        # All conversations share the same per-turn sequence length, so they run as
        # independent batch lanes (one bank per lane, exactly like training).
        if ts is not None:
            ts._new_thought = (
                (lambda H, b=None, p=None: torch.zeros_like(orig_new(H, b, p))) if zero_content else orig_new
            )
        key = (lambda k: [KEY_OFF + k]) if K > 1 else (lambda k: [])
        n = len(convs)
        ces = []
        corr_t = [0] * turns
        tot_t  = [0] * turns
        stick_c, post_c = 0, 0                  # post-switch answers matching OLD rule s1
        mem = None

        def present(kk, mem):
            rows = []
            for ctxs in convs:
                s, ex, _ = ctxs[kk]
                row = key(kk if not sw else 0)
                for xi in ex:
                    row += [SYM_OFF + xi, SYM_OFF + _apply(s, xi)]
                rows.append(row)
            x0 = torch.tensor(rows, device=device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x0, init_mem=mem, compute_logits=False)
            return out["mem_bank"].detach()

        if not ablate:
            for kk in range(1 if sw else K):     # with a switch, s2 is presented mid-conv
                mem = present(kk, mem)
        for t in range(turns):
            if sw and t == sw and not ablate:
                mem = present(1, mem)            # bank carried, no reset: s1 must be dropped
            if sw:
                k, idx = (1, t - sw) if t >= sw else (0, t)
            else:
                k, idx = t % K, t // K
            rows, ys, ys_old = [], [], []
            for ctxs in convs:
                s, _, unseen = ctxs[k]
                q = unseen[idx % len(unseen)]
                rows.append(key(k if not sw else 0) + [SYM_OFF + q])
                ys.append(SYM_OFF + _apply(s, q))
                if sw and t >= sw:
                    ys_old.append(SYM_OFF + _apply(ctxs[0][0], q))
            xq = torch.tensor(rows, device=device)
            y_true = torch.tensor(ys, device=device)
            yq = torch.zeros_like(xq); yq[:, -1] = y_true
            lmq = torch.zeros_like(xq, dtype=torch.bool); lmq[:, -1] = True
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(xq, init_mem=(None if ablate else mem), compute_logits=True)
                _, lg = compute_loss(out, yq, balance_w, ce_chunk, loss_mask=lmq)
            pred = out["logits"][:, -1].argmax(dim=-1)
            corr_t[t] += int((pred == y_true).sum()); tot_t[t] += n
            if sw and t >= sw:
                stick_c += int((pred == torch.tensor(ys_old, device=device)).sum())
                post_c  += n
            if not ablate:
                mem = out["mem_bank"].detach()
            ces.append(lg["ce"])
        if ts is not None:
            ts._new_thought = orig_new
        acc_bt = [c / max(1, nt) for c, nt in zip(corr_t, tot_t)]
        acc = sum(corr_t) / max(1, sum(tot_t))
        run.stick = stick_c / max(1, post_c)
        return (sum(ces) / max(1, len(ces)) if ces else 0.0), acc, mem, acc_bt

    ce_real, acc_real, mem, acc_bt = run(zero_content=False, ablate=False)
    stick_real = run.stick if sw else None
    ce_zero, acc_zero, _, _        = run(zero_content=True,  ablate=False)
    ce_abl,  acc_abl,  _, _        = run(zero_content=False, ablate=True)

    if turns >= 12:
        # long-horizon runs: per-turn accuracy exposes the FIFO-eviction cliff
        tqdm.write("  [horizon] acc/turn: " + " ".join(f"{a:.2f}" for a in acc_bt))

    out = {
        "mem_ablation_gap": ce_abl - ce_real,
        "content_gap":      ce_zero - ce_real,
        "rule_acc":         acc_real,             # accuracy on UNSEEN queries (the verdict)
        "rule_acc_ablate":  acc_abl,
        "mem_eff_rank":     _effective_rank(mem) if mem is not None else 0.0,
        "mem_slots_final":  float(mem.size(1)) if mem is not None else 0.0,
        "mem_turns":        float(1 + turns),
    }
    if turns >= 12:
        q = max(1, turns // 4)
        out["rule_acc_late"] = float(sum(acc_bt[-q:]) / q)   # last quarter: post-eviction
    if sw:
        out["rule_acc_pre"]  = float(sum(acc_bt[:sw]) / sw)
        out["rule_acc_post"] = float(sum(acc_bt[sw:]) / max(1, turns - sw))
        out["rule_stick"]    = float(stick_real)   # post-switch answers still using s1
    if held_pool:
        # generalization arm: shifts NEVER seen in training (held-out pool)
        _, acc_held, _, _ = run(zero_content=False, ablate=False,
                                convs=_make_convs(held_pool))
        out["rule_acc_held"] = acc_held
    blind_pool = sorted(int(v) for v in (d.get("teacher_blind_shifts") or []))
    if blind_pool:
        # intra-run control: TRAIN rules the teacher never touched (no blend,
        # no distill) — installs like the taught ones ⇔ the kick is per-circuit
        _, acc_blind, _, _ = run(zero_content=False, ablate=False,
                                 convs=_make_convs(blind_pool))
        out["rule_acc_blind"] = acc_blind

    if was_training:
        model.train()
    return out


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

    model_cfg = ThoughtBankConfig.from_yaml(cfg_path)
    train_cfg = raw.get("training", {})
    data_cfg  = raw.get("data", {})

    device = _device(train_cfg.get("device", "auto"))
    _set_seed(train_cfg.get("seed", 42))

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    # Synthetic tasks have no text: keep vocab_size from the yaml, no tokenizer.
    if data_cfg.get("task") in ("associative_recall", "latent_context", "multiturn_gist",
                                "multiturn_gist_kv", "multiturn_rule"):
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
        model = ThoughtBankLM(model_cfg).to(device)
        tqdm.write("Architecture: ThoughtBankLM (with memory bank)")
    else:
        model = TrunkLM(model_cfg).to(device)
        tqdm.write("Architecture: TrunkLM (no memory bank)")
    tqdm.write(f"Model: {model.num_params():,} parameters")

    # ── Warm restart (SGDR-style) ─────────────────────────────────────────────
    # Load MODEL weights only from a prior checkpoint; optimizer and LR schedule
    # start fresh. Use case: a cosine that died under a late-engaging circuit —
    # restarting restores LR without re-paying the teacher bootstrap.
    init_from = train_cfg.get("init_from")
    if init_from:
        ckpt = torch.load(init_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        tqdm.write(f"Warm restart ← {init_from} (step {ckpt['step']}); fresh optimizer/schedule")

    # torch.compile: fuses the many tiny kernels (per-slot fast-weight read,
    # Sinkhorn, short turns) that leave the GPU launch-bound on this model size.
    # In-place nn.Module.compile() keeps state_dict keys clean (no _orig_mod.).
    # dynamic=True: turn lengths and bank size vary — avoid a recompile per shape.
    if bool(train_cfg.get("compile", False)):
        model.compile(dynamic=True)
        tqdm.write("torch.compile ON (dynamic=True)")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    muon_lr   = float(train_cfg.get("muon_lr", 0.02))
    adam_lr   = float(train_cfg.get("lr", 3e-4))
    wd        = float(train_cfg.get("weight_decay", 0.1))

    muon_params, adam_params = _split_muon_params(model)
    tqdm.write(
        f"Muon params: {sum(p.numel() for p in muon_params):,}  "
        f"Adam params: {sum(p.numel() for p in adam_params):,}"
    )
    use_muon  = bool(train_cfg.get("use_muon", True))
    rms_match = bool(train_cfg.get("muon_rms_match", False))
    adam_eps  = float(train_cfg.get("adam_eps", 1e-8))
    if rms_match:
        tqdm.write("Muon RMS-match ON: update RMS = 0.2 for all shapes (DSv4 convention)")
    if use_muon:
        opt = Muon(
            muon_params, lr=muon_lr, momentum=0.95, nesterov=True, ns_steps=10, wd=wd,
            rms_match=rms_match,
            adam_params=adam_params, adam_lr=adam_lr, adam_betas=(0.9, 0.95), adam_wd=wd,
            adam_eps=adam_eps,
        )
    else:
        # All-AdamW mode: the Muon group is kept as a lr=0 no-op so the
        # scheduler/checkpoint paths stay identical; every param is actually
        # updated by the bundled AdamW.
        opt = Muon(
            muon_params, lr=0.0, momentum=0.0, nesterov=False, ns_steps=1, wd=0.0,
            adam_params=adam_params + muon_params, adam_lr=adam_lr,
            adam_betas=(0.9, 0.95), adam_wd=wd,
        )
        tqdm.write("use_muon=false: ALL params on AdamW")

    total_steps   = int(train_cfg.get("steps", 10_000))
    warmup_steps  = int(train_cfg.get("warmup_steps", 200))

    lr_schedule = str(train_cfg.get("lr_schedule", "cosine"))
    # WSD (warmup-stable-decay, DSv4): hold peak LR through search AND
    # installation, cosine-decay to the 10% floor only over the final stretch.
    # Antidote to the dying-cosine trap: late-cracking circuits (fissure ~1000+)
    # otherwise consolidate on a fading LR.
    wsd_decay_start = int(train_cfg.get("wsd_decay_start", int(total_steps * 0.75)))
    # Dynamic-pacing companion: when the anneal window is pulled by a trigger
    # (curriculum mastery / ce_below), re-anchor the decay to the anneal END
    # (+offset) — search at full LR under the teacher, cool down only once β=0.
    # The configured wsd_decay_start stays as the fallback if no trigger fires.
    wsd_decay_at_anneal_end = bool(train_cfg.get("wsd_decay_at_anneal_end", False))
    wsd_decay_offset        = int(train_cfg.get("wsd_decay_offset", 0))
    # ReduceLROnPlateau-style trigger (the last clock in the protocol becomes
    # mastery-gated, like the curriculum and the anneal): once β=0, hold peak LR
    # while the post-anneal installation still improves the CE EMA; when the EMA
    # stalls for `patience` steps, pull the decay to NOW. Patience must survive
    # a pre-crack stall — late staircases look like plateaus right before they
    # crack. Combines with the anneal-end re-anchor: whichever fires first.
    wsd_decay_on_plateau = bool(train_cfg.get("wsd_decay_on_plateau", False))
    wsd_plateau_delta    = float(train_cfg.get("wsd_plateau_delta", 0.05))
    wsd_plateau_patience = int(train_cfg.get("wsd_plateau_patience", 300))
    pl_ema = pl_best = None
    pl_best_step = 0

    def _lr_fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if lr_schedule == "constant":
            return 1.0
        if lr_schedule == "wsd":
            if step < wsd_decay_start:
                return 1.0
            progress = (step - wsd_decay_start) / max(1, total_steps - wsd_decay_start)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
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
    # bank viz (TensorBoard figures): writes captured during training (lane 0,
    # zero extra forwards), dumped at the mem-probe cadence.
    viz_buf: list = []
    viz_final = viz_last = None
    _viz_hop_tf = bool(data_cfg.get("hop_teacher", False))
    mem_persist     = bool(data_cfg.get("persist", False))      # carry bank across steps
    mem_multiturn   = data_cfg.get("task") in ("multiturn", "multiturn_gist", "multiturn_gist_kv", "multiturn_rule")  # per-turn forward
    mem_synth_mt    = data_cfg.get("task") in ("multiturn_gist", "multiturn_gist_kv", "multiturn_rule")  # synthetic probe (no tokenizer)
    persist_chunks  = int(train_cfg.get("persist_probe_chunks", 6))
    log_every   = int(train_cfg.get("log_every", 50))
    save_every  = int(train_cfg.get("save_every", 100))
    save_dir    = Path(train_cfg.get("save_dir", f"checkpoints/{run_name}"))
    balance_w   = float(model_cfg.balance_loss_weight)
    write_cost  = float(getattr(model_cfg, "mem_write_cost", 0.0))       # write-sparsity budget
    write_div   = float(getattr(model_cfg, "mem_write_diversity", 0.0))  # novelty-gated write
    write_tgt   = float(getattr(model_cfg, "mem_write_target", 0.0))         # target-rate objective
    write_tgt_w = float(getattr(model_cfg, "mem_write_target_weight", 0.0))  # its weight (0=off)
    # Warmup for the write-sparsity budget: applying it from step 0 kills writing
    # before the bank is useful, so hold at 0 until wc_start then ramp over wc_ramp.
    wc_start    = int(train_cfg.get("write_cost_warmup_start", 0))
    wc_ramp     = int(train_cfg.get("write_cost_warmup_steps", 0))
    def _write_cost_at(s: int) -> float:
        if write_cost <= 0.0 or s < wc_start:
            return 0.0
        return write_cost * min(1.0, (s - wc_start) / max(1, wc_ramp))

    # ── Teacher-forced bank bootstrap (multiturn_rule only) ────────────────────
    # At turn 0 the read consumes β·teacher[s] + (1-β)·w0 and a distill loss pulls
    # the written slot w0 toward the clean teacher code; β anneals 1→0 so the read
    # ends up applying the pure written code. Breaks the ignore-bank fixed point.
    tf_on = (bool(getattr(model_cfg, "mem_teacher_forcing", False))
             and data_cfg.get("task") == "multiturn_rule" and model_cfg.use_dual_stream)
    teacher_emb = teacher_opt = None
    tf_a0 = int(getattr(model_cfg, "mem_teacher_anneal_start", 300))
    tf_a1 = int(getattr(model_cfg, "mem_teacher_anneal_end", 500))
    tf_dw = float(getattr(model_cfg, "mem_teacher_distill_weight", 2.0))
    tf_trigger = str(getattr(model_cfg, "mem_teacher_anneal_trigger", ""))
    tf_alen    = int(getattr(model_cfg, "mem_teacher_anneal_len", 500))
    # ce_below: β stays 1 until the train-CE EMA proves the read exploits the
    # teacher (CE < ln S − margin); the anneal then runs [now, now+len].
    tf_ce_thresh = (math.log(int(data_cfg.get("n_symbols", 32)))
                    - float(getattr(model_cfg, "mem_teacher_anneal_margin", 0.5)))
    tf_ce_ema: Optional[float] = None
    tf_fired = False   # ce_below: PULL-EARLIER only — the fixed [tf_a0,tf_a1] stays
                       # as the fallback (v2: CE sat at ln S through all of β=1 yet
                       # the anneal cracked at ~1000 — organization is silent below
                       # CE, so a CE gate must never be the only way to anneal)
    if tf_on:
        S_rule      = _rule_space(data_cfg)[1]    # rule count (shift: S; affine: φ(S)·S)
        if bool(data_cfg.get("hop_teacher", False)):
            # hop teacher targets are SYMBOL ids (the intermediate mid) — they
            # index the same Fourier table, valid only if it covers [0, S)
            assert S_rule >= int(data_cfg.get("n_symbols", 32)), \
                "hop_teacher needs the Fourier table to cover the symbol circle"
            tqdm.write("Teacher: HOP segments only (Fourier[mid]); presentations untouched")
        teacher_emb = nn.Embedding(S_rule, model_cfg.mem_dim).to(device)
        if bool(getattr(model_cfg, "mem_teacher_fourier", False)):
            _kmax = int(getattr(model_cfg, "mem_teacher_fourier_kmax", 0))
            with torch.no_grad():
                teacher_emb.weight.copy_(_fourier_codes(
                    {**data_cfg, "_fourier_kmax": _kmax}, model_cfg.mem_dim).to(device))
            teacher_emb.weight.requires_grad_(False)
            teacher_opt = None                    # fixed codes: nothing to train
            tqdm.write(f"Teacher codes: FIXED Fourier features ({S_rule} rules, RMS 1, "
                       f"kmax={_kmax or model_cfg.mem_dim // 2})")
        else:
            teacher_lr  = float(train_cfg.get("teacher_lr", 3e-4))
            teacher_opt = optim.AdamW(teacher_emb.parameters(), lr=teacher_lr, weight_decay=wd)
    # Teacher-blind control rules: trained normally (CE, curriculum) but NEVER
    # touched by the blend or the distill — the intra-run arm of the
    # kill-the-crutch test. If they install like their taught neighbours, the
    # teacher kick is per-circuit (one bootstrap organizes the read for every
    # rule); if they stay at chance, the kick is per-rule.
    tf_blind_lut = None
    if tf_on:
        _blind = sorted(int(v) for v in (data_cfg.get("teacher_blind_shifts") or []))
        if _blind:
            _held_b = set(int(v) for v in (data_cfg.get("heldout_shifts") or []))
            assert not (_held_b & set(_blind)), "teacher_blind_shifts must be TRAIN rules"
            tf_blind_lut = torch.zeros(S_rule, dtype=torch.bool, device=device)
            tf_blind_lut[torch.tensor(_blind, device=device)] = True
            tqdm.write(f"Teacher-blind rules: {len(_blind)} train rules excluded from "
                       f"blend+distill (intra-run per-circuit control)")
    # Code-space mixup: EMA dictionary of the presentation code the read actually
    # consumes (post teacher-blend), one slot per shift. Midpoints of trained
    # neighbours are injected with the middle rule's labels (see config).
    mix_p     = float(getattr(model_cfg, "mem_code_mixup_p", 0.0))
    mix_mom   = float(getattr(model_cfg, "mem_code_mixup_momentum", 0.99))
    mix_start = int(getattr(model_cfg, "mem_code_mixup_start", 0))
    mix_ema = mix_cnt = mix_pairs = None
    if tf_on and mix_p > 0.0:
        assert str(data_cfg.get("rule_family", "shift")) == "shift", \
            "code mixup pairs are 1D-circle (shift family) only"
        # v2, GENERALIZED SYMMETRIC PAIRS: for each trained mid s, ALL trained
        # pairs (s-d, s+d), d >= 1 — many geometries per mid, so memorizing the
        # supervised midpoints is costlier than learning "midpoint = mean rule".
        # (v1, d=1 only: 8 supervised mids were memorized — train 0.992, held
        # 0.003.) Held shifts never appear as mid or endpoint: no leakage.
        _held = set(int(v) for v in (data_cfg.get("heldout_shifts") or []))
        _S    = int(data_cfg.get("n_symbols", 32))
        # Span cap: the code manifold is CIRCULAR, so the chord midpoint of
        # (s-d, s+d) only points at s for the short arc (d < S/4); beyond that
        # it points at the antipode. Cap well below S/4 to keep the injected
        # midpoints on the right side and the renormalization well-conditioned.
        _D    = min(6, _S // 4 - 1)
        mix_pairs = torch.zeros(_S, _D, dtype=torch.bool, device=device)  # [s, d-1]
        for _s in range(1, _S):
            if _s in _held:
                continue
            for _d in range(1, _D + 1):
                a, b = _s - _d, _s + _d
                if 1 <= a and b <= _S - 1 and a not in _held and b not in _held:
                    mix_pairs[_s, _d - 1] = True
        mix_ema = torch.zeros(_S, model_cfg.mem_dim, device=device)
        mix_cnt = torch.zeros(_S, device=device)
        _mids = int((mix_pairs.any(dim=1)).sum())
        tqdm.write(f"Code mixup ON (v2 pairs): p={mix_p} mom={mix_mom} "
                   f"eligible_mids={_mids} pairs={int(mix_pairs.sum())}")
    if tf_on:
        _win = (f"[{tf_a0},{tf_a1}] or earlier if ce<{tf_ce_thresh:.2f} (len={tf_alen})"
                if tf_trigger == "ce_below" else f"[{tf_a0},{tf_a1}]")
        tqdm.write(f"Teacher-forcing ON: S={S_rule} anneal={_win} distill_w={tf_dw} "
                   f"gate={'on' if model_cfg.mem_write_gate else 'off'}")
    def _beta_at(s: int) -> float:
        if s <= tf_a0: return 1.0
        if s >= tf_a1: return 0.0
        return 1.0 - (s - tf_a0) / max(1, tf_a1 - tf_a0)
    tf_distill_last = 0.0    # telemetry: last turn-0 distill MSE (alignment progress)

    model.train()
    step  = 0          # optimiser steps (one per grad_accum micro-batches)
    micro = 0          # micro-batches seen since the last optimiser step
    t0    = time.perf_counter()
    toks  = 0          # tokens accumulated across the current optimiser step
    pbar  = tqdm(total=total_steps, desc="Training")
    persist_mem: Optional[torch.Tensor] = None   # bank carried across steps
    window_loss = None                            # multi-turn TBPTT window accumulator
    cur_full = int(data_cfg.get("curriculum_full_step", 0))  # diversity curriculum ramp end
    # Mastery-gated curriculum (dsv4h): the pool DOUBLES only when the train-CE EMA
    # proves the current pool is exploited (dsv4g falsified the clock ramp: frac=step/800
    # gave the 16-rule regime <50 steps — the mastery phase never existed). A stalled
    # stage is a verdict, not a bug: it localizes the lock at that pool size.
    cur_mastery  = str(data_cfg.get("curriculum_mode", "")) == "mastery"
    cur_thresh   = float(data_cfg.get("curriculum_ce_thresh", 5.0))
    cur_dwell    = int(data_cfg.get("curriculum_dwell_min", 150))   # min steps per stage
    cur_full_n   = int(data_cfg.get("curriculum_pool_full_n", 0))   # train-pool size; >0
                                    # couples the anneal to the curriculum: teacher time
                                    # ∝ rule count, so β=1 holds until the FULL pool is
                                    # MASTERED (next CE-EMA gate crossing at full size),
                                    # then the anneal is pulled to [now, now+len].
                                    # curriculum_anneal_margin>0 = legacy fixed offset
                                    # from the pool-full doubling instead (dsv4h: 300).
    cur_post_full = int(data_cfg.get("curriculum_anneal_margin", 0))
    cur_anneal_at_n = int(data_cfg.get("curriculum_anneal_at_n", 0))  # >0: pull the anneal
                                    # at the MASTERY of that pool size (early crutch exit)
                                    # while the curriculum keeps doubling — the remaining
                                    # rules must install teacher-free (write generalization)
    cur_ce_ema: Optional[float] = None
    cur_stage_from = 0                                # step at which the current stage began
    # Structure-randomized cells (turns_range / switch_phase_*): conversations
    # have VARIABLE segment counts — the generator publishes each one's length
    # in raw["_conv"] and the optimiser steps at conversation end; grad_accum
    # only serves as the loss denominator fallback outside this mode.
    struct_rand = bool(data_cfg.get("turns_range") or data_cfg.get("switch_phase_min")
                       or data_cfg.get("n_contexts_range"))
    conv_len    = grad_accum   # current conversation's segment count
    micro_conv  = 0            # segments seen inside the current conversation
    conv_k      = 0            # current conversation's establish-presentation count
    redund_sw   = None         # min redundancy over SWITCH writes since last log line:
                               # the step-line redund samples the conversation's LAST
                               # segment (a rehearsal copy, ~0.99 by construction);
                               # switch re-presentations write novel content (~0.3)
                               # and would otherwise never appear in the logs

    for batch in dl:
        if "_curriculum" in raw:
            if cur_mastery:
                raw["_curriculum"].setdefault("n", int(data_cfg.get("curriculum_rules_start", 16)))
            else:
                raw["_curriculum"]["frac"] = (step / cur_full) if cur_full > 0 else 1.0
        if mem_multiturn:
            # One TURN-SLOT of B lanes (padded). The bank persists across slots
            # within a conversation, resets per-lane at a conversation boundary, and
            # backward runs over a TBPTT window of mem_bptt_window slots.
            if len(batch) == 5:                       # multiturn_rule yields the rule id
                x, y, lmask, reset, rule_s = batch
                rule_s = rule_s.to(device)
            else:
                x, y, lmask, reset = batch
                rule_s = None
            x, y, lmask, reset = x.to(device), y.to(device), lmask.to(device), reset.to(device)
            toks += int(lmask.sum())

            init_mem = persist_mem
            if init_mem is not None and reset.any():
                if bool(reset.all()):
                    # Turn-aligned synthetic tasks reset all lanes together at a
                    # conversation boundary → hand back None so the model reseeds a
                    # fresh random bank (matches the fast-weight seed_bank design).
                    init_mem = None
                else:
                    init_mem = init_mem.masked_fill(reset.view(-1, 1, 1), 0.0)
            if struct_rand and bool(reset.all()):
                if viz_buf:                       # previous conversation complete
                    viz_last, viz_buf = (viz_buf, viz_final), []
                conv_len, micro_conv = int(raw["_conv"]["n_seg"]), 0
                conv_k = int(raw["_conv"].get("k", 0))
            accum = conv_len if struct_rand else grad_accum
            # pad_mask gates which positions the WRITE head can pool. For
            # multiturn_rule, lmask is a LOSS mask (turn 0 supervises only the
            # applied-x positions) but every token is real — the write must see
            # the full window (x AND y symbols) to infer the rule.
            wmask = (x != 0) if data_cfg.get("task") == "multiturn_rule" else lmask
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x, init_mem=init_mem, compute_logits=not fused_ce, pad_mask=wmask)
                loss, logs = compute_loss(
                    out, y, balance_w, ce_chunk, _write_cost_at(step), write_div,
                    write_tgt, write_tgt_w, loss_mask=lmask,
                )
            mem_bank = out["mem_bank"]
            if tf_on and rule_s is not None and bool((rule_s >= 0).all()):
                # Presentation turn (rule id valid; -1 on answer turns): blend the
                # written slot toward a clean teacher code and distill the write
                # toward it; both fade as β→0 (teacher removed).
                beta   = _beta_at(step)
                w0     = mem_bank[:, -1]
                t_s    = teacher_emb(rule_s).to(mem_bank.dtype)
                taught = None if tf_blind_lut is None else ~tf_blind_lut[rule_s]
                if bool(getattr(model_cfg, "mem_teacher_distill_cosine", False)):
                    # direction-only distill: shrinking ‖w0‖ pays nothing
                    per = 1.0 - F.cosine_similarity(
                        w0.float(), t_s.detach().float(), dim=1)
                else:
                    per = ((w0.float() - t_s.detach().float()) ** 2).mean(dim=1)
                if taught is None:
                    distill = per.mean()
                else:
                    # blind lanes: no distill pull, no blend — pure TBPTT gradient
                    distill = (per * taught).sum() / taught.sum().clamp_min(1)
                code   = beta * t_s + (1.0 - beta) * w0
                if taught is not None:
                    code = torch.where(taught.unsqueeze(1), code, w0)
                code   = code.unsqueeze(1)
                mem_bank = torch.cat([mem_bank[:, :-1], code], dim=1)
                if mix_ema is not None:
                    # Track the code the read consumes at presentation (post-blend),
                    # then swap in the neighbours' midpoint for a random subset of
                    # eligible lanes — labels stay those of the middle rule s.
                    slot = mem_bank[:, -1].detach().float()
                    mix_ema[rule_s] = mix_mom * mix_ema[rule_s] + (1.0 - mix_mom) * slot
                    mix_cnt[rule_s] += 1
                    # sample one valid span d per lane, uniformly over its pairs
                    cand = mix_pairs[rule_s]                               # [B, D]
                    r    = torch.rand_like(cand, dtype=torch.float32).masked_fill(~cand, -1.0)
                    dsp  = r.argmax(dim=1) + 1                             # [B] chosen d
                    sm1  = (rule_s - dsp).clamp(0, mix_ema.size(0) - 1)
                    sp1  = (rule_s + dsp).clamp(0, mix_ema.size(0) - 1)
                    ok = (cand.any(dim=1) & (mix_cnt[sm1] > 0) & (mix_cnt[sp1] > 0)
                          & (torch.rand(rule_s.size(0), device=device) < mix_p))
                    if step < mix_start:
                        ok = torch.zeros_like(ok)   # EMA warms up; no injection yet
                    if bool(ok.any()):
                        a, b = mix_ema[sm1], mix_ema[sp1]
                        mid  = 0.5 * (a + b)
                        tgt  = 0.5 * (a.norm(dim=1) + b.norm(dim=1))
                        mid  = mid * (tgt / mid.norm(dim=1).clamp_min(1e-6)).unsqueeze(1)
                        new_slot = torch.where(ok.unsqueeze(1), mid.to(mem_bank.dtype),
                                               mem_bank[:, -1])
                        mem_bank = torch.cat([mem_bank[:, :-1], new_slot.unsqueeze(1)], dim=1)
                # distill is a PER-CONVERSATION loss but rides on the turn-0 micro,
                # which the window divides by the accum denominator — pre-multiply
                # so its effective weight is the configured tf_dw (matches the proof).
                loss = loss + (beta * tf_dw * accum) * distill
                tf_distill_last = float(distill.detach())
            if (struct_rand and rule_s is not None and bool((rule_s >= 0).all())
                    and micro_conv >= conv_k and "write_redund" in logs):
                # this presentation is PAST the establish block = a switch write
                redund_sw = (logs["write_redund"] if redund_sw is None
                             else min(redund_sw, logs["write_redund"]))
            if writer is not None and struct_rand:
                # lane-0 write trail for the bank figures (post-blend code)
                _vk = ("present" if micro_conv < conv_k else
                       (("hop" if _viz_hop_tf else "switch")
                        if (rule_s is not None and bool((rule_s >= 0).all())) else "turn"))
                viz_buf.append((mem_bank[0, -1].detach().float().cpu(), _vk))
                viz_final = mem_bank[0].detach().float().cpu()
            persist_mem = mem_bank                         # keep graph within the window
            seg_loss = scaler.scale(loss / accum)
            window_loss = seg_loss if window_loss is None else window_loss + seg_loss
            micro += 1
            micro_conv += 1
            at_step = ((micro_conv == conv_len) if struct_rand
                       else (micro % grad_accum == 0))
            if (micro_conv if struct_rand else micro) % mem_bptt_window == 0 or at_step:
                # flush window (TBPTT truncation)
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
                write_cost=_write_cost_at(step), write_diversity=write_div,
                write_target=write_tgt, write_target_weight=write_tgt_w,
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
        if teacher_opt is not None:
            scaler.step(teacher_opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if teacher_opt is not None:
            teacher_opt.zero_grad(set_to_none=True)
        sched_step()

        step += 1
        if cur_mastery and "_curriculum" in raw:
            ce_now = float(logs["ce"])
            cur_ce_ema = ce_now if cur_ce_ema is None else 0.98 * cur_ce_ema + 0.02 * ce_now
            if cur_ce_ema < cur_thresh and (step - cur_stage_from) >= cur_dwell:
                n_old = int(raw["_curriculum"]["n"])
                if cur_full_n > 0 and n_old >= cur_full_n:
                    # Pool already full: this gate crossing = FULL POOL MASTERED.
                    # Same judge as the stages decides when the crutch can leave.
                    if tf_on and not tf_fired:
                        tf_fired = True
                        tf_a0, tf_a1 = step, step + tf_alen
                        tqdm.write(f"[curriculum] full pool ({cur_full_n}) mastered "
                                   f"(CE EMA < {cur_thresh:.3f}) @ step {step}: "
                                   f"anneal pulled to [{tf_a0},{tf_a1}]")
                        if wsd_decay_at_anneal_end and lr_schedule == "wsd":
                            wsd_decay_start = tf_a1 + wsd_decay_offset
                            tqdm.write(f"[lr] WSD decay re-anchored to anneal end: "
                                       f"start @ {wsd_decay_start}")
                    cur_stage_from = step             # re-arm the dwell, stop log spam
                    cur_ce_ema = None
                else:
                    raw["_curriculum"]["n"] = n_old * 2   # _cur_pool clamps to the full pool
                    cur_stage_from = step
                    cur_ce_ema = None                 # fresh EMA: judge the NEW pool only
                    tqdm.write(f"[curriculum] CE EMA < {cur_thresh:.3f} @ step {step}: "
                               f"pool {n_old} -> {n_old * 2}")
                    if tf_on and not tf_fired and cur_anneal_at_n > 0 and n_old >= cur_anneal_at_n:
                        # early crutch exit: pool of size n_old just proved mastered
                        tf_fired = True
                        tf_a0, tf_a1 = step, step + tf_alen
                        tqdm.write(f"[curriculum] pool {n_old} mastered >= anneal_at_n "
                                   f"({cur_anneal_at_n}): anneal pulled to [{tf_a0},{tf_a1}] "
                                   f"— remaining rules install teacher-free")
                        if wsd_decay_at_anneal_end and lr_schedule == "wsd":
                            wsd_decay_start = tf_a1 + wsd_decay_offset
                            tqdm.write(f"[lr] WSD decay re-anchored to anneal end: "
                                       f"start @ {wsd_decay_start}")
                    if (tf_on and not tf_fired and cur_post_full > 0
                            and cur_full_n > 0 and n_old * 2 >= cur_full_n):
                        # legacy fixed-margin coupling (dsv4h): anneal at doubling+margin
                        tf_fired = True
                        tf_a0, tf_a1 = step + cur_post_full, step + cur_post_full + tf_alen
                        tqdm.write(f"[curriculum] pool FULL ({cur_full_n}): anneal pulled to "
                                   f"[{tf_a0},{tf_a1}]")
                        if wsd_decay_at_anneal_end and lr_schedule == "wsd":
                            wsd_decay_start = tf_a1 + wsd_decay_offset
                            tqdm.write(f"[lr] WSD decay re-anchored to anneal end: "
                                       f"start @ {wsd_decay_start}")
        if (wsd_decay_on_plateau and lr_schedule == "wsd"
                and step < wsd_decay_start and ((not tf_on) or step >= tf_a1)):
            ce_now = float(logs["ce"])
            pl_ema = ce_now if pl_ema is None else 0.98 * pl_ema + 0.02 * ce_now
            if pl_best is None or pl_best - pl_ema >= wsd_plateau_delta:
                pl_best, pl_best_step = pl_ema, step
            elif step - pl_best_step >= wsd_plateau_patience:
                wsd_decay_start = step
                tqdm.write(f"[lr] CE EMA plateau ({pl_ema:.3f}; no −{wsd_plateau_delta} "
                           f"in {wsd_plateau_patience} steps since best "
                           f"{pl_best:.3f}@{pl_best_step}): WSD decay starts NOW @ {step}")
        if tf_on and tf_trigger == "ce_below" and not tf_fired and step < tf_a0:
            ce_now = float(logs["ce"])
            tf_ce_ema = ce_now if tf_ce_ema is None else 0.98 * tf_ce_ema + 0.02 * ce_now
            if tf_ce_ema < tf_ce_thresh:
                tf_fired = True
                tf_a0, tf_a1 = step, step + tf_alen
                tqdm.write(f"[anneal-trigger] CE EMA {tf_ce_ema:.3f} < {tf_ce_thresh:.3f} "
                           f"@ step {step}: anneal pulled in to [{tf_a0},{tf_a1}]")
                if wsd_decay_at_anneal_end and lr_schedule == "wsd":
                    wsd_decay_start = tf_a1 + wsd_decay_offset
                    tqdm.write(f"[lr] WSD decay re-anchored to anneal end: "
                               f"start @ {wsd_decay_start}")
        dt    = time.perf_counter() - t0
        tok_s = toks / dt
        t0    = time.perf_counter()
        toks  = 0

        pbar.set_postfix(loss=f"{logs['ce']:.3f}", ppl=f"{logs['ppl']:.1f}", tok_s=f"{tok_s:.0f}")
        pbar.update(1)

        # Multi-turn probe: turn-cadenced content_gap + effective rank
        if mem_probe_every and mem_multiturn and step % mem_probe_every == 0:
            if mem_synth_mt:
                _synth_task = data_cfg.get("task")
                probe_fn = {
                    "multiturn_gist_kv": synthetic_multiturn_kv_probe,
                    "multiturn_rule":    synthetic_rule_probe,
                }.get(_synth_task, synthetic_multiturn_probe)
                mp = probe_fn(
                    model, raw, fused_ce=fused_ce, ce_chunk=ce_chunk,
                    balance_w=balance_w, device=device, amp_dtype=amp_dtype, use_amp=use_amp,
                )
            else:
                mp = multiturn_probe(
                    model, tokenizer, raw, fused_ce=fused_ce, ce_chunk=ce_chunk,
                    balance_w=balance_w, device=device, amp_dtype=amp_dtype, use_amp=use_amp,
                )
            logs.update(mp)
            if mp:
                acc = (f"  rule_acc={mp['rule_acc']:.3f}(abl {mp['rule_acc_ablate']:.3f})"
                       if "rule_acc" in mp else "")
                acc += (f"  rule_HELD={mp['rule_acc_held']:.3f}"
                        if "rule_acc_held" in mp else "")
                acc += (f"  rule_BLIND={mp['rule_acc_blind']:.3f}"
                        if "rule_acc_blind" in mp else "")
                acc += (f"  rule_LATE={mp['rule_acc_late']:.3f}"
                        if "rule_acc_late" in mp else "")
                acc += (f"  pre/post={mp['rule_acc_pre']:.3f}/{mp['rule_acc_post']:.3f}"
                        f"  STICK={mp['rule_stick']:.3f}"
                        if "rule_stick" in mp else "")
                tqdm.write(
                    f"  [mt-probe] ablation_gap={mp['mem_ablation_gap']:+.4f}"
                    f"  content_gap={mp['content_gap']:+.4f}"
                    f"  eff_rank={mp['mem_eff_rank']:.2f}/{mp['mem_slots_final']:.0f}"
                    f"  turns={mp['mem_turns']:.0f}{acc}"
                )
            if writer is not None and viz_last is not None:
                from .bank_viz import (bank_content_fig, bank_similarity_fig,
                                       writes_pca_fig)
                _wl, _fb = viz_last
                writer.add_figure("bank/content", bank_content_fig(_fb), step)
                writer.add_figure("bank/similarity", bank_similarity_fig(_fb), step)
                writer.add_figure("bank/writes_pca",
                                  writes_pca_fig(torch.stack([w for w, _ in _wl]),
                                                 [k for _, k in _wl]), step)

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
            if tf_on:
                logs["tf_distill"] = tf_distill_last
                logs["tf_beta"]    = _beta_at(step)
            if redund_sw is not None:
                logs["write_redund_sw"] = redund_sw
                redund_sw = None                 # min over the next log window
            tqdm.write(
                f"step={step:>6}  ce={logs['ce']:.4f}  ppl={logs['ppl']:.2f}"
                f"  balance={logs['balance']:.4f}"
                + (f"  r_hat={logs.get('r_hat', 0):.3f}" if "r_hat" in logs else "")
                + (f"  mem={logs['mem_slots']:.0f}" if "mem_slots" in logs else "")
                + (f"  α={logs['write_alpha']:.3f}" if "write_alpha" in logs else "")
                + (f"  wpen={logs['write_pen']:.3f}" if "write_pen" in logs else "")
                + (f"  redund={logs['write_redund']:.3f}" if "write_redund" in logs else "")
                + (f"  redund_sw={logs['write_redund_sw']:.3f}" if "write_redund_sw" in logs else "")
                + (f"  erank={logs['mem_eff_rank']:.2f}" if "mem_eff_rank" in logs else "")
                + (f"  gap={logs['mem_ablation_gap']:+.3f}" if "mem_ablation_gap" in logs else "")
                + (f"  distill={logs['tf_distill']:.3f}(β={logs['tf_beta']:.2f})" if "tf_distill" in logs else "")
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
