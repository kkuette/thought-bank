from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .model import ThoughtLM
from .tokenization import TokenizerAdapter


# ----------------------------- Data structures -----------------------------


@dataclass
class MCBatch:
    """Batch with answer mask for multi-context memory.

    Attributes
    ----------
    input_ids: torch.LongTensor
        Token ids of shape [B, T].
    labels: torch.LongTensor
        Next-token labels of shape [B, T].
    answer_mask: torch.BoolTensor
        Boolean mask [B, T] where True indicates answer tokens (to compute masked CE/EM).
    """

    input_ids: torch.LongTensor
    labels: torch.LongTensor
    answer_mask: torch.BoolTensor


@dataclass
class MCGenConfig:
    """Config for generating multi-context memory episodes."""

    min_contexts: int = 3
    max_contexts: int = 6
    min_facts: int = 1
    max_facts: int = 3
    min_delay: int = 1  # min contexts between store and query
    max_delay: int = 4  # max contexts between store and query
    distractor_lines: Tuple[int, int] = (1, 3)
    key_len: int = 4
    val_len: int = 6


# ----------------------------- Text generation -----------------------------


CTX_SEP = "<CTX_SEP>\n"
STORE_TMPL = "<STORE id={k}> value={v} </STORE>\n"
QUERY_TMPL = "<QUERY id={k}> -> <ANS> {v} </ANS> </QUERY>\n"
DIST_TMPL = "# DSTR {i}: {payload}\n"


def _rand_ident(n: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join(random.choice(alphabet) for _ in range(n))


def _rand_payload(lines: int) -> str:
    # Simple filler that looks code-like without needing external data
    out: List[str] = []
    for i in range(lines):
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        out.append(f"x = {a} * {b} + {i}")
    return "\n".join(out)


def _encode_concat(tok: TokenizerAdapter, parts: Sequence[str]) -> List[int]:
    """Encode multiple parts, adding BOS only on the very first one.

    Ensures consistent packing across tokenizers.
    """
    ids_all: List[int] = []
    for i, p in enumerate(parts):
        ids = tok.encode(p, add_bos=(i == 0), add_eos=False)
        if i > 0 and tok.kind == "byte":
            # Byte tokenizer doesn't add BOS; no-op. Left for symmetry.
            pass
        ids_all.extend(ids)
    return ids_all


def _find_pattern(seq: List[int], pattern: List[int]) -> Optional[Tuple[int, int]]:
    """Return [start, end) indices for the first pattern occurrence in seq."""
    if not pattern or not seq:
        return None
    n, m = len(seq), len(pattern)
    for i in range(0, n - m + 1):
        if seq[i : i + m] == pattern:
            return i, i + m
    return None


def build_episode(
    tok: TokenizerAdapter, cfg: MCGenConfig
) -> Tuple[List[int], List[Tuple[int, int]], dict]:
    """Create one episode as concatenated contexts with answer spans.

    Returns
    -------
    ids: List[int]
        Token ids for the entire episode (without EOS). BOS is included once at the beginning.
    spans: List[Tuple[int, int]]
        List of [start, end) token spans for answers (mask on labels aligns to positions 1..T-1).
    meta: dict
        Summary info (num_contexts, delays per fact, etc.).
    """
    num_ctx = random.randint(cfg.min_contexts, cfg.max_contexts)
    num_facts = random.randint(cfg.min_facts, cfg.max_facts)

    # Sample facts and their placements
    facts: List[Tuple[str, str]] = [
        (_rand_ident(cfg.key_len), _rand_ident(cfg.val_len)) for _ in range(num_facts)
    ]
    # Assign each fact a store context and query context with a delay
    store_ctx_idx = [random.randint(0, max(0, num_ctx - 2)) for _ in range(num_facts)]
    delays = [random.randint(cfg.min_delay, cfg.max_delay) for _ in range(num_facts)]
    query_ctx_idx = [min(num_ctx - 1, s + d) for s, d in zip(store_ctx_idx, delays)]

    # Build textual contexts
    ctx_texts: List[str] = []
    answer_patterns: List[List[int]] = []
    for i in range(num_ctx):
        parts: List[str] = []
        parts.append(CTX_SEP)
        # Stores scheduled at this context
        for j, (k, v) in enumerate(facts):
            if store_ctx_idx[j] == i:
                parts.append(STORE_TMPL.format(k=k, v=v))
        # Optional distractors
        d_lines = random.randint(cfg.distractor_lines[0], cfg.distractor_lines[1])
        parts.append(DIST_TMPL.format(i=i, payload=_rand_payload(d_lines)))
        # Queries scheduled at this context
        for j, (k, v) in enumerate(facts):
            if query_ctx_idx[j] == i:
                parts.append(QUERY_TMPL.format(k=k, v=v))
        ctx_texts.append("".join(parts))

        # Cache answer patterns for later span find (ANS markers + value)
        # We search for the encoded "<ANS> {v} </ANS>" piece
        ans_text = "<ANS> " + facts[0][1] + " </ANS>"  # will override per fact below
        answer_patterns = []  # reset; we'll fill after the whole text is known

    # Encode whole episode once, then find spans for each fact
    episode_text = "".join(ctx_texts)
    ids = _encode_concat(tok, [episode_text])

    spans: List[Tuple[int, int]] = []
    for k, v in facts:
        ans_piece = "<ANS> " + v + " </ANS>"
        pat = tok.encode(ans_piece, add_bos=False, add_eos=False)
        loc = _find_pattern(ids, pat)
        if loc is None:
            continue
        # Exclude the tag tokens themselves from the mask; keep only the value tokens
        # Determine inner span within pat: find encoded value alone and map to global
        v_ids = tok.encode(v, add_bos=False, add_eos=False)
        # Find v_ids within pat to compute offset
        sub = _find_pattern(pat, v_ids)
        if sub is None:
            # Fallback: keep full pat span
            spans.append(loc)
        else:
            start = loc[0] + sub[0]
            end = loc[0] + sub[1]
            spans.append((start, end))

    meta = {
        "num_contexts": num_ctx,
        "num_facts": num_facts,
        "delays": delays,
        "store_ctx_idx": store_ctx_idx,
        "query_ctx_idx": query_ctx_idx,
    }
    return ids, spans, meta


class MultiContextMemoryDataset(Dataset[Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]):
    """Dataset generating multi-context episodes and masks.

    Each item returns (x, y, answer_mask) with fixed length seq_len.
    """

    def __init__(
        self,
        tokenizer: TokenizerAdapter,
        seq_len: int,
        size: int = 2000,
        gen_cfg: Optional[MCGenConfig] = None,
        seed: int = 123,
    ) -> None:
        super().__init__()
        random.seed(seed)
        torch.manual_seed(seed)
        self.tok = tokenizer
        self.seq_len = int(seq_len)
        self.size = int(size)
        self.gen_cfg = gen_cfg or MCGenConfig()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]:
        ids, spans, _ = build_episode(self.tok, self.gen_cfg)
        # Build x/y of length seq_len and an answer mask aligned to y positions
        if len(ids) < self.seq_len + 1:
            pad = [self.tok.pad_id] * (self.seq_len + 1 - len(ids))
            ids = ids + pad
        else:
            ids = ids[: self.seq_len + 1]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        mask = torch.zeros_like(y, dtype=torch.bool)
        for (s, e) in spans:
            # Mask needs to align to positions of y; token at input index t predicts y[t]
            # So mark [s, e) on y by shifting left by 1.
            s_y = max(0, s - 1)
            e_y = max(0, e - 1)
            if s_y < mask.numel():
                mask[s_y : min(e_y, mask.numel())] = True
        return x, y, mask


def build_mc_dataloader(
    tokenizer: TokenizerAdapter,
    seq_len: int,
    batch_size: int,
    num_workers: int,
    size: int = 2000,
    gen_cfg: Optional[MCGenConfig] = None,
) -> DataLoader[MCBatch]:
    ds = MultiContextMemoryDataset(tokenizer, seq_len=seq_len, size=size, gen_cfg=gen_cfg)

    def collate(batch: Sequence[Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]) -> MCBatch:
        xs, ys, ms = zip(*batch)
        return MCBatch(
            input_ids=torch.stack(list(xs), dim=0),
            labels=torch.stack(list(ys), dim=0),
            answer_mask=torch.stack(list(ms), dim=0),
        )

    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)


# ----------------------------- Evaluation -----------------------------


def masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    """Masked cross entropy over positions where mask is True.

    logits: [B, T, V], targets: [B, T], mask: [B, T]
    Returns a scalar CE.
    """
    ce_all = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")  # [B, T]
    m = mask.float()
    denom = m.sum().clamp_min(1.0)
    return (ce_all * m).sum() / denom


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.BoolTensor) -> float:
    pred = logits.argmax(dim=-1)
    correct = ((pred == targets) & mask).sum().item()
    total = mask.sum().item()
    return float(correct / total) if total > 0 else 0.0


@dataclass
class EvalStats:
    ce_mem: float
    ce_nomem: float
    ppl_mem: float
    ppl_nomem: float
    token_acc: float
    r_hat: float
    write_rate_hard: float


@torch.no_grad()
def evaluate_multi_context(
    model: ThoughtLM,
    dl: DataLoader[MCBatch],
    device: torch.device,
) -> EvalStats:
    model_was_training = model.training
    model.eval()
    ce_mem_sum = 0.0
    ce_nomem_sum = 0.0
    acc_sum = 0.0
    r_hat_sum = 0.0
    wrh_sum = 0.0
    count = 0

    for batch in dl:
        x = batch.input_ids.to(device)
        y = batch.labels.to(device)
        m = batch.answer_mask.to(device)
        out = model(x)
        lm_mem = out["logits_mem"][:, :-1, :]
        lm_nom = out["logits_nomem"][:, :-1, :]
        tgt = y[:, 1:]
        m_shift = m[:, 1:]

        ce_mem = masked_ce(lm_mem, tgt, m_shift)
        ce_nom = masked_ce(lm_nom, tgt, m_shift)
        acc = token_accuracy(lm_mem, tgt, m_shift)

        ce_mem_sum += float(ce_mem.item())
        ce_nomem_sum += float(ce_nom.item())
        acc_sum += float(acc)
        r_hat_sum += float(out["p_gates"].mean().detach().cpu())
        wrh_sum += float(((out["p_gates"] > 0.5).float().mean()).detach().cpu())
        count += 1

    if model_was_training:
        model.train()

    if count == 0:
        return EvalStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    ce_mem_avg = ce_mem_sum / count
    ce_nomem_avg = ce_nomem_sum / count
    ppl_mem = math.exp(ce_mem_avg)
    ppl_nom = math.exp(ce_nomem_avg)
    acc_avg = acc_sum / count
    r_hat = r_hat_sum / count
    wrh = wrh_sum / count

    return EvalStats(
        ce_mem=ce_mem_avg,
        ce_nomem=ce_nomem_avg,
        ppl_mem=ppl_mem,
        ppl_nomem=ppl_nom,
        token_acc=acc_avg,
        r_hat=r_hat,
        write_rate_hard=wrh,
    )


# (CLI moved to train_mc.py to keep this module focused and concise)
