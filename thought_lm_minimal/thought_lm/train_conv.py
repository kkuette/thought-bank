from __future__ import annotations

import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter

from .config import Config
from .model import ThoughtLM
from .tokenization import HFBuildArgs, TokenizerAdapter, build_tokenizer
from .train import JSONLLogger

# Optional progress bar (tqdm); fall back gracefully if not available
try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

# AMP scaler import consistent with train.py
try:  # PyTorch >= 2.1
    from torch.amp import GradScaler as AmpGradScaler  # type: ignore
except Exception:  # pragma: no cover
    AmpGradScaler = None  # type: ignore
try:
    from torch.cuda.amp import GradScaler as CudaGradScaler  # type: ignore
except Exception:  # pragma: no cover
    CudaGradScaler = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from datasets import load_dataset  # type: ignore
    from datasets.utils.logging import disable_progress_bar  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore

    def disable_progress_bar() -> None:  # type: ignore
        return None


logger = logging.getLogger(__name__)

# How many per-pair-index metrics to log (0-based pair index within a conversation)
PAIR_METRICS_MAX = 8


def _select_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _setup_logging() -> None:
    """Configure basic logging for this entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config_from_argv() -> Config:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/default.yaml")
    return Config.from_yaml(cfg_path)


@dataclass
class ConvBatch:
    input_ids: torch.LongTensor  # [B, T]
    labels: torch.LongTensor  # [B, T]
    loss_mask: torch.BoolTensor  # [B, T] -> compute loss only on assistant turn


def _find_subseq_idx(hay: List[int], needle: List[int]) -> Optional[int]:
    """Find first index of subsequence needle in hay. Returns None if not found."""
    if not needle:
        return None
    n = len(needle)
    for i in range(0, len(hay) - n + 1):
        if hay[i : i + n] == needle:
            return i
    return None


class UltraChatTurnStream(
    IterableDataset[Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]
):
    """Deprecated: per-pair stream (kept for reference)."""

    def __init__(
        self,
        tokenizer: TokenizerAdapter,
        *,
        seq_len: int,
        split: str = "train_gen",
        streaming: bool = True,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if load_dataset is None:
            raise RuntimeError("datasets is not installed. Please install 'datasets'.")
        self.tok = tokenizer
        self.seq_len = int(seq_len)
        self.split = split
        self.streaming = streaming
        self.max_samples = max_samples
        _set_seed(seed)
        self.user_tok_ids = tokenizer.encode("<|user|> ", add_bos=False, add_eos=False)
        self.assist_tok_ids = tokenizer.encode("<|assistant|> ", add_bos=False, add_eos=False)
        self.pad_id = tokenizer.pad_id

    def _records(self):
        ds = load_dataset(
            "HuggingFaceH4/ultrachat_200k", split=self.split, streaming=self.streaming
        )
        count = 0
        for rec in ds:  # type: ignore[assignment]
            msgs = rec.get("messages") or rec.get("conversations")
            if not isinstance(msgs, list):
                continue
            pending_user: Optional[str] = None
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or m.get("from") or "").lower()
                text = m.get("content") or m.get("value") or m.get("text") or ""
                if not isinstance(text, str) or not text.strip():
                    continue
                if role in ("user", "human", "prompt"):
                    pending_user = text
                elif role in ("assistant", "gpt", "model", "bot") and pending_user is not None:
                    yield pending_user, text
                    pending_user = None
                    count += 1
                    if self.max_samples and count >= self.max_samples:
                        return

    def __iter__(self) -> Iterator[Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]:  # type: ignore[override]
        for user_text, assistant_text in self._records():
            text = f"<|user|> {user_text}\n<|assistant|> {assistant_text}\n"
            ids: List[int] = self.tok.encode(text, add_bos=True, add_eos=True)
            a_idx = _find_subseq_idx(ids, self.assist_tok_ids)
            if a_idx is None:
                continue
            a_len = max(1, len(self.assist_tok_ids))
            limit = self.seq_len + 1
            if len(ids) > limit:
                start = max(0, min(len(ids) - limit, a_idx))
                ids = ids[start : start + limit]
                a_idx = a_idx - start
                if a_idx < 0 or a_idx + a_len > len(ids):
                    continue
            elif len(ids) < limit:
                ids = ids + [self.pad_id] * (limit - len(ids))
            x_ids = torch.tensor(ids[:-1], dtype=torch.long)
            y_ids = torch.tensor(ids[1:], dtype=torch.long)
            mask = torch.zeros_like(y_ids, dtype=torch.bool)
            mask_start = max(0, a_idx + a_len - 1)
            if mask_start < mask.numel():
                mask[mask_start:] = True
            else:
                continue
            yield x_ids, y_ids, mask


def _find_all(hay: List[int], needle: List[int]) -> List[int]:
    if not needle:
        return []
    n = len(needle)
    out: List[int] = []
    for i in range(0, len(hay) - n + 1):
        if hay[i : i + n] == needle:
            out.append(i)
    return out


class UltraChatConversationStream(
    IterableDataset[Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]
):
    """Stream full conversations; loss on all assistant messages; memory persists across pairs."""

    def __init__(
        self,
        tokenizer: TokenizerAdapter,
        *,
        seq_len: int,
        split: str = "train_gen",
        streaming: bool = True,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if load_dataset is None:
            raise RuntimeError("datasets is not installed. Please install 'datasets'.")
        self.tok = tokenizer
        self.seq_len = int(seq_len)
        self.split = split
        self.streaming = streaming
        self.max_samples = max_samples
        _set_seed(seed)
        self.user_tok_ids = tokenizer.encode("<|user|> ", add_bos=False, add_eos=False)
        self.assist_tok_ids = tokenizer.encode("<|assistant|> ", add_bos=False, add_eos=False)
        self.pad_id = tokenizer.pad_id

    def _records(self):
        ds = load_dataset(
            "HuggingFaceH4/ultrachat_200k", split=self.split, streaming=self.streaming
        )
        count = 0
        for rec in ds:  # type: ignore[assignment]
            msgs = rec.get("messages") or rec.get("conversations")
            if not isinstance(msgs, list):
                continue
            parts: List[str] = []
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or m.get("from") or "").lower()
                text = m.get("content") or m.get("value") or m.get("text") or ""
                if not isinstance(text, str) or not text.strip():
                    continue
                if role in ("user", "human", "prompt"):
                    parts.append(f"<|user|> {text}\n")
                elif role in ("assistant", "gpt", "model", "bot"):
                    parts.append(f"<|assistant|> {text}\n")
            if not parts:
                continue
            yield "".join(parts)
            count += 1
            if self.max_samples and count >= self.max_samples:
                return

    def __iter__(self) -> Iterator[Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]:  # type: ignore[override]
        for text in self._records():
            ids: List[int] = self.tok.encode(text, add_bos=True, add_eos=True)
            limit = self.seq_len + 1
            if len(ids) > limit:
                ids = ids[:limit]
            elif len(ids) < limit:
                ids = ids + [self.pad_id] * (limit - len(ids))

            # Role marker positions on current truncated sequence
            assist_pos = _find_all(ids, self.assist_tok_ids)
            a_len = max(1, len(self.assist_tok_ids))

            x_ids = torch.tensor(ids[:-1], dtype=torch.long)
            y_ids = torch.tensor(ids[1:], dtype=torch.long)
            mask = torch.zeros_like(y_ids, dtype=torch.bool)

            # For each assistant marker, mask until the next role marker or EOS
            # We don't strictly need user positions for end boundaries; EOS or next role suffices.
            role_marks = sorted(assist_pos)
            for i, pos in enumerate(role_marks):
                start = pos + a_len - 1
                end_excl = (role_marks[i + 1] - 1) if (i + 1 < len(role_marks)) else (len(ids) - 1)
                if start < mask.numel() and end_excl > start:
                    lo = max(0, start)
                    hi = min(mask.numel(), end_excl)
                    if hi > lo:
                        mask[lo:hi] = True

            if mask.any():
                yield x_ids, y_ids, mask


class UltraChatPairSeqStream(
    IterableDataset[list[Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]]
):
    """Yield each conversation as a list of (x,y,mask) pairs (user→assistant), in order.

    Each pair contains only the current user/assistant tokens in the text context.
    Memory should be carried across pairs by the training loop using init_mem/mem_bank.
    """

    def __init__(
        self,
        tokenizer: TokenizerAdapter,
        *,
        seq_len: int,
        split: str = "train_gen",
        streaming: bool = True,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if load_dataset is None:
            raise RuntimeError("datasets is not installed. Please install 'datasets'.")
        self.tok = tokenizer
        self.seq_len = int(seq_len)
        self.split = split
        self.streaming = streaming
        self.max_samples = max_samples
        _set_seed(seed)
        self.user_tok_ids = tokenizer.encode("<|user|> ", add_bos=False, add_eos=False)
        self.assist_tok_ids = tokenizer.encode("<|assistant|> ", add_bos=False, add_eos=False)
        # Robust single special token id for assistant marker
        _assist_ids = tokenizer.encode("<|assistant|>", add_bos=False, add_eos=False)
        self.assist_token_id: Optional[int] = _assist_ids[0] if len(_assist_ids) >= 1 else None
        # Optional fallback variant with newline
        self._assist_ids_nl = tokenizer.encode("<|assistant|>\n", add_bos=False, add_eos=False)
        self.pad_id = tokenizer.pad_id

    def _pairs_from_msgs(self, msgs: list[dict]) -> list[Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]:
        pairs: list[Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]] = []
        pending_user: Optional[str] = None
        raw_pairs = 0
        drop_marker_miss = 0
        drop_windowing = 0
        drop_no_room = 0

        def _norm_role(v: str) -> str:
            v = (v or "").lower()
            if v in ("user", "human", "prompt", "prompter"):
                return "user"
            if v in ("assistant", "gpt", "model", "bot", "assistant_selected"):
                return "assistant"
            return v

        def _extract_text_field(obj: Any) -> str:
            # Accept str, list of str/dicts, or dict with common keys
            if isinstance(obj, str):
                return obj
            if isinstance(obj, list):
                parts: List[str] = []
                for it in obj:
                    if isinstance(it, str):
                        parts.append(it)
                    elif isinstance(it, dict):
                        for k in ("text", "content", "value"):
                            val = it.get(k)
                            if isinstance(val, str) and val.strip():
                                parts.append(val)
                                break
                return "\n".join(p for p in parts if p.strip())
            if isinstance(obj, dict):
                for k in ("text", "content", "value"):
                    val = obj.get(k)
                    if isinstance(val, str) and val.strip():
                        return val
                    if isinstance(val, list):
                        joined = _extract_text_field(val)
                        if joined:
                            return joined
            return ""

        for m in msgs:
            if not isinstance(m, dict):
                continue
            role_raw = m.get("role") or m.get("from") or m.get("speaker") or ""
            role = _norm_role(str(role_raw))
            raw = m.get("content")
            text = _extract_text_field(raw)
            if not text:
                # try fallback keys
                text = _extract_text_field(m.get("value")) or _extract_text_field(m.get("text"))
            if not isinstance(text, str) or not text.strip():
                continue
            if role == "user":
                pending_user = text
            elif role == "assistant" and pending_user is not None:
                raw_pairs += 1
                text_pair = f"<|user|> {pending_user}\n<|assistant|> {text}\n"
                ids: List[int] = self.tok.encode(text_pair, add_bos=True, add_eos=True)
                # Robust assistant marker matching
                a_idx: Optional[int] = None
                a_len: int = 1
                if self.assist_token_id is not None:
                    try:
                        a_idx = ids.index(self.assist_token_id)
                    except ValueError:
                        a_idx = None
                if a_idx is None:
                    # Optional fallbacks: try with space (legacy) and newline variants
                    for needle in (self.assist_tok_ids, self._assist_ids_nl):
                        if needle:
                            pos = _find_subseq_idx(ids, needle)
                            if pos is not None:
                                a_idx = pos
                                a_len = max(1, len(needle))
                                break
                if a_idx is None:
                    drop_marker_miss += 1
                    pending_user = None
                    continue
                limit = self.seq_len + 1
                if len(ids) > limit:
                    start = max(0, min(len(ids) - limit, a_idx))
                    ids = ids[start : start + limit]
                    a_idx = a_idx - start
                    if a_idx < 0 or a_idx + a_len > len(ids):
                        drop_windowing += 1
                        pending_user = None
                        continue
                elif len(ids) < limit:
                    ids = ids + [self.pad_id] * (limit - len(ids))
                x_ids = torch.tensor(ids[:-1], dtype=torch.long)
                y_ids = torch.tensor(ids[1:], dtype=torch.long)
                mask = torch.zeros_like(y_ids, dtype=torch.bool)
                mask_start = max(0, a_idx + a_len - 1)
                if mask_start < mask.numel():
                    mask[mask_start:] = True
                    pairs.append((x_ids, y_ids, mask))
                else:
                    drop_no_room += 1
                pending_user = None
        # Debug/log: raw pairs seen vs kept after truncation/mask
        try:
            logger.debug(
                "pairseq: raw_pairs=%d kept_pairs=%d msgs=%d drop_marker_miss=%d drop_window=%d drop_no_room=%d",
                raw_pairs,
                len(pairs),
                len(msgs),
                drop_marker_miss,
                drop_windowing,
                drop_no_room,
            )
        except Exception:
            pass
        return pairs

    def __iter__(self):  # type: ignore[override]
        ds = load_dataset(
            "HuggingFaceH4/ultrachat_200k", split=self.split, streaming=self.streaming
        )
        count = 0
        for rec in ds:  # type: ignore[assignment]
            msgs = rec.get("messages") or rec.get("conversations")
            if not isinstance(msgs, list):
                continue
            conv_pairs = self._pairs_from_msgs(msgs)
            if conv_pairs:
                yield conv_pairs
                count += 1
                if self.max_samples and count >= self.max_samples:
                    return


def build_ultrachat_pairseq_dataloader(
    tokenizer: TokenizerAdapter,
    *,
    seq_len: int,
    batch_size: int,
    num_workers: int,
    split: str,
    streaming: bool,
    max_samples: Optional[int],
):
    ds = UltraChatPairSeqStream(
        tokenizer,
        seq_len=seq_len,
        split=split,
        streaming=streaming,
        max_samples=max_samples,
    )

    def collate(batch):  # list of conversations; each is list of (x,y,mask)
        return batch  # pass through

    nw = num_workers
    if streaming:
        nw = max(0, min(int(num_workers), 1))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=nw, collate_fn=collate)


def _masked_ce(
    logits: torch.Tensor, targets: torch.LongTensor, mask: torch.BoolTensor
) -> torch.Tensor:
    # logits: [B, T, V], targets: [B, T], mask: [B, T]
    B, T, V = logits.shape
    logits = logits.reshape(B * T, V)
    targets = targets.reshape(B * T)
    mask = mask.reshape(B * T)
    idx = mask.nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return torch.zeros((), device=logits.device)
    return F.cross_entropy(logits.index_select(0, idx), targets.index_select(0, idx))


def _compute_sft_losses(
    out: dict,
    targets: torch.LongTensor,
    loss_mask: torch.BoolTensor,
    r_target: float,
) -> Tuple[torch.Tensor, dict]:
    """SFT loss on assistant tokens only (memory logits), plus a budget regularizer.

    Logs include ce_mem (nats), ppl_mem, sft_reward_nats (=-ce_mem), r_hat, loss_budget, n_mask_tokens.
    """
    logits_mem: torch.Tensor = out["logits_mem"][:, :-1, :]
    tgt: torch.LongTensor = targets[:, 1:]
    mask = loss_mask[:, :-1]

    ce_mem = _masked_ce(logits_mem, tgt, mask)

    p_gates: torch.Tensor = out["p_gates"][:, :-1]
    if mask.any():
        r_hat = (p_gates[mask]).mean()
    else:
        r_hat = p_gates.new_tensor(0.0)
    loss_budget = (r_hat - r_target) ** 2
    loss = ce_mem + loss_budget * 0.1  # small regularization by default

    ppl_mem = torch.exp(ce_mem)
    n_mask_tokens = int(mask.sum().item())
    logs = {
        "loss": float(loss.detach().cpu()),
        "ce_mem": float(ce_mem.detach().cpu()),
        "ppl_mem": float(ppl_mem.detach().cpu()),
        "sft_reward_nats": float((-ce_mem).detach().cpu()),
        "r_hat": float(r_hat.detach().cpu()),
        "loss_budget": float(loss_budget.detach().cpu()),
        "n_mask_tokens": float(n_mask_tokens),
    }
    return loss, logs


def _decode(tok: TokenizerAdapter, ids: List[int]) -> str:
    try:
        return tok.decode(ids)
    except Exception:
        # Best-effort fallback for byte tokenizer
        return " ".join(str(i) for i in ids)


@torch.no_grad()
def _eval_sft_pairseq_loop(
    model: ThoughtLM, dl: DataLoader, device: torch.device, max_batches: int
) -> dict:
    was_training = model.training
    model.eval()
    sum_nll = 0.0
    sum_tokens = 0.0
    sum_rhat_w = 0.0
    convs = 0
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        conversations = batch if isinstance(batch, list) else [batch]
        # Per-pair aggregates
        pair_sum_nll: dict[int, float] = {}
        pair_sum_tok: dict[int, float] = {}
        pair_sum_rhat: dict[int, float] = {}
        for conv in conversations:
            mem = None
            for p_idx, (x_cpu, y_cpu, mask_cpu) in enumerate(conv):
                x = x_cpu.to(device)
                y = y_cpu.to(device)
                mask = mask_cpu.to(device)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                    y = y.unsqueeze(0)
                    mask = mask.unsqueeze(0)
                out = model(x, init_mem=mem)
                logits_mem = out["logits_mem"][:, :-1, :]
                tgt = y[:, 1:]
                mk_y = mask[:, :-1]
                per_tok_nll = torch.nn.functional.cross_entropy(
                    logits_mem.transpose(1, 2), tgt, reduction="none"
                )
                nll_sel = per_tok_nll[mk_y]
                cnt = float(mk_y.sum().item())
                sum_nll += float(nll_sel.sum().item())
                sum_tokens += cnt
                if cnt > 0:
                    p = out["p_gates"][:, :-1]
                    r = float(p[mk_y].mean().item())
                    sum_rhat_w += r * cnt
                    # Pair-index aggregates
                    pair_sum_nll[p_idx] = pair_sum_nll.get(p_idx, 0.0) + float(nll_sel.sum().item())
                    pair_sum_tok[p_idx] = pair_sum_tok.get(p_idx, 0.0) + cnt
                    pair_sum_rhat[p_idx] = pair_sum_rhat.get(p_idx, 0.0) + r * cnt
                mem_t = out.get("mem_bank", None)
                mem = mem_t.detach() if isinstance(mem_t, torch.Tensor) else None
            convs += 1
    if was_training:
        model.train()
    if sum_tokens == 0:
        return {
            "ce_mem": 0.0,
            "ppl_mem": float("inf"),
            "sft_reward_nats": 0.0,
            "sft_quality": 0.0,
            "r_hat": 0.0,
        }
    ce_mem = sum_nll / sum_tokens
    ppl_mem = math.exp(ce_mem)
    sft_reward_nats = -ce_mem
    sft_quality = math.exp(-ce_mem)
    r_hat = (sum_rhat_w / sum_tokens) if sum_tokens > 0 else 0.0
    # Flatten per-pair metrics (limit to PAIR_METRICS_MAX)
    max_idx = min(PAIR_METRICS_MAX, (max(pair_sum_tok.keys()) + 1) if pair_sum_tok else 0)
    flat_pairs = {}
    for i in range(max_idx):
        tok_i = pair_sum_tok.get(i, 0.0)
        if tok_i > 0:
            ce_i = pair_sum_nll[i] / tok_i
            r_i = pair_sum_rhat[i] / tok_i
            flat_pairs[f"pair_ce_mem_{i}"] = float(ce_i)
            flat_pairs[f"pair_r_hat_{i}"] = float(r_i)

    # Pair summaries (first3/last3, slope)
    pair_first3 = pair_last3 = pair_delta = pair_slope = None
    if pair_sum_tok:
        idxs_all = sorted(pair_sum_tok.keys())
        ce_by_idx = {
            i: (pair_sum_nll[i] / pair_sum_tok[i]) for i in idxs_all if pair_sum_tok[i] > 0
        }

        def _wavg(idxs):
            w = sum(pair_sum_tok.get(i, 0.0) for i in idxs)
            if w <= 0:
                return None
            return sum(ce_by_idx[i] * pair_sum_tok.get(i, 0.0) for i in idxs if i in ce_by_idx) / w

        first_idxs = idxs_all[:3]
        last_idxs = idxs_all[-3:]
        f3 = _wavg(first_idxs)
        l3 = _wavg(last_idxs)
        if f3 is not None:
            pair_first3 = float(f3)
        if l3 is not None:
            pair_last3 = float(l3)
        if f3 is not None and l3 is not None:
            pair_delta = float(l3 - f3)
        W = sum(pair_sum_tok.get(i, 0.0) for i in ce_by_idx.keys())
        if W > 0:
            mx = sum(i * pair_sum_tok[i] for i in ce_by_idx.keys()) / W
            my = sum(ce_by_idx[i] * pair_sum_tok[i] for i in ce_by_idx.keys()) / W
            num = sum(pair_sum_tok[i] * (i - mx) * (ce_by_idx[i] - my) for i in ce_by_idx.keys())
            den = sum(pair_sum_tok[i] * (i - mx) * (i - mx) for i in ce_by_idx.keys()) + 1e-9
            pair_slope = float(num / den)

    out = {
        "ce_mem": float(ce_mem),
        "ppl_mem": float(ppl_mem),
        "sft_reward_nats": float(sft_reward_nats),
        "sft_quality": float(sft_quality),
        "r_hat": float(r_hat),
        **flat_pairs,
    }
    if pair_first3 is not None:
        out["pair_ce_mem_first3"] = pair_first3
    if pair_last3 is not None:
        out["pair_ce_mem_last3"] = pair_last3
    if pair_delta is not None:
        out["pair_ce_mem_delta_last_first"] = pair_delta
    if pair_slope is not None:
        out["pair_ce_mem_slope"] = pair_slope
    return out


def main() -> None:
    _setup_logging()
    # Avoid HF tokenizers multi-thread warning after DataLoader forks
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    disable_progress_bar()  # silence HF internal progress output
    cfg = _load_config_from_argv()
    _set_seed(cfg.seed)
    device = _select_device(cfg.run.device)

    logger.info(
        "Starting train_conv (SFT) | device=%s | precision=%s", device.type, cfg.run.precision
    )

    # TensorBoard writer
    writer: Optional[SummaryWriter] = None
    if cfg.run.enable_tb:
        log_dir = Path(cfg.run.tb_log_dir) / f"{cfg.run.run_name}_conv_sft"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        logger.info("TensorBoard: %s", str(log_dir))

    # Metrics file for this run
    metrics_path = Path(cfg.run.metrics_file)
    if metrics_path.name == "metrics.jsonl":
        metrics_path = metrics_path.with_name("metrics_conv_sft.jsonl")
    metrics = JSONLLogger(metrics_path)

    # Tokenizer (ensure role tokens exist)
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

    # Data: UltraChat turns (user->assistant). Only current turn as context; loss on assistant.
    split_from_cfg = (
        cfg.data.hf_dataset.split
        if (cfg.data.hf_dataset and cfg.data.hf_dataset.split)
        else "train_gen"
    )
    split = "train_gen" if split_from_cfg == "train" else split_from_cfg
    dl = build_ultrachat_pairseq_dataloader(
        tok,
        seq_len=cfg.data.seq_len,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        split=split,
        streaming=True,
        max_samples=cfg.data.hf_dataset.max_samples if cfg.data.hf_dataset else None,
    )
    # Build eval loader (default to test_gen if no override)
    eval_split = (
        cfg.eval.hf_split if cfg.eval.hf_split else ("test_gen" if split == "train_gen" else split)
    )
    dl_eval = build_ultrachat_pairseq_dataloader(
        tok,
        seq_len=cfg.data.seq_len,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        split=eval_split,
        streaming=True,
        max_samples=cfg.eval.max_batches,  # number of conversations to sample
    )
    logger.info(
        "Dataset: HuggingFaceH4/ultrachat_200k | train_split=%s | eval_split=%s | batch=%d | seq_len=%d (pair-seq, mem carry)",
        split,
        eval_split,
        cfg.data.batch_size,
        cfg.data.seq_len,
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
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model: ThoughtLM | params=%.2fM | d_model=%d | layers=%d | heads=%d | mem_dim=%d",
        n_params / 1e6,
        cfg.model.d_model,
        cfg.model.n_layers,
        cfg.model.n_heads,
        cfg.model.mem_dim,
    )

    # Optimizer & LR schedule
    opt = optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        weight_decay=cfg.optim.weight_decay,
    )
    sched = optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: min(1.0, (s + 1) / max(1, cfg.optim.warmup_steps))
    )

    # AMP scaler
    use_fp16 = device.type == "cuda" and cfg.run.precision == "fp16"
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

    # Training hyperparams (SFT only)
    sft_weight = 1.0

    model.train()
    step = 0
    window_tokens = 0
    window_time = 0.0
    pbar = (
        tqdm(total=int(cfg.optim.steps), desc="train_conv_sft", unit="step")
        if tqdm is not None
        else None
    )

    while step < cfg.optim.steps:
        for batch in dl:  # batch is List[List[(x,y,mask)]] with len=batch_size
            t0 = time.perf_counter()
            step += 1

            # Aggregate loss and metrics over all pairs across conversations in this batch
            total = None
            sum_nll = 0.0
            sum_tokens = 0.0
            sum_rhat_w = 0.0
            sum_budget = 0.0
            pairs_count = 0
            pred_tokens = 0

            # Handle both batch_size=1 and >1
            conversations = batch if isinstance(batch, list) else [batch]

            with torch.autocast(
                device_type=device.type,
                dtype=(
                    (torch.bfloat16 if cfg.run.precision == "bf16" else torch.float16)
                    if device.type == "cuda"
                    else torch.float32
                ),
            ):
                # Per-pair-index aggregations across the batch
                pair_sum_nll: dict[int, float] = {}
                pair_sum_tok: dict[int, float] = {}
                pair_sum_rhat: dict[int, float] = {}
                # Pairs-per-conversation stats
                pairs_counts: list[int] = []

                for conv in conversations:
                    mem = None
                    pairs_counts.append(len(conv))
                    for p_idx, (x_cpu, y_cpu, mask_cpu) in enumerate(conv):
                        x = x_cpu.to(device)
                        y = y_cpu.to(device)
                        mask = mask_cpu.to(device)
                        if x.dim() == 1:
                            x = x.unsqueeze(0)
                            y = y.unsqueeze(0)
                            mask = mask.unsqueeze(0)
                        out = model(x, init_mem=mem)
                        sft_loss, logs_pair = _compute_sft_losses(
                            out, y, mask, r_target=cfg.loss.r_target
                        )
                        if total is None:
                            total = sft_weight * sft_loss
                        else:
                            total = total + sft_weight * sft_loss
                        n_tok = float(logs_pair.get("n_mask_tokens", 0.0))
                        sum_nll += float(logs_pair["ce_mem"]) * n_tok
                        sum_tokens += n_tok
                        sum_rhat_w += float(logs_pair["r_hat"]) * n_tok
                        sum_budget += (
                            float(logs_pair["loss_budget"]) if "loss_budget" in logs_pair else 0.0
                        )
                        pred_tokens += int(n_tok)
                        pairs_count += 1
                        # Pair-index aggregates
                        if n_tok > 0:
                            pair_sum_nll[p_idx] = (
                                pair_sum_nll.get(p_idx, 0.0) + float(logs_pair["ce_mem"]) * n_tok
                            )
                            pair_sum_tok[p_idx] = pair_sum_tok.get(p_idx, 0.0) + n_tok
                            pair_sum_rhat[p_idx] = (
                                pair_sum_rhat.get(p_idx, 0.0) + float(logs_pair["r_hat"]) * n_tok
                            )
                        # Detach memory between pairs
                        mem_t = out.get("mem_bank", None)
                        mem = mem_t.detach() if isinstance(mem_t, torch.Tensor) else None
                if total is None:
                    # no valid pairs; skip
                    continue

            scaler.scale(total).backward()
            if hasattr(scaler, "unscale_"):
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()

            # Throughput
            t1 = time.perf_counter()
            step_time = t1 - t0
            window_time += step_time
            window_tokens += pred_tokens

            # Aggregate per-step metrics
            if sum_tokens > 0:
                ce_mem_avg = sum_nll / sum_tokens
                ppl_mem = float(math.exp(max(1e-8, ce_mem_avg)))
                r_hat_avg = sum_rhat_w / sum_tokens
            else:
                ce_mem_avg = 0.0
                ppl_mem = float("inf")
                r_hat_avg = 0.0
            # Pairs-per-conversation metrics
            convs_this_step = len(conversations)
            pairs_total = int(sum(pairs_counts)) if 'pairs_counts' in locals() and pairs_counts else 0
            if convs_this_step > 0 and 'pairs_counts' in locals() and pairs_counts:
                pairs_per_conv_mean = float(pairs_total / convs_this_step)
                pairs_per_conv_min = int(min(pairs_counts))
                pairs_per_conv_max = int(max(pairs_counts))
            else:
                pairs_per_conv_mean = 0.0
                pairs_per_conv_min = 0
                pairs_per_conv_max = 0
            # Small histogram for pairs per conv (1..PAIR_METRICS_MAX)
            pairs_hist = {k: 0 for k in range(1, PAIR_METRICS_MAX + 1)}
            if 'pairs_counts' in locals() and pairs_counts:
                for c in pairs_counts:
                    if 1 <= c <= PAIR_METRICS_MAX:
                        pairs_hist[c] += 1

            # Progress bar
            if pbar is not None:
                try:
                    pbar.update(1)
                    tok_s = float(pred_tokens / step_time) if step_time > 0 else float("nan")
                    pbar.set_postfix(
                        {
                            "loss": f"{float(total.detach().cpu()):.4f}",
                            "sft": f"{-ce_mem_avg:.4f}",
                            "tok/s": f"{tok_s:.0f}",
                        }
                    )
                except Exception:
                    pass

            # Logging
            if step % cfg.run.log_interval == 0 or step == 1:
                tok_s_window = (
                    float(window_tokens / window_time) if window_time > 0 else float("nan")
                )
                msg = (
                    f"SFT step={int(step)} total={float(total.detach().cpu()):.4f} "
                    f"sft_reward={float(-ce_mem_avg):.4f} ce_mem={float(ce_mem_avg):.4f} "
                    f"r_hat={float(r_hat_avg):.3f} ppl_mem={ppl_mem:.2f} tok/s={f'{tok_s_window:.0f}'}"
                )
                if pbar is not None:
                    try:
                        pbar.write(msg)
                    except Exception:
                        logger.info(msg)
                else:
                    logger.info(msg)

                sft_quality = (
                    float(math.exp(-ce_mem_avg)) if ce_mem_avg < 50 else 0.0
                )  # guard overflow
                # Build flattened per-pair metrics (limit to PAIR_METRICS_MAX)
                flat_pair_metrics = {}
                if "pair_sum_nll" in locals():
                    max_idx = min(
                        PAIR_METRICS_MAX, (max(pair_sum_tok.keys()) + 1) if pair_sum_tok else 0
                    )
                    for i in range(max_idx):
                        tok_i = pair_sum_tok.get(i, 0.0)
                        if tok_i > 0:
                            ce_i = pair_sum_nll[i] / tok_i
                            r_i = pair_sum_rhat[i] / tok_i
                            flat_pair_metrics[f"pair_ce_mem_{i}"] = float(ce_i)
                            flat_pair_metrics[f"pair_r_hat_{i}"] = float(r_i)
                # Pair summaries (first3/last3 and slope)
                pair_first3 = pair_last3 = pair_delta = pair_slope = None
                if "pair_sum_tok" in locals() and pair_sum_tok:
                    idxs_all = sorted(pair_sum_tok.keys())
                    # Compute per-index ce
                    ce_by_idx = {}
                    for i in idxs_all:
                        tok_i = pair_sum_tok.get(i, 0.0)
                        if tok_i > 0:
                            ce_by_idx[i] = pair_sum_nll[i] / tok_i

                    # First3 / Last3 weighted by tokens
                    def _wavg(idxs):
                        w = sum(pair_sum_tok.get(i, 0.0) for i in idxs)
                        if w <= 0:
                            return None
                        return (
                            sum(
                                ce_by_idx[i] * pair_sum_tok.get(i, 0.0)
                                for i in idxs
                                if i in ce_by_idx
                            )
                            / w
                        )

                    first_idxs = idxs_all[:3]
                    last_idxs = idxs_all[-3:]
                    f3 = _wavg(first_idxs)
                    l3 = _wavg(last_idxs)
                    if f3 is not None:
                        pair_first3 = float(f3)
                    if l3 is not None:
                        pair_last3 = float(l3)
                    if f3 is not None and l3 is not None:
                        pair_delta = float(l3 - f3)
                    # Weighted slope across indices
                    W = sum(pair_sum_tok.get(i, 0.0) for i in ce_by_idx.keys())
                    if W > 0:
                        mx = sum(i * pair_sum_tok[i] for i in ce_by_idx.keys()) / W
                        my = sum(ce_by_idx[i] * pair_sum_tok[i] for i in ce_by_idx.keys()) / W
                        num = sum(
                            pair_sum_tok[i] * (i - mx) * (ce_by_idx[i] - my)
                            for i in ce_by_idx.keys()
                        )
                        den = (
                            sum(pair_sum_tok[i] * (i - mx) * (i - mx) for i in ce_by_idx.keys())
                            + 1e-9
                        )
                        pair_slope = float(num / den)
                rec = {
                    "phase": "train_conv_sft",
                    "step": int(step),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "loss": float(total.detach().cpu()),
                    "ce_mem": float(ce_mem_avg),
                    "ppl_mem": float(ppl_mem),
                    "sft_reward_nats": float(-ce_mem_avg),
                    "r_hat": float(r_hat_avg),
                    "loss_budget": float(sum_budget / max(1, pairs_count)),
                    "sft_quality": sft_quality,
                    "tokens_per_s": tok_s_window,
                    "pairs_per_conv_mean": pairs_per_conv_mean,
                    "pairs_per_conv_min": float(pairs_per_conv_min),
                    "pairs_per_conv_max": float(pairs_per_conv_max),
                    "pairs_total": float(pairs_total),
                    "convs": float(convs_this_step),
                    **flat_pair_metrics,
                }
                # add histogram buckets
                for k, v in pairs_hist.items():
                    rec[f"pairs_hist_{k}"] = float(v)
                if pair_first3 is not None:
                    rec["pair_ce_mem_first3"] = pair_first3
                if pair_last3 is not None:
                    rec["pair_ce_mem_last3"] = pair_last3
                if pair_delta is not None:
                    rec["pair_ce_mem_delta_last_first"] = pair_delta
                if pair_slope is not None:
                    rec["pair_ce_mem_slope"] = pair_slope
                metrics.log(rec)
                if writer is not None:
                    writer.add_scalar("train_conv_sft/total", float(total.detach().cpu()), step)
                    writer.add_scalar("train_conv_sft/sft_reward", float(-ce_mem_avg), step)
                    writer.add_scalar("train_conv_sft/ce_mem", float(ce_mem_avg), step)
                    writer.add_scalar("train_conv_sft/ppl_mem", float(ppl_mem), step)
                    writer.add_scalar("train_conv_sft/sft_quality", sft_quality, step)
                    writer.add_scalar("train_conv_sft/r_hat", float(r_hat_avg), step)
                    # Pair-index TB scalars
                    if "pair_sum_nll" in locals():
                        max_idx = min(
                            PAIR_METRICS_MAX, (max(pair_sum_tok.keys()) + 1) if pair_sum_tok else 0
                        )
                        for i in range(max_idx):
                            tok_i = pair_sum_tok.get(i, 0.0)
                            if tok_i > 0:
                                ce_i = pair_sum_nll[i] / tok_i
                                r_i = pair_sum_rhat[i] / tok_i
                                writer.add_scalar(
                                    f"train_conv_sft/pair/ce_mem@{i}", float(ce_i), step
                                )
                                writer.add_scalar(
                                    f"train_conv_sft/pair/r_hat@{i}", float(r_i), step
                                )
                    # Pair summaries TB scalars
                    if pair_first3 is not None:
                        writer.add_scalar("train_conv_sft/pair/ce_first3", pair_first3, step)
                    if pair_last3 is not None:
                        writer.add_scalar("train_conv_sft/pair/ce_last3", pair_last3, step)
                    if pair_delta is not None:
                        writer.add_scalar(
                            "train_conv_sft/pair/ce_delta_last_first", pair_delta, step
                        )
                    if pair_slope is not None:
                        writer.add_scalar("train_conv_sft/pair/ce_slope", pair_slope, step)
                    # Pairs-per-conversation TB scalars
                    writer.add_scalar("train_conv_sft/pairs/per_conv_mean", pairs_per_conv_mean, step)
                    writer.add_scalar("train_conv_sft/pairs/per_conv_min", pairs_per_conv_min, step)
                    writer.add_scalar("train_conv_sft/pairs/per_conv_max", pairs_per_conv_max, step)
                    writer.add_scalar("train_conv_sft/pairs/total", pairs_total, step)
                    writer.add_scalar("train_conv_sft/pairs/convs", convs_this_step, step)
                    # small histogram buckets as separate scalars
                    for k, v in pairs_hist.items():
                        writer.add_scalar(f"train_conv_sft/pairs/hist@{k}", v, step)
                    writer.add_scalar("train_conv_sft/tokens_per_s", tok_s_window, step)
                    writer.add_scalar("train_conv_sft/lr", float(opt.param_groups[0]["lr"]), step)
                    try:
                        writer.flush()
                    except Exception:
                        pass
                window_time = 0.0
                window_tokens = 0

            # Periodic evaluation (SFT-only metrics)
            if cfg.eval.enabled and (step % cfg.eval.every == 0):
                eval_stats = _eval_sft_pairseq_loop(model, dl_eval, device, cfg.eval.max_batches)
                emsg = (
                    f"EVAL SFT step={int(step)} ce_mem={eval_stats['ce_mem']:.4f} ppl_mem={eval_stats['ppl_mem']:.2f} "
                    f"sft_reward={eval_stats['sft_reward_nats']:.4f} sft_quality={eval_stats['sft_quality']:.4f} r_hat={eval_stats['r_hat']:.3f}"
                )
                if pbar is not None:
                    try:
                        pbar.write(emsg)
                    except Exception:
                        logger.info(emsg)
                else:
                    logger.info(emsg)
                metrics.log(
                    {
                        "phase": "eval_conv_sft",
                        "step": int(step),
                        **{k: float(v) for k, v in eval_stats.items()},
                    }
                )
                if writer is not None:
                    writer.add_scalar("eval_conv_sft/ce_mem", eval_stats["ce_mem"], step)
                    writer.add_scalar("eval_conv_sft/ppl_mem", eval_stats["ppl_mem"], step)
                    writer.add_scalar(
                        "eval_conv_sft/sft_reward", eval_stats["sft_reward_nats"], step
                    )
                    writer.add_scalar("eval_conv_sft/sft_quality", eval_stats["sft_quality"], step)
                    writer.add_scalar("eval_conv_sft/r_hat", eval_stats["r_hat"], step)
                    # Per-pair eval TB scalars
                    for i in range(PAIR_METRICS_MAX):
                        k_ce = f"pair_ce_mem_{i}"
                        k_r = f"pair_r_hat_{i}"
                        if k_ce in eval_stats:
                            writer.add_scalar(
                                f"eval_conv_sft/pair/ce_mem@{i}", eval_stats[k_ce], step
                            )
                        if k_r in eval_stats:
                            writer.add_scalar(
                                f"eval_conv_sft/pair/r_hat@{i}", eval_stats[k_r], step
                            )
                    # Pair summaries
                    for k in (
                        "pair_ce_mem_first3",
                        "pair_ce_mem_last3",
                        "pair_ce_mem_delta_last_first",
                        "pair_ce_mem_slope",
                    ):
                        if k in eval_stats:
                            tb_name = k.replace("pair_ce_mem_", "eval_conv_sft/pair/ce_")
                            writer.add_scalar(tb_name, float(eval_stats[k]), step)
                    try:
                        writer.flush()
                    except Exception:
                        pass

            if step >= cfg.optim.steps:
                break

    if pbar is not None:
        try:
            pbar.close()
        except Exception:
            pass
    metrics.close()
    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
