from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, IterableDataset

from .tokenization import TokenizerAdapter
from .data import PackedBatch

try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore


def _buffered_shuffle(it: "Iterable", buf_size: int):
    buf: List = []
    for x in it:
        buf.append(x)
        if len(buf) >= buf_size:
            random.shuffle(buf)
            while buf:
                yield buf.pop()
    random.shuffle(buf)
    while buf:
        yield buf.pop()


def _normalize_role(v: str) -> str:
    v = (v or "").lower()
    if v in ("human", "user", "prompt"):
        return "user"
    if v in ("assistant", "gpt", "model", "bot"):
        return "assistant"
    return v or "user"


def _extract_turns(rec: dict) -> Optional[List[Tuple[str, str]]]:
    # Try common keys used by HF chat datasets
    for key in ("messages", "conversations", "dialogue", "dialog"):
        seq = rec.get(key)
        if isinstance(seq, list):
            turns: List[Tuple[str, str]] = []
            for t in seq:
                if not isinstance(t, dict):
                    continue
                role = t.get("role")
                if role is None:
                    role = t.get("from")
                text = (
                    t.get("content")
                    or t.get("value")
                    or t.get("text")
                    or t.get("utterance")
                )
                if isinstance(text, str) and text.strip():
                    turns.append((_normalize_role(str(role or "user")), text))
            return turns if len(turns) >= 2 else None
    return None


class HFConversationStream(IterableDataset[Tuple[torch.LongTensor, torch.LongTensor]]):
    """Stream multi-turn conversations from a HF dataset and pack to fixed length.

    No explicit context separator is inserted; we only include role tokens if provided
    in the tokenizer (e.g., <|user|>, <|assistant|>). Each dialogue becomes plain text:
        <|user|> ...\n<|assistant|> ...\n...
    """

    def __init__(
        self,
        name: str,
        split: str,
        tokenizer: TokenizerAdapter,
        seq_len: int,
        streaming: bool = True,
        shuffle_buffer: int = 10000,
        max_samples: Optional[int] = None,
        filter_long: bool = True,
        seed: int = 42,
        user_token: str = "<|user|>",
        assistant_token: str = "<|assistant|>",
    ) -> None:
        super().__init__()
        if load_dataset is None:
            raise RuntimeError("datasets is not installed. Please install 'datasets'.")
        self.name = name
        self.split = split
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.streaming = streaming
        self.shuffle_buffer = int(shuffle_buffer)
        self.max_samples = max_samples
        self.filter_long = filter_long
        self.user_token = user_token
        self.assistant_token = assistant_token
        random.seed(seed)
        torch.manual_seed(seed)

    def __iter__(self):  # type: ignore[override]
        ds = load_dataset(self.name, split=self.split, streaming=self.streaming)
        it = (rec for rec in ds)
        if self.shuffle_buffer and self.shuffle_buffer > 0:
            it = _buffered_shuffle(it, self.shuffle_buffer)
        count = 0
        for rec in it:
            turns = _extract_turns(rec)
            if not turns:
                continue
            # Build simple dialogue text without explicit separators
            parts: List[str] = []
            for role, text in turns:
                tok = self.user_token if role == "user" else self.assistant_token
                parts.append(f"{tok} {text}\n")
            text = "".join(parts)
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            if self.filter_long and len(ids) > self.seq_len + 1:
                continue
            if len(ids) < self.seq_len + 1:
                ids = ids + [self.tokenizer.pad_id] * (self.seq_len + 1 - len(ids))
            else:
                ids = ids[: self.seq_len + 1]
            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            yield x, y
            count += 1
            if self.max_samples and count >= self.max_samples:
                break


def build_hf_conv_dataloader(
    *,
    name: str,
    split: str,
    tokenizer: TokenizerAdapter,
    seq_len: int,
    batch_size: int,
    num_workers: int,
    streaming: bool = True,
    shuffle_buffer: int = 10000,
    max_samples: Optional[int] = None,
    filter_long: bool = True,
) -> DataLoader[PackedBatch]:
    ds = HFConversationStream(
        name=name,
        split=split,
        tokenizer=tokenizer,
        seq_len=seq_len,
        streaming=streaming,
        shuffle_buffer=shuffle_buffer,
        max_samples=max_samples,
        filter_long=filter_long,
    )

    def collate(batch: Sequence[Tuple[torch.LongTensor, torch.LongTensor]]) -> PackedBatch:
        xs, ys = zip(*batch)
        return PackedBatch(input_ids=torch.stack(list(xs), dim=0), labels=torch.stack(list(ys), dim=0))

    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

