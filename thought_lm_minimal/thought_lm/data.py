from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .tokenization import TokenizerAdapter


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class PackedBatch:
    input_ids: torch.LongTensor  # [B, T]
    labels: torch.LongTensor  # [B, T]


class CodeTextDataset(Dataset[Tuple[torch.LongTensor, torch.LongTensor]]):
    """Packs code text into fixed-length LM training sequences.

    If train_dir is None or empty, generates synthetic code snippets.
    """

    def __init__(
        self,
        tokenizer: TokenizerAdapter,
        seq_len: int,
        train_dir: Optional[str] = None,
        synthetic_samples: int = 5000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        set_seed(seed)
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.samples: List[List[int]] = []

        if train_dir and any(Path(train_dir).glob("**/*.txt")):
            texts = _read_text_files(train_dir)
        else:
            texts = _generate_synthetic_code(synthetic_samples)

        for txt in texts:
            ids = tokenizer.encode(txt, add_bos=True, add_eos=True)
            # chunk into sequences of length seq_len+1 to build next-token labels
            for i in range(0, max(0, len(ids) - 1), self.seq_len):
                chunk = ids[i : i + self.seq_len + 1]
                if len(chunk) < self.seq_len + 1:
                    pad = [tokenizer.pad_id] * (self.seq_len + 1 - len(chunk))
                    chunk = chunk + pad
                self.samples.append(chunk)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        item = self.samples[idx]
        x = torch.tensor(item[:-1], dtype=torch.long)
        y = torch.tensor(item[1:], dtype=torch.long)
        return x, y


def _read_text_files(train_dir: str) -> List[str]:
    paths = list(Path(train_dir).glob("**/*.txt"))
    texts: List[str] = []
    for p in paths:
        try:
            texts.append(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return texts


# -------------------- HF streaming (the-stack-smol) --------------------

try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - optional dependency
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


class HFCodeStream(IterableDataset[Tuple[torch.LongTensor, torch.LongTensor]]):
    """Stream code samples from an HF dataset, filtering long sequences.

    - Filters by language if a language field exists.
    - Drops samples whose tokenized length > seq_len+1 when filter_long is True.
    - Pads shorter sequences to seq_len+1, then yields (x, y) of length seq_len.
    """

    def __init__(
        self,
        name: str,
        split: str,
        text_field: str,
        tokenizer: TokenizerAdapter,
        seq_len: int,
        languages: Optional[List[str]] = None,
        streaming: bool = True,
        shuffle_buffer: int = 10000,
        max_samples: Optional[int] = None,
        filter_long: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.name = name
        self.split = split
        self.text_field = text_field
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.languages = [l.lower() for l in languages] if languages else None
        self.streaming = streaming
        self.shuffle_buffer = int(shuffle_buffer)
        self.max_samples = max_samples
        self.filter_long = filter_long
        set_seed(seed)

    def _ok_lang(self, rec: dict) -> bool:
        if self.languages is None:
            return True
        # Try common keys
        for key in ("language", "lang", "programming_language"):
            if key in rec:
                val = rec[key]
                if isinstance(val, list):
                    return any(str(v).lower() in self.languages for v in val)
                return str(val).lower() in self.languages
        # If no language info, drop
        return False

    def _records(self):
        if load_dataset is None:
            raise RuntimeError("datasets is not installed. Please install 'datasets'.")
        ds = load_dataset(self.name, split=self.split, streaming=self.streaming)
        it = (rec for rec in ds if self._ok_lang(rec))
        if self.shuffle_buffer and self.shuffle_buffer > 0:
            it = _buffered_shuffle(it, self.shuffle_buffer)
        count = 0
        for rec in it:
            txt = rec.get(self.text_field, None)
            if isinstance(txt, str):
                yield txt
                count += 1
                if self.max_samples and count >= self.max_samples:
                    break

    def __iter__(self):  # type: ignore[override]
        pad_id = self.tokenizer.pad_id
        for txt in self._records():
            ids = self.tokenizer.encode(txt, add_bos=True, add_eos=True)
            # Filter long sequences if requested
            if self.filter_long and len(ids) > self.seq_len + 1:
                continue
            # If shorter, pad to fixed length; if longer and not filtering, truncate
            if len(ids) < self.seq_len + 1:
                ids = ids + [pad_id] * (self.seq_len + 1 - len(ids))
            else:
                ids = ids[: self.seq_len + 1]
            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            yield x, y


def build_hf_stream_dataloader(
    name: str,
    split: str,
    text_field: str,
    tokenizer: TokenizerAdapter,
    seq_len: int,
    batch_size: int,
    num_workers: int,
    languages: Optional[List[str]] = None,
    streaming: bool = True,
    shuffle_buffer: int = 10000,
    max_samples: Optional[int] = None,
    filter_long: bool = True,
) -> DataLoader[PackedBatch]:
    ds = HFCodeStream(
        name=name,
        split=split,
        text_field=text_field,
        tokenizer=tokenizer,
        seq_len=seq_len,
        languages=languages,
        streaming=streaming,
        shuffle_buffer=shuffle_buffer,
        max_samples=max_samples,
        filter_long=filter_long,
    )
    return build_dataloader(ds, batch_size=batch_size, num_workers=num_workers)


def _generate_synthetic_code(n: int) -> List[str]:
    """Create tiny Python-like snippets to exercise structure tokens.

    This is intentionally simple and deterministic enough to train quickly.
    """
    texts: List[str] = []
    for i in range(n):
        fname = f"func_{i%100}"
        arg = f"x{i%7}"
        body = "\n    ".join(
            [
                f"res = 0",
                f"for i in range({(i%5)+1}):",
                f"    res += {i%7} * i",
                f"if res % 2 == 0:",
                f"    return res",
                f"return res - 1",
            ]
        )
        text = f"def {fname}({arg}: int) -> int:\n    {body}\n"
        texts.append(text)
    return texts


def build_dataloader(
    dataset: Dataset[Tuple[torch.LongTensor, torch.LongTensor]] | IterableDataset[Tuple[torch.LongTensor, torch.LongTensor]],
    batch_size: int,
    num_workers: int,
) -> DataLoader[PackedBatch]:
    def collate(batch: Sequence[Tuple[torch.LongTensor, torch.LongTensor]]) -> PackedBatch:
        xs, ys = zip(*batch)
        x = torch.stack(list(xs), dim=0)
        y = torch.stack(list(ys), dim=0)
        return PackedBatch(x, y)

    is_iterable = isinstance(dataset, IterableDataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not is_iterable),
        num_workers=num_workers,
        collate_fn=collate,
    )

