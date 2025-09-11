from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class ByteVocab:
    """Byte-level vocabulary with 4 specials.

    ids:
      0: <pad>
      1: <bos>
      2: <eos>
      3: <nop>
      4..259: raw bytes 0..255
    """

    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    nop_id: int = 3

    @property
    def size(self) -> int:
        return 260


class ByteTokenizer:
    """Byte-level tokenizer with deterministic mapping.

    This keeps dependencies minimal for prototyping.
    """

    def __init__(self) -> None:
        self.vocab = ByteVocab()

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.vocab.bos_id)
        for b in text.encode("utf-8", errors="replace"):
            ids.append(4 + int(b))
        if add_eos:
            ids.append(self.vocab.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        bytes_out = bytearray()
        for i in ids:
            if i >= 4:
                bytes_out.append(i - 4)
        return bytes_out.decode("utf-8", errors="replace")

