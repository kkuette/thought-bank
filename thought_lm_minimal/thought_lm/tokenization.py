from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore

from .tokenizer import ByteTokenizer


class TokenizerAdapter:
    """Unified tokenizer interface for byte-level and HF tokenizers."""

    def __init__(self, kind: str, byte_tok: Optional[ByteTokenizer] = None, hf_tok=None) -> None:
        self.kind = kind
        self._byte = byte_tok
        self._hf = hf_tok

    @property
    def pad_id(self) -> int:
        if self.kind == "byte":
            return self._byte.vocab.pad_id  # type: ignore[union-attr]
        assert self._hf is not None
        return int(self._hf.pad_token_id)

    @property
    def bos_id(self) -> Optional[int]:
        if self.kind == "byte":
            return self._byte.vocab.bos_id  # type: ignore[union-attr]
        assert self._hf is not None
        return int(self._hf.bos_token_id) if self._hf.bos_token_id is not None else None

    @property
    def eos_id(self) -> Optional[int]:
        if self.kind == "byte":
            return self._byte.vocab.eos_id  # type: ignore[union-attr]
        assert self._hf is not None
        return int(self._hf.eos_token_id) if self._hf.eos_token_id is not None else None

    @property
    def vocab_size(self) -> int:
        if self.kind == "byte":
            return self._byte.vocab.size  # type: ignore[union-attr]
        assert self._hf is not None
        # For HF tokenizers, vocab_size may exclude newly added special tokens (e.g., a pad token).
        # len(tokenizer) reflects the full size including added tokens, which is what the embedding needs.
        return int(len(self._hf))

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        if self.kind == "byte":
            return self._byte.encode(text, add_bos=add_bos, add_eos=add_eos)  # type: ignore[union-attr]
        assert self._hf is not None
        ids: List[int] = self._hf.encode(text, add_special_tokens=False)
        if add_bos and self.bos_id is not None:
            ids = [int(self.bos_id)] + ids
        if add_eos and self.eos_id is not None:
            ids = ids + [int(self.eos_id)]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        if self.kind == "byte":
            return self._byte.decode(ids)  # type: ignore[union-attr]
        assert self._hf is not None
        return self._hf.decode(list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=False)


@dataclass
class HFBuildArgs:
    name: str
    trust_remote_code: bool
    add_pad_if_missing: bool


def build_tokenizer(kind: str, hf_args: Optional[HFBuildArgs] = None, extra_special_tokens: Optional[List[str]] = None) -> TokenizerAdapter:
    if kind == "byte":
        return TokenizerAdapter(kind="byte", byte_tok=ByteTokenizer())
    if kind == "hf":
        if AutoTokenizer is None:
            raise RuntimeError("transformers is not installed. Please install 'transformers'.")
        assert hf_args is not None
        tok = AutoTokenizer.from_pretrained(hf_args.name, use_fast=True, trust_remote_code=hf_args.trust_remote_code)
        # Ensure pad token exists
        if tok.pad_token_id is None:
            if hf_args.add_pad_if_missing:
                try:
                    tok.add_special_tokens({"pad_token": "<|pad|>"})
                    tok.pad_token = "<|pad|>"
                except Exception:
                    # Fallback: reuse eos as pad
                    if tok.eos_token is not None:
                        tok.pad_token = tok.eos_token
                    else:
                        raise RuntimeError("Tokenizer has no pad/eos token and cannot add one.")
            else:
                # Use eos as pad if present
                if tok.eos_token is not None:
                    tok.pad_token = tok.eos_token
                else:
                    raise RuntimeError("Tokenizer has no pad token and add_pad_if_missing is False.")
        # Add requested additional special tokens (e.g., roles and separators)
        if extra_special_tokens:
            try:
                tok.add_special_tokens({"additional_special_tokens": list(extra_special_tokens)})
            except Exception:
                pass
        return TokenizerAdapter(kind="hf", hf_tok=tok)
    raise ValueError(f"Unknown tokenizer kind: {kind}")

