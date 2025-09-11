"""
Tokenizer utilities: ensure special tokens exist and return their IDs.

Functions are small, typed, and follow PEP8/PEP257.
"""
from __future__ import annotations

from typing import Tuple

from transformers import PreTrainedTokenizerBase


def ensure_think_token(
    tokenizer: PreTrainedTokenizerBase,
    think_token_text: str = "<think>",
) -> Tuple[PreTrainedTokenizerBase, int, bool]:
    """Ensure a special think token exists in the tokenizer.

    If absent, add it as an additional special token. Returns the tokenizer
    (possibly mutated), the token id, and a boolean indicating if the token was added.
    """
    # Check if token already exists
    token_id = tokenizer.convert_tokens_to_ids(think_token_text)
    if token_id is not None and token_id != tokenizer.unk_token_id and token_id >= 0:
        return tokenizer, int(token_id), False

    # Add as additional special token
    added = tokenizer.add_special_tokens({
        "additional_special_tokens": [think_token_text]
    })
    token_id = tokenizer.convert_tokens_to_ids(think_token_text)
    if token_id is None or token_id < 0:
        # Fallback: try regular add_tokens if special failed
        tokenizer.add_tokens([think_token_text])
        token_id = tokenizer.convert_tokens_to_ids(think_token_text)
        if token_id is None or token_id < 0:
            raise ValueError(f"Failed to register think token: {think_token_text}")
    return tokenizer, int(token_id), added > 0

