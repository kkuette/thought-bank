"""
TRL wrapper for ARThoughtModel providing a Transformers-compatible interface for SFT.

- Computes supervised next-token loss over sequences by unrolling AR steps
- Provides generate() for evaluation
- Keeps files < 400 LOC and functions < 100 LOC
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

try:
    from .ar_thought_model import ARThoughtModel, ARConfig
except ImportError:  # pragma: no cover
    from ar_thought_model import ARThoughtModel, ARConfig


class ARHFConfig(PretrainedConfig):
    model_type = "ar_thought"

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 2048,
        thought_dim: int = 128,
        max_seq_len: int = 2048,
        max_thoughts: int = 64,
        gradient_checkpointing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.thought_dim = thought_dim
        self.max_seq_len = max_seq_len
        self.max_thoughts = max_thoughts
        self.gradient_checkpointing = gradient_checkpointing


class ARForCausalLM(PreTrainedModel):
    """HF-compatible wrapper for ARThoughtModel with SFT loss.

    Forward contract:
    - If labels provided, returns dict with `loss` (scalar). Labels should be
      shaped (B, T) with -100 for ignored positions (standard Trainer behavior).
    - Optionally returns last-position logits shaped (B, 1, vocab) for logging.
    """

    config_class = ARHFConfig
    supports_gradient_checkpointing = True  # allow Trainer to enable/disable GC

    def __init__(self, config: ARHFConfig) -> None:
        super().__init__(config)
        cfg = ARConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            thought_dim=config.thought_dim,
            max_thoughts=config.max_thoughts,
            tie_weights=True,
            gradient_checkpointing=getattr(config, "gradient_checkpointing", False),
        )
        self.base = ARThoughtModel(cfg)

        # Ensure Transformers is aware of our tied weights when saving.
        # Our token head reuses the input embedding weights via F.linear with
        # base.token_embed.weight. Register both parameter names as tied so
        # safetensors can drop the duplicate safely at save time.
        tied_patterns = [
            r"^base\.token_embed\.weight$",
            r"^base\.token_head\.token_embed\.weight$",
        ]
        # Static list of tied keys (used by HF on save)
        self._tied_weights_keys = getattr(self, "_tied_weights_keys", []) + tied_patterns
        # Dynamic fallback (newer HF error message recommends this attribute)
        self._dynamic_tied_weights_keys = getattr(self, "_dynamic_tied_weights_keys", []) + tied_patterns
        # Hint in config that word embeddings are tied
        if hasattr(self.config.get_text_config(), "tie_word_embeddings"):
            self.config.get_text_config().tie_word_embeddings = True

        self._init_weights()

    # --- Gradient checkpointing control hooks expected by HF Trainer ---
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None) -> None:  # noqa: D401
        """Enable gradient checkpointing in the underlying AR model.

        HF's Trainer calls this to toggle checkpointing at runtime. Our base model
        reads a simple boolean flag from its config inside forward().
        """
        self.base.cfg.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:  # noqa: D401
        """Disable gradient checkpointing in the underlying AR model."""
        self.base.cfg.gradient_checkpointing = False

    def _init_weights(self) -> None:
        # Use base model's init; nothing extra here
        pass

    def get_input_embeddings(self) -> nn.Embedding:  # noqa: D401
        return self.base.token_embed

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:  # noqa: D401
        self.base.token_embed = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        thought_memory: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        """Compute SFT loss by unrolling AR steps over the input sequence.

        We compute next-token loss only where labels != -100. For step t, the model
        consumes tokens[:, :t] and is trained to predict tokens[:, t]. Thought
        memory is updated per step.
        """
        B, T = input_ids.shape
        device = input_ids.device
        mem = thought_memory
        if mem is None:
            mem = self.base.memory.initialize(B, device)

        # If no labels, produce last-position logits for logging/greedy gen
        if labels is None:
            out = self.base(input_ids, mem, update_memory=False)
            return {"logits": out["logits"].unsqueeze(1)}

        loss_accum = 0.0
        steps = 0
        for t in range(1, T):
            target = labels[:, t]
            mask = target != -100
            if not mask.any():
                continue
            out = self.base(input_ids[:, :t], mem, update_memory=True)
            logits = out["logits"][mask]
            step_loss = F.cross_entropy(logits, target[mask])
            loss_accum = loss_accum + step_loss
            steps += 1
            mem = out["updated_memory"]

        loss = loss_accum / max(steps, 1)
        # Also return a dummy last-position logits for logging convenience
        out_last = self.base(input_ids, mem, update_memory=False)
        return {"loss": loss, "logits": out_last["logits"].unsqueeze(1)}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        think_token_id: Optional[int] = None,
        max_new_thoughts: int = 64,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        thought_memory: Optional[torch.Tensor] = None,
        eos_token_id: Optional[int] = None,
        include_think_tokens_in_output: bool = True,
        think_logit_bias: float = 0.0,
        min_thoughts_first: int = 0,
    ) -> torch.Tensor:
        """Interleaved thinking/output decoding controlled by a special think token.

        The loop maintains two budgets:
        - thought budget (max_new_thoughts): maximum number of think tokens allowed
        - output budget (max_new_tokens): maximum number of non-think tokens to emit

        Generation stops if:
        - eos_token_id is produced (for all batch elements), or
        - output budget is exhausted, or
        - model tries to produce a think token after the thought budget is exhausted

        Think tokens are appended to maintain context; you can drop them from the
        returned sequence by setting include_think_tokens_in_output=False.
        """
        device = input_ids.device
        B = input_ids.size(0)
        mem = thought_memory or self.base.memory.initialize(B, device)
        tokens = input_ids

        def apply_think_bias(logits: torch.Tensor) -> torch.Tensor:
            if think_token_id is not None and think_logit_bias != 0.0:
                logits = logits.clone()
                logits[:, think_token_id] = logits[:, think_token_id] + think_logit_bias
            return logits

        def sample_next(logits: torch.Tensor) -> torch.Tensor:
            logits = apply_think_bias(logits) / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                topk_vals, topk_idx = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                probs = F.softmax(topk_vals, dim=-1)
                next_local = torch.multinomial(probs, 1).squeeze(-1)
                return topk_idx.gather(-1, next_local.unsqueeze(-1)).squeeze(-1)
            return torch.argmax(logits, dim=-1)

        thoughts_used = 0
        outputs_used = 0
        while True:
            # Stop if output budget exhausted
            if outputs_used >= max_new_tokens:
                break
            out = self.base(tokens, mem, update_memory=True)

            # Force initial thought steps if requested
            if min_thoughts_first > 0 and thoughts_used < min_thoughts_first and think_token_id is not None:
                next_token = torch.full((B,), think_token_id, dtype=tokens.dtype, device=device)
            else:
                next_token = sample_next(out["logits"])

            # For now, assume generation is run with batch_size=1 (callbacks do this)
            # Use a safe check that works for B==1 and general case
            is_think = (
                think_token_id is not None and (next_token == think_token_id).all().item() is True
            )

            if is_think:
                # Respect thought budget; if exhausted, stop generation (timeout)
                if thoughts_used >= max_new_thoughts:
                    break
                thoughts_used += 1
                tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
                mem = out["updated_memory"]
                continue

            # Non-think (output) token
            outputs_used += 1
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            mem = out["updated_memory"]
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

        if include_think_tokens_in_output or think_token_id is None:
            return tokens
        # Strip think tokens while preserving batch shape via right-padding
        mask = tokens != think_token_id
        filtered_rows = [row[m] for row, m in zip(tokens, mask)]
        max_len = max(r.size(0) for r in filtered_rows)
        pad_id = eos_token_id if eos_token_id is not None else 0
        out_batch = []
        for r in filtered_rows:
            if r.size(0) < max_len:
                pad = torch.full((max_len - r.size(0),), pad_id, dtype=r.dtype, device=r.device)
                r = torch.cat([r, pad], dim=0)
            out_batch.append(r.unsqueeze(0))
        return torch.cat(out_batch, dim=0)

    @torch.no_grad()
    def generate_with_trace(
        self,
        input_ids: torch.Tensor,
        *,
        think_token_id: Optional[int] = None,
        max_new_thoughts: int = 64,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        thought_memory: Optional[torch.Tensor] = None,
        eos_token_id: Optional[int] = None,
        include_think_tokens_in_output: bool = True,
        think_logit_bias: float = 0.0,
        min_thoughts_first: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Same as generate(), but also returns a thought trace tensor.

        Returns dict with keys:
        - tokens: Tensor[B, T_out]
        - thought_trace: Tensor[steps, B, thought_dim]
        """
        device = input_ids.device
        B = input_ids.size(0)
        mem = thought_memory or self.base.memory.initialize(B, device)
        tokens = input_ids
        trace: list[torch.Tensor] = []

        def apply_think_bias(logits: torch.Tensor) -> torch.Tensor:
            if think_token_id is not None and think_logit_bias != 0.0:
                logits = logits.clone()
                logits[:, think_token_id] = logits[:, think_token_id] + think_logit_bias
            return logits

        def sample_next(logits: torch.Tensor) -> torch.Tensor:
            logits = apply_think_bias(logits) / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                topk_vals, topk_idx = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                probs = F.softmax(topk_vals, dim=-1)
                next_local = torch.multinomial(probs, 1).squeeze(-1)
                return topk_idx.gather(-1, next_local.unsqueeze(-1)).squeeze(-1)
            return torch.argmax(logits, dim=-1)

        thoughts_used = 0
        outputs_used = 0
        while True:
            if outputs_used >= max_new_tokens:
                break
            out = self.base(tokens, mem, update_memory=True)
            trace.append(out["thought"])  # (B, thought_dim)

            if min_thoughts_first > 0 and thoughts_used < min_thoughts_first and think_token_id is not None:
                next_token = torch.full((B,), think_token_id, dtype=tokens.dtype, device=device)
            else:
                next_token = sample_next(out["logits"])

            is_think = (
                think_token_id is not None and (next_token == think_token_id).all().item() is True
            )
            if is_think:
                if thoughts_used >= max_new_thoughts:
                    break
                thoughts_used += 1
                tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
                mem = out["updated_memory"]
                continue
            outputs_used += 1
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            mem = out["updated_memory"]
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

        if include_think_tokens_in_output or think_token_id is None:
            toks = tokens
        else:
            mask = tokens != think_token_id
            filtered_rows = [row[m] for row, m in zip(tokens, mask)]
            max_len = max(r.size(0) for r in filtered_rows)
            pad_id = eos_token_id if eos_token_id is not None else 0
            out_batch = []
            for r in filtered_rows:
                if r.size(0) < max_len:
                    pad = torch.full((max_len - r.size(0),), pad_id, dtype=r.dtype, device=r.device)
                    r = torch.cat([r, pad], dim=0)
                out_batch.append(r.unsqueeze(0))
            toks = torch.cat(out_batch, dim=0)

        thought_trace = torch.stack(trace, dim=0) if len(trace) > 0 else torch.empty(0, B, self.config.thought_dim, device=device)
        return {"tokens": toks, "thought_trace": thought_trace}

