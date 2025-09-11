"""
Autoregressive Thought-Token Model

This model produces two outputs per forward step:
- thought: a continuous vector in R^thought_dim (the next thought)
- token: next-token prediction (logits and optionally the predicted token embedding)

Inputs include an accumulating array of thought vectors (thought memory). On each
forward pass, the new thought can be appended to this memory to influence future steps.

Design goals
- Files < 400 LOC, functions < 100 LOC, follow PEP8/PEP257 and type annotations
- SOLID and modular components (separate memory, attention, and heads)
- No reliance on diffusion masking; strictly autoregressive

Usage sketch
    model = ARThoughtModel(vocab_size=30522, d_model=512, n_layers=8, n_heads=8,
                           thought_dim=128, max_seq_len=2048)
    tokens = torch.randint(0, model.vocab_size, (B, T))
    thought_mem = None  # or prior memory (B, L, thought_dim)
    out = model(tokens, thought_mem)
    logits = out["logits"]          # (B, vocab_size)
    thought = out["thought"]        # (B, thought_dim)
    next_mem = out["updated_memory"]

Training
- Standard next-token cross-entropy on logits
- Thought loss can be MSE/cosine against a teacher signal or learned via a
  self-supervised objective externally
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ARConfig:
    """Configuration for the autoregressive thought-token model.

    Args:
        vocab_size: Token vocabulary size
        d_model: Model hidden size
        n_layers: Number of transformer blocks
        n_heads: Attention heads per block
        d_ff: Feed-forward inner dimension
        max_seq_len: Maximum supported sequence length (for positional embeddings)
        dropout: Dropout rate
        thought_dim: Dimensionality of thought vectors
        max_thoughts: Max number of thought vectors to keep in memory
        max_embeddings: Max number of token embeddings to keep in memory
        use_embedding_memory: Whether to reuse predicted token embeddings
        tie_weights: Whether to tie token head to token embeddings
        return_token_embedding: Whether to also return a predicted token embedding
        gradient_checkpointing: Enable gradient checkpointing across transformer blocks
    """

    vocab_size: int = 50257
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 2048
    dropout: float = 0.1
    thought_dim: int = 128
    max_thoughts: int = 64
    max_embeddings: int = 64
    use_embedding_memory: bool = True  # reuse predicted token embeddings
    tie_weights: bool = True  # tie token output head to embedding weights
    return_token_embedding: bool = False  # also return predicted next-token embedding
    gradient_checkpointing: bool = False
    use_thought_cross_attention: bool = True


class CausalSelfAttention(nn.Module):
    """Standard causal multi-head self-attention (no kv cache; batch_first)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
        # causal mask
        mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril_()
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.out(y))
        return y


class TransformerBlock(nn.Module):
    """Causal Transformer block with MHA and MLP."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ThoughtMemory(nn.Module):
    """FIFO memory buffer for accumulating thought vectors."""

    def __init__(self, thought_dim: int, max_thoughts: int) -> None:
        super().__init__()
        self.thought_dim = thought_dim
        self.max_thoughts = max_thoughts

    @torch.no_grad()
    def initialize(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create an empty memory tensor with shape (B, 0, D_th)."""
        return torch.empty(batch_size, 0, self.thought_dim, device=device)

    def push(self, memory: torch.Tensor, new_thought: torch.Tensor) -> torch.Tensor:
        """Append new_thought (B, D_th) to memory (B, L, D_th) with FIFO trim."""
        if memory.numel() == 0:
            updated = new_thought.unsqueeze(1)
        else:
            updated = torch.cat([memory, new_thought.unsqueeze(1)], dim=1)
        if updated.size(1) > self.max_thoughts:
            updated = updated[:, -self.max_thoughts :, :]
        return updated


class ThoughtContextEncoder(nn.Module):
    """Encode the thought memory into a context vector in the token space.

    Uses a simple MLP to map thoughts to d_model and attends with a query from the
    last token hidden state. If memory is empty, returns zeros.
    """

    def __init__(self, thought_dim: int, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.map = nn.Sequential(
            nn.Linear(thought_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, last_hidden: torch.Tensor, memory: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute context given last token hidden and memory.

        Args:
            last_hidden: (B, d_model) last token representation
            memory: (B, L, thought_dim) or None
        Returns:
            ctx: (B, d_model)
        """
        if memory is None or memory.numel() == 0:
            return torch.zeros_like(last_hidden)
        mem_proj = self.map(memory)  # (B, L, d_model)
        q = last_hidden.unsqueeze(1)  # (B, 1, d_model)
        ctx, _ = self.attn(self.ln(q), mem_proj, mem_proj)
        return ctx.squeeze(1)


class ThoughtCrossAttention(nn.Module):
    """Sequence-level cross-attention over thought memory.

    Projects thought memory into token hidden space and lets all token
    positions attend to it. If memory is empty, returns zeros so the
    residual connection is a no-op.
    """

    def __init__(self, thought_dim: int, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.map = nn.Linear(thought_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor]) -> torch.Tensor:
        """Return cross-attended context per token position.

        Args:
            x: (B, T, d_model) token hidden states
            memory: (B, L, thought_dim) or None
        Returns:
            ctx: (B, T, d_model) cross-attended context
        """
        if memory is None or memory.numel() == 0:
            return torch.zeros_like(x)
        mem_proj = self.map(memory)  # (B, L, d_model)
        q = self.ln(x)  # (B, T, d_model)
        ctx, _ = self.attn(q, mem_proj, mem_proj)
        return self.drop(ctx)


class TokenHead(nn.Module):
    """Token prediction head that can emit logits and optionally an embedding."""

    def __init__(self, d_model: int, vocab_size: int, tie_weights: bool, token_embed: nn.Embedding) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.vocab_size = vocab_size
        self.tie_weights = tie_weights
        self.token_embed = token_embed
        if not tie_weights:
            self.out = nn.Linear(d_model, vocab_size)
        self.embed_proj = nn.Linear(d_model, token_embed.embedding_dim)

    def forward(self, h: torch.Tensor, return_embedding: bool) -> Dict[str, torch.Tensor]:
        h = self.norm(h)
        outputs: Dict[str, torch.Tensor] = {}
        if self.tie_weights:
            logits = F.linear(h, self.token_embed.weight)  # (B, vocab)
        else:
            logits = self.out(h)
        outputs["logits"] = logits
        if return_embedding:
            outputs["token_embedding"] = self.embed_proj(h)
        return outputs


class ThoughtHead(nn.Module):
    """Predict the next thought vector."""

    def __init__(self, d_model: int, thought_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, thought_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class ARThoughtModel(nn.Module):
    """Autoregressive model that outputs a next thought vector and next-token prediction.

    Inputs:
        tokens: (B, T) token ids
        thought_memory: Optional (B, L, thought_dim) sequence of prior thoughts

    Outputs dict:
        logits: (B, vocab_size) next-token logits for position T-1
        thought: (B, thought_dim) next thought vector
        token_embedding: (B, d_model) optional predicted token embedding (if enabled)
        updated_memory: (B, L', thought_dim) memory after appending the new thought
        fused_last_hidden: (B, d_model) last hidden fused with thought context (for value heads)
    """

    def __init__(self, cfg: Optional[ARConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ARConfig()
        c = self.cfg

        # Embeddings
        self.token_embed = nn.Embedding(c.vocab_size, c.d_model)
        self.pos_embed = nn.Embedding(c.max_seq_len, c.d_model)

        # Backbone
        self.blocks = nn.ModuleList(
            [TransformerBlock(c.d_model, c.n_heads, c.d_ff, c.dropout) for _ in range(c.n_layers)]
        )

        # Memory and context
        self.memory = ThoughtMemory(c.thought_dim, c.max_thoughts)
        self.ctx = ThoughtContextEncoder(c.thought_dim, c.d_model, c.n_heads, c.dropout)
        self.xattn = (
            ThoughtCrossAttention(c.thought_dim, c.d_model, c.n_heads, c.dropout)
            if c.use_thought_cross_attention
            else None
        )

        # Heads
        self.token_head = TokenHead(c.d_model, c.vocab_size, c.tie_weights, self.token_embed)
        self.thought_head = ThoughtHead(c.d_model, c.thought_dim, c.dropout)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        thought_memory: Optional[torch.Tensor] = None,
        *,
        return_token_embedding: Optional[bool] = None,
        update_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        c = self.cfg
        B, T = tokens.shape
        device = tokens.device
        if thought_memory is None:
            thought_memory = self.memory.initialize(B, device)

        # Truncate to max_seq_len to avoid positional embedding OOB
        if T > c.max_seq_len:
            tokens = tokens[:, -c.max_seq_len :]
            T = tokens.shape[1]

        # Embed tokens with positions
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_embed(tokens) + self.pos_embed(pos)

        # Causal Transformer
        if self.training and self.cfg.gradient_checkpointing:
            # Activation checkpoint each block to reduce memory
            from torch.utils.checkpoint import checkpoint as _ckpt

            for blk in self.blocks:
                x = _ckpt(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        # Thought cross-attention integration (sequence-level)
        if self.xattn is not None:
            x = x + self.xattn(x, thought_memory)

        # Last hidden state after integration
        fused = x[:, -1, :]  # (B, d_model)

        # Heads
        token_out = self.token_head(
            fused, return_embedding=c.return_token_embedding if return_token_embedding is None else return_token_embedding
        )
        thought = self.thought_head(fused)

        # Memory update
        updated_memory = self.memory.push(thought_memory, thought) if update_memory else thought_memory

        return {
            **token_out,
            "thought": thought,
            "updated_memory": updated_memory,
            "fused_last_hidden": fused,
        }


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Minimal smoke test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ARConfig(vocab_size=1000, d_model=256, n_layers=4, n_heads=8, thought_dim=64, max_seq_len=128)
    model = ARThoughtModel(cfg).to(device)
    B, T = 2, 16
    tokens = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    out = model(tokens)
    print("logits:", out["logits"].shape)
    print("thought:", out["thought"].shape)
    print("updated_memory:", out["updated_memory"].shape)
    if cfg.return_token_embedding:
        print("token_embedding:", out["token_embedding"].shape)
    print(f"params: {count_parameters(model)/1e6:.2f}M")

