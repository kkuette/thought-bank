from __future__ import annotations

from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Normalization and activations ----


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias).

    Keeps parameter count low and is standard in modern decoders.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # <=100 LOC
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(norm + self.eps)
        return self.weight * x_norm


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.w12 = nn.Linear(dim, hidden * 2, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # <=100 LOC
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.drop(F.silu(a) * b))


# ---- Rotary embeddings ----


def _build_rope_cache(seq_len: int, head_dim: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create cos/sin caches for RoPE (interleaved frequencies)."""
    theta = 10000.0
    idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (idx / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("t,f->tf", t, inv_freq)  # [T, D/2]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to last dimension pairs of x.

    x: [B, H, T, D]
    cos/sin: [T, D/2]
    """
    _, _, T, _ = x.shape
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    cos_t = cos[:T].unsqueeze(0).unsqueeze(0)
    sin_t = sin[:T].unsqueeze(0).unsqueeze(0)
    x_rot_even = x_even * cos_t - x_odd * sin_t
    x_rot_odd = x_even * sin_t + x_odd * cos_t
    x_out = torch.empty_like(x)
    x_out[..., ::2] = x_rot_even
    x_out[..., 1::2] = x_rot_odd
    return x_out


# ---- Attention blocks ----


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:  # <=100 LOC
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        q, k, v = qkv.unbind(dim=3)  # [B,H,T,D]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.full((T, T), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        att = att + mask  # broadcast over B,H
        w = F.softmax(att, dim=-1)
        w = self.drop(w)
        y = w @ v  # [B,H,T,D]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, d_ff, dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:  # <=100 LOC
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class MemoryCrossAttention(nn.Module):
    def __init__(self, dim: int, mem_dim: int, n_heads: int) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(mem_dim, dim, bias=False)
        self.v_proj = nn.Linear(mem_dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor]) -> torch.Tensor:  # <=100 LOC
        if mem is None or mem.size(1) == 0:
            return x
        B, T, H = x.shape
        M = mem.size(1)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(mem).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(mem).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        w = F.softmax(att, dim=-1)
        z = w @ v
        z = z.transpose(1, 2).contiguous().view(B, T, H)
        return x + self.o_proj(z)


class WriteGate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, 1, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # <=100 LOC
        return torch.sigmoid(self.lin(h))  # [B, 1]


class ThoughtHead(nn.Module):
    def __init__(self, dim: int, mem_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, mem_dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # <=100 LOC
        return self.proj(h)  # [B, D]


# ---- Decoder backbone and full model ----


class TinyDecoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int, n_layers: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.dim = dim
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:  # <=100 LOC
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        x = self.drop(x)
        cos, sin = _build_rope_cache(T, self.blocks[0].attn.head_dim, x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.norm(x)  # [B, T, H]


class ThoughtLM(nn.Module):
    """Decoder with external thought memory read/write.

    - Compute base hidden states H with the decoder.
    - For each timestep t, augment h_t via cross-attn to current memory bank M_{<t}.
    - Predict next-token logits from augmented h_t and decide whether to write a thought vector.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        mem_dim: int,
        max_mem: int,
    ) -> None:
        super().__init__()
        self.decoder = TinyDecoder(vocab_size, dim, n_layers, n_heads, d_ff, dropout)
        self.mem_attn = MemoryCrossAttention(dim, mem_dim, n_heads)
        self.gate = WriteGate(dim)
        self.thought = ThoughtHead(dim, mem_dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.max_mem = int(max_mem)

    def forward(self, input_ids: torch.LongTensor, init_mem: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:  # <=100 LOC
        B, T = input_ids.shape
        H_base = self.decoder(input_ids)  # [B, T, H]
        # No-memory logits for ablation (retain for completeness)
        logits_nomem = self.lm_head(H_base)

        # Build memory incrementally and compute memory-augmented logits
        mem_list: List[torch.Tensor] = []  # writes in this call, each: [B,1,D]
        logits_list: List[torch.Tensor] = []
        p_list: List[torch.Tensor] = []
        m_raw_list: List[torch.Tensor] = []
        delta_list: List[torch.Tensor] = []  # [B,1] per step

        mem_bank: Optional[torch.Tensor] = init_mem  # carry-in memory
        for t in range(T):
            h_t = H_base[:, t : t + 1, :]  # [B,1,H]
            h_aug = self.mem_attn(h_t, mem_bank)
            logit_t = self.lm_head(h_aug)  # [B,1,V]
            logits_list.append(logit_t)
            # Write decision and vector
            p_t = self.gate(h_aug.squeeze(1))  # [B,1]
            m_t = self.thought(h_aug.squeeze(1))  # [B,D]
            p_list.append(p_t)
            m_raw_list.append(m_t)
            m_write = (p_t * m_t).unsqueeze(1)  # [B,1,D]
            mem_list.append(m_write)
            # Track per-step memory effect magnitude (L2 of residual add)
            delta = (h_aug - h_t).pow(2).mean(dim=-1)  # [B,1]
            delta_list.append(delta)

            # Update carry memory bank
            mem_bank = m_write if mem_bank is None else torch.cat([mem_bank, m_write], dim=1)
            if mem_bank.size(1) > self.max_mem:
                mem_bank = mem_bank[:, -self.max_mem :, :]

        logits_mem = torch.cat(logits_list, dim=1)
        p_gates = torch.cat(p_list, dim=1).squeeze(-1)  # [B,T]
        m_raw = torch.cat(m_raw_list, dim=1)  # [B,T,D]
        m_used = torch.cat(mem_list, dim=1)  # [B,T,D] (includes zeros if p small)
        mem_delta = torch.cat(delta_list, dim=1).squeeze(-1)  # [B,T]

        return {
            "h_base": H_base,
            "logits_mem": logits_mem,
            "logits_nomem": logits_nomem,
            "p_gates": p_gates,
            "m_raw": m_raw,
            "m_used": m_used,
            "mem_delta": mem_delta,
            "mem_bank": mem_bank,  # carry-out memory bank
        }

