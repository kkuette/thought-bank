"""Thought-bank graft onto a pretrained HF causal LM (SmolLM2 target).

Phase-1 real-data plumbing (design: memory dsv4mini-real-data-sft-smollm):
the same write head (ThoughtStream, reused as-is) and the same fast-weight
read (standalone copy of DualModalBlock._cross_modal) bolted onto a frozen
or SFT-able HF model. Segment semantics are identical to dsv4mini: ONE write
per forward, bank carried across segments by the caller (TBPTT window lives
in the training loop, not here).

Graft safety (dsv4l lesson): fw_o is ZERO-INITIALISED, so at init the read
delta is exactly 0 and the host LM is bit-identical to the ungrafted model.
The write head trains from the LM loss through the read path only once fw_o
moves — no cold-start damage to the host.

Read placement: a forward hook on ONE decoder layer adds the read delta to
its output hidden states (low layer by default — mem_read_layers [0] was the
winning placement at 3M scale).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mhc import RMSNorm
from .memory import ThoughtStream


@dataclass
class GraftConfig:
    """Duck-typed stand-in for ThoughtBankConfig — only the fields the bank
    modules actually touch. d_model must match the HOST's hidden size."""
    d_model: int = 576                 # SmolLM2-135M hidden size
    mem_dim: int = 32
    max_mem: int = 8
    mem_seed_slots: int = 4
    mem_read_rank: int = 16
    mem_read_dropout: float = 0.0
    mem_read_swiglu: bool = True
    mem_write_noise: float = 0.0
    mem_write_gate: bool = False
    mem_write_gate_novelty: bool = False
    mem_write_gate_delta: bool = False
    mem_write_gate_merge: bool = True   # dsv5b semantics: scratch can't evict rules
    mem_write_merge_tau: float = 0.85
    read_layer: int = 0                 # decoder layer whose OUTPUT gets the read
    read_cap: float = 0.5               # cap ‖read delta‖ ≤ cap·‖h‖ per token (0 = uncapped).
                                        # Structural bound: an unbounded read blew to 3.6·‖h‖
                                        # and cratered the host LM (v8/v9). Grad still flows.


class BankRead(nn.Module):
    """Fast-weight read, standalone (math identical to DualModalBlock._cross_modal):
    each bank slot is expanded by a hypernet into a low-rank (SwiGLU) MLP layer;
    the stream is passed through the M layers sequentially; the net effect is a
    residual delta fw_o(y - y0). fw_o starts at ZERO → exact no-op at init."""

    def __init__(self, cfg: GraftConfig) -> None:
        super().__init__()
        d, r = cfg.d_model, cfg.mem_read_rank
        self.read_rank = r
        self.fw_swiglu = bool(cfg.mem_read_swiglu)
        _na = 2 if self.fw_swiglu else 1
        self.fw_A    = nn.Linear(cfg.mem_dim, _na * r * d, bias=False)
        self.fw_B    = nn.Linear(cfg.mem_dim, d * r, bias=False)
        self.fw_o    = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.fw_o.weight)          # graft no-op at init (dsv4l)
        self.norm_fw = RMSNorm(d)
        self.fw_act  = nn.GELU()
        self.fw_drop = nn.Dropout(cfg.mem_read_dropout)

    def forward(self, h: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        """h [B,T,d], bank [B,M,mem_dim] → read delta [B,T,d] (to be added to h)."""
        B, M, _ = bank.shape
        d, r = h.size(-1), self.read_rank
        _na = 2 if self.fw_swiglu else 1
        A  = self.fw_A(bank).view(B, M, _na, r, d)
        Bm = self.fw_B(bank).view(B, M, d, r)
        ds, rs = d ** -0.5, r ** -0.5
        y0 = self.norm_fw(h)
        y  = y0
        for i in range(M):
            if self.fw_swiglu:
                zg = torch.einsum("brd,btd->btr", A[:, i, 0], y) * ds
                zv = torch.einsum("brd,btd->btr", A[:, i, 1], y) * ds
                z  = (F.silu(zg) * zv).clamp(-8.0, 8.0)
            else:
                z = self.fw_act(torch.einsum("brd,btd->btr", A[:, i, 0], y) * ds)
            y = y + self.fw_drop(torch.einsum("bdr,btr->btd", Bm[:, i], z) * rs)
        return self.fw_o(y - y0)


class SmolBankLM(nn.Module):
    """HF causal LM + thought bank. Same public surface as ThoughtBankLM where
    it matters to the training loop:

        out = model(input_ids, attention_mask=..., init_mem=bank, labels=...)
        out["logits"], out["loss"], out["mem_bank"]

    One write per forward; the caller carries mem_bank across segments and
    detaches at its TBPTT boundary. pad positions (attention_mask==0) are
    excluded from the write pool.
    """

    def __init__(self, host: nn.Module, cfg: GraftConfig) -> None:
        super().__init__()
        hidden = host.config.hidden_size
        assert hidden == cfg.d_model, \
            f"GraftConfig.d_model={cfg.d_model} must match host hidden_size={hidden}"
        self.cfg   = cfg
        self.host  = host
        self.read  = BankRead(cfg)
        self.write = ThoughtStream(cfg)           # reused as-is (duck-typed cfg)
        _dt = next(host.parameters()).dtype       # follow the host's dtype (bf16 HF loads)
        self.read.to(_dt); self.write.to(_dt)
        self._bank: Optional[torch.Tensor] = None
        self._last_read_rel: Optional[float] = None   # ‖read delta‖/‖h‖ (diagnostic)
        layers = self.host.model.layers
        assert 0 <= cfg.read_layer < len(layers), "read_layer out of range"
        layers[cfg.read_layer].register_forward_hook(self._read_hook)

    def _read_hook(self, module, args, output):
        if self._bank is None or self._bank.size(1) == 0:
            return output
        h = output[0] if isinstance(output, tuple) else output
        delta = self.read(h, self._bank)
        cap = float(self.cfg.read_cap)
        if cap > 0.0:                             # structural bound: ‖delta‖ ≤ cap·‖h‖ per token
            dn = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            hn = h.norm(dim=-1, keepdim=True)
            delta = delta * (cap * hn / dn).clamp(max=1.0)   # scale down only when over cap
        with torch.no_grad():                     # diagnostic: injection magnitude
            self._last_read_rel = float((delta.norm(dim=-1) /
                                         h.norm(dim=-1).clamp_min(1e-6)).mean())
        if isinstance(output, tuple):             # transformers ≤4.x layer API
            return (h + delta,) + output[1:]
        return h + delta

    def forward(
        self,
        input_ids: torch.Tensor,                  # [B, T]
        attention_mask: Optional[torch.Tensor] = None,
        init_mem: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        B = input_ids.size(0)
        p = next(self.host.parameters())
        bank = (init_mem if init_mem is not None
                else self.write.seed_bank(B, input_ids.device, p.dtype))
        self._bank = bank
        try:
            out = self.host(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels, output_hidden_states=True)
        finally:
            self._bank = None                     # never leak into a foreign forward
        h_last = out.hidden_states[-1]            # [B, T, d]
        new_bank = self.write._write(h_last, bank, pad_mask=attention_mask)
        return {"logits": out.logits, "loss": out.loss, "mem_bank": new_bank}
