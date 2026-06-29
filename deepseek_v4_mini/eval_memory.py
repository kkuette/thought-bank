"""
Architecture comparison: DeepSeekV4Mini (no bank) vs DualModalDeepSeekV4Mini (with bank).

Measures per chunk of text:
  - PPL (no_mem)      : DualModal with init_mem=None each chunk (no carry-over)
  - PPL (with_mem)    : DualModal with accumulated memory bank (carry-over)
  - PPL (no_bank)     : baseline DeepSeekV4Mini (no memory at all)
  - logit_drift       : ||logits_with_mem - logits_no_mem||  per token
  - bank diagnostics  : size, vector norms, intra-bank cosine similarity

Usage:
    # Random weights (mechanism test):
    python -m deepseek_v4_mini.eval_memory

    # With checkpoints (both architectures trained separately):
    python -m deepseek_v4_mini.eval_memory \\
        --cfg_mem  deepseek_v4_mini/configs/tiny_with_mem.yaml \\
        --cfg_base deepseek_v4_mini/configs/tiny_no_mem.yaml \\
        --ckpt_mem checkpoints/tiny_with_mem/final.pt \\
        --ckpt_base checkpoints/tiny_no_mem/final.pt
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .config import DeepSeekV4MiniConfig
from .model import DeepSeekV4Mini, DualModalDeepSeekV4Mini


# ── Synthetic text ────────────────────────────────────────────────────────────

_SYNTHETIC = (
    "The capital of France is Paris. Paris is known for the Eiffel Tower. "
    "The tower was built in 1889 and stands 330 metres tall. "
    "France borders Germany, Spain, Italy, and Belgium. "
    "The French language is spoken by over 300 million people worldwide. "
    "The capital of Germany is Berlin. Berlin has a population of 3.6 million. "
    "The capital of France is Paris. Paris has many famous museums. "
    "The Louvre in Paris holds the Mona Lisa by Leonardo da Vinci. "
    "France is also famous for its cuisine and wine production. "
    "The capital of France is Paris, a major European hub. "
    "The Eiffel Tower in Paris receives millions of tourists every year. "
) * 8


# ── Write-gate capture ────────────────────────────────────────────────────────

class _WriteGateCapture:
    def __init__(self, thought_stream) -> None:
        self.values: list[float] = []
        self._h = thought_stream.write_gate.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        p = torch.sigmoid(out).detach().float()
        self.values.append(float(p.mean()))

    def remove(self) -> None:
        self._h.remove()

    def mean(self) -> float:
        return float(sum(self.values) / len(self.values)) if self.values else 0.0

    def reset(self) -> None:
        self.values.clear()


# ── Tokenisation ──────────────────────────────────────────────────────────────

def _tokenise(text: str, vocab_size: int) -> torch.LongTensor:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        ids = tok.encode(text)
        ids = [min(i, vocab_size - 1) for i in ids]
    except Exception:
        ids = [ord(c) % vocab_size for c in text]
    return torch.tensor(ids, dtype=torch.long)


# ── Per-chunk helpers ─────────────────────────────────────────────────────────

def _ppl(logits: torch.Tensor, targets: torch.LongTensor) -> float:
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
    return float(math.exp(min(float(ce), 20)))


def _bank_stats(bank: Optional[torch.Tensor]):
    """Returns (size, mean_norm, mean_cosine_similarity)."""
    if bank is None or bank.size(1) == 0:
        return 0, 0.0, 0.0
    bank = bank[0].float()
    M = bank.size(0)
    norm_mean = float(bank.norm(dim=-1).mean())
    if M < 2:
        return M, norm_mean, 0.0
    normed = F.normalize(bank, dim=-1)
    sim = normed @ normed.T
    mask = ~torch.eye(M, dtype=torch.bool, device=sim.device)
    return M, norm_mean, float(sim[mask].mean())


# ── Core evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    dual_model: DualModalDeepSeekV4Mini,
    base_model: Optional[DeepSeekV4Mini],
    tokens: torch.LongTensor,
    seq_len: int,
    device: torch.device,
) -> list[dict]:
    """
    Process tokens in non-overlapping chunks of seq_len.

    For each chunk records:
      ppl_with_mem    DualModal with accumulated bank
      ppl_no_mem      DualModal cold-start (no bank)
      ppl_no_bank     baseline DeepSeekV4Mini (if provided)
      logit_drift     mean L2 between with/without memory logits
      bank_size / bank_norm / bank_diversity  memory bank health
      write_gate      mean write-gate probability
    """
    dual_model.eval()
    if base_model is not None:
        base_model.eval()
    tokens = tokens.to(device)

    n_chunks = (len(tokens) - 1) // seq_len
    if n_chunks < 2:
        raise ValueError(
            f"Need at least 2 chunks; got {n_chunks} (seq_len={seq_len}, tokens={len(tokens)})"
        )

    gate_capture = _WriteGateCapture(dual_model.thought_stream)
    records: list[dict] = []
    mem_bank: Optional[torch.Tensor] = None

    for i in range(n_chunks):
        start = i * seq_len
        x = tokens[start : start + seq_len].unsqueeze(0)
        y = tokens[start + 1 : start + seq_len + 1].unsqueeze(0)

        # DualModal — with accumulated memory
        gate_capture.reset()
        out_mem  = dual_model(x, init_mem=mem_bank)
        ppl_with = _ppl(out_mem["logits"], y)
        new_bank  = out_mem["mem_bank"]
        gate_val  = gate_capture.mean()

        # DualModal — cold start (no memory)
        out_cold = dual_model(x, init_mem=None)
        ppl_cold = _ppl(out_cold["logits"], y)

        # Logit drift: how much does the memory change predictions?
        drift = (out_mem["logits"] - out_cold["logits"]).norm(dim=-1).mean().item()

        # Baseline: DeepSeekV4Mini (no bank at all)
        ppl_base = None
        if base_model is not None:
            out_base = base_model(x)
            ppl_base = _ppl(out_base["logits"], y)

        # Bank diagnostics
        bank_size, norm_mean, diversity = _bank_stats(new_bank)

        records.append({
            "chunk":          i,
            "ppl_with_mem":   ppl_with,
            "ppl_no_mem":     ppl_cold,
            "ppl_delta":      ppl_cold - ppl_with,
            "ppl_no_bank":    ppl_base,
            "logit_drift":    drift,
            "bank_size":      bank_size,
            "bank_norm":      norm_mean,
            "bank_diversity": diversity,
            "write_gate":     gate_val,
        })

        mem_bank = new_bank

    gate_capture.remove()
    return records


# ── Reporting ─────────────────────────────────────────────────────────────────

def _mean(xs: list) -> float:
    xs = [v for v in xs if v is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _print_report(
    records: list[dict],
    dual_cfg: DeepSeekV4MiniConfig,
    dual_params: int,
    base_params: Optional[int],
) -> None:
    has_base = records[0]["ppl_no_bank"] is not None

    print("\n" + "=" * 76)
    print("  Architecture Comparison: DeepSeekV4Mini vs DualModalDeepSeekV4Mini")
    print("=" * 76)
    print(
        f"  DualModal  : {dual_params:,} params  |  max_mem={dual_cfg.max_mem}"
        f"  consolidate_k={dual_cfg.consolidate_k}  mem_dim={dual_cfg.mem_dim}"
    )
    if base_params is not None:
        print(f"  Baseline   : {base_params:,} params  (no memory bank)")
    print("-" * 76)

    # Header
    base_col = f"{'PPL(base)':>10}" if has_base else ""
    print(
        f"{'Chunk':>5}  {'PPL(mem)':>9}  {'PPL(cold)':>9}  "
        f"{'ΔPPL':>7}  {base_col}  "
        f"{'Drift':>6}  {'Bank':>4}  {'Norm':>6}  {'Sim↓':>6}  {'Gate':>6}"
    )
    print("-" * 76)

    for r in records:
        base_str = f"{r['ppl_no_bank']:>10.2f}" if has_base and r["ppl_no_bank"] is not None else " " * (10 if has_base else 0)
        print(
            f"{r['chunk']:>5}  {r['ppl_with_mem']:>9.2f}  {r['ppl_no_mem']:>9.2f}  "
            f"{r['ppl_delta']:>+7.2f}  {base_str}  "
            f"{r['logit_drift']:>6.3f}  {r['bank_size']:>4}  "
            f"{r['bank_norm']:>6.3f}  {r['bank_diversity']:>6.3f}  {r['write_gate']:>6.3f}"
        )

    print("-" * 76)

    ppl_deltas  = [r["ppl_delta"]   for r in records[1:]]
    drifts      = [r["logit_drift"] for r in records]
    norms       = [r["bank_norm"]   for r in records]
    gates       = [r["write_gate"]  for r in records]
    divs        = [r["bank_diversity"] for r in records if r["bank_size"] >= 2]
    ppls_base   = [r["ppl_no_bank"] for r in records if r["ppl_no_bank"] is not None]
    ppls_mem    = [r["ppl_with_mem"] for r in records]

    print(f"\n  Chunks evaluated      : {len(records)}")
    print(
        f"  Mean PPL (with mem)   : {_mean(ppls_mem):.2f}  vs  "
        f"cold-start {_mean([r['ppl_no_mem'] for r in records]):.2f}"
        + (f"  vs  no-bank {_mean(ppls_base):.2f}" if has_base else "")
    )
    print(
        f"  Mean PPL delta (mem)  : {_mean(ppl_deltas):+.3f}  "
        f"({'memory helps' if _mean(ppl_deltas) > 0 else 'memory hurts / neutral'})"
    )
    print(
        f"  Mean logit drift      : {_mean(drifts):.4f}  "
        f"({'memory influences predictions' if _mean(drifts) > 0.01 else 'marginal effect'})"
    )
    print(
        f"  Mean bank norm        : {_mean(norms):.4f}  "
        f"({'active' if _mean(norms) > 0.05 else 'near-zero — check write gate'})"
    )
    print(
        f"  Mean write gate       : {_mean(gates):.4f}  "
        f"({'writing' if _mean(gates) > 0.3 else 'suppressed'})"
    )
    print(
        f"  Mean intra-bank sim   : {_mean(divs):.4f}  "
        f"({'diverse' if _mean(divs) < 0.7 else 'redundant slots'})"
    )

    max_size  = max(r["bank_size"] for r in records) if records else 0
    n_consol  = sum(
        1 for i in range(1, len(records))
        if records[i]["bank_size"] <= records[i - 1]["bank_size"]
    )
    print(f"  Peak bank size        : {max_size}  (max_mem={dual_cfg.max_mem})")
    print(f"  Consolidation events  : {n_consol}")
    print("=" * 76)

    # Verdict
    drift_ok = _mean(drifts) > 1e-3
    gate_ok  = _mean(gates)  > 0.05
    norm_ok  = _mean(norms)  > 1e-3

    print("\n  VERDICT")
    print(f"    Write gate active   : {'✓' if gate_ok else '✗'}  ({_mean(gates):.3f})")
    print(f"    Bank non-trivial    : {'✓' if norm_ok else '✗'}  (norm {_mean(norms):.4f})")
    print(f"    Memory shifts logits: {'✓' if drift_ok else '✗'}  (drift {_mean(drifts):.4f})")

    if gate_ok and norm_ok and drift_ok:
        print("\n    → Bank is ACTIVE and influences predictions.")
        print("      Train both configs to see real PPL gap (random weights are noisy).")
    elif not gate_ok:
        print("\n    → Write gate stuck near 0. Bank is not being written.")
    elif not norm_ok:
        print("\n    → Thought vectors near zero: write head may have collapsed.")
    else:
        print("\n    → Bank is written but has marginal effect on logits so far.")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Compare memory vs no-memory architectures")
    p.add_argument("--cfg_mem",  type=Path, default=None,
                   help="Config for DualModalDeepSeekV4Mini (with memory bank)")
    p.add_argument("--cfg_base", type=Path, default=None,
                   help="Config for DeepSeekV4Mini baseline (no bank)")
    p.add_argument("--ckpt_mem",  type=Path, default=None,
                   help="Checkpoint for the DualModal model")
    p.add_argument("--ckpt_base", type=Path, default=None,
                   help="Checkpoint for the baseline model")
    p.add_argument("--seq_len", type=int, default=64,
                   help="Chunk length for evaluation (default 64)")
    # Legacy positional args for backwards compatibility
    p.add_argument("legacy_cfg",  nargs="?", type=Path)
    p.add_argument("legacy_ckpt", nargs="?", type=Path)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Backwards-compatible: if positional args used, treat as cfg_mem / ckpt_mem
    cfg_mem_path  = args.cfg_mem  or args.legacy_cfg
    ckpt_mem_path = args.ckpt_mem or args.legacy_ckpt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── DualModal (with memory bank) ──────────────────────────────────────────
    if cfg_mem_path and cfg_mem_path.exists():
        dual_cfg = DeepSeekV4MiniConfig.from_yaml(cfg_mem_path)
        print(f"DualModal config loaded from {cfg_mem_path}")
    else:
        dual_cfg = DeepSeekV4MiniConfig.tiny()
        dual_cfg.use_dual_stream = True
        print("DualModal: using tiny config (random weights — mechanism test only)")

    dual_model = DualModalDeepSeekV4Mini(dual_cfg).to(device)

    if ckpt_mem_path and ckpt_mem_path.exists():
        state = torch.load(ckpt_mem_path, map_location=device)
        dual_model.load_state_dict(state["model"])
        print(f"DualModal checkpoint loaded from {ckpt_mem_path}")
    else:
        print("DualModal: no checkpoint — using random weights")

    # ── Baseline (no memory bank) ─────────────────────────────────────────────
    base_model: Optional[DeepSeekV4Mini] = None
    if args.cfg_base and args.cfg_base.exists():
        base_cfg   = DeepSeekV4MiniConfig.from_yaml(args.cfg_base)
        base_model = DeepSeekV4Mini(base_cfg).to(device)
        if args.ckpt_base and args.ckpt_base.exists():
            state = torch.load(args.ckpt_base, map_location=device)
            base_model.load_state_dict(state["model"])
            print(f"Baseline checkpoint loaded from {args.ckpt_base}")
        else:
            print("Baseline: no checkpoint — using random weights")
    else:
        print("Baseline: no config provided — skipping no-bank column")

    tokens  = _tokenise(_SYNTHETIC, dual_cfg.vocab_size)
    seq_len = min(args.seq_len, dual_cfg.max_seq_len)

    print(
        f"\nTokens: {len(tokens)}  |  Chunks: {(len(tokens)-1)//seq_len}"
        f"  |  Seq len: {seq_len}  |  Device: {device}\n"
    )

    records = evaluate(dual_model, base_model, tokens, seq_len, device)
    _print_report(
        records,
        dual_cfg,
        dual_model.num_params(),
        base_model.num_params() if base_model is not None else None,
    )


if __name__ == "__main__":
    main()
