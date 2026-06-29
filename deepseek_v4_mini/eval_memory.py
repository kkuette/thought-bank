"""
Memory bank diagnostic for DualModalDeepSeekV4Mini.

Measures:
  1. PPL delta  — perplexity with_memory vs no_memory across successive chunks
  2. Write gate — are thought vectors actually written (gate near 1) or suppressed?
  3. Bank norms — are thought vectors non-trivial or near zero?
  4. Diversity  — mean pairwise cosine sim within bank (low = diverse = good)
  5. Logit drift — how much do logits shift when memory is provided vs not?
  6. Consolidation events

Usage:
    python -m deepseek_v4_mini.eval_memory [config.yaml] [checkpoint.pt]

Without a checkpoint, runs on random weights to verify the mechanism works.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .config import DeepSeekV4MiniConfig
from .model import DualModalDeepSeekV4Mini


# ── Synthetic text (always available, no external deps needed) ────────────────

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
) * 8   # repeat to give the model multiple chunks to process


# ── Hooks ─────────────────────────────────────────────────────────────────────

class _WriteGateCapture:
    """Register a hook on ThoughtStream.write_gate to capture sigmoid outputs."""

    def __init__(self, thought_stream) -> None:
        self.values: list[float] = []
        self._h = thought_stream.write_gate.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        # out is the raw linear output; sigmoid is applied afterwards in _new_thought
        p = torch.sigmoid(out).detach().float()
        self.values.append(float(p.mean()))

    def remove(self) -> None:
        self._h.remove()

    def mean(self) -> float:
        return float(sum(self.values) / len(self.values)) if self.values else 0.0

    def reset(self) -> None:
        self.values.clear()


# ── Tokenisation (minimal, char-level fallback if no transformers) ────────────

def _tokenise(text: str, vocab_size: int) -> torch.LongTensor:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        ids = tok.encode(text)
        # Clip any id that exceeds the model's vocab (e.g. tiny uses 32k, gpt2 has 50k)
        ids = [min(i, vocab_size - 1) for i in ids]
    except Exception:
        ids = [ord(c) % vocab_size for c in text]
    return torch.tensor(ids, dtype=torch.long)


# ── Core evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: DualModalDeepSeekV4Mini,
    tokens: torch.LongTensor,
    seq_len: int,
    device: torch.device,
) -> dict:
    """
    Process `tokens` in non-overlapping chunks of `seq_len`.

    Returns a dict of per-chunk statistics:
      ppl_with_mem   perplexity using memory from previous chunk
      ppl_no_mem     perplexity with fresh start (no memory)
      logit_drift    mean L2 distance between logits with / without memory
      bank_size      how many slots are in the bank at end of each chunk
      bank_norm_mean mean L2 norm of thought vectors in the bank
      bank_diversity mean pairwise cosine similarity within the bank (lower = more diverse)
      write_gate     mean write-gate probability
    """
    model.eval()
    tokens = tokens.to(device)

    n_chunks = (len(tokens) - 1) // seq_len
    if n_chunks < 2:
        raise ValueError(f"Need at least 2 chunks; got {n_chunks} (seq_len={seq_len}, tokens={len(tokens)})")

    gate_capture = _WriteGateCapture(model.thought_stream)

    records: list[dict] = []
    mem_bank: Optional[torch.Tensor] = None   # carry-over bank

    for i in range(n_chunks):
        start = i * seq_len
        x = tokens[start : start + seq_len].unsqueeze(0)          # [1, T]
        y = tokens[start + 1 : start + seq_len + 1].unsqueeze(0)  # [1, T]

        # ── Pass WITH memory from previous chunk ──────────────────────────────
        gate_capture.reset()
        out_mem  = model(x, init_mem=mem_bank)
        gate_mem = gate_capture.mean()

        ppl_with = _ppl(out_mem["logits"], y)
        new_bank  = out_mem["mem_bank"]

        # ── Pass WITHOUT memory (fresh start) ────────────────────────────────
        gate_capture.reset()
        out_nomem = model(x, init_mem=None)

        ppl_no   = _ppl(out_nomem["logits"], y)

        # ── Logit drift ───────────────────────────────────────────────────────
        drift = (out_mem["logits"] - out_nomem["logits"]).norm(dim=-1).mean().item()

        # ── Bank diagnostics ──────────────────────────────────────────────────
        bank_size, norm_mean, diversity = _bank_stats(new_bank)

        records.append({
            "chunk":         i,
            "ppl_with_mem":  ppl_with,
            "ppl_no_mem":    ppl_no,
            "ppl_delta":     ppl_no - ppl_with,   # positive = memory helps
            "logit_drift":   drift,
            "bank_size":     bank_size,
            "bank_norm":     norm_mean,
            "bank_diversity": diversity,
            "write_gate":    gate_mem,
        })

        mem_bank = new_bank   # carry the bank forward

    gate_capture.remove()
    return records


def _ppl(logits: torch.Tensor, targets: torch.LongTensor) -> float:
    """Cross-entropy → perplexity."""
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
    return float(math.exp(min(float(ce), 20)))   # cap at e^20 for sanity


def _bank_stats(bank: Optional[torch.Tensor]):
    """Returns (size, mean_norm, mean_cosine_similarity)."""
    if bank is None or bank.size(1) == 0:
        return 0, 0.0, 0.0
    bank = bank[0].float()      # [M, mem_dim]
    M    = bank.size(0)
    norm_mean = float(bank.norm(dim=-1).mean())

    if M < 2:
        return M, norm_mean, 0.0

    # Pairwise cosine similarity
    normed = F.normalize(bank, dim=-1)            # [M, mem_dim]
    sim    = normed @ normed.T                    # [M, M]
    mask   = ~torch.eye(M, dtype=torch.bool, device=sim.device)
    diversity = float(sim[mask].mean())           # lower = more diverse

    return M, norm_mean, diversity


# ── Pretty printing ───────────────────────────────────────────────────────────

def _print_report(records: list[dict], cfg: DeepSeekV4MiniConfig) -> None:
    print("\n" + "="*72)
    print(" Memory Bank Diagnostic Report")
    print("="*72)
    print(f" Model: DualModalDeepSeekV4Mini  |  max_mem={cfg.max_mem}  "
          f"consolidate_k={cfg.consolidate_k}  mem_dim={cfg.mem_dim}")
    print("-"*72)

    header = (
        f"{'Chunk':>5}  {'PPL(mem)':>9}  {'PPL(no)':>8}  "
        f"{'Δ PPL':>7}  {'Drift':>6}  "
        f"{'Bank':>4}  {'Norm':>6}  {'Sim↓':>6}  {'Gate':>6}"
    )
    print(header)
    print("-"*72)

    for r in records:
        delta_str = f"{r['ppl_delta']:+.2f}"
        print(
            f"{r['chunk']:>5}  {r['ppl_with_mem']:>9.2f}  {r['ppl_no_mem']:>8.2f}  "
            f"{delta_str:>7}  {r['logit_drift']:>6.3f}  "
            f"{r['bank_size']:>4}  {r['bank_norm']:>6.3f}  "
            f"{r['bank_diversity']:>6.3f}  {r['write_gate']:>6.3f}"
        )

    print("-"*72)

    # Summary stats
    ppl_deltas   = [r["ppl_delta"] for r in records[1:]]   # skip first (no prior mem)
    drifts       = [r["logit_drift"] for r in records]
    norms        = [r["bank_norm"] for r in records]
    gates        = [r["write_gate"] for r in records]
    diversities  = [r["bank_diversity"] for r in records if r["bank_size"] >= 2]

    print(f"\n Chunks evaluated    : {len(records)}")
    print(f" Mean PPL delta      : {_mean(ppl_deltas):+.3f}  "
          f"({'memory helps' if _mean(ppl_deltas) > 0 else 'memory hurts / neutral'} on untrained model)")
    print(f" Mean logit drift    : {_mean(drifts):.4f}  "
          f"({'memory influences predictions' if _mean(drifts) > 0.01 else 'memory has little effect'})")
    print(f" Mean bank norm      : {_mean(norms):.4f}  "
          f"({'active' if _mean(norms) > 0.05 else 'near-zero — check write gate'})")
    print(f" Mean write gate     : {_mean(gates):.4f}  "
          f"({'writing' if _mean(gates) > 0.3 else 'suppressed — model not writing'})")
    print(f" Mean intra-bank sim : {_mean(diversities):.4f}  "
          f"({'diverse' if _mean(diversities) < 0.7 else 'redundant slots'})")

    # Consolidation count
    max_size = max(r["bank_size"] for r in records) if records else 0
    n_consol  = sum(1 for i in range(1, len(records))
                    if records[i]["bank_size"] <= records[i-1]["bank_size"])
    print(f" Peak bank size      : {max_size}  (max_mem={cfg.max_mem})")
    print(f" Consolidation events: {n_consol}")

    print("="*72)

    # Verdict
    drift_ok = _mean(drifts) > 1e-3
    gate_ok  = _mean(gates) > 0.05
    norm_ok  = _mean(norms) > 1e-3

    print("\n VERDICT")
    print(f"  Write gate active  : {'✓' if gate_ok else '✗'}  ({_mean(gates):.3f})")
    print(f"  Bank non-trivial   : {'✓' if norm_ok else '✗'}  (mean norm {_mean(norms):.4f})")
    print(f"  Memory shifts logits: {'✓' if drift_ok else '✗'}  (mean drift {_mean(drifts):.4f})")
    print()
    if gate_ok and norm_ok and drift_ok:
        print("  → Bank is ACTIVE and influences predictions.")
        print("    PPL delta on an untrained model is noisy — train to see real effect.")
    elif not gate_ok:
        print("  → Write gate stuck near 0: model is not writing to the bank.")
        print("    Consider lowering write_gate bias init or adding a write-loss term.")
    elif not norm_ok:
        print("  → Thought vectors are near-zero: the write head is collapsed.")
    else:
        print("  → Bank is written but has marginal effect on logits.")
    print()


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    cfg_path  = Path(args[0]) if len(args) > 0 else None
    ckpt_path = Path(args[1]) if len(args) > 1 else None

    if cfg_path and cfg_path.exists():
        cfg = DeepSeekV4MiniConfig.from_yaml(cfg_path)
        print(f"Config loaded from {cfg_path}")
    else:
        cfg = DeepSeekV4MiniConfig.tiny()
        print("Using tiny config (random weights — mechanism test only)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DualModalDeepSeekV4Mini(cfg).to(device)

    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        print(f"Checkpoint loaded from {ckpt_path}")
    else:
        print("No checkpoint — using random weights")

    tokens  = _tokenise(_SYNTHETIC, cfg.vocab_size)
    seq_len = min(64, cfg.max_seq_len)

    print(f"Tokens: {len(tokens)}  |  Chunks: {(len(tokens)-1)//seq_len}  "
          f"|  Seq len: {seq_len}  |  Device: {device}\n")

    records = evaluate(model, tokens, seq_len, device)
    _print_report(records, cfg)


if __name__ == "__main__":
    main()
