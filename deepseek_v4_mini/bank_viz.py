"""Bank visualizations for TensorBoard (add_figure).

Three views of ONE eval conversation (lane 0, bank carried):

  1. bank/content    — final bank heatmap [slots × mem_dim], per-slot RMS-
                       normalized: what the read actually consumes.
  2. bank/similarity — M×M cosine matrix of the final bank: redundancy and
                       block structure at a glance (merge territory).
  3. bank/writes_pca — every write of the conversation projected on the 2
                       principal axes, colored by segment kind (present /
                       query / distract / switch / replay), numbered by
                       order. THE eviction/interference picture: do rule
                       writes cluster away from real-text noise?

All figures are matplotlib (Agg), cheap, called only at eval cadence.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_KIND_COLORS = {"present": "tab:red", "switch": "tab:orange", "query": "tab:blue",
                "distract": "tab:gray", "replay": "tab:green",
                "turn": "tab:blue", "hop": "tab:purple"}


def bank_content_fig(bank: torch.Tensor):
    """bank [M, D] → heatmap figure (per-slot RMS normalized)."""
    b = bank.detach().float()
    b = b / b.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(b.numpy(), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xlabel("mem dim"); ax.set_ylabel("slot (0 = oldest)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    return fig


def bank_similarity_fig(bank: torch.Tensor):
    """bank [M, D] → M×M cosine matrix figure."""
    b = torch.nn.functional.normalize(bank.detach().float(), dim=1)
    cos = (b @ b.T).numpy()
    M = cos.shape[0]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(M)); ax.set_yticks(range(M))
    ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return fig


def writes_pca_fig(writes: torch.Tensor, kinds: list[str]):
    """writes [N, D] (one per segment, order = time) + segment kinds →
    2-PC scatter, numbered by write order."""
    w = writes.detach().float()
    w0 = w - w.mean(dim=0, keepdim=True)
    # PCA via SVD (N small)
    _, _, V = torch.linalg.svd(w0, full_matrices=False)
    xy = (w0 @ V[:2].T).numpy()
    fig, ax = plt.subplots(figsize=(5, 4))
    seen = set()
    for i, k in enumerate(kinds):
        c = _KIND_COLORS.get(k, "black")
        ax.scatter(xy[i, 0], xy[i, 1], c=c, s=60,
                   label=(k if k not in seen else None), zorder=3)
        ax.annotate(str(i), (xy[i, 0], xy[i, 1]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
        seen.add(k)
    ax.plot(xy[:, 0], xy[:, 1], color="lightgray", lw=0.7, zorder=1)
    ax.legend(fontsize=7, loc="best")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout()
    return fig


@torch.no_grad()
def log_bank_figures(writer, model, segs: list[dict], device, step: int,
                     amp_dtype=torch.bfloat16) -> None:
    """Run ONE conversation (lane 0 shown), carried bank, and log the three
    figures. `segs` = a conversation from the eval stream."""
    bank, writes, kinds = None, [], []
    for s in segs:
        x  = s["input_ids"].to(device)
        am = s["attention_mask"].to(device)
        with torch.autocast(device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda"):
            o = model(x, attention_mask=am, init_mem=bank)
        bank = o["mem_bank"]
        writes.append(bank[0, -1].float().cpu())   # newest slot = this write
        kinds.append(s["kind"])
    final = bank[0].float().cpu()
    writer.add_figure("bank/content",    bank_content_fig(final), step)
    writer.add_figure("bank/similarity", bank_similarity_fig(final), step)
    writer.add_figure("bank/writes_pca", writes_pca_fig(torch.stack(writes), kinds), step)
    plt.close("all")
