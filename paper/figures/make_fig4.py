"""Figure 4 — the superposition mechanism, both policy seeds.

Panel A: mean slot-slot cosine matrix at the end of a conversation (seed 42)
— the 8 slots carry near-copies of one superposed vector (bank effective
rank ~1, both seeds). Panel B: PCA of the clean write codes across all 127
rules (seed 42), coloured by the rule offset s — the code space stays
high-dimensional and rule-separable. Panel C: per-write redundancy (max
cosine vs the resident bank) across a 16-turn conversation with a switch at
turn 8, for both seeds — steady-state rehearsal writes are near-duplicates;
the switch write is novel content, and its redundancy separates the two
replacement attractors of §9 (selective update: 0.50; flush-and-rewrite:
-0.10).

Inputs: JSON dumps of superposition_probe.py for seeds 42 and 43.
Usage: python make_fig4.py <fig4_s42.json> <fig4_s43.json> [--out PATH]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_S42 = "#2e7d32"
C_S43 = "#6a1fb1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("s42_json")
    ap.add_argument("s43_json")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "fig4_superposition"))
    args = ap.parse_args()
    d42 = json.load(open(args.s42_json))
    d43 = json.load(open(args.s43_json))

    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(10.8, 3.4), gridspec_kw={"width_ratios": [0.9, 1.1, 1.4]}
    )

    # --- A: slot-slot cosine (seed 42) --------------------------------------
    sim = np.array(d42["bank_slot_sim"])
    im = axA.imshow(sim, vmin=0, vmax=1, cmap="viridis")
    axA.set_xticks(range(8)); axA.set_yticks(range(8))
    axA.tick_params(labelsize=7)
    axA.set_xlabel("slot"); axA.set_ylabel("slot")
    axA.set_title(f"A — end-of-conv. bank (s42)\nslot cosine; eff. rank "
                  f"{d42['bank_eff_rank_mean']:.2f}/8\n(s43: {d43['bank_eff_rank_mean']:.2f}/8)",
                  fontsize=8.5)
    fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

    # --- B: PCA of rule codes (seed 42) --------------------------------------
    codes = np.array(d42["rule_codes"])
    rids = np.array(d42["rule_ids"])
    held0 = d42["held_idx_start"]
    X = codes - codes.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    P = X @ vt[:2].T
    sc = axB.scatter(P[:held0, 0], P[:held0, 1], c=rids[:held0], cmap="twilight",
                     s=14, lw=0)
    axB.scatter(P[held0:, 0], P[held0:, 1], c=rids[held0:], cmap="twilight",
                s=42, marker="*", edgecolors="k", linewidths=0.4, label="held rules")
    axB.set_title(f"B — write codes across rules, s42\nPCA (eff. rank "
                  f"{d42['rule_eff_rank']:.1f}/32; s43 {d43['rule_eff_rank']:.1f}/32)",
                  fontsize=8.5)
    axB.set_xlabel("PC 1"); axB.set_ylabel("PC 2")
    axB.tick_params(labelsize=7)
    axB.legend(fontsize=7, frameon=False, loc="best")
    fig.colorbar(sc, ax=axB, fraction=0.046, pad=0.04, label="offset s").ax.tick_params(labelsize=7)

    # --- C: per-write redundancy across a switch, both seeds -----------------
    for d, color, seed in ((d42, C_S42, 42), (d43, C_S43, 43)):
        red = d["write_redundancy"]
        labels = d["write_labels"]
        x = np.arange(len(red))
        axC.plot(x, red, "o-", color=color, ms=4, lw=1.3, label=f"seed {seed}")
        sw = labels.index("SWITCH k0")
        axC.plot(sw, red[sw], "o", color=color, ms=8, mfc="white", mew=1.6)
    axC.axhline(0, color="grey", lw=0.8, ls=":")
    axC.axvline(sw, color="#c62828", lw=0.9, ls=":")
    axC.text(2.4, -0.27, "switch write (novel content):  s42 +0.50 — selective update\n"
             "s43 −0.10 — flush-and-rewrite (§9)",
             fontsize=7.2, color="#c62828", ha="left", va="bottom")
    axC.text(3.6, 1.06, "rehearsal writes (near-copies)", fontsize=7.2, color="#455a64")
    axC.set_ylim(-0.3, 1.22)
    axC.set_xlabel("write # (16-turn conversation,\nswitch at turn 8)")
    axC.set_ylabel("redundancy with resident bank\n(max cosine)")
    axC.tick_params(labelsize=7)
    axC.legend(fontsize=7.5, frameon=False, loc="center right")
    axC.set_title("C — every write, one conversation", fontsize=8.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
    print("wrote", args.out + ".png/.pdf")


if __name__ == "__main__":
    main()
