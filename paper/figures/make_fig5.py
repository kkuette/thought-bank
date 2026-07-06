"""Figure 5 — memory policy is a trained behaviour.

Panel A: per-turn key-0 accuracy around a mid-conversation rule switch
(switch at turn 8 of 16), old rule vs new rule, for the fixed-structure
(zero-shot) model and both structure-randomized (policy-trained) seeds.
Panel B: sweep of the switch position 2-14: new-rule accuracy and old-rule
persistence on the first post-switch query.

Inputs: the three JSON dumps of switch_probe_k2.py --sweep --dump.
Usage: python make_fig5.py <dsv4m.json> <dsv4w_s42.json> <dsv4w_s43.json> [--out PATH]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_ZS = "#c62828"   # zero-shot / fixed-structure
C_42 = "#2e7d32"   # policy-trained, seed 42
C_43 = "#6a1fb1"   # policy-trained, seed 43


def key0_turns(arm):
    """Conversation turn index of each key-0 query (keys alternate, key0 even)."""
    return list(range(0, arm["turns"], 2))


def first_post_switch(arm, series):
    turns = key0_turns(arm)
    for turn, v in zip(turns, series):
        if turn >= arm["sw"]:
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zs_json")
    ap.add_argument("s42_json")
    ap.add_argument("s43_json")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "fig5_policy_trained"))
    args = ap.parse_args()

    zs = json.load(open(args.zs_json))["arms"]
    s42 = json.load(open(args.s42_json))["arms"]
    s43 = json.load(open(args.s43_json))["arms"]
    models = ((zs, C_ZS, "fixed-structure (zero-shot)"),
              (s42, C_42, "structure-randomized s42"),
              (s43, C_43, "structure-randomized s43"))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # --- Panel A: per-turn traces, switch at turn 8 of 16 -------------------
    for arms, color, label in models:
        arm = arms["sw@8"]
        t = key0_turns(arm)
        axA.plot(t, arm["key0"], "o-", color=color, lw=1.4, ms=4,
                 label=f"{label} — current rule")
        axA.plot(t, arm["key0_old_rule"], "o--", color=color, lw=1.1, ms=4,
                 mfc="white", label=f"{label} — old rule")
    axA.axvline(8, color="k", lw=0.9, ls=":")
    axA.text(8.15, 0.62, "switch", rotation=90, fontsize=8, va="center")
    axA.set_xlabel("conversation turn (key-0 queries)")
    axA.set_ylabel("key-0 accuracy")
    axA.set_ylim(-0.05, 1.08)
    axA.set_title("A — around a switch at turn 8 (16-turn conv.)", fontsize=9.5)
    axA.legend(fontsize=6.4, loc="lower left", bbox_to_anchor=(0.01, 0.13), frameon=False)

    # --- Panel B: sweep over switch positions ------------------------------
    pos = sorted(int(k.split("@")[1]) for k in zs)
    for arms, color, _label in models:
        new_acc, old_first = [], []
        for p in pos:
            arm = arms[f"sw@{p}"]
            new_acc.append(first_post_switch(arm, arm["key0"]))
            old_first.append(first_post_switch(arm, arm["key0_old_rule"]))
        axB.plot(pos, new_acc, "o-", color=color, lw=1.4, ms=4, label="new rule")
        axB.plot(pos, old_first, "o--", color=color, lw=1.1, ms=4, mfc="white",
                 label="old rule (persistence)")
    axB.axhline(1 / 128, color="grey", ls=":", lw=0.9)
    axB.text(pos[-1], 1 / 128 + 0.015, "chance", ha="right", fontsize=7, color="grey")
    axB.set_xlabel("switch position (turn)")
    axB.set_ylabel("first post-switch key-0 query")
    axB.set_ylim(-0.05, 1.08)
    axB.set_title("B — sweep of the switch position", fontsize=9.5)

    # shared legend semantics: colour = model, linestyle = which rule
    handles = [
        plt.Line2D([], [], color=C_ZS, lw=1.6, label="fixed-structure (zero-shot)"),
        plt.Line2D([], [], color=C_42, lw=1.6, label="structure-randomized s42"),
        plt.Line2D([], [], color=C_43, lw=1.6, label="structure-randomized s43"),
        plt.Line2D([], [], color="k", lw=1.2, marker="o", ms=4, label="answers new rule"),
        plt.Line2D([], [], color="k", lw=1.0, ls="--", marker="o", ms=4, mfc="white",
                   label="answers old rule"),
    ]
    axB.legend(handles=handles, fontsize=6.6, loc="center right", frameon=False)

    fig.suptitle("Same architecture, same competence — the switch policy is decided by the "
                 "training distribution", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
    print("wrote", args.out + ".png/.pdf")


if __name__ == "__main__":
    main()
