"""Figure 3 — training dynamics of the policy cell (dsv4w), seeds 42 and 43.

Top row: cross-entropy (log scale) with curriculum milestones and the
teacher-anneal window. Bottom row: rule accuracy on unseen queries, train
pool vs held-out rules, with the write-distillation loss on a twin axis.

Data: runs/dsv4mini_multiturn_rule_k2_inter_s128_dsv4w{,_s43}/metrics.jsonl
Usage: python make_fig3.py [--runs-root PATH] [--out PATH]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "runs")

SEEDS = {
    42: {
        "dir": "dsv4mini_multiturn_rule_k2_inter_s128_dsv4w",
        # pool doublings 16->32->64 (mastery-gated curriculum), from train log
        "milestones": [553, 814, 1004],
        "eval_step": 3000,
    },
    43: {
        "dir": "dsv4mini_multiturn_rule_k2_inter_s128_dsv4w_s43",
        "milestones": None,  # not logged for this run
        "eval_step": 4000,
    },
}


def load(runs_root, subdir):
    path = os.path.join(runs_root, subdir, "metrics.jsonl")
    return [json.loads(line) for line in open(path)]


def anneal_window(rows):
    steps = [r["step"] for r in rows if 0.0 < r["tf_beta"] < 1.0]
    if not steps:
        return None
    # metrics are sampled every 50 steps; widen to the enclosing samples
    return steps[0] - 50, steps[-1] + 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=DEFAULT_RUNS)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "fig3_training_dynamics"))
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.6), sharex="col")

    for col, (seed, meta) in enumerate(SEEDS.items()):
        rows = load(args.runs_root, meta["dir"])
        steps = [r["step"] for r in rows]
        ce = [r["ce"] for r in rows]
        distill = [r["tf_distill"] for r in rows]
        ev = [(r["step"], r["rule_acc"], r["rule_acc_held"]) for r in rows if "rule_acc" in r]
        ev_s, ev_tr, ev_hd = zip(*ev)
        win = anneal_window(rows)

        ax = axes[0][col]
        ax.plot(steps, ce, color="#1f4e79", lw=1.4, label="CE")
        ax.axhline(4.852, color="grey", ls=":", lw=0.9)
        ax.text(steps[-1] * 0.99, 4.852 * 1.12, "ln 128", va="bottom", ha="right", fontsize=7, color="grey")
        ax.set_yscale("log")
        ax.set_title(f"seed {seed}", fontsize=10)
        if col == 0:
            ax.set_ylabel("cross-entropy")

        ax2 = axes[1][col]
        ax2.plot(ev_s, ev_tr, "o-", color="#2e7d32", ms=3, lw=1.2, label="train rules")
        ax2.plot(ev_s, ev_hd, "s-", color="#c62828", ms=3, lw=1.2, label="held rules")
        ax2.axhline(1 / 128, color="grey", ls=":", lw=0.9)
        ax2.text(steps[-1], 1 / 128, " chance", va="bottom", ha="left", fontsize=7, color="grey")
        ax2.set_ylim(-0.04, 1.06)
        ax2.set_xlabel("training step")
        if col == 0:
            ax2.set_ylabel("rule accuracy\n(unseen queries)")

        ax3 = ax2.twinx()
        ax3.plot(steps, distill, color="#9e9e9e", lw=1.0, alpha=0.8, label="write distill")
        ax3.set_ylim(0, 1.6)
        ax3.tick_params(axis="y", labelsize=7, colors="#757575")
        if col == 1:
            ax3.set_ylabel("distill (cos)", fontsize=8, color="#757575")

        for a in (ax, ax2):
            if win:
                a.axvspan(*win, color="#ffb74d", alpha=0.25, lw=0)
            if meta["milestones"]:
                for m in meta["milestones"]:
                    a.axvline(m, color="#7b1fa2", ls="--", lw=0.8, alpha=0.6)
        if win:
            axes[1][col].text(
                sum(win) / 2, 1.01, "β anneal",
                ha="center", va="bottom", fontsize=7, color="#e65100",
            )
        if meta["milestones"]:
            axes[1][col].text(
                meta["milestones"][0] - 60, 1.01, "curriculum\ndoublings",
                ha="right", va="bottom", fontsize=7, color="#7b1fa2",
            )
        # final headline point
        ax2.annotate(
            f"held {ev_hd[-1]:.2f}", (ev_s[-1], ev_hd[-1]),
            textcoords="offset points", xytext=(-6, -14), fontsize=8, color="#c62828", ha="right",
        )

    handles, labels = axes[1][0].get_legend_handles_labels()
    h3, l3 = axes[1][0].get_shared_x_axes(), None
    axes[1][0].legend(loc="center left", fontsize=8, frameon=False)
    fig.suptitle("Training dynamics of the policy cell (structure-randomized, K=2)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
    print("wrote", args.out + ".png/.pdf")


if __name__ == "__main__":
    main()
