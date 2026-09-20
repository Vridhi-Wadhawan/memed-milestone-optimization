"""Draw the charts used in the one-pager. Reads outputs/*.csv, so run backtest.py first.

Run: python src/make_charts.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs"
CH = OUT / "charts"
MANUAL, PROPOSED = "#9a9a9a", "#1f6feb"


def budget_chart(res):
    fig, ax = plt.subplots(figsize=(6, 5))
    x = res.manual_paid_as_paid / res.budget
    y = res.proposed_paid_clean / res.budget
    ax.scatter(x, y, s=22, color=PROPOSED, alpha=0.7)
    lim = 2.0
    ax.plot([0, lim], [0, lim], color="#bbbbbb", lw=1, ls="--")
    ax.axhline(1, color="#c0392b", lw=1)
    ax.axvline(1, color="#c0392b", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Manual ladder: payout as share of budget")
    ax.set_ylabel("Proposed ladder: payout as share of budget")
    n_over_m, n_over_p = int((x > 1).sum()), int((y > 1).sum())
    ax.set_title(f"Over budget: {n_over_m} campaigns (manual) vs {n_over_p} (proposed)",
                 fontsize=11)
    ax.text(1.03, 0.05, "manual over budget", color="#c0392b", fontsize=8)
    fig.tight_layout()
    fig.savefig(CH / "budget_use.png", dpi=150)
    plt.close(fig)


def tier_chart(tiers):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    idx = np.arange(len(tiers))
    ax.axhspan(0.30, 0.70, color="#e8f0fe", zorder=0)
    ax.bar(idx - 0.2, tiers.manual_completion, 0.4, color=MANUAL, label="Manual")
    ax.bar(idx + 0.2, tiers.proposed_completion, 0.4, color=PROPOSED, label="Proposed")
    ax.set_xticks(idx)
    ax.set_xticklabels(tiers.creator_tier)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Creators reaching the first milestone")
    ax.set_title("Manual ladders: trivial for macro, out of reach for nano\n(shaded = 30-70% target band)",
                 fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(CH / "reach_by_tier.png", dpi=150)
    plt.close(fig)


def continuation_chart(summary):
    manual = [float(v) for v in summary.loc["continuation_manual", "value"].split(" / ")]
    prop = [float(v) for v in summary.loc["continuation_proposed", "value"].split(" / ")]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    idx = np.arange(3)
    ax.axhspan(0.25, 0.65, color="#e8f0fe", zorder=0)
    ax.bar(idx - 0.2, manual, 0.4, color=MANUAL, label="Manual")
    ax.bar(idx + 0.2, prop, 0.4, color=PROPOSED, label="Proposed")
    ax.set_xticks(idx)
    ax.set_xticklabels(["rung 1 to 2", "rung 2 to 3", "rung 3 to 4"])
    ax.set_ylim(0, 0.8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Creators at a rung who reach the next one")
    ax.set_title("Manual ladders get much steeper at the top\n(shaded = 25-65% target band)", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(CH / "continuation.png", dpi=150)
    plt.close(fig)


def ladder_chart(lad):
    order = list(lad.campaign_id.unique())   # picked by rule in backtest.py
    ids = [order[1], order[3]]               # a manual overrun, and a typical campaign
    fig, axes = plt.subplots(1, len(ids), figsize=(6 * len(ids), 4.2))
    axes = np.atleast_1d(axes)
    for ax, cid in zip(axes, ids):
        g = lad[lad.campaign_id == cid]
        x_end = max(g.manual_threshold.max(), g.proposed_threshold.max()) * 1.8
        x_start = min(g.manual_threshold.min(), g.proposed_threshold.min()) / 2
        for prefix, colour, label in (("manual", MANUAL, "Manual"), ("proposed", PROPOSED, "Proposed")):
            x = np.concatenate([[x_start], g[f"{prefix}_threshold"], [x_end]])
            y = np.concatenate([[0], g[f"{prefix}_payout"], [g[f"{prefix}_payout"].iloc[-1]]])
            ax.step(x, y, where="post", color=colour, lw=2, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("Views")
        ax.set_ylabel("Payout (INR)")
        ax.set_title(cid, fontsize=10)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(CH / "ladders_example.png", dpi=150)
    plt.close(fig)


def main():
    CH.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(OUT / "backtest_all_campaigns.csv")
    budget_chart(res)
    tier_chart(pd.read_csv(OUT / "tier_completion.csv"))
    continuation_chart(pd.read_csv(OUT / "summary.csv", index_col=0))
    ladder_chart(pd.read_csv(OUT / "ladders_selected.csv"))
    print("charts written to", CH)


if __name__ == "__main__":
    main()
