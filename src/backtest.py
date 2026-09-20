"""Backtest: run the method on every historical campaign without letting it see
that campaign, then compare with the ladder that was actually used.

For each campaign the model is fitted on the other 99, a ladder is proposed
from the campaign's inputs only (category, platform, tier, budget, creator
count), and both ladders are applied to the views the campaign really got.

Comparison basis, so nothing is flattered:
  * Manual ladder, "as paid": what Meme'd actually paid, flagged posts included.
  * Both ladders on clean posts only: like-for-like ladder comparison.
  * Proposed ladder without a fraud gate: what it would pay if flagged posts
    were paid on raw views.

Run: python src/backtest.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from model import (CONTINUE_RANGE, M1_REACH_RANGE, TIERS, _hits, load_data, media_cpm,
                   optimize_ladder, payout_for_views)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
N_SIMS = 400


def run_campaign(c, data, exclude_category=False):
    kwargs = ({"exclude_category": c["category"]} if exclude_category
              else {"exclude_campaign": c["campaign_id"]})
    res, (views, present, _) = optimize_ladder(c, data, n_sims=N_SIMS, seed=1000 + int(c["campaign_id"][2:]),
                                               return_sim=True, **kwargs)
    manual = data.ladders[data.ladders.campaign_id == c["campaign_id"]].sort_values("milestone_rank")
    m_th, m_pay = manual.view_threshold.values, manual.payout_amount.values
    p_th = [m["view_threshold"] for m in res["milestones"]]
    p_pay = [m["payout_amount"] for m in res["milestones"]]

    posts = data.posts[data.posts.campaign_id == c["campaign_id"]]
    clean = posts[~posts.flagged_suspicious]
    v_clean, v_all = clean.views_final.values, posts.views_final.values

    # simulated total under the MANUAL ladder: used to check the model is calibrated
    hits = _hits(views, present, m_th)
    sim_manual = hits @ np.diff(np.concatenate([[0], m_pay]))
    cpm = media_cpm(c)

    def completion(v, th):
        return float((v >= th[0]).mean())

    row = {
        "campaign_id": c["campaign_id"], "category": c["category"], "platform": c["platform"],
        "tier": c["target_creator_tier"], "budget": int(c["total_budget"]),
        "creators": len(posts), "clean_creators": len(clean),
        "manual_paid_as_paid": int(posts.total_payout_earned.sum()),
        "manual_paid_clean": int(payout_for_views(v_clean, m_th, m_pay).sum()),
        "proposed_paid_clean": int(payout_for_views(v_clean, p_th, p_pay).sum()),
        "proposed_paid_raw": int(payout_for_views(v_all, p_th, p_pay).sum()),
        "manual_completion": completion(v_clean, m_th),
        "proposed_completion": completion(v_clean, p_th),
        "manual_first_threshold": int(m_th[0]), "proposed_first_threshold": int(p_th[0]),
        "manual_payout_share_of_cpm": None, "proposed_payout_share_of_cpm": None,
        "sim_p90_proposed": res["p90_total_payout"],
        "sim_p10_manual": float(np.quantile(sim_manual, 0.10)),
        "sim_p90_manual": float(np.quantile(sim_manual, 0.90)),
        "payout_rate_of_media_value": res["payout_rate_of_media_value"],
        "cold_start": res["cold_start"], "n_warnings": len(res["warnings"]),
    }
    per_1k = v_clean.sum() / 1000
    row["manual_payout_share_of_cpm"] = row["manual_paid_clean"] / per_1k / cpm
    row["proposed_payout_share_of_cpm"] = row["proposed_paid_clean"] / per_1k / cpm
    row["manual_over_budget"] = row["manual_paid_as_paid"] > row["budget"]
    row["proposed_over_budget"] = row["proposed_paid_clean"] > row["budget"]
    row["proposed_over_budget_ungated"] = row["proposed_paid_raw"] > row["budget"]
    row["proposed_within_sim_p90"] = row["proposed_paid_clean"] <= row["sim_p90_proposed"]
    row["manual_in_sim_p10_p90"] = (row["sim_p10_manual"] <= row["manual_paid_clean"]
                                    <= row["sim_p90_manual"])
    return row, res, (m_th, m_pay, p_th, p_pay)


def pooled_continuation(data, ladders_by_campaign, which):
    """P(reach rung k+1 | reached rung k) across all clean posts, per rung."""
    clean = data.posts[~data.posts.flagged_suspicious]
    reached = np.zeros(4)
    for cid, g in clean.groupby("campaign_id"):
        th = ladders_by_campaign[cid][which]
        reached += (g.views_final.values[:, None] >= np.asarray(th)[None, :]).sum(axis=0)
    return reached[1:] / reached[:-1], reached[0] / len(clean)


def tier_completion(data, ladders_by_campaign):
    clean = data.posts[~data.posts.flagged_suspicious].merge(
        data.creators[["creator_id", "tier"]], on="creator_id")
    rows = []
    for tier in TIERS:
        g = clean[clean.tier == tier]
        hit = {"manual": 0, "proposed": 0}
        for cid, gg in g.groupby("campaign_id"):
            for which, key in (("manual", 0), ("proposed", 2)):
                hit[which] += (gg.views_final.values >= ladders_by_campaign[cid][key][0]).sum()
        rows.append({"creator_tier": tier, "posts": len(g),
                     "manual_completion": hit["manual"] / len(g),
                     "proposed_completion": hit["proposed"] / len(g)})
    return pd.DataFrame(rows)


def pick_illustrative(res):
    """Five campaigns chosen by rule, not by hand."""
    r = res.assign(util=res.manual_paid_as_paid / res.budget)
    picks = list(r.nlargest(2, "util").campaign_id)                 # manual blew the budget most
    picks.append(r.nsmallest(1, "util").campaign_id.iloc[0])        # manual left most budget unspent
    mid = r[~r.campaign_id.isin(picks)]
    med = mid.util.median()
    picks += list(mid.iloc[(mid.util - med).abs().argsort()[:2]].campaign_id)  # two typical ones
    return picks


def md_table(df):
    head = "| " + " | ".join(df.columns) + " |\n"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|\n"
    body = "".join("| " + " | ".join(str(x) for x in row) + " |\n" for row in df.values)
    return head + sep + body


def main():
    OUT.mkdir(exist_ok=True)
    data = load_data()
    rows, ladders = [], {}
    for _, c in data.campaigns.iterrows():
        row, _, lad = run_campaign(c.to_dict(), data)
        rows.append(row)
        ladders[c.campaign_id] = lad
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "backtest_all_campaigns.csv", index=False)

    # ---- headline numbers -------------------------------------------------
    over = lambda paid, b: float(np.maximum(paid - b, 0).sum())
    cont_m, m1_m = pooled_continuation(data, ladders, 0)
    cont_p, m1_p = pooled_continuation(data, ladders, 2)
    lo, hi = M1_REACH_RANGE
    summary = pd.DataFrame([{
        "campaigns": len(res),
        "manual_over_budget": int(res.manual_over_budget.sum()),
        "proposed_over_budget_gated": int(res.proposed_over_budget.sum()),
        "proposed_over_budget_ungated": int(res.proposed_over_budget_ungated.sum()),
        "manual_overspend_inr": over(res.manual_paid_as_paid, res.budget),
        "proposed_overspend_gated_inr": over(res.proposed_paid_clean, res.budget),
        "proposed_overspend_ungated_inr": over(res.proposed_paid_raw, res.budget),
        "manual_median_budget_used": float((res.manual_paid_as_paid / res.budget).median()),
        "proposed_median_budget_used": float((res.proposed_paid_clean / res.budget).median()),
        "manual_total_paid_as_paid": int(res.manual_paid_as_paid.sum()),
        "manual_total_paid_clean": int(res.manual_paid_clean.sum()),
        "proposed_total_paid_clean": int(res.proposed_paid_clean.sum()),
        "manual_paid_to_flagged_posts": int((res.manual_paid_as_paid - res.manual_paid_clean).sum()),
        "manual_mean_completion": float(res.manual_completion.mean()),
        "proposed_mean_completion": float(res.proposed_completion.mean()),
        "manual_share_m1_in_target_band": float(res.manual_completion.between(lo, hi).mean()),
        "proposed_share_m1_in_target_band": float(res.proposed_completion.between(lo, hi).mean()),
        "manual_median_payout_share_of_cpm": float(res.manual_payout_share_of_cpm.median()),
        "proposed_median_payout_share_of_cpm": float(res.proposed_payout_share_of_cpm.median()),
        "proposed_within_sim_p90_share": float(res.proposed_within_sim_p90.mean()),
        "manual_in_sim_p10_p90_share": float(res.manual_in_sim_p10_p90.mean()),
        "continuation_manual": " / ".join(f"{x:.2f}" for x in cont_m),
        "continuation_proposed": " / ".join(f"{x:.2f}" for x in cont_p),
    }]).T.rename(columns={0: "value"})
    summary.to_csv(OUT / "summary.csv")

    tiers = tier_completion(data, ladders)
    tiers.to_csv(OUT / "tier_completion.csv", index=False)

    # ---- five illustrative campaigns -------------------------------------
    picks = pick_illustrative(res)
    sel = res[res.campaign_id.isin(picks)].set_index("campaign_id").loc[picks].reset_index()
    sel.to_csv(OUT / "backtest_selected.csv", index=False)
    lad_rows = []
    for cid in picks:
        m_th, m_pay, p_th, p_pay = ladders[cid]
        for k in range(4):
            lad_rows.append({"campaign_id": cid, "milestone_rank": k + 1,
                             "manual_threshold": int(m_th[k]), "manual_payout": int(m_pay[k]),
                             "proposed_threshold": int(p_th[k]), "proposed_payout": int(p_pay[k])})
    pd.DataFrame(lad_rows).to_csv(OUT / "ladders_selected.csv", index=False)

    # ---- cold start: hide the whole category ------------------------------
    cold_rows = []
    for cat, g in data.campaigns.groupby("category"):
        out = [run_campaign(c.to_dict(), data, exclude_category=True)[0] for _, c in g.iterrows()]
        o = pd.DataFrame(out)
        cold_rows.append({"hidden_category": cat, "campaigns": len(o),
                          "flagged_cold_start": int(o.cold_start.sum()),
                          "proposed_over_budget": int(o.proposed_over_budget.sum()),
                          "within_sim_p90_share": round(float(o.proposed_within_sim_p90.mean()), 3),
                          "mean_completion": round(float(o.proposed_completion.mean()), 3)})
    cold = pd.DataFrame(cold_rows)
    cold.to_csv(OUT / "cold_start_check.csv", index=False)

    write_report(res, summary, sel, ladders, tiers, cold, picks)
    print(summary.to_string())
    print(tiers.round(3).to_string(index=False))
    print(cold.to_string(index=False))
    print(sel[["campaign_id", "budget", "manual_paid_as_paid", "proposed_paid_clean",
               "manual_completion", "proposed_completion"]].round(3).to_string(index=False))


def write_report(res, summary, sel, ladders, tiers, cold, picks):
    s = summary["value"]
    inr = lambda x: f"₹{int(round(float(x))):,}"
    pct = lambda x: f"{float(x):.0%}"
    lines = ["# Backtest results (generated by src/backtest.py, do not edit by hand)\n",
             "Leave-one-campaign-out on all 100 synthetic campaigns. Numbers come from "
             "`outputs/backtest_all_campaigns.csv`. Synthetic data, so read these as a test "
             "of the method under my assumptions, not as evidence about real Meme'd campaigns.\n",
             "## All 100 campaigns\n",
             md_table(pd.DataFrame([
                 ["Campaigns over budget", s.manual_over_budget, s.proposed_over_budget_gated,
                  s.proposed_over_budget_ungated],
                 ["Total overspend", inr(s.manual_overspend_inr), inr(s.proposed_overspend_gated_inr),
                  inr(s.proposed_overspend_ungated_inr)],
                 ["Median share of budget used", pct(s.manual_median_budget_used),
                  pct(s.proposed_median_budget_used), "-"],
                 ["Mean share of creators reaching rung 1", pct(s.manual_mean_completion),
                  pct(s.proposed_mean_completion), "-"],
                 ["Campaigns with rung-1 reach in 30-70% band", pct(s.manual_share_m1_in_target_band),
                  pct(s.proposed_share_m1_in_target_band), "-"],
                 ["Median payout as share of paid-media CPM", pct(s.manual_median_payout_share_of_cpm),
                  pct(s.proposed_median_payout_share_of_cpm), "-"],
                 ["Continuation rung1>2 / 2>3 / 3>4", s.continuation_manual, s.continuation_proposed, "-"],
             ], columns=["Measure", "Manual ladder (as paid)", "Proposed (fraud gate on)",
                         "Proposed (no fraud gate)"])),
             f"\nThe manual ladders paid {inr(s.manual_paid_to_flagged_posts)} to flagged posts in total. "
             "The proposed ladders are compared on clean posts, so that saving is not credited to the ladder itself.\n",
             "## Model calibration on held-out campaigns\n",
             f"- Realised payout of the proposed ladder was at or below the simulated P90 in "
             f"{pct(s.proposed_within_sim_p90_share)} of campaigns (target 90%).\n"
             f"- Realised payout of the manual ladder fell inside the simulated P10 to P90 range in "
             f"{pct(s.manual_in_sim_p10_p90_share)} of campaigns (target 80%).\n",
             "## Rung-1 reach by creator tier (all campaigns pooled)\n",
             md_table(tiers.assign(posts=tiers.posts,
                                   manual_completion=tiers.manual_completion.map(pct),
                                   proposed_completion=tiers.proposed_completion.map(pct))),
             "\n## Five illustrative campaigns (picked by rule)\n",
             "Rule: the two where the manual ladder went furthest over budget, the one where it left the most "
             "budget unspent, and the two closest to the median.\n",
             md_table(pd.DataFrame({
                 "Campaign": sel.campaign_id,
                 "Category / tier": sel.category + " / " + sel.tier,
                 "Budget": sel.budget.map(inr),
                 "Manual paid": sel.manual_paid_as_paid.map(inr),
                 "Proposed paid (clean)": sel.proposed_paid_clean.map(inr),
                 "Proposed paid (no gate)": sel.proposed_paid_raw.map(inr),
                 "Sim P90": sel.sim_p90_proposed.map(inr),
                 "Rung-1 reach manual": sel.manual_completion.map(pct),
                 "Rung-1 reach proposed": sel.proposed_completion.map(pct)})),
             "\n### Ladders side by side (views: payout)\n"]
    for cid in picks:
        m_th, m_pay, p_th, p_pay = ladders[cid]
        fmt = lambda th, pay: ", ".join(f"{int(t):,}: ₹{int(p):,}" for t, p in zip(th, pay))
        lines.append(f"**{cid}**\n- manual: {fmt(m_th, m_pay)}\n- proposed: {fmt(p_th, p_pay)}\n")
    lines += ["\n## Hiding a whole category (cold start)\n",
              "Every campaign in a category is run with that category removed from the training data.\n",
              md_table(cold)]
    (OUT / "backtest_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
