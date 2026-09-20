"""Sanity checks on the ladder logic. Run: python tests/test_ladder.py  (or pytest)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model as M  # noqa: E402

DATA = M.load_data()
CAMPAIGNS = [
    {"campaign_id": "T1", "category": "gaming", "platform": "instagram",
     "target_creator_tier": "micro", "total_budget": 150_000, "expected_creator_count": 60},
    {"campaign_id": "T2", "category": "finance", "platform": "youtube",
     "target_creator_tier": "mid", "total_budget": 400_000, "expected_creator_count": 50},
    {"campaign_id": "T3", "category": "crypto", "platform": "instagram",   # unseen category
     "target_creator_tier": "micro", "total_budget": 100_000, "expected_creator_count": 40},
]


def test_ladder_is_well_formed():
    for c in CAMPAIGNS:
        rungs = M.optimize_ladder(c, DATA, n_sims=200)["milestones"]
        th = [r["view_threshold"] for r in rungs]
        pay = [r["payout_amount"] for r in rungs]
        assert th == sorted(th) and len(set(th)) == len(th), "thresholds must strictly increase"
        assert all(b > a for a, b in zip(pay, pay[1:])), "payouts must strictly increase"
        assert th[0] >= M.MIN_FIRST_THRESHOLD


def test_roi_cap_holds():
    c = CAMPAIGNS[0]
    res = M.optimize_ladder(c, DATA, n_sims=200)
    for r in res["milestones"]:
        assert r["payout_amount"] <= M.ROI_CAP * r["view_threshold"] / 1000 * res["assumed_media_cpm"] + 100


def test_budget_confidence_is_met_when_feasible():
    for c in CAMPAIGNS:
        res = M.optimize_ladder(c, DATA, n_sims=200)
        if not any("Budget" in w or "No ladder" in w for w in res["warnings"]):
            assert res["prob_within_budget"] >= res["budget_confidence_target"] - 0.01


def test_unseen_category_is_flagged_as_cold_start():
    res = M.optimize_ladder(CAMPAIGNS[2], DATA, n_sims=200)
    assert res["cold_start"] and res["budget_confidence_target"] > M.BUDGET_CONFIDENCE


def test_tiny_budget_warns_instead_of_pretending():
    c = {**CAMPAIGNS[1], "total_budget": 2_000}
    assert M.optimize_ladder(c, DATA, n_sims=200)["warnings"]


def test_payout_for_views_is_cumulative_not_additive():
    got = M.payout_for_views(np.array([1_000, 6_000, 25_000]), [5_000, 20_000], [300, 900])
    assert got.tolist() == [0, 300, 900]


def test_variance_components_recovers_known_values():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(200), 40)
    resid = rng.normal(0, 1.0, groups.size) + np.repeat(rng.normal(0, 0.3, 200), 40)
    post, camp = M.variance_components(resid, groups)
    assert abs(post - 1.0) < 0.05 and abs(camp - 0.3) < 0.06


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
