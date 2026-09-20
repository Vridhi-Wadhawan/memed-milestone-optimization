"""Propose a milestone ladder for a new campaign.

Examples
    python run.py
    python run.py --category finance --platform youtube --tier mid --budget 400000 --creators 60
    python run.py --category crypto --platform instagram --tier micro --budget 150000 --creators 40
    python run.py --json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from model import load_data, optimize_ladder  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Milestone ladder for a new Meme'd campaign")
    p.add_argument("--category", default="gaming",
                   help="gaming / FMCG / finance / D2C / entertainment (anything else is treated as a cold start)")
    p.add_argument("--platform", default="instagram", choices=["instagram", "youtube"])
    p.add_argument("--tier", default="micro", choices=["nano", "micro", "mid", "macro"],
                   help="target creator tier")
    p.add_argument("--budget", type=float, default=150_000, help="total campaign budget in INR")
    p.add_argument("--creators", type=int, default=60, help="expected number of creators")
    p.add_argument("--sims", type=int, default=400, help="number of simulated campaigns")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    return p.parse_args()


def main():
    a = parse_args()
    campaign = {"campaign_id": "NEW", "category": a.category, "platform": a.platform,
                "target_creator_tier": a.tier, "total_budget": a.budget,
                "expected_creator_count": a.creators}
    result = optimize_ladder(campaign, load_data(), n_sims=a.sims)
    if a.json:
        print(json.dumps(result, indent=2))
        return

    print(f"\n{a.category} | {a.platform} | {a.tier} creators | {a.creators} creators | "
          f"budget INR {a.budget:,.0f}\n")
    print(f"{'Rung':<6}{'Views':>12}{'Payout (INR)':>15}{'Marginal INR/1K views':>24}{'vs paid CPM':>13}")
    for m in result["milestones"]:
        print(f"{m['milestone_rank']:<6}{m['view_threshold']:>12,}{m['payout_amount']:>15,}"
              f"{m['marginal_payout_per_1k_views']:>24,.0f}{m['share_of_paid_media_cpm']:>13.0%}")
    print(f"\nPays {result['payout_rate_of_media_value']:.0%} of the media value of each threshold "
          f"(assumed paid CPM INR {result['assumed_media_cpm']}/1K views)")
    print(f"Simulated total payout: median INR {result['p50_total_payout']:,}, "
          f"P90 INR {result['p90_total_payout']:,}  "
          f"(P(within budget) = {result['prob_within_budget']:.0%}, target {result['budget_confidence_target']:.0%})")
    print(f"Creators reaching rung 1: {result['first_milestone_reach']:.0%} | "
          f"rung-to-rung continuation: {', '.join(f'{x:.0%}' for x in result['continuation_probabilities'])}")
    print("Rung-1 reach by tier: " + ", ".join(f"{t} {r:.0%}"
                                              for t, r in result["first_milestone_reach_by_tier"].items()))
    for w in result["warnings"]:
        print(f"\nWARNING: {w}")


if __name__ == "__main__":
    main()
