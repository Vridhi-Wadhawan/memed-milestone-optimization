"""Generate the synthetic Meme'd history: campaigns, creators, posts, ladders.

Everything here is an assumption I made up, not a measurement. The key ones
are listed in methodology.md (section "What I baked into the data").
Run: python src/generate_data.py   (fixed seed, so output is reproducible)
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

CATEGORIES = ["gaming", "FMCG", "finance", "D2C", "entertainment"]
PLATFORMS = ["instagram", "youtube"]
TIERS = ["nano", "micro", "mid", "macro"]
FORMATS = ["reel", "short", "long_form", "carousel"]

TIER_RANGES = {
    "nano": (1_500, 9_500),
    "micro": (10_000, 95_000),
    "mid": (100_000, 900_000),
    "macro": (1_000_000, 5_000_000),
}
CATEGORY_EFFECT = {"gaming": 1.20, "FMCG": 0.95, "finance": 0.70,
                   "D2C": 0.90, "entertainment": 1.10}
PLATFORM_EFFECT = {"instagram": 1.05, "youtube": 1.00}
FORMAT_EFFECT = {"reel": 1.15, "short": 1.10, "long_form": 0.90, "carousel": 0.55}
FORMAT_MIX = {"instagram": [0.55, 0.25, 0.12, 0.08],
              "youtube": [0.15, 0.55, 0.25, 0.05]}

# Assumed paid-media CPM in INR per 1,000 views. Replace with real benchmarks.
MEDIA_CPM = {
    ("instagram", "gaming"): 180, ("instagram", "FMCG"): 150,
    ("instagram", "finance"): 190, ("instagram", "D2C"): 145,
    ("instagram", "entertainment"): 170,
    ("youtube", "gaming"): 160, ("youtube", "FMCG"): 135,
    ("youtube", "finance"): 175, ("youtube", "D2C"): 130,
    ("youtube", "entertainment"): 150,
}

POST_NOISE_SD = 1.05      # post-level noise on log views
CAMPAIGN_SHOCK_SD = 0.30  # one shock per campaign, shared by all its posts
FRAUD_RATE = 0.055


def make_manual_ladder(category, tier, platform, campaign_no):
    """Mimic an ops person setting a ladder by feel.

    Thresholds follow a fixed 1x / 4x / 10x / 35x shape scaled by tier.
    The payout rate (share of media value) wanders between campaigns, and
    every 7th campaign is far too generous, every 11th far too stingy.
    """
    base = {"nano": 0.65, "micro": 0.85, "mid": 1.10, "macro": 1.35}[tier]
    cat_factor = {"gaming": 1.10, "FMCG": 1.00, "finance": 0.90,
                  "D2C": 0.95, "entertainment": 1.05}[category]
    rate = 0.28 + 0.06 * ((campaign_no * 7) % 5)
    if campaign_no % 7 == 0:
        base *= 1.65
        rate *= 1.55
    elif campaign_no % 11 == 0:
        base *= 0.62
        rate *= 0.65

    first = max(5_000, int(10_000 * base * cat_factor))
    thresholds = [int(first * r) for r in (1, 4, 10, 35)]
    cpm = MEDIA_CPM[(platform, category)]
    payouts = [max(250, int(round(t / 1000 * cpm * rate / 50) * 50))
               for t in thresholds]
    return list(zip(thresholds, np.maximum.accumulate(payouts)))


def make_creators(rng, n=1200):
    rows = []
    for i in range(n):
        tier = rng.choice(TIERS, p=[0.28, 0.43, 0.24, 0.05])
        lo, hi = TIER_RANGES[tier]
        followers = int(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        quality = float(np.exp(rng.normal(0, 0.38)))  # hidden: size is not quality
        avg_views = followers * 0.18 * quality * rng.lognormal(0, 0.45)
        rows.append({
            "creator_id": f"CR{i + 1:04d}",
            "platform": rng.choice(PLATFORMS, p=[0.68, 0.32]),
            "follower_count": followers,
            "tier": tier,
            "account_age_months": int(np.clip(rng.lognormal(3.5, 0.55), 3, 120)),
            "historical_avg_views_per_post": int(np.clip(avg_views, 300, 4_000_000)),
            "_quality": quality,
        })
    return pd.DataFrame(rows)


def make_campaigns(rng, n=100):
    """Campaign parameters and the manual ladder. Budgets are set later."""
    camp_rows, ladder_rows = [], []
    start0 = pd.Timestamp("2025-01-01")
    for i in range(n):
        start = start0 + pd.Timedelta(days=3 * i)
        end = start + pd.Timedelta(days=int(rng.choice([14, 21, 30, 45])))
        category = rng.choice(CATEGORIES)
        platform = rng.choice(PLATFORMS)
        tier = rng.choice(TIERS, p=[0.25, 0.40, 0.28, 0.07])
        cid = f"CA{i + 1:03d}"
        camp_rows.append({
            "campaign_id": cid, "brand": f"Brand_{i + 1:03d}",
            "category": category, "platform": platform, "total_budget": 0,
            "start_date": start.date(), "end_date": end.date(),
            "target_creator_tier": tier,
            "expected_creator_count": int(rng.integers(30, 201)),
        })
        for rank, (thr, pay) in enumerate(
                make_manual_ladder(category, tier, platform, i + 1), 1):
            ladder_rows.append({"campaign_id": cid, "milestone_rank": rank,
                                "view_threshold": int(thr),
                                "payout_amount": int(pay)})
    return pd.DataFrame(camp_rows), pd.DataFrame(ladder_rows)


def make_posts(rng, campaigns, creators, ladders):
    post_rows, post_id = [], 1
    for _, c in campaigns.iterrows():
        pool = creators[creators.platform == c.platform]
        # 80% of a campaign's creators come from the target tier, 5% from each other tier
        probs = np.array([0.80 if t == c.target_creator_tier else 0.05 for t in TIERS])
        chosen = rng.choice(TIERS, size=int(c.expected_creator_count), p=probs / probs.sum())
        campaign_shock = rng.normal(0, CAMPAIGN_SHOCK_SD)  # drawn once per campaign
        lad = ladders[ladders.campaign_id == c.campaign_id].sort_values("milestone_rank")
        thr, pay = lad.view_threshold.values, lad.payout_amount.values

        # each creator posts at most once per campaign, unless a tier has fewer
        # creators than the campaign wants (then the remainder repeat)
        picked = []
        for tier in TIERS:
            members = pool[pool.tier == tier]
            need = int((chosen == tier).sum())
            if need == 0:
                continue
            first = members.sample(min(need, len(members)), replace=False,
                                   random_state=int(rng.integers(1e9)))
            extra = (members.sample(need - len(first), replace=True,
                                    random_state=int(rng.integers(1e9)))
                     if need > len(first) else members.iloc[:0])
            picked.append(pd.concat([first, extra]))
        campaign_creators = pd.concat(picked).sample(frac=1, random_state=int(rng.integers(1e9)))

        for _, cr in campaign_creators.iterrows():
            fmt = rng.choice(FORMATS, p=FORMAT_MIX[c.platform])
            mean = (np.log(1800)
                    + 0.72 * np.log(max(cr.follower_count, 1000) / 10_000)
                    + np.log(CATEGORY_EFFECT[c.category])
                    + np.log(PLATFORM_EFFECT[c.platform])
                    + np.log(FORMAT_EFFECT[fmt])
                    + np.log(cr["_quality"]))
            final = int(np.clip(np.exp(mean + campaign_shock + rng.normal(0, POST_NOISE_SD)),
                                100, 15_000_000))

            # normal growth curve vs. suspicious (flat start, late spike, inflated total)
            suspicious = bool(rng.random() < FRAUD_RATE)
            if suspicious:
                s24, s7, s30 = (rng.uniform(.01, .05), rng.uniform(.08, .20),
                                rng.uniform(.75, .95))
                final = int(final * rng.uniform(1.3, 2.3))
            else:
                s24, s7, s30 = (rng.uniform(.08, .28), rng.uniform(.35, .70),
                                rng.uniform(.80, 1.03))
            v24 = int(max(100, final * s24))
            v7 = int(max(v24, final * s7))
            v30 = int(max(v7, final * s30))

            hit = np.where(final >= thr)[0]
            payout = int(pay[hit.max()]) if len(hit) else 0
            post_rows.append({
                "post_id": f"PO{post_id:05d}", "campaign_id": c.campaign_id,
                "creator_id": cr.creator_id, "post_date": c.start_date,
                "platform": c.platform, "format": fmt,
                "views_at_24h": v24, "views_at_7d": v7, "views_at_30d": v30,
                "views_final": final, "total_payout_earned": payout,
                "flagged_suspicious": suspicious,
            })
            post_id += 1
    return pd.DataFrame(post_rows)


def set_budgets(rng, campaigns, posts):
    """Budget = what the manual ladder actually paid x a slack factor.

    Real budgets come from the brand, not from the outcome, but I need budgets
    that are the right order of magnitude for each campaign, otherwise the
    budget constraint never matters. About 20% of campaigns get slack < 1 (the
    manual ladder blew through the budget), 60% get a comfortable margin and
    20% are heavily over-budgeted.
    """
    paid = posts.groupby("campaign_id").total_payout_earned.sum()
    out = campaigns.copy()
    budgets = []
    for cid in out.campaign_id:
        kind = rng.choice(["tight", "normal", "loose"], p=[0.20, 0.60, 0.20])
        slack = {"tight": rng.uniform(0.60, 1.00),
                 "normal": rng.uniform(1.15, 1.80),
                 "loose": rng.uniform(2.50, 5.00)}[kind]
        budgets.append(max(10_000, int(round(paid.get(cid, 0) * slack / 5_000) * 5_000)))
    out["total_budget"] = budgets
    return out


def main():
    rng = np.random.default_rng(SEED)
    DATA.mkdir(exist_ok=True)
    creators = make_creators(rng)
    campaigns, ladders = make_campaigns(rng)
    posts = make_posts(rng, campaigns, creators, ladders)
    campaigns = set_budgets(rng, campaigns, posts)

    completion = (posts.assign(hit=posts.total_payout_earned > 0)
                       .groupby("creator_id").hit.mean().rename("historical_completion_rate"))
    creators = (creators.drop(columns="_quality")
                        .merge(completion, on="creator_id", how="left"))
    creators["historical_completion_rate"] = creators.historical_completion_rate.fillna(0)

    campaigns.to_csv(DATA / "campaigns.csv", index=False)
    ladders.to_csv(DATA / "milestone_ladders.csv", index=False)
    creators.to_csv(DATA / "creators.csv", index=False)
    posts.to_csv(DATA / "posts.csv", index=False)
    print(f"{len(campaigns)} campaigns, {len(creators)} creators, {len(posts)} posts")


if __name__ == "__main__":
    main()
