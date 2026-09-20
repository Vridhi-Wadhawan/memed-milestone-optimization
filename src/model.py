"""View model and milestone-ladder optimiser.

Flow:
  1. fit_view_model: Ridge on log(final views) using clean (non-flagged) posts.
     Also estimates how much of the leftover noise is shared by a whole
     campaign (which drives budget risk) and how much is per post.
  2. simulate_views: simulate the campaign many times (who joins, which
     format, campaign-wide shock, post-level noise).
  3. optimize_ladder: search candidate ladders, keep those that pass the
     constraints, pick the one closest to the target "shape", then set the
     payouts as high as the budget and the ROI cap allow.

Every constant a reviewer might want to argue with is at the top.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TIERS = ["nano", "micro", "mid", "macro"]
FORMATS = ["reel", "short", "long_form", "carousel"]

# Assumed paid-media CPM (INR per 1,000 views). Placeholder until Meme'd gives real ones.
MEDIA_CPM = {
    ("instagram", "gaming"): 180, ("instagram", "FMCG"): 150,
    ("instagram", "finance"): 190, ("instagram", "D2C"): 145,
    ("instagram", "entertainment"): 170,
    ("youtube", "gaming"): 160, ("youtube", "FMCG"): 135,
    ("youtube", "finance"): 175, ("youtube", "D2C"): 130,
    ("youtube", "entertainment"): 150,
}
DEFAULT_CPM = 150

# ---- What "optimised" means (see methodology.md, section 2) -----------------
BUDGET_CONFIDENCE = 0.90       # P(total payout <= budget) must be at least this
COLD_START_CONFIDENCE = 0.95   # stricter when we have little history
ROI_CAP = 0.40                 # never pay more than 40% of what the reach costs on paid media
RATE_MIN = 0.15                # below 15% of media value creators will not bother
RATE_GRID = np.arange(0.10, ROI_CAP + 1e-9, 0.025)
TARGET_M1_REACH = 0.50         # share of creators who should reach the first milestone
TARGET_CONTINUE = 0.40         # share of creators at rung k who go on to reach rung k+1
M1_REACH_RANGE = (0.30, 0.70)
CONTINUE_RANGE = (0.25, 0.65)  # below: creators give up. above: rung is trivial to climb
MIN_FIRST_THRESHOLD = 3_000    # below this a tiny account can grind the first rung cheaply
RATIO_GRID = (2.0, 2.5, 3.0, 4.0)   # allowed threshold multiples between rungs
N_RUNGS = 4
COLD_START_MIN_POSTS = 60      # fewer similar historical posts than this = cold start
COLD_START_SIGMA_BOOST = 1.25


@dataclass
class Data:
    campaigns: pd.DataFrame
    creators: pd.DataFrame
    posts: pd.DataFrame
    ladders: pd.DataFrame


def load_data(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    d = root / "data"
    return Data(pd.read_csv(d / "campaigns.csv"), pd.read_csv(d / "creators.csv"),
                pd.read_csv(d / "posts.csv"), pd.read_csv(d / "milestone_ladders.csv"))


def media_cpm(campaign):
    return MEDIA_CPM.get((campaign["platform"], campaign["category"]), DEFAULT_CPM)


# ---------------------------------------------------------------- view model
@dataclass
class ViewModel:
    pipe: Pipeline
    sigma_post: float        # per-post noise (log scale)
    sigma_campaign: float    # noise shared by every post in a campaign (log scale)
    flag_rate: float         # share of historical posts flagged as suspicious
    tier_mix: dict           # target tier -> share of creators from each tier
    format_mix: dict         # platform -> share of posts per format
    train_posts: pd.DataFrame


FEATURE_CAT = ["tier", "platform", "format", "category"]
FEATURE_NUM = ["log_followers", "log_hist_avg"]


def _design(df):
    out = df.copy()
    out["log_followers"] = np.log1p(out.follower_count)
    out["log_hist_avg"] = np.log1p(out.historical_avg_views_per_post)
    return out


def _training_frame(data, exclude_campaign=None, exclude_category=None):
    p = data.posts.merge(
        data.creators[["creator_id", "follower_count", "tier", "historical_avg_views_per_post"]],
        on="creator_id")
    p = p.merge(data.campaigns[["campaign_id", "category", "target_creator_tier"]],
                on="campaign_id")
    if exclude_campaign is not None:
        p = p[p.campaign_id != exclude_campaign]
    if exclude_category is not None:
        p = p[p.category != exclude_category]
    return p


def variance_components(resid, groups):
    """Split residual variance into per-post noise and a per-campaign shared part.

    Method of moments: the variance of campaign-mean residuals equals
    sigma_campaign^2 + sigma_post^2 / n_posts, so subtract the second term.
    """
    df = pd.DataFrame({"r": resid, "g": groups})
    means = df.groupby("g").r.transform("mean")
    n = df.groupby("g").r.transform("count")
    within = ((df.r - means) ** 2).sum() / (len(df) - df.g.nunique())
    per_campaign = df.groupby("g").r.agg(["mean", "count"])
    var_means = per_campaign["mean"].var(ddof=1)
    camp_var = max(0.0, var_means - float((within / per_campaign["count"]).mean()))
    return float(np.sqrt(within)), float(np.sqrt(camp_var))


def fit_view_model(data, exclude_campaign=None, exclude_category=None):
    train = _training_frame(data, exclude_campaign, exclude_category)
    clean = _design(train[~train.flagged_suspicious])
    y = np.log1p(clean.views_final.values)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_CAT),
        ("num", "passthrough", FEATURE_NUM),
    ])
    pipe = Pipeline([("pre", pre), ("ridge", Ridge(alpha=2.0))])
    pipe.fit(clean[FEATURE_CAT + FEATURE_NUM], y)
    resid = y - pipe.predict(clean[FEATURE_CAT + FEATURE_NUM])
    sigma_post, sigma_camp = variance_components(resid, clean.campaign_id.values)

    # who joins a campaign depends on its target tier, and formats depend on the platform:
    # both are read off the history rather than hard-coded
    tier_mix = {}
    for target, g in train.groupby("target_creator_tier"):
        share = g.tier.value_counts(normalize=True).reindex(TIERS).fillna(0.0)
        tier_mix[target] = share.values
    format_mix = {plat: g.format.value_counts(normalize=True).reindex(FORMATS).fillna(0.0).values
                  for plat, g in train.groupby("platform")}
    flag_rate = float(train.flagged_suspicious.mean())
    return ViewModel(pipe, sigma_post, sigma_camp, flag_rate, tier_mix, format_mix, train)


def count_similar_posts(model, campaign):
    t = model.train_posts
    return int(((t.platform == campaign["platform"]) & (t.category == campaign["category"])
                & (t.target_creator_tier == campaign["target_creator_tier"])).sum())


# ---------------------------------------------------------------- simulation
def simulate_views(campaign, model, creators, n_sims=400, seed=123, sigma_boost=1.0):
    """Return (views, present, tiers), each shaped (n_sims, n_max).

    Per simulation: how many clean creators join (Poisson around the planned count),
    which creators, which format, one shock shared by the whole campaign, and
    post-level noise. `present` marks the creators who actually joined.
    """
    rng = np.random.default_rng(seed)
    n_mean = int(campaign["expected_creator_count"])
    n_max = 2 * n_mean + 10
    pool = creators[creators.platform == campaign["platform"]].reset_index(drop=True)

    # expected log views for every (creator, format) pair, predicted once
    grid = pd.concat([pool.assign(format=f) for f in FORMATS], ignore_index=True)
    grid["category"] = campaign["category"]
    grid["platform"] = campaign["platform"]
    pred = model.pipe.predict(_design(grid)[FEATURE_CAT + FEATURE_NUM])
    pred = pred.reshape(len(FORMATS), len(pool)).T          # (creator, format)

    mix = model.tier_mix.get(campaign["target_creator_tier"], np.full(4, 0.25))
    mix = np.array([m if (pool.tier == t).any() else 0.0 for m, t in zip(mix, TIERS)])
    mix = mix / mix.sum()
    tier_draw = rng.choice(len(TIERS), size=(n_sims, n_max), p=mix)
    creator_idx = np.zeros((n_sims, n_max), dtype=int)
    for k, t in enumerate(TIERS):
        members = np.where(pool.tier.values == t)[0]
        mask = tier_draw == k
        if len(members) and mask.any():
            creator_idx[mask] = rng.choice(members, size=int(mask.sum()))
    fmt_idx = rng.choice(len(FORMATS), size=(n_sims, n_max),
                         p=model.format_mix[campaign["platform"]])

    campaign_shock = rng.normal(0, model.sigma_campaign * sigma_boost, size=(n_sims, 1))
    post_noise = rng.normal(0, model.sigma_post * sigma_boost, size=(n_sims, n_max))
    log_views = pred[creator_idx, fmt_idx] + campaign_shock + post_noise
    views = np.clip(np.expm1(log_views), 100, 15_000_000)

    # flagged posts are not paid, so only the clean share of creators counts
    joined = rng.poisson(n_mean * (1 - model.flag_rate), size=n_sims)
    present = np.arange(n_max)[None, :] < joined[:, None]
    return views, present, np.array(TIERS)[tier_draw]


# ---------------------------------------------------------------- ladders
def nice_round(x):
    """Round to a threshold a human would write down: 3K, 5K, 7.5K, 10K, 25K ..."""
    steps = np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5, 10])
    mag = 10 ** np.floor(np.log10(x))
    return int(mag * steps[np.argmin(np.abs(np.log(steps) - np.log(x / mag)))])


def payouts_for(thresholds, rate, cpm):
    """Cumulative payout at each rung = rate x media value of the threshold.

    Floored to the nearest Rs 50 (never rounded up, so the ROI cap holds) with
    a Rs 100 minimum, and each rung pays at least Rs 50 more than the one below.
    """
    raw = np.array([t / 1000 * cpm * rate for t in thresholds])
    pay = np.maximum(100, np.floor(raw / 50) * 50)
    for k in range(1, len(pay)):                      # every rung pays at least Rs 50 more
        pay[k] = max(pay[k], pay[k - 1] + 50)
    return pay.astype(int)


def payout_for_views(views, thresholds, payouts):
    """A creator earns the payout of the highest rung reached (cumulative ladder)."""
    idx = np.searchsorted(np.asarray(thresholds), np.asarray(views), side="right") - 1
    return np.where(idx >= 0, np.asarray(payouts)[np.maximum(idx, 0)], 0)


def _hits(views, present, thresholds):
    """Creators at or above each threshold, per simulation: shape (n_sims, n_rungs)."""
    return ((views[:, :, None] >= np.asarray(thresholds)[None, None, :])
            & present[:, :, None]).sum(axis=1)


def _candidate_structures(views, present, floor):
    flat = views[present]
    seen = set()
    for reach in (0.30, 0.40, 0.50, 0.60, 0.70):
        first = np.quantile(flat, 1 - reach)
        floor_bound = first < floor
        first = max(first, floor)
        for r2 in RATIO_GRID:
            for r3 in RATIO_GRID:
                for r4 in RATIO_GRID:
                    th = [nice_round(first)]
                    for r in (r2, r3, r4):
                        th.append(nice_round(th[-1] * r))
                    key = tuple(th)
                    if key in seen or any(th[i + 1] / th[i] < 1.5 for i in range(N_RUNGS - 1)):
                        continue
                    seen.add(key)
                    yield th, floor_bound


def optimize_ladder(campaign, data, n_sims=400, seed=123,
                    exclude_campaign=None, exclude_category=None, return_sim=False):
    """Propose a four-rung ladder for one campaign. Returns a dict.

    exclude_campaign / exclude_category hide history from the model (used by the
    backtest). return_sim=True also returns the simulated (views, present, tiers).
    """
    model = fit_view_model(data, exclude_campaign, exclude_category)
    n_similar = count_similar_posts(model, campaign)
    cold = n_similar < COLD_START_MIN_POSTS
    conf = COLD_START_CONFIDENCE if cold else BUDGET_CONFIDENCE
    views, present, tiers = simulate_views(
        campaign, model, data.creators, n_sims, seed,
        sigma_boost=COLD_START_SIGMA_BOOST if cold else 1.0)

    budget = float(campaign["total_budget"])
    cpm = media_cpm(campaign)
    n_join = present.sum(axis=1)
    candidates = []
    for th, floor_bound in _candidate_structures(views, present, MIN_FIRST_THRESHOLD):
        hits = _hits(views, present, th)                       # (sims, rungs)
        reach = hits / n_join[:, None]
        r_pool = hits.sum(axis=0) / n_join.sum()               # pooled reach per rung
        m1 = float(r_pool[0])
        cont = r_pool[1:] / r_pool[:-1]
        if m1 > M1_REACH_RANGE[1] or (m1 < M1_REACH_RANGE[0] and not floor_bound):
            continue
        if cont.min() < CONTINUE_RANGE[0] or cont.max() > CONTINUE_RANGE[1]:
            continue
        shape_gap = abs(m1 - TARGET_M1_REACH) + float(np.mean(np.abs(cont - TARGET_CONTINUE)))

        # highest payout rate that still meets the budget confidence
        best = None
        for rate in RATE_GRID[::-1]:
            pay = payouts_for(th, rate, cpm)
            step = np.diff(np.concatenate([[0], pay]))         # extra pay for each new rung
            total = hits @ step
            if np.quantile(total, conf) <= budget:
                best = (rate, pay, total)
                break
        if best is None:
            continue
        rate, pay, total = best
        candidates.append(dict(th=th, pay=pay, rate=float(rate), total=total, m1=m1, cont=cont,
                               reach=reach, gap=shape_gap, floor_bound=floor_bound))

    warnings = []
    feasible = [c for c in candidates if c["rate"] >= RATE_MIN]
    if not feasible and candidates:
        # budget only supports a payout rate below the floor: use the highest rate available
        top_rate = max(c["rate"] for c in candidates)
        feasible = [c for c in candidates if c["rate"] == top_rate]
        warnings.append(f"Budget is tight: at {conf:.0%} confidence it only supports paying "
                        f"{top_rate:.0%} of media value, below the {RATE_MIN:.0%} floor. "
                        "Raise the budget or cut the number of creators.")
    if feasible:
        best_gap = min(c["gap"] for c in feasible)
        near = [c for c in feasible if c["gap"] <= best_gap + 0.05]
        chosen = max(near, key=lambda c: (c["rate"], -c["gap"]))
    else:
        chosen = _cheapest_fallback(views, present, cpm, budget, conf)
        warnings.append("No ladder fits this budget at the required confidence even at the "
                        "lowest payout rate. Showing the cheapest ladder found.")
    th, pay, total = chosen["th"], chosen["pay"], chosen["total"]
    if chosen["floor_bound"]:
        warnings.append("First milestone is held at the minimum threshold, so it is "
                        "hard to reach for most of this pool. A flat participation fee "
                        "may work better than a ladder for these creators.")
    if cold:
        warnings.append(f"Cold start: only {n_similar} similar historical posts. Uncertainty "
                        "widened and budget confidence raised.")

    # who can actually reach the first rung, by creator tier
    flat_v, flat_t = views[present], tiers[present]
    tier_reach = {t: round(float((flat_v[flat_t == t] >= th[0]).mean()), 3)
                  for t in TIERS if (flat_t == t).any()}
    for t, r in tier_reach.items():
        if (flat_t == t).mean() > 0.10 and (r < 0.10 or r > 0.90):
            warnings.append(f"Ladder is {'out of reach for' if r < 0.10 else 'trivial for'} "
                            f"{t} creators (first rung reached by {r:.0%}). Consider a "
                            "separate ladder for that tier.")
    rungs = []
    prev_t, prev_p = 0, 0
    for k, (t, p) in enumerate(zip(th, pay), 1):
        marginal = (p - prev_p) / ((t - prev_t) / 1000)
        rungs.append({"milestone_rank": k, "view_threshold": int(t), "payout_amount": int(p),
                      "payout_per_1k_views_at_threshold": round(p / (t / 1000), 1),
                      "marginal_payout_per_1k_views": round(marginal, 1),
                      "share_of_paid_media_cpm": round(marginal / cpm, 3)})
        prev_t, prev_p = t, p
    result = {
        "milestones": rungs,
        "budget": budget,
        "p50_total_payout": round(float(np.quantile(total, 0.5))),
        "p90_total_payout": round(float(np.quantile(total, 0.9))),
        "expected_total_payout": round(float(total.mean())),
        "prob_within_budget": round(float((total <= budget).mean()), 3),
        "budget_confidence_target": conf,
        "payout_rate_of_media_value": round(chosen["rate"], 3),
        "first_milestone_reach": round(float(chosen["m1"]), 3),
        "continuation_probabilities": [round(float(x), 3) for x in chosen["cont"]],
        "first_milestone_reach_by_tier": tier_reach,
        "assumed_media_cpm": cpm,
        "cold_start": cold,
        "similar_historical_posts": n_similar,
        "warnings": warnings,
    }
    return (result, (views, present, tiers)) if return_sim else result


def _cheapest_fallback(views, present, cpm, budget, conf):
    """When nothing fits the budget, return the lowest-cost structure at the lowest rate."""
    best = None
    for th, floor_bound in _candidate_structures(views, present, MIN_FIRST_THRESHOLD):
        pay = payouts_for(th, RATE_GRID[0], cpm)
        hits = _hits(views, present, th)
        total = hits @ np.diff(np.concatenate([[0], pay]))
        cost = float(np.quantile(total, conf))
        if best is None or cost < best[0]:
            r_pool = hits.sum(axis=0) / present.sum()
            best = (cost, dict(th=th, pay=pay, total=total, rate=float(RATE_GRID[0]),
                               m1=float(r_pool[0]), cont=r_pool[1:] / r_pool[:-1],
                               floor_bound=floor_bound))
    return best[1]
