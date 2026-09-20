"""Check the view model before trusting the ladders built on top of it.

1. Out-of-sample fit: 5-fold cross-validation, folds split by campaign, so the
   model never sees any post from the campaign it is scored on.
2. Interval coverage: do the model's 50/80/90% prediction intervals contain
   about that share of held-out posts?
3. Effect recovery: does the fitted model find the category and format effects
   I built into the synthetic data? (A sanity check, not independent evidence.)
4. Fraud signal: how well does early growth (24h views / 30-day views) separate
   flagged from clean posts? Also circular on synthetic data. It shows which
   fields Meme'd would need to log.

Run: python src/validate.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import model as M
from generate_data import CATEGORY_EFFECT, FORMAT_EFFECT

OUT = Path(__file__).resolve().parents[1] / "outputs"


def make_pipe():
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), M.FEATURE_CAT),
                             ("num", "passthrough", M.FEATURE_NUM)])
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=2.0))])


def cross_validate(clean):
    X = clean[M.FEATURE_CAT + M.FEATURE_NUM]
    y = np.log1p(clean.views_final.values)
    pred, sigma = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, clean.campaign_id):
        pipe = make_pipe().fit(X.iloc[tr], y[tr])
        pred[te] = pipe.predict(X.iloc[te])
        # same total-noise estimate the simulator uses: post noise + campaign-shared noise
        sp, sc = M.variance_components(y[tr] - pipe.predict(X.iloc[tr]), clean.campaign_id.values[tr])
        sigma[te] = np.sqrt(sp ** 2 + sc ** 2)
    resid = y - pred
    rows = [{"check": "out-of-sample R2 (log views)",
             "value": round(1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum(), 3)},
            {"check": "R2 with follower count only",
             "value": round(_r2_followers_only(clean, y), 3)}]
    for level in (0.5, 0.8, 0.9):
        z = norm.ppf(0.5 + level / 2)
        rows.append({"check": f"{level:.0%} interval coverage (target {level:.0%})",
                     "value": round(float((np.abs(resid / sigma) < z).mean()), 3)})
    for tier, g in clean.assign(resid=resid).groupby("tier"):
        rows.append({"check": f"mean residual, {tier} (0 = unbiased)", "value": round(g.resid.mean(), 3)})
    return rows


def _r2_followers_only(clean, y):
    x = clean.log_followers.values
    slope, intercept = np.polyfit(x, y, 1)
    return 1 - ((y - (slope * x + intercept)) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def effect_recovery(clean):
    """Fitted vs true log-effect of category and format, holding everything else fixed."""
    pipe = make_pipe().fit(clean[M.FEATURE_CAT + M.FEATURE_NUM], np.log1p(clean.views_final.values))
    base = clean[M.FEATURE_CAT + M.FEATURE_NUM].copy()

    def shift(col, a, b):
        return (pipe.predict(base.assign(**{col: b})) - pipe.predict(base.assign(**{col: a}))).mean()

    rows = []
    for cat in ("gaming", "entertainment", "D2C", "finance"):
        rows.append({"check": f"category effect {cat} vs FMCG (fitted / true)",
                     "value": f"{shift('category', 'FMCG', cat):+.2f} / "
                              f"{np.log(CATEGORY_EFFECT[cat] / CATEGORY_EFFECT['FMCG']):+.2f}"})
    rows.append({"check": "format effect reel vs carousel (fitted / true)",
                 "value": f"{shift('format', 'carousel', 'reel'):+.2f} / "
                          f"{np.log(FORMAT_EFFECT['reel'] / FORMAT_EFFECT['carousel']):+.2f}"})
    return rows


def fraud_signal(posts):
    p = posts.assign(early=posts.views_at_24h / posts.views_at_30d)
    auc = roc_auc_score(p.flagged_suspicious, -p.early)
    cutoff = p.loc[~p.flagged_suspicious, "early"].quantile(0.01)   # 1% of clean posts get caught
    recall = float((p.loc[p.flagged_suspicious, "early"] < cutoff).mean())
    return [{"check": "fraud signal AUC: 24h views / 30d views", "value": round(auc, 3)},
            {"check": "share of flagged posts caught at 1% false-positive rate", "value": round(recall, 3)}]


def main():
    data = M.load_data()
    train = M._training_frame(data)
    clean = M._design(train[~train.flagged_suspicious]).reset_index(drop=True)
    rows = cross_validate(clean) + effect_recovery(clean) + fraud_signal(data.posts)
    model = M.fit_view_model(data)
    rows += [{"check": "estimated per-post noise (log sd)", "value": round(model.sigma_post, 3)},
             {"check": "estimated campaign-shared noise (log sd)", "value": round(model.sigma_campaign, 3)},
             {"check": "true values used in the generator (post, campaign)", "value": "1.05, 0.30"}]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "model_validation.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
