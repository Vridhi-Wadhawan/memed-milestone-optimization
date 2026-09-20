"""How much do the results move if the judgement calls in model.py change?

Runs the backtest on every 4th campaign (25 of 100) under different values of
the ROI cap, the target reach and continuation rates, and the budget confidence.

Run: python src/sensitivity.py     (about 5 minutes)
"""
from pathlib import Path

import numpy as np
import pandas as pd

import model as M
from backtest import run_campaign

OUT = Path(__file__).resolve().parents[1] / "outputs"

SETTINGS = [
    ("baseline (cap 40%, continue 40%, confidence 90%)", {}),
    ("ROI cap 30%", {"ROI_CAP": 0.30}),
    ("ROI cap 50%", {"ROI_CAP": 0.50}),
    ("target rung-1 reach 40%", {"TARGET_M1_REACH": 0.40}),
    ("target rung-1 reach 60%", {"TARGET_M1_REACH": 0.60}),
    ("target continuation 30%", {"TARGET_CONTINUE": 0.30}),
    ("target continuation 50%", {"TARGET_CONTINUE": 0.50}),
    ("budget confidence 80%", {"BUDGET_CONFIDENCE": 0.80}),
    ("budget confidence 95%", {"BUDGET_CONFIDENCE": 0.95}),
]


def apply(overrides):
    defaults = {"ROI_CAP": 0.40, "TARGET_CONTINUE": 0.40, "BUDGET_CONFIDENCE": 0.90,
                "TARGET_M1_REACH": 0.50}
    values = {**defaults, **overrides}
    M.ROI_CAP = values["ROI_CAP"]
    M.RATE_GRID = np.arange(0.10, values["ROI_CAP"] + 1e-9, 0.025)
    M.TARGET_CONTINUE = values["TARGET_CONTINUE"]
    M.TARGET_M1_REACH = values["TARGET_M1_REACH"]
    M.BUDGET_CONFIDENCE = values["BUDGET_CONFIDENCE"]


def main():
    data = M.load_data()
    sample = data.campaigns.iloc[::4]
    rows = []
    for name, overrides in SETTINGS:
        apply(overrides)
        res = pd.DataFrame([run_campaign(c.to_dict(), data)[0] for _, c in sample.iterrows()])
        rows.append({
            "setting": name,
            "over_budget": int(res.proposed_over_budget.sum()),
            "median_budget_used": f"{(res.proposed_paid_clean / res.budget).median():.0%}",
            "mean_rung1_reach": f"{res.proposed_completion.mean():.0%}",
            "median_payout_share_of_cpm": f"{res.proposed_payout_share_of_cpm.median():.0%}",
            "within_sim_p90": f"{res.proposed_within_sim_p90.mean():.0%}",
        })
        print(rows[-1])
    apply({})
    pd.DataFrame(rows).to_csv(OUT / "sensitivity.csv", index=False)
    print(f"({len(sample)} campaigns per setting)")


if __name__ == "__main__":
    main()
