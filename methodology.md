# Methodology

## 1. What "optimized" means here

A ladder is a bet placed before anyone has posted. So I treat it as a decision under uncertainty, not a forecast: I don't need to know how many views each creator will get, I need to know the range, and pick a ladder that holds up across that range.

My definition, in priority order:

1. **Stay inside the budget with 90% confidence.** The 90th percentile of simulated total payout must be at or below the budget. This is a hard constraint. It is 95% if we have little history for the campaign type.
2. **Never pay more than 40% of what the same reach costs on paid media.** This is my ROI ceiling. Payout per extra 1,000 views can't go above 40% of the assumed CPM.
3. **Keep the ladder climbable but not trivial.** I call each milestone in the ladder a "rung". About half of creators should reach the first rung, and of the creators who reach any rung, 25% to 65% should reach the next. Below that, people give up. Above it, the ladder is easy to grind.
4. **Then pay as much as 1 and 2 allow.** Once the shape is fixed, payouts are scaled up until the budget or the ROI ceiling binds.

Budget comes first because it is the one failure a brand cannot forgive. The ROI ceiling comes second because without it the tool would happily spend the whole budget just because it can. Budget is a ceiling, not a target. If the ROI ceiling binds first, the money is left unspent, and I think that is the right answer (the brand can put it into more creators).

How each tradeoff in the brief is handled:

| Tradeoff | What I did |
|---|---|
| Budget adherence | P90 of simulated total payout must be within budget (rule 1) |
| Motivation and retention | Target reach for rung 1 and rung-to-rung continuation bands (rule 3) |
| Fairness across tiers | One ladder per campaign, built on the mix of tiers that campaign actually attracts. Reach by tier is reported, with a warning if a tier is shut out (section 6) |
| Fraud | Flagged posts are excluded from training. Payouts assume validated views only (section 7) |
| Marginal ROI | Payout is tied to the media value of each threshold and capped (rule 2) |

## 2. Data, and what I baked into it

There is no Meme'd data, so I generated 100 campaigns, 1,200 creators and 11,339 posts (`src/generate_data.py`, fixed seed). The results depend on these choices, so here they are:

- Views are lognormal. Expected views scale with follower count to the power 0.72, so bigger accounts get more views but less than proportionally.
- Category, platform and format multiply expected views (gaming 1.2x, finance 0.7x; reels 1.15x, carousels 0.55x).
- Each creator has a hidden quality factor, so follower count is a noisy predictor.
- Noise has two parts: post-level noise (log sd 1.05) and a shock shared by every post in a campaign (log sd 0.30). The second part matters most for budget risk, because it moves all creators the same way and does not average out.
- 5.5% of posts are suspicious: a flat first 24 hours, a late spike, and a final count 1.3x to 2.3x too high.
- The "manual" ladders follow a fixed 1x / 4x / 10x / 35x shape. Their payout rate wanders, and every 7th campaign is far too generous and every 11th far too stingy.
- **Budgets:** I set each budget after the fact as what the manual ladder actually paid times a slack factor. About 20% of campaigns get slack below 1 (the manual ladder blew the budget), 60% a comfortable margin, 20% a lot of headroom. This is the assumption that most shapes the budget results. My first version had budgets about 30 times too large, so the budget constraint never mattered and the backtest could not test it.
- Paid-media CPM is a made-up table (₹130 to ₹190 per 1,000 views). It needs replacing with real numbers.

## 3. View model

For each post I predict log(final views) with a Ridge regression on: log followers, log of the creator's historical average views, tier, platform, format and category. It is trained on non-flagged posts only.

Then I split what the model cannot explain into two parts (method of moments on campaign-level residuals): noise per post, and noise shared by the whole campaign. On the full data the estimates are 1.08 and 0.31, against true values of 1.05 and 0.30 in the generator.

For a new campaign I simulate 400 versions of it. In each one the number of creators is Poisson around the planned count, creators are drawn using the tier mix that past campaigns with the same target tier attracted, formats follow the platform's historical mix, and views come from the prediction plus one campaign-wide shock plus post noise. Simulated total payout for any ladder comes from those 400 runs.

**Validation** (`src/validate.py`, 5-fold cross-validation split by campaign):

- Out-of-sample R² is 0.67. Follower count alone gives 0.64, so the extra features add little. Most of the variation is noise.
- The 50%, 80% and 90% prediction intervals contain 49.6%, 79.3% and 89.6% of held-out posts.
- The model recovers the reel and finance effects well. It underestimates gaming (+0.08 fitted against +0.23 true). With only about 20 campaigns per category and a 0.30 campaign shock, category effects are only pinned down to roughly ±0.07, so this is noise I should expect, not a bug.

The recovery check is circular, since I wrote both the generator and the model. All it tells me is that the code is doing what it is meant to do.

## 4. From distribution to thresholds

I search over candidate ladders and keep the ones that pass the rules in section 1.

- Rung 1 is set so that 30% to 70% of the simulated pool reaches it, with a floor of 3,000 views. The floor exists because a tiny threshold can be ground out cheaply. It is an assumption; ops should set the real number.
- Rungs 2 to 4 multiply the previous threshold by 2, 2.5, 3 or 4. Thresholds are rounded to numbers a person would write down (5K, 7.5K, 10K, 25K).
- The gap rule uses the probability of reaching rung k+1 given rung k, which I call continuation. It has to sit between 25% and 65%. This is how I turn "gaps shouldn't be too big or too small" into a number.
- Among ladders that pass, I pick the one closest to the target shape (50% reach for rung 1, 40% continuation after that). This also answers "why 250K and not 200K": each threshold comes with a stated share of creators expected to reach it, and the alternatives fail a stated rule.

## 5. Payouts

Payout at each rung = payout rate × (threshold ÷ 1,000) × assumed CPM. Payouts are cumulative, so a creator who reaches rung 3 gets the rung 3 amount, not 1 + 2 + 3. The rate is the highest value between 10% and 40% that still keeps P90 of total payout inside the budget. Below 15% I add a warning, because I suspect creators won't bother, but I have no data on that.

This makes payout per extra view the same at every rung. I did that on purpose, so nobody can argue the top rung is priced differently to the bottom one. The brief's own example ladder is close to this.

## 6. Tiers

I did not build a separate ladder per tier inside one campaign. Creators compare payouts, and two ladders in one campaign invite arguments and multi-account tricks. Instead the ladder is built on the actual mix of tiers a campaign of that type draws, and the tool reports the share of each tier that reaches rung 1. If a tier with more than 10% of the pool is below 10% or above 90%, it warns and suggests a separate campaign or ladder for that tier.

It does not fully solve nano creators. Their median post is around 1,000 views, so with a 3,000-view floor most cannot reach rung 1 (16% do). For them I think a flat participation fee beats a ladder, and the tool says so.

## 7. Fraud

Two things are in the code. Flagged posts are left out of training, so a bot-inflated post does not push future thresholds up. And the backtest pays only on non-flagged posts, which stands for a rule that only validated views count.

What is not in the code is detection. The flag is a given label. On this data, 24h views divided by 30-day views separates flagged from clean posts well (AUC 0.94, 85% caught at a 1% false alarm rate). That result is circular, because I built the flagged posts to look that way. The useful part is the list of fields Meme'd would need to log: growth curve, audience geography, watch time, engagement per view, traffic source.

If payouts were made on raw views, the proposed ladders would have paid ₹28,100 over budget instead of ₹14,600, and the manual ladders paid ₹1.34M (8% of everything paid) to flagged posts.

## 8. Cold start

A campaign type the model has not seen (fewer than 60 similar historical posts) is flagged. For those, uncertainty is widened by 25% and the budget confidence goes from 90% to 95%. Unknown categories get the average category effect. Unknown platforms or creator types are not handled and would need a manual benchmark.

To test this I hid each category in turn and ran every campaign in it (all 100 campaigns end up flagged as cold starts). The realised payout stayed at or below the simulated P90 in 89% to 100% of campaigns, and only one of 100 went over budget. Each category has only 18 to 22 campaigns, so treat that as a rough check.

## 9. Backtest

`src/backtest.py` runs every one of the 100 campaigns with that campaign hidden from the model, using only its category, platform, tier, budget and planned creator count. Both ladders are then applied to the views the campaign actually got. The full tables are in `outputs/backtest_report.md`.

| | Manual | Proposed |
|---|---|---|
| Campaigns over budget | 17 | 1 |
| Total overspend | ₹225,650 | ₹14,600 |
| Median share of budget used | 67% | 47% |
| Total paid on non-flagged posts | ₹14.85M | ₹11.51M |
| Creators reaching rung 1 (mean) | 40% | 44% |
| Continuation, rung 3 to 4 | 21% | 42% |

Reach by creator tier: nano 5% to 16%, micro 29% to 47%, mid 68% to 57%, macro 95% to 72%.

How to read this:

- The 17 manual overruns come from how I built the budgets (section 2). They only show that the method responds to a budget. They say nothing about how often Meme'd really overspends.
- The number that tests the method is calibration. The realised payout was at or below the simulated P90 in 91% of campaigns (target 90%). The manual ladder's realised payout landed inside the simulated 10th to 90th percentile range in 75% of campaigns (target 80%). So the model is a little too confident.
- Completion went up in 70 campaigns and down in 30. It went down mostly where the manual ladder was too easy: in CA061 (gaming, macro) 81% of creators hit rung 1 under the manual ladder and 43% under the proposed one. I count that as intended, but a brand may not like the headline.
- The share of campaigns with rung-1 reach in my 30% to 70% band went from 45% to 77%. I set that band myself, so this checks my own definition and proves nothing independent.
- Completion only looks at thresholds. It cannot tell whether the rupee amounts are attractive. That needs real participation data.
- In 6 campaigns the budget was too small to pay even 15% of media value at 90% confidence. The tool says so instead of returning a ladder that looks fine.

`src/sensitivity.py` reruns 25 campaigns with the ROI ceiling at 30% and 50%, the reach target at 40% and 60%, the continuation target at 30% and 50%, and budget confidence at 80% and 95%. Nothing flips. The ROI ceiling mainly moves how much budget is used (38% to 51%), and the reach target moves rung-1 reach (36% to 47%).

## 10. Limits, and what would break it

- The generator and the model share a structure (lognormal, multiplicative effects), so the backtest is friendlier than real data will be. Real views have viral spikes, seasonality and algorithm changes.
- Budgets and CPMs are invented, and both drive the results directly.
- I don't model whether a creator posts at all, or drops out after missing a rung. "Reached rung 1" is a stand-in for motivation.
- Only final views count. A creator at 40K views on day 10 behaves differently from one who plateaued at 40K on day 2, and the ladder can't tell them apart.
- I fixed the ladder at four rungs. More or fewer might work better.
- In my data, small tiers force some creators to appear in the same campaign twice.
- I assume the creator count is known. In practice it depends on the ladder itself.
  
Data I would ask Meme'd for first: creator-level post history by platform and format, real paid CPMs, verified views with growth curves, timestamps of when creators stopped posting in a campaign, and past budget overruns and underspends.

## 11. What I'd do next

With another week, I'd first swap the invented CPM table for real numbers, since the payouts depend on it directly. Then I'd test the ladder against actual drop-off data, so "reached rung 1" stops being a stand-in for motivation. Only after that would I look at a fancier model. I doubt it would help much, because most of the variation in views is noise (R² of 0.67, against 0.64 from follower count alone), so better inputs matter more than a better model.
