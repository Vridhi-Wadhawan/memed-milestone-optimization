# Milestone ladders that hold up: one page

## The problem

Today, ops sets the view thresholds and payouts by feel. Some campaigns overspend. Some pay so little that creators walk away. And when a brand asks "why 250K and not 200K?", nobody has a real answer.

## What this does

You give it a campaign: category, platform, creator size, budget and how many creators you expect. It returns four milestones (views and payout) and tells you three things about them:

- **How likely you are to stay in budget.** It simulates the campaign 400 times, with different creators, different formats and good and bad luck for the whole campaign. The ladder is only accepted if the budget holds in at least 90 out of 100 of those runs.
- **How many creators should reach each milestone.** The aim is about half for the first one, and roughly 4 in 10 of those who reach a milestone to reach the next. Too hard and people quit. Too easy and it is cheap to game.
- **What the reach is worth.** No milestone pays more than 40% of what the same views would cost as paid ads. So the top rung can't quietly cost far more than the reach is worth.

So the answer to "why 250K?" becomes something like: "about 1 in 3 creators like these gets there, and at that level we still fit inside your budget nine times out of ten."

## What we found when we tested it

I ran it on 100 simulated past campaigns, hiding each campaign's results from the tool first, then compared its ladder with the one "actually used".

| | Manual ladders | This method |
|---|---|---|
| Campaigns over budget | 17 of 100 | 1 of 100 |
| Total overspend | ₹2.26 lakh | ₹14,600 |
| Creators reaching the first milestone | 40% | 44% |
| Jump from the 3rd to the 4th milestone | 21% get there | 42% get there |

Two examples:
- **CA086** (entertainment, ₹40,000 budget): the manual ladder paid ₹67,350, which is 168% of budget. This method would have paid ₹30,150 and more creators would have hit the first milestone (49% instead of 41%).
- **CA061** (gaming, big creators): the manual first milestone was so easy that 81% of creators hit it. This method sets it higher (43% hit it) and would have paid ₹5.35 lakh instead of ₹7.61 lakh.

Creator sizes are treated more evenly too. Under the manual ladders 95% of the largest creators hit the first milestone and 5% of the smallest. With this method it is 72% and 16%. And the "90 out of 100" promise held up: payouts stayed under the tool's 90% estimate in 91 of 100 campaigns.

**Read this before you trust those numbers.** All of it comes from data I made up. It shows the method behaves as designed. It does not show what will happen on Meme'd campaigns. The "17 overspent" figure in particular comes from how I built the fake budgets.

## Where it can still get it wrong

- **Small creators.** The smallest accounts get about 1,000 views a post. No fair ladder gets them to a 3,000-view first milestone. They may need a flat fee instead, and the tool warns you.
- **Something new.** A category or platform it has not seen gets a more cautious ladder, but a less accurate one.
- **Viral surprises.** One post that explodes can push a campaign over budget. A 90% promise means about 1 campaign in 10 can still go over.
- **Bought views.** The tool assumes only verified views get paid. It does not detect fraud itself.
- **Made-up inputs.** The paid ad prices are placeholders, and "reaches the first milestone" is only a stand-in for staying motivated.

## What I'd need to make it real

Past campaigns with creator-level results, real ad prices, verified views over time, and the point where creators stopped posting in a campaign.
