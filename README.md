# XGBoost NFL Playoff Odds

Every NFL team's chances to make the playoffs, win its division, earn the #1 seed and
win the Super Bowl. The app plays out the rest of the season **10,000 times** with an
**XGBoost** game model and the NFL's real **tiebreakers**, and updates every week.

It's a showcase of XGBoost, Monte Carlo simulation, SHAP and Streamlit on free public
data, and a sibling of the [XGBoost NASCAR Predictor](https://xgboost-nascar-predictor.streamlit.app/).

## What's in the app

- **Playoff odds:** every team's chances, plus a heatmap of how likely each seed is.
- **This week:** each game as a TV-style score bug, with the model's pick next to the
  betting line and what the game means for both teams' playoff odds.
- **What if?:** set any of a team's games to a win or a loss and watch the odds,
  and its rivals' odds, move.
- **Odds over time:** how a division's race has shifted, week by week.
- **How good is it?:** the honest scorecard, below.

## How it works

1. **Data.** [nflverse](https://github.com/nflverse), via `nflreadpy`: every game since
   2002 with scores and betting lines, and every play with its **EPA** (expected
   points added).
2. **Team strength before each game.** Built only from earlier games:
   - an Elo rating
   - running averages of offensive and defensive EPA per play, pass EPA, success rate
     and margin
   - the starting quarterback's own EPA per dropback, so a backup starting shows up as
     a real drop in quality
3. **The game model.** An `XGBRegressor` predicts the home team's margin from the
   home-minus-away differences in those ratings, plus rest, division game and neutral
   site. Real margins land about 12.7 points either side of the prediction, and a
   normal curve turns the margin into a win chance. **SHAP** values explain each pick in
   points.
4. **The season simulation.** Each simulated season:
   - draws every remaining game from that curve
   - gives each team a random **strength shock** for the whole season, because the
     model can be wrong about a team for months, not just one game
   - seeds 7 teams per conference
   - plays out the bracket
   10,000 seasons take about 7 seconds.
5. **Tiebreakers.** The NFL's procedure: head-to-head, division record, common games,
   conference record, strength of victory and strength of schedule, including the
   rule that tied teams "start again" whenever one drops out. Rebuilt from their real
   results, the simulation reproduces **all 22 real playoff brackets from 2004 to
   2025**. The late points-based steps are simplified.

## Honest results

Everything is tested walk-forward: each forecast uses only what was known at the time.

**Single games** (1,247 games, 2022 to week 7 of 2026):

| | XGBoost | Betting line | Elo | Home field only |
|---|---|---|---|---|
| Picked the winner | 63.5% | **67.3%** | 63.8% | 53.5% |
| Win log loss (lower is better) | 0.629 | **0.611** | 0.637 | 0.689 |

The model edges Elo and clearly beats home field, but **the betting market is better**.
It knows about injuries, weather and news the model never sees.

**Season odds.** For 2020–2025, each season was forecast as if it were week 5, 9 or 13,
and compared with a baseline that knows the standings but treats every remaining game
as a coin flip:

- **The model wins** on division, #1 seed and Super Bowl odds at every checkpoint.
- **Plain playoff odds are level** with the baseline from week 9 on.
- **The backtest set the strength shock.** With less uncertainty, the odds were
  overconfident: two 2025 teams given 96%+ playoff chances both missed.

## Run it yourself

On Windows, with Python 3.12:

```
setup.bat                                  build .venv (once per computer)
.venv\Scripts\python game_data.py          download and build the game table
.venv\Scripts\python game_train.py         backtest and train the game model
.venv\Scripts\python season_forecast.py    simulate the season
run_app.bat                                open the app
```

Every code file has a plain-English guide in [`docs/`](docs).

## Built by

Tony Honesto · [LinkedIn](https://www.linkedin.com/in/tony-honesto-4195023) ·
[GitHub](https://github.com/atonyhonesto)

An independent fan project built on nflverse's free public data. Not affiliated with or
endorsed by the NFL, any team or any sponsor. Forecasts are for fun, not betting advice.
