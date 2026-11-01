# The game model (`game_model.py`)

This is the heart of the forecast: for any game, how much should the home team win
by, and how likely is it to win at all?

## The model

**One `XGBRegressor`** predicts the home team's margin in points (home score minus
away score). It's trained on every game since 2004; 2002 and 2003 are only used to
warm up the ratings. It reads 12 features from [game_data.md](game_data.md):

```
elo_diff, net_epa_diff, pass_net_diff, succ_net_diff, margin_diff,
qb_diff, home_qb_shift, away_qb_shift, rest_diff, div_game, neutral, week
```

```python
n_estimators=500, learning_rate=0.02, max_depth=3, min_child_weight=30,
subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0
```

Football scores are noisy and there are only about 6,000 games, so the settings are
cautious: shallow trees, slow learning, big leaves and strong regularisation.

## From a margin to a win chance

Real margins land around the prediction with a spread **sigma** of about
**12.7 points**. That's measured on games the model hadn't seen, so it isn't
flattering. A normal curve then gives

```
P(home wins) = Phi(predicted margin / sigma)
```

So a 3-point favourite wins about 59% of the time and a 7-point favourite about 71%.
The season simulation will draw each game's margin from this same curve.

## How good is it? The backtest

`backtest(df)` walks forward through **every week since 2022** (96 weeks, 1,247
games): train on everything before that week, then forecast the week. It compares
the model with three baselines:

- **The betting spread**: the market's line, the toughest benchmark there is.
- **Elo**: the Elo difference plus home field, turned into points.
- **Home field only**: the home team by its average margin, every game.

Results as of week 8 of 2026:

| | XGBoost | Betting spread | Elo | Home field |
|---|---|---|---|---|
| Margin error (points) | 9.92 | **9.52** | 9.93 | 10.65 |
| Picked the winner | 63.5% | **67.3%** | 63.8% | 53.5% |
| Win log loss | 0.629 | **0.611** | 0.637 | 0.689 |
| Win Brier score | 0.219 | **0.211** | 0.222 | 0.247 |

**In plain English:** the model beats home field and edges out Elo on win chances,
but the betting market is still better. That's normal. The market includes
injuries, weather and inside information this model never sees, and very few public
models beat it. What the playoff simulation needs is honest, well-calibrated game
chances, and the calibration report shows predicted chances lining up with how often
teams really won.

`score(bt, by="season")` gives the same table season by season, and
`calibration(bt)` bins the predicted chances against what really happened.

## What the model leans on

Share of the model's total gain:

- **recent margin** 35%
- **Elo** 26%
- **QB rating** 12%
- **success rate** 7%
- **EPA** 4%
- **pass EPA** 4%

`contributions(model, rows)` gives **SHAP values in points**: for each game, how far
each factor pushed the predicted margin. For example, "the QB edge adds +2.1, home
rest takes away 0.4". They add up exactly to the prediction.

## The building blocks

- **`load_table()`, `played()`** read `data/games.csv`.
- **`fit()`, `predict()`, `win_prob()`, `fit_sigma()`** train, predict margins and
  turn them into chances.
- **`save()` / `load()`** write and read `models/game/`.
- **`baseline_margins()`, `backtest()`, `score()`, `calibration()`** are the honesty checks.
