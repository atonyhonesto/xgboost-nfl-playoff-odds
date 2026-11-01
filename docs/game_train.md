# Training the game model (`game_train.py`)

One script does the whole job after each NFL week, once `game_data.py` has
refreshed the data:

```
.venv\Scripts\python game_train.py      (about 1–2 minutes)
```

## What it does

1. **Backtests**: walks forward through every week since 2022, training only on
   earlier games, and scores the model against the betting spread, Elo and home
   field (see [game_model.md](game_model.md)).
2. **Fits sigma**: how far real margins land from the prediction (about 12.7
   points), measured on those out-of-sample games.
3. **Trains the final model** on every completed game and saves it.
4. **Forecasts the next week**: every game still to play in the earliest unfinished
   week, with the model's margin next to the betting spread and the home team's win
   chance.

## What it writes

| File | What's in it |
|---|---|
| `models/game/margin.json` | The trained XGBoost model |
| `models/game/meta.json` | Sigma, the feature list, and how many games it was trained on |
| `reports/game_backtest.csv` | Every backtested game: each predictor's margin and win chance, and the result |
| `reports/game_metrics.csv` | The scorecard: model vs baselines |
| `reports/game_metrics_by_season.csv` | The same, season by season |
| `reports/game_calibration.csv` | Predicted vs actual home win rate, in bins |
| `reports/game_next.csv` | Next week's forecast, with SHAP values for every game |
