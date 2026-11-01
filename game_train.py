"""Backtest and train the game model, then forecast the next week's games.

  1. Walk-forward backtest over every week since 2022: for each week, train on all
     earlier games, forecast the week, and score the model against the betting spread,
     Elo and home field only.
  2. Fit sigma (how far real margins land from the prediction) on those out-of-sample games.
  3. Train the final model on every completed game and save it.
  4. Forecast the next week of the current season.

Outputs:
  models/game/margin.json, meta.json      saved model + sigma
  reports/game_backtest.csv               per game, per tested week
  reports/game_metrics.csv                model vs baselines scorecard
  reports/game_metrics_by_season.csv      the same, season by season
  reports/game_calibration.csv            predicted vs actual home win rate
  reports/game_next.csv                   the next week's forecast, with SHAP values

Usage:  python game_train.py      (run game_data.py first)
"""
import game_model as gm


def main():
    df = gm.load_table()
    games = gm.played(df)
    gm.REPORTS.mkdir(exist_ok=True)
    print(f"{len(games):,} games from {gm.TRAIN_FROM} through {games['gameday'].max()}.")
    print(f"Backtesting every week since {gm.BACKTEST_FROM}...")

    bt = gm.backtest(df, progress=lambda i, n, label: print(f"  {i}/{n} {label}") if i % 20 == 0 or i == n else None)
    scores, by_season = gm.score(bt), gm.score(bt, by="season")
    bt.to_csv(gm.REPORTS / "game_backtest.csv", index=False)
    scores.to_csv(gm.REPORTS / "game_metrics.csv", index=False)
    by_season.to_csv(gm.REPORTS / "game_metrics_by_season.csv", index=False)
    gm.calibration(bt).to_csv(gm.REPORTS / "game_calibration.csv", index=False)
    print(f"\n{len(bt):,} out-of-sample games:")
    print(scores.drop(columns="games").to_string(index=False, float_format="%.3f"))
    print("\nWin log loss by season (lower is better):")
    ll = by_season[by_season["metric"].str.startswith("Win log")]
    print(ll.drop(columns=["metric"]).to_string(index=False, float_format="%.3f"))

    sigma = gm.fit_sigma(bt["result"] - bt["model_margin"])
    model = gm.fit(games)
    gm.save(model, sigma, {"trained_games": int(len(games)), "trained_through": games["gameday"].max()})
    top = gm.importance(model).head(6)
    print(f"\nSigma: {sigma:.2f} points")
    print("Model leans most on: " + ", ".join(f"{f} ({v:.0%})" for f, v in top.items()))

    upcoming = df[df["result"].isna()]
    if upcoming.empty:
        print("\nNo games left on the schedule.")
        return
    nxt = upcoming[(upcoming["season"] == upcoming["season"].min()) & (upcoming["week"] == upcoming["week"].min())].copy()
    nxt["model_margin"] = gm.predict(model, nxt)
    nxt["model_p"] = gm.win_prob(nxt["model_margin"], sigma)
    shap, base = gm.contributions(model, nxt)
    nxt = nxt.join(shap.add_prefix("shap_")).assign(shap_base=base)
    nxt.to_csv(gm.REPORTS / "game_next.csv", index=False)
    print(f"\n{nxt['season'].iloc[0]} week {nxt['week'].iloc[0]} (home margin: model vs betting spread):")
    for _, r in nxt.iterrows():
        print(f"  {r['away_team']:>3} @ {r['home_team']:<3}  model {r['model_margin']:+5.1f}  "
              f"spread {r['spread_line']:+5.1f}  home wins {r['model_p']:4.0%}")


if __name__ == "__main__":
    main()
