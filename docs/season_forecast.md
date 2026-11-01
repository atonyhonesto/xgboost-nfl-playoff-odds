# The weekly forecast (`season_forecast.py`)

The last step of the weekly routine. It simulates the rest of the season and saves
everything the app shows:

```
.venv\Scripts\python game_data.py         refresh the data
.venv\Scripts\python game_train.py        retrain the game model
.venv\Scripts\python season_forecast.py   simulate the season (1–2 minutes)
```

Add `--backtest` to also re-run the season backtest (about 3 more minutes). It
re-checks the choice of tau, as explained in [season_sim.md](season_sim.md).

## What it does

1. Loads the games, the saved game model and its sigma.
2. Simulates the rest of the season and the playoffs 10,000 times.
3. Prints both conferences: record, expected wins, and the chances to make the
   playoffs, win the division, get the bye, reach the Super Bowl and win it.
4. Saves this week's odds to the **odds-over-time history**.
   - Each snapshot is labelled with the last *complete* week, so running it on a
     Sunday afternoon counts as the week before. Re-running a week replaces that
     week's snapshot.
   - **Missing weeks are re-created**, for example the first time it runs in a season,
     or after a missed week. For each one, results after that week are hidden, ratings
     are frozen there, and the game model is retrained on earlier games only. These
     snapshots are marked `recreated`, and the app shades them.
5. **Works out what's at stake** in the coming week. For each game, it simulates the
   season once with the home team winning and once with it losing (2,000 seasons each,
   with the same random numbers). Every team's odds are saved for both cases. Games
   are ranked by how much the two teams' playoff odds swing.
6. Runs the seeding check on every finished season since 2004.

## What it writes

| File | What's in it |
|---|---|
| `reports/season_odds.csv` | Every team's odds now, plus the chance of each seed 1–7 |
| `reports/odds_history.csv` | One snapshot per week, for the odds-over-time chart |
| `reports/stakes.csv` | The coming week's games: both teams' odds after a win and after a loss |
| `reports/seeding_check.csv` | Rebuilt vs real playoff brackets, season by season |
| `reports/season_backtest.csv` | (`--backtest`) Past forecasts next to what happened |
| `reports/season_metrics.csv` | (`--backtest`) Log loss and Brier for each kind of odds |
