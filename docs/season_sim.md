# The season simulation (`season_sim.py`)

This turns single-game chances into what fans care about: each team's chances to
make the playoffs, win its division, earn the #1 seed and win the Super Bowl. It
plays out the rest of the season **10,000 times** and counts what happens.

## One simulated season

1. **The remaining games.** Each game's margin is drawn from the game model
   ([game_model.md](game_model.md)): the predicted margin plus random noise. A
   positive margin is a home win. Games already played keep their real result,
   including ties.
2. **The standings.** Every team's record, division record, conference record,
   strength of victory and strength of schedule, for all 10,000 seasons at once.
3. **The seeds.** In each conference, the 4 division winners get seeds 1–4 and the
   3 best other teams get seeds 5–7 (the wild cards). Ties go through the NFL's
   tiebreakers ([tiebreakers.md](tiebreakers.md)).
4. **The playoffs.** 2 v 7, 3 v 6 and 4 v 5, while the #1 seed rests. Next, the #1
   seed hosts the lowest seed left and the other two play. The better seed hosts the
   conference final, and the Super Bowl is at a neutral site. Playoff games use the
   same model, with every team rated as it stands now.

10,000 seasons take about 7 seconds. The app's what-ifs use 2,000 (`WHAT_IF_SIMS`),
which take about 1.5 seconds.

## The strength shock: why the odds aren't overconfident

The simplest simulation treats every game as a separate coin toss. That's
overconfident, because the model's view of a team can be *wrong for the whole
season*. A quarterback gets hurt, a defence falls apart, or a lucky start turns out
to be luck.

So in each simulated season every team also gets a **strength shock**, a random
number of points (spread **tau = 5**) added to all its remaining games. In some
simulations a team is better than the model thinks, in others worse. Each game's
noise shrinks to match, so a single game is exactly as uncertain as before; the
team's games just move *together*.

### Choosing tau: the season backtest

`backtest()` goes back to 2020–2025 and forecasts each season **as if it were week
5, 9 or 13**. Results after that date are hidden, every team keeps its rating as of
that date, and the game model is retrained on earlier games only. It then compares
the forecasts with what really happened.

Playoff-odds log loss (lower is better):

| | From week 5 | From week 9 | From week 13 |
|---|---|---|---|
| Home field only (every game ~50/50) | 0.509 | 0.425 | 0.343 |
| Model, tau 3 | 0.493 | 0.440 | 0.355 |
| Model, **tau 5** (used) | 0.495 | 0.426 | 0.345 |
| Model, tau 7 | 0.505 | 0.428 | 0.339 |

Smaller values of tau are overconfident. With tau 3, Tampa Bay (6-2) and
Indianapolis (7-1) got 96%+ playoff chances in 2025, and both missed. Tau 5 is the
best single value across the three checkpoints.

### Model vs "every game a coin flip", with tau 5

Log loss, lower is better:

| Odds of… | From week 5 | From week 9 | From week 13 |
|---|---|---|---|
| Making the playoffs | **0.495** vs 0.509 | 0.426 vs **0.425** | 0.345 vs **0.343** |
| Winning the division | **0.405** vs 0.434 | **0.376** vs 0.392 | **0.324** vs 0.326 |
| The #1 seed | **0.164** vs 0.184 | **0.148** vs 0.157 | **0.119** vs 0.128 |
| Reaching the Super Bowl | **0.191** vs 0.201 | **0.160** vs 0.181 | **0.163** vs 0.172 |
| Winning it all | **0.123** vs 0.127 | **0.106** vs 0.117 | **0.117** vs 0.118 |

**In plain English:**

- **Where the model wins clearly:** division, #1 seed, Super Bowl and title odds, at
  every checkpoint. Those depend on who's actually good, not just today's standings.
- **Where it doesn't:** once the standings are half-built, plain *playoff* odds are
  nearly as good from "every game is a coin flip". From week 9 on the two are level.
- **Expected wins:** the model is clearly better, off by 1.5 wins at week 9 against
  1.8 for the baseline.

The full results are in `reports/season_metrics.csv`.

## The building blocks

- **`prepare()`** gathers everything that stays fixed: the schedule, results so far,
  the model's prediction for every remaining game, and predicted margins for every
  possible playoff matchup.
- **`simulate()`** returns every team's odds, and takes `forced={game_id: 1 or 0}` to
  fix games for a what-if.
- **`standings()`** works out the records.
- **`seed()`** produces seeds 1–7, taking the quick path when no tie matters.
- **`freeze()`, `backtest()`, `score()`** are the honesty checks.
- **`real_bracket()`, `check_seeding()`** test the seeding rules on real seasons.
