"""Simulate the rest of this season 10,000 times and save every team's odds.

  1. Load the game table, the saved game model and its sigma.
  2. Simulate the remaining regular season and the playoffs (season_sim.py).
  3. Save the odds, and add this week's snapshot to the odds-over-time history.
     Weeks missing from the history (the first run of a season, or a missed week)
     are re-created as they would have looked then, and marked as re-created.
  4. What's at stake: for each game in the coming week, every team's odds if the
     home team wins and if it loses.
  5. Check the seeding rules: rebuild every finished season since 2004 from its
     results and compare with the real playoff bracket.

With --backtest it also re-runs the season backtest (about 3 minutes): forecasts of
2020-25 from weeks 5, 9 and 13, for several values of tau, against home field only.

Outputs:
  reports/season_odds.csv          every team's odds now
  reports/odds_history.csv         one snapshot per week (re-running a week replaces it)
  reports/stakes.csv               the coming week's games: odds after a win and after a loss
  reports/seeding_check.csv        rebuilt vs real playoff brackets
  reports/season_backtest.csv      (--backtest) per team, season, cutoff week and tau
  reports/season_metrics.csv       (--backtest) log loss and Brier per kind of odds

Usage:  python season_forecast.py [--backtest]     (run game_train.py first; about 1-2 minutes)
"""
import sys
from datetime import date

import pandas as pd

import game_model as gm
import season_sim as ss
from game_data import current_season

HISTORY = gm.REPORTS / "odds_history.csv"
STAKES = gm.REPORTS / "stakes.csv"
SNAPSHOT = ["team", "exp_wins", "p_playoffs", "p_division", "p_bye", "p_super_bowl", "p_champion"]


def regular_season(games, season):
    return games[(games["season"] == season) & (games["game_type"] == "REG")]


def progress_point(games, season):
    """(last full week played, date of the last result) for the odds-over-time history."""
    reg = regular_season(games, season)
    done = reg.groupby("week")["result"].apply(lambda r: r.notna().all())
    full = done[done].index
    return (int(full.max()) if len(full) else 0), reg.loc[reg["result"].notna(), "gameday"].max()


def load_history():
    return pd.read_csv(HISTORY) if HISTORY.exists() else pd.DataFrame(columns=SNAPSHOT + ["season", "week"])


def save_snapshot(odds, season, week, as_of, recreated=False):
    snap = odds[SNAPSHOT].assign(season=season, week=week, as_of=as_of, run_on=date.today().isoformat(),
                                 recreated=recreated)
    old = load_history()
    snap = pd.concat([old[~((old["season"] == season) & (old["week"] == week))], snap], ignore_index=True)
    snap.sort_values(["season", "week", "team"]).to_csv(HISTORY, index=False)


def recreate_missing_weeks(games, teams, sigma, season, week):
    """Snapshots for earlier weeks of this season that the history doesn't have yet.

    Week w means "after week w": results hidden from week w+1 on, ratings frozen there,
    and the game model retrained on the games before (week 0 = before the season).
    """
    have = set(load_history().query("season == @season")["week"])
    reg = regular_season(games, season)
    for w in [w for w in range(week) if w not in have]:
        cutoff = reg.loc[reg["week"] == w + 1, "gameday"].min()
        g = ss.freeze(games, season, cutoff)
        model = gm.fit(gm.played(g[g["season"] >= gm.TRAIN_FROM]))
        odds, _, _ = ss.simulate(ss.prepare(g, season, teams, model), sigma, seed_=w)
        last = reg.loc[(reg["gameday"] < cutoff) & reg["result"].notna(), "gameday"].max()
        save_snapshot(odds, season, w, last if pd.notna(last) else "", recreated=True)
        print(f"  re-created the snapshot after week {w}")


def stakes(ctx, sigma, odds_cols=("p_playoffs", "p_division", "p_bye")):
    """For each game in the coming week, both teams' odds after a home win and after an away win.

    Both runs use the same random numbers, so the difference is the game, not noise.
    """
    reg = ctx["games"]
    upcoming = reg[~ctx["played"]]
    if upcoming.empty:
        return pd.DataFrame()
    week = upcoming["week"].min()
    rows = []
    for g in upcoming[upcoming["week"] == week].itertuples(index=False):
        runs = {}
        for value, label in ((1, "home_wins"), (0, "away_wins")):
            s, _, _ = ss.simulate(ctx, sigma, n_sims=ss.WHAT_IF_SIMS, seed_=1, forced={g.game_id: value})
            runs[label] = s.set_index("team")
        row = {"game_id": g.game_id, "week": week, "gameday": g.gameday,
               "home_team": g.home_team, "away_team": g.away_team}
        for side in ("home", "away"):
            team = getattr(g, f"{side}_team")
            win, lose = ("home_wins", "away_wins") if side == "home" else ("away_wins", "home_wins")
            for col in odds_cols:
                row[f"{side}_{col}_win"] = runs[win].at[team, col]
                row[f"{side}_{col}_loss"] = runs[lose].at[team, col]
        rows.append(row)
    out = pd.DataFrame(rows)
    out["swing"] = (out["home_p_playoffs_win"] - out["home_p_playoffs_loss"]
                    + out["away_p_playoffs_win"] - out["away_p_playoffs_loss"])
    return out.sort_values("swing", ascending=False)


def main(run_backtest=False):
    games = pd.read_csv(gm.TABLE)
    teams = ss.load_teams()
    model, meta = gm.load()
    season = current_season()
    week, as_of = progress_point(games, season)
    ctx = ss.prepare(games, season, teams, model)
    print(f"{season}: {int(ctx['played'].sum())} games played, {int((~ctx['played']).sum())} to go "
          f"(through week {week}, last result {as_of}).")
    print(f"Simulating {ss.N_SIMS:,} seasons (sigma {meta['sigma']:.1f}, tau {ss.TAU:g})...")

    odds, _, _ = ss.simulate(ctx, meta["sigma"])
    odds = odds.sort_values(["p_champion", "p_playoffs"], ascending=False)
    odds.to_csv(gm.REPORTS / "season_odds.csv", index=False)
    save_snapshot(odds, season, week, as_of)
    for conf in ss.CONFS:
        print(f"\n{conf}   W-L-T   exp W  playoffs  division   bye   Super Bowl  title")
        for _, r in odds[odds["conf"] == conf].sort_values("p_playoffs", ascending=False).iterrows():
            print(f"  {r['team']:<4} {r['wins']:>2.0f}-{r['losses']:.0f}-{r['ties']:.0f}  {r['exp_wins']:5.1f}  "
                  f"{r['p_playoffs']:7.0%}  {r['p_division']:8.0%}  {r['p_bye']:5.0%}  "
                  f"{r['p_super_bowl']:9.0%}  {r['p_champion']:5.0%}")

    print("\nOdds-over-time history...")
    recreate_missing_weeks(games, teams, meta["sigma"], season, week)

    print("What's at stake in the coming week...")
    st = stakes(ctx, meta["sigma"])
    st.to_csv(STAKES, index=False)
    for _, r in st.head(5).iterrows():
        print(f"  {r['away_team']} @ {r['home_team']}: playoff odds swing "
              f"{r['home_team']} {r['home_p_playoffs_loss']:.0%}->{r['home_p_playoffs_win']:.0%}, "
              f"{r['away_team']} {r['away_p_playoffs_loss']:.0%}->{r['away_p_playoffs_win']:.0%}")

    check = ss.check_seeding(games, teams, range(gm.TRAIN_FROM, season))
    check.to_csv(gm.REPORTS / "seeding_check.csv", index=False)
    print(f"\nSeeding check: rebuilt the real playoff bracket in {check['match'].sum()} of {len(check)} "
          f"seasons since {gm.TRAIN_FROM}.")

    if run_backtest:
        print("\nSeason backtest (2020-25, from weeks 5, 9 and 13)...")
        bt = ss.backtest(games, teams, meta["sigma"], taus=ss.TAUS,
                         progress=lambda s, w: print(f"  {s} week {w}") if w == ss.BACKTEST_WEEKS[-1] else None)
        metrics = ss.score(bt)
        bt.to_csv(gm.REPORTS / "season_backtest.csv", index=False)
        metrics.to_csv(gm.REPORTS / "season_metrics.csv", index=False)
        print("Playoff-odds log loss (lower is better):")
        ll = metrics[metrics["event"] == "made_playoffs"].pivot(index="predictor", columns="week", values="log_loss")
        print(ll.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main(run_backtest="--backtest" in sys.argv)
