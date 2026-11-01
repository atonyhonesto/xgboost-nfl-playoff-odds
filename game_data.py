"""Build the game table: every NFL game since 2002, with what was known about both teams beforehand.

This is the dataset behind the game model. One row per game, from nflverse (via
nflreadpy): the schedule with scores and betting lines, plus team strength measured
from every play. Everything about a team enters only as it stood *before* kickoff:

  elo     a simple Elo rating (wins and margins), 1/3 of the way back to average each offseason
  epa     running averages of offensive and defensive EPA per play, pass EPA and success
          rate, and point margin, with the newest game weighted ALPHA; each offseason keeps
          CARRY of a team's edge over the league average
  qb      the starting QB's own EPA per dropback (career, recent games weighted most),
          shrunk towards a backup's level until he has a few hundred dropbacks
  rest    days since the last game; division game; neutral site

Games not yet played get the same features, as things stand now, for the season
simulation.

Usage:  python game_data.py         -> data/games.csv (reloads this season's plays only)
        python game_data.py --all   (reloads every season's plays, a few minutes)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# nflreadpy and polars are imported inside the download functions, so the app (which
# only reads the saved table) starts without them.

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = DATA / "games.csv"
TEAM_GAMES = DATA / "team_games.csv"   # per team per game, from the plays (cached)
QB_GAMES = DATA / "qb_games.csv"       # per QB per game, from the plays (cached)
TEAMS = DATA / "teams.csv"

FIRST_SEASON = 2002              # Houston joins: 32 teams from here on
RENAMED = {"OAK": "LV", "SD": "LAC", "STL": "LA"}  # moved franchises keep one name
WP_RANGE = (0.05, 0.95)          # plays outside this win probability (garbage time) are left out

ALPHA = 0.1                      # weight of the newest game in a team's running averages
CARRY = 0.6                      # share of a team's edge over average kept into a new season
STATS = ["off_epa", "def_epa", "pass_off", "pass_def", "succ_off", "succ_def", "margin"]

ELO_MEAN, ELO_K, ELO_HFA, ELO_REVERT = 1505, 20, 48, 1 / 3
ELO_PER_POINT = 25               # 25 Elo points ~ 1 point of margin

QB_DECAY = 0.95                  # per game played: older dropbacks count a little less
QB_PRIOR, QB_PRIOR_DROPBACKS = -0.10, 200  # a new QB starts at a backup's EPA per dropback


# ---------------------------------------------------------------- downloads

def current_season():
    import nflreadpy as nfl
    return int(nfl.get_current_season())


def load_schedule():
    import nflreadpy as nfl
    s = nfl.load_schedules(list(range(FIRST_SEASON, current_season() + 1))).to_pandas()
    for col in ("home_team", "away_team"):
        s[col] = s[col].replace(RENAMED)
    return s


def play_summaries(season):
    """(per team per game, per QB per game) from one season's plays."""
    import nflreadpy as nfl
    import polars as pl
    p = (nfl.load_pbp([season])
         .select(["game_id", "posteam", "defteam", "epa", "success", "pass", "rush", "qb_dropback",
                  "passer_player_id", "rusher_player_id", "wp"])
         .filter((pl.col("pass") == 1) | (pl.col("rush") == 1))
         .filter(pl.col("epa").is_not_null() & pl.col("posteam").is_not_null())
         .filter(pl.col("wp").is_between(*WP_RANGE))
         .to_pandas())
    for col in ("posteam", "defteam"):
        p[col] = p[col].replace(RENAMED)
    p["pass_epa"] = p["epa"].where(p["pass"] == 1)

    def side(team_col, prefix):
        g = p.groupby(["game_id", team_col])
        return pd.DataFrame({f"{prefix}_plays": g.size(), f"{prefix}_epa": g["epa"].mean(),
                             f"pass_{prefix}": g["pass_epa"].mean(), f"succ_{prefix}": g["success"].mean()})

    off = side("posteam", "off").rename_axis(["game_id", "team"])
    dfn = side("defteam", "def").rename_axis(["game_id", "team"])
    teams = off.join(dfn, how="outer").reset_index().assign(season=season)

    db = p[p["qb_dropback"] == 1].copy()
    db["qb_id"] = db["passer_player_id"].fillna(db["rusher_player_id"])
    qbs = (db.dropna(subset=["qb_id"]).groupby(["game_id", "posteam", "qb_id"])
           .agg(dropbacks=("epa", "size"), epa_sum=("epa", "sum")).reset_index()
           .rename(columns={"posteam": "team"}).assign(season=season))
    return teams, qbs


def update_play_cache(seasons):
    """Re-read the plays for `seasons` and merge them into the cached summaries."""
    old_t = pd.read_csv(TEAM_GAMES) if TEAM_GAMES.exists() else pd.DataFrame(columns=["season"])
    old_q = pd.read_csv(QB_GAMES) if QB_GAMES.exists() else pd.DataFrame(columns=["season"])
    new_t, new_q = [], []
    for season in seasons:
        print(f"  plays {season}...")
        t, q = play_summaries(season)
        new_t.append(t)
        new_q.append(q)
    keep = lambda old: old[~old["season"].isin(seasons)]
    teams = pd.concat([keep(old_t), *new_t], ignore_index=True).sort_values(["season", "game_id"])
    qbs = pd.concat([keep(old_q), *new_q], ignore_index=True).sort_values(["season", "game_id"])
    teams.to_csv(TEAM_GAMES, index=False)
    qbs.to_csv(QB_GAMES, index=False)
    return teams, qbs


def save_teams(sched):
    """The 32 teams on the schedule (the teams list also has old and alternate abbreviations)."""
    import nflreadpy as nfl
    t = nfl.load_teams().to_pandas()
    t = t[t["team_abbr"].isin(set(sched["home_team"]) | set(sched["away_team"]))]
    t = t[["team_abbr", "team_name", "team_nick", "team_conf", "team_division"]].drop_duplicates("team_abbr")
    t.to_csv(TEAMS, index=False)
    return t


# ---------------------------------------------------------------- features

def qb_rating(qbs, qb_id):
    num, den = qbs.get(qb_id, (0.0, 0.0))
    return (num + QB_PRIOR * QB_PRIOR_DROPBACKS) / (den + QB_PRIOR_DROPBACKS)


def elo_update(h, a, result, neutral):
    """New (home, away) Elo after a game, with a margin-of-victory multiplier."""
    diff = h - a + (0 if neutral else ELO_HFA)
    expected = 1 / (1 + 10 ** (-diff / 400))
    actual = 1.0 if result > 0 else 0.0 if result < 0 else 0.5
    winner_edge = diff if result > 0 else -diff
    mult = np.log(abs(result) + 1) * 2.2 / (winner_edge * 0.001 + 2.2) if result != 0 else 1.0
    shift = ELO_K * mult * (actual - expected)
    return h + shift, a - shift


def build(sched, team_games, qb_games):
    """Walk through every game in date order, recording each team's state before kickoff."""
    sched = sched.sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)
    tg = team_games.set_index(["game_id", "team"])
    qb_by_game = {k: g for k, g in qb_games.groupby(["game_id", "team"])}
    league = {s: float(team_games[s].mean()) if s != "margin" else 0.0 for s in STATS}  # newcomers start here

    teams, qbs, rows, season = {}, {}, [], None
    for g in sched.itertuples(index=False):
        if g.season != season:   # new season: everyone drifts back towards average
            season = g.season
            if teams:
                avg = {s: np.mean([t[s] for t in teams.values()]) for s in STATS}
                for t in teams.values():
                    for s in STATS:
                        t[s] = avg[s] + CARRY * (t[s] - avg[s])
                    t["elo"] = ELO_MEAN + (1 - ELO_REVERT) * (t["elo"] - ELO_MEAN)
                    t["games"] = 0
        played = not pd.isna(g.result)
        neutral = int(g.location == "Neutral")
        row = {"game_id": g.game_id, "season": g.season, "week": g.week, "game_type": g.game_type,
               "gameday": g.gameday, "home_team": g.home_team, "away_team": g.away_team,
               "home_score": g.home_score, "away_score": g.away_score, "result": g.result,
               "spread_line": g.spread_line, "total_line": g.total_line,
               "home_moneyline": g.home_moneyline, "away_moneyline": g.away_moneyline,
               "div_game": g.div_game, "neutral": neutral, "roof": g.roof,
               "home_rest": g.home_rest, "away_rest": g.away_rest}
        starters = {}
        for side, team in (("home", g.home_team), ("away", g.away_team)):
            t = teams.setdefault(team, {**league, "elo": ELO_MEAN, "games": 0, "last_qb": None})
            qb_id, qb_name = getattr(g, f"{side}_qb_id"), getattr(g, f"{side}_qb_name")
            if pd.isna(qb_id) and (g.game_id, team) in qb_by_game:   # fall back on the plays
                qb_id = qb_by_game[(g.game_id, team)].sort_values("dropbacks")["qb_id"].iloc[-1]
            if pd.isna(qb_id):                                         # not announced: last starter
                qb_id = t["last_qb"]
            starters[team] = qb_id
            rating = qb_rating(qbs, qb_id)
            usual = qb_rating(qbs, t["last_qb"]) if t["last_qb"] is not None else rating
            row.update({f"{side}_elo": t["elo"], f"{side}_games": t["games"], f"{side}_qb_id": qb_id,
                        f"{side}_qb_name": qb_name, f"{side}_qb_epa": rating,
                        f"{side}_qb_shift": rating - usual,
                        **{f"{side}_{s}": t[s] for s in STATS}})
        rows.append(row)
        if not played:
            continue

        h, a = teams[g.home_team], teams[g.away_team]
        h["elo"], a["elo"] = elo_update(h["elo"], a["elo"], g.result, neutral)
        for side, team, margin in (("home", g.home_team, g.result), ("away", g.away_team, -g.result)):
            t = teams[team]
            stats = tg.loc[(g.game_id, team)].to_dict() if (g.game_id, team) in tg.index else {}
            stats["margin"] = margin
            for s in STATS:
                if not pd.isna(stats.get(s, np.nan)):
                    t[s] += ALPHA * (stats[s] - t[s])
            t["games"] += 1
            for q in qb_by_game.get((g.game_id, team), pd.DataFrame()).itertuples(index=False):
                num, den = qbs.get(q.qb_id, (0.0, 0.0))
                qbs[q.qb_id] = (num * QB_DECAY + q.epa_sum, den * QB_DECAY + q.dropbacks)
            t["last_qb"] = starters[team]

    games = pd.DataFrame(rows)
    return add_matchup_features(games)


def add_matchup_features(games):
    """Home-minus-away differences: the form the model reads them in."""
    g = games
    net = lambda side, off, dfn: g[f"{side}_{off}"] - g[f"{side}_{dfn}"]
    g["elo_diff"] = g["home_elo"] - g["away_elo"]
    g["net_epa_diff"] = net("home", "off_epa", "def_epa") - net("away", "off_epa", "def_epa")
    g["pass_net_diff"] = net("home", "pass_off", "pass_def") - net("away", "pass_off", "pass_def")
    g["succ_net_diff"] = net("home", "succ_off", "succ_def") - net("away", "succ_off", "succ_def")
    g["margin_diff"] = g["home_margin"] - g["away_margin"]
    g["qb_diff"] = g["home_qb_epa"] - g["away_qb_epa"]
    g["rest_diff"] = (g["home_rest"] - g["away_rest"]).clip(-7, 7)
    return g


def main(reload_all=False):
    DATA.mkdir(exist_ok=True)
    season = current_season()
    print("Schedule and teams...")
    sched = load_schedule()
    save_teams(sched)
    cached = set(pd.read_csv(TEAM_GAMES, usecols=["season"])["season"]) if TEAM_GAMES.exists() else set()
    seasons = list(range(FIRST_SEASON, season + 1))
    todo = seasons if reload_all else [s for s in seasons if s not in cached or s == season]
    team_games, qb_games = update_play_cache(todo)
    games = build(sched, team_games, qb_games)
    games.to_csv(OUT, index=False)
    done = games[games["result"].notna()]
    now = games[games["season"] == season]
    print(f"{len(done):,} games played since {FIRST_SEASON} -> {OUT.name}. "
          f"{season}: {now['result'].notna().sum()} of {len(now)} played, through week "
          f"{int(now.loc[now['result'].notna(), 'week'].max())}.")


if __name__ == "__main__":
    main(reload_all="--all" in sys.argv)
