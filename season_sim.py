"""Play out the rest of the season 10,000 times: standings, seeding and the playoff bracket.

Each simulated season:
  1. every regular-season game still to play gets a margin from the game model:
     predicted margin + both teams' strength shocks + game noise;
  2. standings, then 7 seeds per conference: the 4 division winners (seeds 1-4) and
     the 3 best other teams (wild cards, seeds 5-7), with the tiebreakers below;
  3. the bracket: 2 v 7, 3 v 6, 4 v 5 with the #1 seed resting; then the #1 seed hosts
     the lowest seed left; the better seed hosts the conference final; the Super Bowl
     is at a neutral site.

The strength shock (one per team per simulated season, spread TAU) makes a team's
games move together: if a team is really better than the model thinks, it's better
every week. Game noise shrinks to match, so each single game still has spread sigma.
TAU was chosen by the season backtest below (2020-25, from weeks 5, 9 and 13).

Ties in the standings are broken by the NFL's procedure (tiebreakers.py).

Games and standings work on all simulations at once as (simulations x teams) arrays;
the tiebreakers go season by season, because they depend on which teams are tied.
"""
import numpy as np
import pandas as pd

import game_model as gm
from game_data import TEAMS, add_matchup_features
from tiebreakers import Tiebreaker, schedule_index

N_SIMS = 10_000
WHAT_IF_SIMS = 2_000             # quicker re-runs for the app's what-ifs (~1.5 s)
TAU = 5.0                       # points; smaller values were overconfident in the backtest
TAUS = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)   # tried by `season_forecast.py --backtest`
BACKTEST_SEASONS = range(2020, 2026)    # the 7-team playoff format
BACKTEST_WEEKS = (5, 9, 13)
PLAYOFF_WEEK = 19
CONFS = ("AFC", "NFC")
STATE = ["elo", "off_epa", "def_epa", "pass_off", "pass_def", "succ_off", "succ_def", "margin", "qb_epa"]
ROUND_PAIRS = ((2, 7), (3, 6), (4, 5))


def load_teams(path=TEAMS):
    return pd.read_csv(path).sort_values("team_abbr").reset_index(drop=True)


# ---------------------------------------------------------------- setup

def team_state(games, season, names):
    """Each team as it stands now: the ratings on its next game, or on its last if none is left."""
    g = games[games["season"] == season]
    sides = []
    for side in ("home", "away"):
        s = g[["gameday", "result", f"{side}_team"] + [f"{side}_{c}" for c in STATE]]
        sides.append(s.set_axis(["gameday", "result", "team"] + STATE, axis=1))
    long = pd.concat(sides).sort_values("gameday")
    upcoming = long[long["result"].isna()].groupby("team").first()
    latest = long.groupby("team").last()
    return upcoming[STATE].combine_first(latest[STATE]).reindex(names)


def matchup_margins(model, state, division, neutral):
    """Predicted margin for every possible playoff game: row team hosts column team."""
    k = len(state)
    i, j = (x.ravel() for x in np.meshgrid(np.arange(k), np.arange(k), indexing="ij"))
    s = {c: state[c].to_numpy() for c in STATE}
    net = lambda off, dfn: s[off] - s[dfn]
    rows = pd.DataFrame({
        "elo_diff": s["elo"][i] - s["elo"][j],
        "net_epa_diff": net("off_epa", "def_epa")[i] - net("off_epa", "def_epa")[j],
        "pass_net_diff": net("pass_off", "pass_def")[i] - net("pass_off", "pass_def")[j],
        "succ_net_diff": net("succ_off", "succ_def")[i] - net("succ_off", "succ_def")[j],
        "margin_diff": s["margin"][i] - s["margin"][j],
        "qb_diff": s["qb_epa"][i] - s["qb_epa"][j],
        "home_qb_shift": 0.0, "away_qb_shift": 0.0, "rest_diff": 0.0,
        "div_game": (division[i] == division[j]).astype(int), "neutral": neutral, "week": PLAYOFF_WEEK,
    })
    return gm.predict(model, rows).reshape(k, k)


def prepare(games, season, teams, model=None, played_only=False):
    """Everything about one season that stays the same across simulations.

    `played_only` drops games with no result (e.g. the cancelled 2022 BUF-CIN game),
    for checking seeding on a finished season. Without a model there are no predictions.
    """
    reg = games[(games["season"] == season) & (games["game_type"] == "REG")]
    if played_only:
        reg = reg[reg["result"].notna()]
    reg = reg.reset_index(drop=True)
    names = teams["team_abbr"].to_numpy()
    idx = {t: i for i, t in enumerate(names)}
    h, a = reg["home_team"].map(idx).to_numpy(), reg["away_team"].map(idx).to_numpy()
    eye = np.eye(len(names))
    division, conf = teams["team_division"].to_numpy(), teams["team_conf"].to_numpy()
    result = reg["result"].to_numpy(dtype=float)
    ctx = {
        "season": season, "names": names, "division": division, "conf": conf,
        "game_ids": reg["game_id"].to_numpy(), "games": reg, "h": h, "a": a,
        "H": eye[h], "A": eye[a],  # games x teams: who was home, who was away
        "same_div": (division[h] == division[a]).astype(float),
        "same_conf": (conf[h] == conf[a]).astype(float),
        "played": ~np.isnan(result), "result": np.nan_to_num(result),
        "fixed": np.where(result > 0, 1.0, np.where(result < 0, 0.0, 0.5)),  # home win, tie = half
        "schedule": schedule_index(h, a, len(names)),
    }
    if model is not None:
        ctx["pred"] = gm.predict(model, reg) if len(reg) else np.zeros(0)
        state = team_state(games, season, names)
        ctx["m_home"] = matchup_margins(model, state, division, neutral=0)
        ctx["m_neutral"] = matchup_margins(model, state, division, neutral=1)
    return ctx


# ---------------------------------------------------------------- standings and seeding

def standings(ctx, hw):
    """Records from home-win results (simulations x games): win %, division, conference, SOV, SOS."""
    H, A = ctx["H"], ctx["A"]

    def record(mask):
        w = hw * mask
        wins = w @ H + (mask - w) @ A
        n = mask @ H + mask @ A
        return wins, np.where(n > 0, wins / np.maximum(n, 1), 0.0)

    ones = np.ones(len(H))
    wins, pct = record(ones)
    _, div_pct = record(ctx["same_div"])
    _, conf_pct = record(ctx["same_conf"])
    h, a = ctx["h"], ctx["a"]
    beaten = (hw * pct[:, a]) @ H + ((1 - hw) * pct[:, h]) @ A   # win % of teams beaten, per win
    sov = np.where(wins > 0, beaten / np.maximum(wins, 1e-9), 0.0)
    sos = (pct[:, a] @ H + pct[:, h] @ A) / (ones @ H + ones @ A)
    return {"wins": wins, "pct": pct, "div_pct": div_pct, "conf_pct": conf_pct, "sov": sov, "sos": sos}


def seed(ctx, hw, margin, st, rng):
    """Seeds per conference: {conf: (simulations x 7) team index for seeds 1-7}.

    Win % alone settles most simulated seasons: no tie for a division lead, among the
    division winners, or around the wild-card places. Those are seeded all at once;
    only seasons with a tie that matters go through the full tiebreaker procedure.
    """
    tb = Tiebreaker(ctx, hw, margin, st, rng)
    pct = st["pct"]
    level = lambda x: np.abs(np.diff(x, axis=1)) < 1e-9   # neighbours with the same record
    seeds = {}
    for conf in CONFS:
        members = [int(t) for t in np.flatnonzero(ctx["conf"] == conf)]
        divisions = [[t for t in members if ctx["division"][t] == d]
                     for d in sorted(set(ctx["division"][members]))]
        tied = np.zeros(len(pct), bool)
        winners = []
        for d in divisions:
            top2 = np.sort(pct[:, d], axis=1)[:, -2:]
            tied |= level(top2)[:, 0]
            winners.append(np.asarray(d)[np.argmax(pct[:, d], axis=1)])
        winners = np.column_stack(winners)
        win_pct = np.take_along_axis(pct, winners, 1)
        by_record = np.argsort(-win_pct, axis=1, kind="stable")
        tied |= level(np.take_along_axis(win_pct, by_record, 1)).any(1)
        is_winner = np.zeros(pct.shape, bool)
        np.put_along_axis(is_winner, winners, True, 1)
        others = np.where(is_winner[:, members], -1.0, pct[:, members])
        wild = np.argsort(-others, axis=1, kind="stable")[:, :4]
        tied |= level(np.take_along_axis(others, wild, 1)).any(1)
        out = np.column_stack([np.take_along_axis(winners, by_record, 1), np.asarray(members)[wild[:, :3]]])
        for s in np.flatnonzero(tied):
            out[s] = tb.seed(s, members, divisions)
        seeds[conf] = out
    return seeds


# ---------------------------------------------------------------- the season

def simulate(ctx, sigma, tau=TAU, n_sims=N_SIMS, seed_=0, forced=None):
    """Simulate the rest of the season and the playoffs.

    `forced` = {game_id: 1 (home wins), 0 (away wins) or 0.5 (tie)} fixes games for a what-if.
    Returns (per-team summary, seeds per conference, regular-season wins per simulation).
    """
    rng = np.random.default_rng(seed_)
    k, h, a = len(ctx["names"]), ctx["h"], ctx["a"]
    shock = rng.normal(0, tau, (n_sims, k))
    noise = np.sqrt(max(sigma ** 2 - 2 * tau ** 2, 1.0))
    margin = ctx["pred"] + shock[:, h] - shock[:, a] + rng.normal(0, noise, (n_sims, len(h)))
    margin[:, ctx["played"]] = ctx["result"][ctx["played"]]
    col = {g: i for i, g in enumerate(ctx["game_ids"])}
    for game_id, value in (forced or {}).items():   # keep the margin's size, set its sign
        margin[:, col[game_id]] = np.abs(margin[:, col[game_id]]) * {1: 1, 0: -1, 0.5: 0}[value]
    hw = np.where(margin > 0, 1.0, np.where(margin < 0, 0.0, 0.5))

    st = standings(ctx, hw)
    seeds = seed(ctx, hw, margin, st, rng)

    def play(home, away, m):
        """True where the home team (or first-named, at a neutral site) wins."""
        s = lambda t: np.take_along_axis(shock, t[:, None], 1)[:, 0]
        return m[home, away] + s(home) - s(away) + rng.normal(0, noise, len(home)) > 0

    champs = {}
    reached_div, reached_conf = np.zeros(k), np.zeros(k)
    for conf in CONFS:
        by_seed = seeds[conf]
        team = lambda s: np.take_along_axis(by_seed, (s - 1)[:, None], 1)[:, 0]
        left = [np.ones(n_sims, dtype=int)]
        for hs, as_ in ROUND_PAIRS:
            hs, as_ = np.full(n_sims, hs), np.full(n_sims, as_)
            left.append(np.where(play(team(hs), team(as_), ctx["m_home"]), hs, as_))
        left = np.sort(np.column_stack(left), axis=1)           # seeds still alive, best first
        reached_div += np.bincount(np.take_along_axis(by_seed, left - 1, 1).ravel(), minlength=k)
        d1 = np.where(play(team(left[:, 0]), team(left[:, 3]), ctx["m_home"]), left[:, 0], left[:, 3])
        d2 = np.where(play(team(left[:, 1]), team(left[:, 2]), ctx["m_home"]), left[:, 1], left[:, 2])
        hi, lo = np.minimum(d1, d2), np.maximum(d1, d2)
        reached_conf += np.bincount(team(hi), minlength=k) + np.bincount(team(lo), minlength=k)
        champs[conf] = team(np.where(play(team(hi), team(lo), ctx["m_home"]), hi, lo))
    afc, nfc = champs["AFC"], champs["NFC"]
    sb = np.where(play(afc, nfc, ctx["m_neutral"]), afc, nfc)

    all_seeds = np.zeros((n_sims, k), dtype=int)
    for conf in CONFS:
        np.put_along_axis(all_seeds, seeds[conf], np.arange(1, 8)[None, :].repeat(n_sims, 0), 1)
    share = lambda x: np.bincount(x, minlength=k) / n_sims
    fixed_hw = np.where(ctx["played"], ctx["fixed"], 0.0)
    played = ctx["played"].astype(float)
    summary = pd.DataFrame({
        "team": ctx["names"], "conf": ctx["conf"], "division": ctx["division"],
        "wins": fixed_hw @ ctx["H"] + (played - fixed_hw) @ ctx["A"],
        "games": played @ ctx["H"] + played @ ctx["A"],
        "exp_wins": st["wins"].mean(0),
        "p_playoffs": (all_seeds > 0).mean(0),
        "p_division": ((all_seeds >= 1) & (all_seeds <= 4)).mean(0),
        "p_bye": (all_seeds == 1).mean(0),
        "p_div_round": reached_div / n_sims,
        "p_conf_final": reached_conf / n_sims,
        "p_super_bowl": share(np.concatenate([afc, nfc])),
        "p_champion": share(sb),
        **{f"seed_{s}": (all_seeds == s).mean(0) for s in range(1, 8)},
    })
    ties = ((ctx["fixed"] == 0.5) & ctx["played"]).astype(float)
    summary["ties"] = ties @ ctx["H"] + ties @ ctx["A"]
    summary["losses"] = summary["games"] - summary["wins"] - summary["ties"] / 2
    summary["wins"] -= summary["ties"] / 2
    return summary, seeds, st["wins"]


# ---------------------------------------------------------------- checking the seeding

def real_bracket(games, season):
    """Teams with a first-round bye and the wild-card matchups (home, away) that really happened."""
    po = games[games["season"] == season]
    wc, div = po[po["game_type"] == "WC"], po[po["game_type"] == "DIV"]
    in_wc = set(wc["home_team"]) | set(wc["away_team"])
    byes = (set(div["home_team"]) | set(div["away_team"])) - in_wc
    return byes, set(zip(wc["home_team"], wc["away_team"]))


def check_seeding(games, teams, seasons, seed_=0):
    """Rebuild each finished season's bracket from its results and compare with the real one.

    Before 2020 there were 6 seeds per conference, with byes for seeds 1 and 2.
    """
    rows = []
    for season in seasons:
        ctx = prepare(games, season, teams, played_only=True)
        hw = ctx["fixed"][None, :]
        seeds = seed(ctx, hw, ctx["result"][None, :], standings(ctx, hw), np.random.default_rng(seed_))
        names = ctx["names"]
        n_bye = 1 if season >= 2020 else 2
        pairs = ROUND_PAIRS if season >= 2020 else ROUND_PAIRS[1:]
        byes, matchups = set(), set()
        for conf in CONFS:
            s = names[seeds[conf][0]]
            byes |= set(s[:n_bye])
            matchups |= {(s[x - 1], s[y - 1]) for x, y in pairs}
        real_byes, real_matchups = real_bracket(games, season)
        rows.append({"season": season, "match": byes == real_byes and matchups == real_matchups,
                     "ours_only": sorted(map(str, (byes - real_byes) | (matchups - real_matchups))),
                     "real_only": sorted(map(str, (real_byes - byes) | (real_matchups - matchups)))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- backtest

def freeze(games, season, cutoff):
    """The season as it looked on `cutoff`: later results hidden, later games rated as of then."""
    g = games.copy()
    later = (g["season"] == season) & (g["gameday"] >= cutoff)
    g.loc[later, ["result", "home_score", "away_score"]] = np.nan
    state = team_state(g, season, sorted(set(g["home_team"])))
    for side in ("home", "away"):
        g.loc[later, [f"{side}_{c}" for c in STATE]] = state.loc[g.loc[later, f"{side}_team"], STATE].to_numpy()
        g.loc[later, f"{side}_qb_shift"] = 0.0
    return add_matchup_features(g)


def real_outcomes(games, season, names):
    """What really happened: playoff teams, division winners, #1 seeds, Super Bowl teams and champion."""
    po = games[(games["season"] == season) & (games["game_type"] != "REG")]
    wc, div, sb = (po[po["game_type"] == t] for t in ("WC", "DIV", "SB"))
    playoff = set(po["home_team"]) | set(po["away_team"])
    byes, _ = real_bracket(games, season)
    division = set(wc["home_team"]) | byes   # in the 7-team format, division winners host or rest
    champ = set(np.where(sb["result"] > 0, sb["home_team"], sb["away_team"]))
    in_sb = set(sb["home_team"]) | set(sb["away_team"])
    sets = {"made_playoffs": playoff, "won_division": division, "got_bye": byes,
            "made_super_bowl": in_sb, "won_title": champ}
    return pd.DataFrame({"team": names, **{c: [float(t in s) for t in names] for c, s in sets.items()}})


def backtest(games, teams, sigma, seasons=BACKTEST_SEASONS, weeks=BACKTEST_WEEKS, taus=(TAU,),
             n_sims=1000, progress=None):
    """Forecast past seasons from part-way through, as if it were then, and record what happened.

    For each season and cutoff week: train the game model on earlier games only, freeze
    the ratings, simulate the rest of the season with each tau, and set the odds next to
    the real outcome. Also a baseline: every remaining game decided by home field alone.
    """
    out, real, names = [], [], teams["team_abbr"].to_numpy()
    for season in seasons:
        real.append(real_outcomes(games, season, names).assign(season=season))
        for week in weeks:
            reg = games[(games["season"] == season) & (games["game_type"] == "REG")]
            cutoff = reg.loc[reg["week"] == week, "gameday"].min()
            g = freeze(games, season, cutoff)
            train = gm.played(g[g["season"] >= gm.TRAIN_FROM])
            model = gm.fit(train)
            ctx = prepare(g, season, teams, model)
            for tau in taus:
                s, _, _ = simulate(ctx, sigma, tau, n_sims=n_sims, seed_=season * 100 + week)
                out.append(s.assign(season=season, week=week, predictor=f"model, tau {tau:g}"))
            hfa = train.loc[train["neutral"] == 0, "result"].mean()
            base = {**ctx, "pred": np.where(ctx["games"]["neutral"] == 1, 0.0, hfa),
                    "m_home": np.full_like(ctx["m_home"], hfa), "m_neutral": np.zeros_like(ctx["m_neutral"])}
            s, _, _ = simulate(base, sigma, 0.0, n_sims=n_sims, seed_=season * 100 + week)
            out.append(s.assign(season=season, week=week, predictor="home field only"))
            if progress:
                progress(season, week)
    return pd.concat(out, ignore_index=True).merge(pd.concat(real), on=["season", "team"])


EVENTS = {"made_playoffs": "p_playoffs", "won_division": "p_division", "got_bye": "p_bye",
          "made_super_bowl": "p_super_bowl", "won_title": "p_champion"}


def score(bt):
    """Log loss and Brier score of each kind of odds, per predictor and cutoff week."""
    rows = []
    for (pred, week), g in bt.groupby(["predictor", "week"]):
        for event, col in EVENTS.items():
            rows.append({"predictor": pred, "week": week, "event": event,
                         "log_loss": gm.logloss(g[col], g[event]),
                         "brier": float(((g[col] - g[event]) ** 2).mean()), "team_seasons": len(g)})
    return pd.DataFrame(rows)
