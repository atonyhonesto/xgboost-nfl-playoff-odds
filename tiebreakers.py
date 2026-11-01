"""NFL tiebreakers: who finishes ahead when teams have the same record.

The NFL's procedure, applied inside every simulated season:

Division ties (who wins the division, and the order inside it)
  1. head-to-head record among the tied teams
  2. division record
  3. record in common games
  4. conference record
  5. strength of victory      6. strength of schedule
  7. net points in all games  (stands in for the NFL's points-ranking steps)
  8. coin flip

Ties between teams from different divisions (wild cards, and seeding the division winners)
  0. only the best team from each division (by the division tiebreaker) goes on
  1. head-to-head: two teams, if they met; three or more, only if one swept the rest
  2. conference record
  3. common games, if each team played at least 4
  4. strength of victory      5. strength of schedule
  6. net points               7. coin flip

Whenever a step separates some teams but not all, the teams still tied start again
at step 1 (the two-team version if two are left). Once a team is placed, the rest
start over too.

Simplified: the NFL's steps that rank points scored and allowed (and net points in
common games, and net touchdowns) are folded into one "net points in all games" step.
"""
import numpy as np

EPS = 1e-9
MIN_COMMON = 4   # common games each team needs before that step counts (between divisions)


def schedule_index(h, a, k):
    """The season's schedule as matrices: which games are (home team, away team) pairs."""
    n_games = len(h)
    home_pair, away_pair = np.zeros((n_games, k * k), np.float32), np.zeros((n_games, k * k), np.float32)
    home_pair[np.arange(n_games), h * k + a] = 1   # a home win counts for (home, away)
    away_pair[np.arange(n_games), a * k + h] = 1   # an away win counts for (away, home)
    met = (home_pair + away_pair).sum(0).reshape(k, k)
    return {"home_pair": home_pair, "away_pair": away_pair, "met": met,
            "opponents": [set(np.flatnonzero(met[t])) for t in range(k)]}


class Tiebreaker:
    """Standings and tiebreakers for every simulated season.

    `hw` = home-win results (simulations x games, tie = 0.5), `margin` = home margins,
    `st` = the records from season_sim.standings().
    """

    def __init__(self, ctx, hw, margin, st, rng):
        sched, k = ctx["schedule"], len(ctx["division"])
        hw32 = hw.astype(np.float32)
        # vs[s, t, u] = team t's wins over team u in simulation s (a tie = half a win)
        self.vs = (hw32 @ sched["home_pair"] + (1 - hw32) @ sched["away_pair"]).reshape(len(hw), k, k)
        self.met, self.opponents = sched["met"], sched["opponents"]
        self.st, self.division = st, ctx["division"]
        self.net = margin @ ctx["H"] - margin @ ctx["A"]
        self.coin = rng.random(st["pct"].shape)

    # ------------------------------------------------------------ records

    def record(self, s, t, opponents):
        """(win %, games) for team t in simulation s against `opponents`."""
        games = self.met[t, opponents].sum()
        return (float(self.vs[s, t, opponents].sum() / games), games) if games else (np.nan, 0)

    def common_opponents(self, group):
        return sorted(set.intersection(*(self.opponents[t] for t in group)))

    # ------------------------------------------------------------ steps (None = doesn't apply)

    def head_to_head(self, s, group):
        """Each team's record against the rest of the group; only if every team met the others."""
        recs = [self.record(s, t, [u for u in group if u != t]) for t in group]
        return None if any(n == 0 for _, n in recs) else [p for p, _ in recs]

    def sweep(self, s, group):
        """Three or more teams: only counts if one beat each of the others, or lost to each."""
        rec = {(t, u): self.record(s, t, [u]) for t in group for u in group if t != u}
        for result, value in ((1.0, 1.0), (0.0, -1.0)):
            for t in group:
                if all(rec[t, u][1] > 0 and rec[t, u][0] == result for u in group if u != t):
                    return [value if x == t else 0.0 for x in group]
        return None

    def common_games(self, s, group, minimum=1):
        opponents = self.common_opponents(group)
        if not opponents:
            return None
        recs = [self.record(s, t, opponents) for t in group]
        return None if any(n < minimum for _, n in recs) else [p for p, _ in recs]

    def stat(self, name):
        return lambda s, group: [self.st[name][s, t] for t in group]

    def net_points(self, s, group):
        return [self.net[s, t] for t in group]

    def coin_flip(self, s, group):
        return [self.coin[s, t] for t in group]

    # ------------------------------------------------------------ procedures

    def decide(self, s, group, steps, again):
        """Go through the steps until one separates the teams; the best team comes out."""
        for step in steps:
            values = step(s, group)
            if values is None:
                continue
            best = max(values)
            ahead = [t for t, v in zip(group, values) if v >= best - EPS]
            if len(ahead) < len(group):
                return ahead[0] if len(ahead) == 1 else again(s, ahead)
        return group[0]

    def best_in_division(self, s, group):
        if len(group) == 1:
            return group[0]
        steps = [self.head_to_head, self.stat("div_pct"), self.common_games, self.stat("conf_pct"),
                 self.stat("sov"), self.stat("sos"), self.net_points, self.coin_flip]
        return self.decide(s, group, steps, self.best_in_division)

    def best_in_conference(self, s, group):
        by_div = {}
        for t in group:
            by_div.setdefault(self.division[t], []).append(t)
        group = [self.best_in_division(s, g) for g in by_div.values()]
        if len(group) == 1:
            return group[0]
        steps = [self.head_to_head if len(group) == 2 else self.sweep, self.stat("conf_pct"),
                 lambda s_, g: self.common_games(s_, g, MIN_COMMON),
                 self.stat("sov"), self.stat("sos"), self.net_points, self.coin_flip]
        return self.decide(s, group, steps, self.best_in_conference)

    def order(self, s, teams, best, places):
        """The first `places` of `teams`: best record first, equal records split by `best`."""
        pct = self.pct_row
        left, out = list(teams), []
        while left and len(out) < places:
            top = max(pct[t] for t in left)
            group = [t for t in left if pct[t] >= top - EPS]
            pick = group[0] if len(group) == 1 else best(s, group)
            out.append(pick)
            left.remove(pick)
        return out

    def seed(self, s, members, divisions):
        """Seeds 1-7 of one conference: division winners 1-4, then three wild cards."""
        self.pct_row = self.st["pct"][s].tolist()   # plain floats: much quicker to compare one by one
        winners = [self.order(s, d, self.best_in_division, 1)[0] for d in divisions]
        others = [t for t in members if t not in winners]
        return self.order(s, winners, self.best_in_conference, 4) + self.order(s, others, self.best_in_conference, 3)
