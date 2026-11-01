# The tiebreakers (`tiebreakers.py`)

With 17 games, records bunch up. In most simulated seasons at least one playoff
place, division title or seed comes down to teams with the same record. This file
settles those ties the way the NFL does.

## The two procedures

**Inside a division** (who wins it, and the order behind the winner):

1. Head-to-head record among the tied teams
2. Division record
3. Record in common games (against opponents they all played)
4. Conference record
5. Strength of victory (the combined win % of the teams they beat)
6. Strength of schedule (the combined win % of everyone they played)
7. Net points in all games
8. Coin flip

**Between divisions** (wild cards, and the order of the four division winners):

0. Only the top team from each division (by the list above) stays in
1. Head-to-head: for two teams, only if they met; for three or more, only if one team
   beat all the others (or lost to all of them)
2. Conference record
3. Common games, only if each team played at least 4
4. Strength of victory
5. Strength of schedule
6. Net points in all games
7. Coin flip

## The "start again" rule

The NFL's rule that trips up most simple versions: when a step separates *some*
tied teams but not all of them, the ones still level **go back to step 1**, using the
two-team version if only two are left. In the same way, once a team is placed (say
the first wild card), everyone else still tied starts over for the next place.

For example, take a three-way tie for a division. If division record drops one team,
the other two don't move on to common games: they go back to head-to-head between
just the two of them.

## What's simplified

The NFL's later steps rank teams by points scored and allowed, then use net points in
common games, then net touchdowns. The simulation draws margins, not full scores, so
those steps are folded into one: **net points in all games**. Ties almost never get
that far.

## How we know it works

`season_sim.check_seeding()` rebuilds every finished season since 2004 from its real
results and compares the playoff bracket (byes and wild-card matchups) with what
really happened.

- With the simple rules from step 2 (no head-to-head or common games): **7 of 22**
  seasons matched.
- With this file: **22 of 22**. That covers two-way, three-way and wild-card ties, and
  both the old 6-team and current 7-team formats.

## Speed

Head-to-head and common games depend on *which* teams are tied, so this runs one
simulated season at a time rather than all 10,000 at once. Two things keep it quick:

- **A results table per season.** For every simulated season, a table of "team A's
  wins over team B" is built once with two matrix multiplications. After that, any
  head-to-head or common-games record is a quick lookup.
- **A shortcut for seasons without a deciding tie.** If no division lead, division
  winner or wild-card place is tied, win % alone sets the seeds, and those seasons are
  seeded all at once. It gives exactly the same answers.

10,000 seasons take about 7 seconds, and the app's 2,000-season what-ifs about 1.5.

## The building blocks

- **`schedule_index()`** turns the season's games into the matrices the results table
  is built from.
- **`Tiebreaker.record()`** gives a team's record against any set of opponents in one
  simulated season.
- **`head_to_head()`, `sweep()`, `common_games()`, `net_points()`, `coin_flip()`** are
  the individual steps. Each returns nothing when the step doesn't apply.
- **`decide()`** runs the steps and handles "start again".
- **`best_in_division()`, `best_in_conference()`** are the two procedures.
- **`order()`, `seed()`** place teams one at a time and produce seeds 1–7.
