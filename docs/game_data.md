# The game table (`game_data.py`)

This builds the dataset the game model learns from: **every NFL game since 2002**,
one row each, with what was known about both teams *before kickoff*. Games not yet
played get a row too, with each team as it stands today, ready for the season
simulation.

Run it after every NFL week:

```
.venv\Scripts\python game_data.py         # reloads this season's plays only (~30 s)
.venv\Scripts\python game_data.py --all   # reloads every season since 2002 (a few minutes)
```

## Where the data comes from

Everything comes from [nflverse](https://github.com/nflverse), through the
`nflreadpy` package:

- **The schedule** (`load_schedules`): scores, the betting spread and total,
  moneylines, rest days, division game, neutral site and the starting QBs.
- **Every play** (`load_pbp`): about 50,000 plays a season, each with its
  **EPA** (expected points added: how much a play changed the expected score).
  The plays are boiled down to one line per team per game (`data/team_games.csv`)
  and one per QB per game (`data/qb_games.csv`), so later runs only reload the
  current season.
- **The teams** (`load_teams`): conference and division, saved to `data/teams.csv`.

Teams that moved keep one name throughout: Oakland → **LV**, San Diego → **LAC**,
St. Louis → **LA**.

Plays where one side had less than a 5% chance to win (garbage time) are left out,
because a team protecting a 28-point lead tells you little about how good it is.

## The features

| Feature | What it is |
|---|---|
| **Elo** | A simple rating that goes up with wins and down with losses, more for bigger margins. Each offseason it moves a third of the way back to average. |
| **EPA and success rate** | Running averages of offensive and defensive EPA per play, pass EPA, and success rate. The newest game carries 10% of the weight, so the last six or seven games count most. |
| **Recent margin** | The same kind of running average of points scored minus points allowed. |
| **The starting QB** | That QB's own EPA per dropback over their career, with recent games counting most. A QB with few dropbacks starts near a backup's level (−0.10) until they've earned more. |
| **QB shift** | How much better or worse today's starter is than the team's last starter. It's zero if it's the same player, and clearly negative when a backup steps in. |
| **Rest, division, neutral site** | Days since each team's last game, whether it's a division game, and whether it's at a neutral ground (Super Bowl, international games). |

Each offseason, team averages keep 60% of their edge over the league average,
because rosters and coaches change.

The model reads them as **home-minus-away differences** (`elo_diff`,
`net_epa_diff`, `qb_diff`…), which keeps the model simple and lets every game
teach it about both teams.

## No peeking

The script walks through the games in date order. It writes down each team's
numbers *before* a game and only then updates them with that game's result, so no
row can see its own outcome.
