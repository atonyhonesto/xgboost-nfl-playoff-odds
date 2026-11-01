# The app (`app.py`)

The Streamlit app everyone sees. It reads the files the weekly routine writes
(`game_data.py` → `game_train.py` → `season_forecast.py`), so it opens quickly. Only
the *What if?* tab simulates live.

Run it with `run_app.bat`, or `.venv\Scripts\python.exe -m streamlit run app.py`.

## The tabs

| Tab | What's on it |
|---|---|
| **Playoff odds** | Four headline tiles (the Super Bowl favourite, each conference's #1-seed favourite, the tightest division race). A table of every team's chances for one conference, sorted by playoff odds or grouped by division. A heatmap of each team's chance of every seed, 1–7 or out. |
| **This week** | Every game left this week as a TV-style score bug: both teams' win chances, the model's margin next to the betting line, and each team's playoff odds with a win or a loss. *Inside one game* explains the pick with SHAP values in points, and charts what's at stake for both teams. It starts on the game with the most at stake. |
| **What if?** | Pick a team and set any of its remaining games to *Win* or *Lose*. The odds update in about 1.5 seconds (2,000 simulated seasons), with *Who else moves* showing the knock-on effect on other teams. |
| **Odds over time** | One division at a time: how each team's odds have moved week by week, and the biggest movers since last week. Weeks re-created after the fact are shaded and labelled. |
| **How good is it?** | The game model against the betting line, Elo and home field; its calibration and what it relies on. The season backtest (would the playoff odds have been right?). The tiebreaker check (22 of 22 real brackets), and what the model can't know. |
| **Data** | Team ratings now (Elo, EPA, the latest starting QB), and the full game table with a download. |

## Honest by design

- **Every model number has a baseline next to it**, including where the model loses:
  the betting line beats it on single games, and coin flips match it on plain playoff
  odds late in the season.
- **What-ifs use the same random numbers every time**, so a change in the odds comes
  from the games you set, not from simulation noise.
- **Re-created history is labelled.** Weeks the forecast didn't run live were rebuilt
  afterwards using only what was known then, and the chart shades them.
- **The limits are spelled out** in the app itself: injuries other than a QB change,
  weather, motivation late in the season, and the simplified points-based tiebreakers.

## Look and feel

The look is a football broadcast, not a league brand:

- **The banner:** a night-game scoreboard. A navy gradient with faint yard lines, a red
  and gold slash, and a turf strip with hash marks along the bottom.
- **Type:** headings, tabs and tile values in **Oswald**, a condensed block typeface
  (from Google Fonts).
- **Score bugs:** this week's games are drawn like the score graphic on a broadcast.
- **What it avoids:** no NFL shield, team logos, official team colours or league fonts.
  Teams appear as plain abbreviations and names. The colours are our own shades, and
  the footer says it's an independent fan project.
- **Themes:** light and dark are both defined in `.streamlit/config.toml`.

The title is never uppercased with CSS, because the keep-awake check looks for the
exact text *XGBoost NFL Playoff Odds*.

## Chart colours

One set of hex values has to work in both light and dark mode, because Plotly charts
don't switch with the theme. The categorical colours, always in this order:

| Slot | Colour | Used for |
|---|---|---|
| 1 | blue `#2a78d6` | the model; the first team in a division |
| 2 | orange `#d95926` | the betting line; "if they lose" |
| 3 | green `#199e70` | Elo |
| 4 | amber `#c98500` | home field only |

SHAP charts use blue (towards the home team) against red `#e34948` (towards the away
team).

All of these pass `check_colours.py` on both the white and the navy (`#0f1726`)
background:

- **Lightness band:** passes.
- **Chroma floor:** passes.
- **Colour-blind separation:** the worst pair is ΔE 8.4, against a target of 8 or more.
- **Normal-vision separation:** the worst pair is ΔE 19.8, against a floor of 15.
- **Contrast:** at least 3:1 everywhere.

The first attempt used the reference palette's brighter orange and yellow. Those failed
the dark-mode lightness band, so the palette's own darker steps are used instead.

The seed heatmap uses a single blue scale. In dark mode it switches to an opaque
navy-to-blue scale with white numbers, because Plotly would otherwise pick dark text
for pale cells on a dark page.

## Checking it

- `streamlit.testing.v1.AppTest` runs the app, flips a what-if game, switches the
  conference and grouping, and changes the odds-over-time measure, with no errors.
- Screenshots of every tab in light and dark mode are taken with Playwright and Edge,
  using a tall viewport and waiting for a stat tile before capturing.
