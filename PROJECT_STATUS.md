# Project status: XGBoost NFL Playoff Odds

Snapshot for future sessions, 2026-11-11. Treat this as the source of truth; see
`CLAUDE.md` for working rules and `docs/*.md` for plain-English detail per file.

## 1. Purpose

A Streamlit app that simulates the rest of the NFL season 10,000 times and shows each
team's chances to make the playoffs, win its division, earn the #1 seed and win the
Super Bowl. An XGBoost game model plus Monte Carlo simulation and the real NFL
tiebreakers, backtested walk-forward against the betting line, Elo and home field.
Live at https://xgboost-nfl-playoff-odds.streamlit.app/, repo
https://github.com/atonyhonesto/xgboost-nfl-playoff-odds (public, branch `main`).

## 2. Architecture

- `game_data.py`: downloads nflverse data (nflreadpy) and builds pre-game features → `data/`.
- `game_model.py`: XGBoost margin model, win chance via sigma, SHAP, backtest and baselines.
- `game_train.py`: runs the walk-forward backtest, trains, saves `models/game/`, writes `reports/game_*`.
- `season_sim.py`: vectorised season + playoff simulation, seeding, season backtest.
- `tiebreakers.py`: the NFL's division and wild-card tiebreak procedures.
- `season_forecast.py`: the weekly forecast; writes `reports/season_odds`, `odds_history`, `stakes`, `seeding_check`.
- `app.py`: the six-tab Streamlit app; reads only files the routine wrote (what-ifs simulate live).
- `data/`: `games.csv`, `teams.csv` (in repo); `team_games.csv`, `qb_games.csv` (local caches, gitignored).
- `models/game/`: `margin.json` (XGBoost), `meta.json` (sigma etc.).
- `reports/`: CSVs the app shows.
- `docs/`: one plain-English guide per code file.
- `.github/workflows/keep-awake.yml`: visits the live app every 6 h and checks the title text.

## 3. Key files

- `app.py`: UI, cached loaders (keyed on `data_stamp()`), what-if tab.
- `season_forecast.py`: weekly run, history snapshots (re-creates missed weeks), stakes.
- `season_sim.py`: constants (`N_SIMS`, `WHAT_IF_SIMS`, `TAU`, `PLAYOFF_WEEK`), `simulate`, `seed`.
- `tiebreakers.py`: `Tiebreaker` class; points steps simplified to net points.
- `game_model.py`: `FEATURES`, `PARAMS`, `TRAIN_FROM`, `BACKTEST_FROM`, `load`/`save`.
- `game_data.py`: paths (`OUT`, `TEAMS`), `RENAMED`, `add_matchup_features`; `--all` rebuilds every season.
- `game_train.py`: prints the metrics and next week's picks.
- `README.md`: showcase plus a results table that is refreshed by hand.
- `docs/app.md`: the app's behaviour, palette and checks.
- `requirements.txt`: lower-bound pins only.
- `.streamlit/config.toml`: light/dark theme colours.
- `.github/workflows/keep-awake.yml`: `APP_URL` and the exact title string.
- `CLAUDE.md`: working rules and progress log (local only, gitignored).

## 4. Conventions

- Plain functions over classes (except `Tiebreaker`); module constants in UPPER_CASE.
- Paths come from `game_data` (`ROOT`, `DATA`, `OUT`, `TEAMS`) and `game_model` (`FOLDER`, `REPORTS`).
- Features use only data from before each game; every backtest is walk-forward.
- Every model number is shown next to a baseline, including where the model loses.
- Missing files raise `FileNotFoundError` with the script to run; the app shows it and stops.
- Heavy imports (nflreadpy, polars) are lazy so the app starts fast.
- Chart colours: fixed palette validated with `D:\Python\Motorsports Fuel Strategy\check_colours.py`.
- Testing: no test suite in the repo. Checks are `streamlit.testing.v1.AppTest` runs,
  Playwright/Edge screenshots (tall viewport, wait for `[data-testid="stMetric"]`),
  `check_seeding()` (22 of 22 brackets) and the backtests.
- Every code file has a `docs/<file>.md`, kept in step.
- Commits use the no-reply email; ask Tony before every push.

## 5. Build / run / test commands

```
setup.bat                                          rebuild .venv (once per PC)
run_app.bat                                        start the app locally
.venv\Scripts\python.exe game_data.py              refresh the current season (--all: every season)
.venv\Scripts\python.exe game_train.py             backtest + retrain (~1 min)
.venv\Scripts\python.exe season_forecast.py        simulate + save reports (~30 s-2 min)
.venv\Scripts\python.exe season_forecast.py --backtest   also re-run the season backtest (+3 min)
.venv\Scripts\python.exe -m streamlit run app.py   same as run_app.bat
git push                                           redeploys the live app (ask first)
```
`git`/`gh` live in `C:\Program Files\Git\cmd` and `C:\Program Files\GitHub CLI` if PATH is stale.
AppTest: `AppTest.from_file(r"D:\Python\NFL Playoff Odds\app.py", default_timeout=180).run()`.

## 6. Completed

- Game data and features since 2002 (Elo, EPA, success rate, QB rating, rest).
- XGBoost margin model with SHAP; walk-forward backtest vs betting line, Elo, home field.
- Season + playoff Monte Carlo with strength shock (tau = 5, chosen by backtest).
- NFL tiebreakers; rebuilds 22 of 22 real brackets (2004-25).
- Weekly forecast with odds history, re-created missed weeks and per-game stakes.
- Six-tab app with football-broadcast look, light/dark, validated chart palette.
- Docs for every code file; README with honest results.
- GitHub repo (topics, website link) and Streamlit Cloud deploy; keep-awake action passing.
- Weekly update through week 9 (2026-11-10), pushed and live.
- Data-only pushes now refresh the live app without a reboot (2026-11-11).

## 7. Remaining work

1. ~~**Finish the cache-staleness fix**~~ (`app.py`, `docs/app.md`). Done 2026-11-11,
   commit `f970044`: loaders keyed on `data_stamp()`; AppTest clean; live app
   confirmed showing "Through week 9" after the push.
2. **Weekly routine** each NFL week through week 18 (`game_data.py` → `game_train.py`
   → `season_forecast.py`), refresh the README results table (`README.md`), commit, push.
3. **Playoff weeks untested** (`season_forecast.py`, `season_sim.py`, `app.py`): behaviour
   once the regular season ends and real playoff games are played hasn't been run.
4. **Next-season rollover untested** (`game_data.py` `current_season`, `season_forecast.py`
   history): what happens when nflverse switches to 2027.
5. **No tests in the repo**: AppTest and screenshot scripts were kept in the session
   scratchpad only. Consider a small `tests/` folder (Playwright stays out of `requirements.txt`).
6. **Keep-awake Node 20 warning** (`.github/workflows/keep-awake.yml`): `actions/setup-python@v5`
   is being forced onto Node 24; bump the version before GitHub removes Node 20.
7. **Unused NASCAR copies** (`simulation.py`, `docs/simulation.md`): gitignored but still on
   disk; delete or keep as reference.

## 8. Known risks / gotchas

- **Caching:** any new data-file loader in `app.py` must take `stamp`, or a data-only
  push won't show until a reboot.
- **Title text** "XGBoost NFL Playoff Odds" is checked by keep-awake; never change or
  uppercase it with CSS.
- **Streamlit Cloud:** main file `app.py` can't be changed; Python 3.12 is fixed at deploy.
  `requirements.txt` has lower bounds only, so a reboot can pull newer libraries.
- **2026-11-03 outage:** Streamlit's static files failed intermittently for both of Tony's
  apps (not our code). If keep-awake fails, compare with the NASCAR app before debugging.
- **SSD damage:** after `chkdsk` some files were emptied or filled with garbage; use
  `D:\Python\Dev Hub\check_files.py` if anything fails oddly.
- **Line endings:** `core.autocrlf=true`; `.gitattributes` keeps `.bat` as CRLF.
- **Missed weeks** are re-created by `season_forecast.py` with a retrained model; this
  takes longer (about 30 s per week) and is labelled `recreated` in the history.
- **tau** was tuned on 2020-25; changing the game model's sigma or features means
  re-running `--backtest`.
- **Data volume:** keep each data file under 25 MB (`games.csv` is about 4 MB).
- **`CLAUDE.md` is gitignored**, so it lives only on the SSD; this file is in the repo
  unless ignored too.
