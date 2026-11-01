"""XGBoost NFL Playoff Odds: every team's chances to make the playoffs and win it all.

Run:  run_app.bat   (or .venv\\Scripts\\python.exe -m streamlit run app.py)

Everything here is read from files the weekly routine writes (game_data.py ->
game_train.py -> season_forecast.py). Only the "What if?" tab simulates live.

Tabs: Playoff odds · This week · What if? · Odds over time · How good is it? · Data
"""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import game_model as gm
import season_sim as ss

# Chart palette: one set of hexes validated for colour-blind separation on both the
# light (#ffffff) and dark (#0f1726) surfaces with check_colours.py; see docs/app.md.
# Categorical slots go in this fixed order and follow the entity, never its rank.
SERIES = ["#2a78d6", "#d95926", "#199e70", "#c98500"]   # blue, orange, green, amber
BLUE, ORANGE, GREEN, AMBER = SERIES
RED = "#e34948"                  # diverging partner of BLUE (SHAP: favours the away team)
MUTED = "#898781"
GRID = "rgba(137,135,129,0.25)"

LINKEDIN_URL = "https://www.linkedin.com/in/tony-honesto-4195023"
GITHUB_URL = "https://github.com/atonyhonesto"
TITLE = "XGBoost NFL Playoff Odds"

# Page styling: a football-broadcast feel (navy scoreboard, yard lines, turf, condensed
# block type) without any league or team marks. Theme colours for Streamlit's own widgets
# live in .streamlit/config.toml. Fixed-colour blocks and translucent fills read in both
# light and dark mode.
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&display=swap');
:root { --navy: #0d1b33; --navy2: #1a3a6b; --ink: #081226; --red: #c62f37; --gold: #e3b448;
        --turf: #2f6b3a; --sport: 'Oswald', 'Arial Narrow', sans-serif; }

/* Banner: night-game scoreboard over a field */
.hero { position: relative; overflow: hidden; border-radius: 6px; padding: 24px 28px 36px;
        background: linear-gradient(110deg, var(--ink) 0%, var(--navy) 50%, var(--navy2) 100%); }
.hero::before { content: ""; position: absolute; inset: 0;   /* yard lines */
                background: repeating-linear-gradient(90deg, rgba(255,255,255,.07) 0 2px, transparent 2px 70px); }
.hero::after { content: ""; position: absolute; top: 0; right: -40px; width: min(220px, 30%); height: 100%;
               background: linear-gradient(90deg, var(--red) 0 70%, var(--gold) 70% 78%, transparent 78%);
               transform: skewX(-18deg); opacity: .9; }
.hero .name { position: relative; z-index: 1; font-family: var(--sport); font-weight: 700; letter-spacing: .5px;
              line-height: 1.05; font-size: clamp(2rem, 5vw, 3.2rem); color: #fff; }
.hero .name span { color: var(--gold); }
.hero .tag { position: relative; z-index: 1; margin-top: 6px; color: #c9d4e8; font-size: 1rem; }
.hero .by { position: relative; z-index: 1; margin-top: 10px; font-size: .95rem; color: #dfe6f3; }
.hero .by a { color: var(--gold); font-weight: 600; text-decoration: none; }
.hero .by a:hover { text-decoration: underline; }
.hero .turf { position: absolute; left: 0; right: 0; bottom: 0; height: 12px; z-index: 1;   /* hash marks */
              background: repeating-linear-gradient(90deg, rgba(255,255,255,.6) 0 2px, transparent 2px 18px) var(--turf); }

/* Section headings: condensed block type behind a red flag */
[data-testid="stMain"] h3 { font-family: var(--sport); font-weight: 600; letter-spacing: .3px; }
[data-testid="stMain"] h3::before { content: ""; display: inline-block; width: 6px; height: .8em; margin-right: 10px;
    vertical-align: -.05em; background: var(--red); box-shadow: 3px 0 0 var(--gold); }

/* Tabs */
[data-testid="stTab"] p { font-family: var(--sport); font-weight: 600; font-size: 1.1rem;
    text-transform: uppercase; letter-spacing: .8px; }
@media (max-width: 640px) { [data-testid="stTab"] p { font-size: .9rem; letter-spacing: .2px; } }

/* Panels around charts and tables: a scoreboard stripe across the top */
[class*="st-key-panel_"] { position: relative; overflow: hidden; }
[class*="st-key-panel_"]::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, var(--navy2) 0 70%, var(--red) 70% 92%, var(--gold) 92%); }

/* Stat tiles */
[data-testid="stMetric"] { border-left: 6px solid var(--red) !important; }
[data-testid="stMetricValue"] { font-family: var(--sport); font-weight: 600; }
[data-testid="stMetricLabel"] p { text-transform: uppercase; letter-spacing: .6px; font-size: .78rem; font-weight: 600; }

/* Info chips */
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 8px; }
.chip { font-size: .85rem; padding: 3px 11px; border-radius: 999px; white-space: nowrap;
        background: rgba(26,58,107,.10); border: 1px solid rgba(26,58,107,.40); }
.chip.hot { color: #fff; border: none; font-weight: 600; background: linear-gradient(90deg, var(--navy2), var(--red)); }

/* Score bugs: this week's games, styled like a broadcast score graphic */
.bugs { display: grid; grid-template-columns: repeat(auto-fill, minmax(270px, 1fr)); gap: 12px; margin: 6px 0 4px; }
.bug { background: var(--navy); color: #fff; border-radius: 6px; overflow: hidden; border-bottom: 4px solid var(--red);
       box-shadow: 0 0 0 1px rgba(137,135,129,.35); }   /* outline keeps the card visible on a dark page */
.bug .when { font-size: .75rem; letter-spacing: .6px; text-transform: uppercase; color: #aebbd4;
             padding: 6px 12px 4px; background: var(--ink); }
.bug .row { display: flex; align-items: center; gap: 10px; padding: 6px 12px; }
.bug .row + .row { border-top: 1px solid rgba(255,255,255,.08); }
.bug .abbr { font-family: var(--sport); font-weight: 700; font-size: 1.35rem; width: 52px; }
.bug .rec { font-size: .8rem; color: #aebbd4; flex: 1; }
.bug .pct { font-family: var(--sport); font-weight: 600; font-size: 1.2rem; padding: 0 8px; border-radius: 3px; }
.bug .fav .pct { background: var(--gold); color: var(--ink); }
.bug .bfoot { font-size: .8rem; line-height: 1.45; color: #dfe6f3; padding: 6px 12px 8px;
              background: rgba(255,255,255,.05); }
.bug .bfoot b { color: var(--gold); font-weight: 600; }

/* Footer */
.foot { margin-top: 36px; padding-top: 16px; font-size: .85rem; position: relative;
        display: flex; flex-wrap: wrap; gap: 10px 24px; justify-content: space-between; align-items: center; }
.foot::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 6px; opacity: .55;
                background: repeating-linear-gradient(90deg, #898781 0 2px, transparent 2px 14px); }
.foot a { color: var(--red); font-weight: 600; text-decoration: none; }
.foot .stack span { font-family: var(--sport); font-weight: 600; letter-spacing: .4px;
                    padding: 2px 9px; margin-right: 6px; border-radius: 3px; border: 1px solid rgba(137,135,129,.45); }
.foot .note { opacity: .65; flex-basis: 100%; }
</style>
"""

HERO = """
<div class="hero">
  <div class="name"><span>XGBoost</span> NFL Playoff Odds</div>
  <div class="tag">The rest of the season, simulated 10,000 times</div>
  <div class="by">Built by <b>Tony Honesto</b> &middot;
    <a href="{linkedin}" target="_blank">LinkedIn</a> &middot;
    <a href="{github}" target="_blank">GitHub</a></div>
  <div class="turf"></div>
</div>
"""

FOOTER = """
<div class="foot">
  <div>Built by <b>Tony Honesto</b> &middot; <a href="{linkedin}" target="_blank">LinkedIn</a> &middot;
       <a href="{github}" target="_blank">GitHub</a></div>
  <div class="stack"><span>XGBoost</span><span>Monte Carlo</span><span>SHAP</span><span>Plotly</span><span>Streamlit</span></div>
  <div class="note">An independent fan project built on nflverse's free public data. Not affiliated with or
       endorsed by the NFL, any team or any sponsor. Forecasts are for fun, not betting advice.</div>
</div>
"""

FEATURE_NAMES = {
    "elo_diff": "Elo rating edge",
    "net_epa_diff": "EPA per play edge (offence minus defence)",
    "pass_net_diff": "Passing EPA edge",
    "succ_net_diff": "Success-rate edge",
    "margin_diff": "Recent point-margin edge",
    "qb_diff": "Starting QB edge (EPA per dropback)",
    "home_qb_shift": "Home QB change vs usual starter",
    "away_qb_shift": "Away QB change vs usual starter",
    "rest_diff": "Rest-day edge",
    "div_game": "Division game",
    "neutral": "Neutral site",
    "week": "Week of the season",
}
ODDS = {"p_playoffs": "Make the playoffs", "p_division": "Win the division", "p_bye": "#1 seed (bye)",
        "p_super_bowl": "Reach the Super Bowl", "p_champion": "Win the Super Bowl"}
EVENT_NAMES = {"made_playoffs": "Make the playoffs", "won_division": "Win the division", "got_bye": "#1 seed (bye)",
               "made_super_bowl": "Reach the Super Bowl", "won_title": "Win the Super Bowl"}

st.set_page_config(page_title=TITLE, page_icon="🏈", layout="wide")


# ---------------------------------------------------------------- data

@st.cache_data
def get_games():
    return pd.read_csv(gm.TABLE)


@st.cache_data
def get_teams():
    return ss.load_teams()


@st.cache_resource
def get_model():
    return gm.load()


@st.cache_data
def get_reports():
    names = ("season_odds", "odds_history", "stakes", "game_next", "game_metrics", "game_metrics_by_season",
             "game_calibration", "season_metrics", "seeding_check")
    paths = {k: gm.REPORTS / f"{k}.csv" for k in names}
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing " + ", ".join(missing) + ". Run game_train.py and season_forecast.py.")
    return {k: pd.read_csv(p) for k, p in paths.items()}


@st.cache_data(show_spinner=False)
def games_csv():
    return get_games().to_csv(index=False)


@st.cache_resource(show_spinner=False)
def get_context(season):
    """Everything the what-if simulation needs (schedule, results, the model's predictions)."""
    model, meta = get_model()
    return ss.prepare(get_games(), season, get_teams(), model), meta["sigma"]


@st.cache_data(show_spinner=False, max_entries=64)
def what_if(season, forced_items):
    """Odds with some games fixed. Same random numbers every time, so changes come from the games."""
    ctx, sigma = get_context(season)
    odds, _, _ = ss.simulate(ctx, sigma, n_sims=ss.WHAT_IF_SIMS, seed_=1, forced=dict(forced_items))
    return odds


# ---------------------------------------------------------------- helpers

def style(fig, height=380, y_title=None, x_title=None):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif'),
    )
    fig.update_xaxes(title=x_title, showgrid=False)
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False)
    return fig


def panel(key):
    """A bordered card with a scoreboard stripe across the top (styled in STYLE)."""
    return st.container(border=True, key=f"panel_{key}")


def record(r):
    w, l, t = int(r["wins"]), int(r["losses"]), int(r["ties"])
    return f"{w}-{l}" + (f"-{t}" if t else "")


def chips(items, hot=None):
    out = [f'<span class="chip">{html.escape(c)}</span>' for c in items]
    if hot:
        out.append(f'<span class="chip hot">{html.escape(hot)}</span>')
    return '<div class="chips">' + "".join(out) + "</div>"


def by(team, margin):
    """'TB by 1.4' from a team and a margin in its favour."""
    return f"{team} by {abs(margin):.1f}" if round(abs(margin), 1) > 0 else "Pick 'em"


def pick_text(home, away, margin):
    return by(home, margin) if margin >= 0 else by(away, -margin)


def pct(label, top=100, help_=None):
    """A progress-bar column for a chance already scaled to 0-100 (see as_pct)."""
    return st.column_config.ProgressColumn(label, format="%.0f%%", min_value=0, max_value=top, help=help_)


def points(label):
    """A column for a change in percentage points (already scaled to 0-100)."""
    return st.column_config.NumberColumn(label, format="%+.1f pts")


def as_pct(df, cols):
    return df.assign(**{c: df[c] * 100 for c in cols})


def end_labels(fig, ends, gap, text):
    """Direct labels at the end of each line, nudged apart so they don't overlap.

    `ends` = [(x, name, y)]; `gap` = the smallest vertical distance between labels.
    """
    placed = []
    for x, name, y in sorted(ends, key=lambda e: e[2]):
        ly = max(y, placed[-1] + gap) if placed else y
        placed.append(ly)
        fig.add_annotation(x=x, y=ly, text=text(name, y), showarrow=False, xanchor="left", xshift=10,
                           yanchor="middle", font=dict(size=11))


def odds_table(odds, names, conf, group):
    d = odds[odds["conf"] == conf].copy()
    d["Team"] = d["team"] + " · " + d["team"].map(names)
    d["Record"] = d.apply(record, axis=1)
    d["Division"] = d["division"].str.replace(f"{conf} ", "", regex=False)
    d = (d.sort_values(["division", "p_division"], ascending=[True, False]) if group
         else d.sort_values(["p_playoffs", "p_champion"], ascending=False))
    show = as_pct(d[["Team", "Division", "Record", "exp_wins"] + list(ODDS)], ODDS).rename(
        columns={"exp_wins": "Exp. wins"})
    st.dataframe(show, hide_index=True, height=36 * len(show) + 40, column_config={
        "Exp. wins": st.column_config.NumberColumn(format="%.1f", help="Average final wins across the simulations"),
        "p_playoffs": pct("Playoffs"), "p_division": pct("Division"), "p_bye": pct("#1 seed"),
        "p_super_bowl": pct("Super Bowl", max(30, show["p_super_bowl"].max())),
        "p_champion": pct("Title", max(15, show["p_champion"].max())),
    })


def seed_heatmap(odds, conf):
    """Chance of each seed, team by team: one-hue sequential scale, numbers in the cells."""
    d = odds[odds["conf"] == conf].sort_values(["p_playoffs", "p_bye"], ascending=False)
    cols = [f"seed_{s}" for s in range(1, 8)]
    z = np.column_stack([d[cols].to_numpy(), 1 - d["p_playoffs"].to_numpy()])
    text = [[f"{v:.0%}" if v >= 0.005 else "" for v in row] for row in z]
    x = [f"#{s}" for s in range(1, 8)] + ["Out"]
    # Plotly picks cell text colours as if the page were white, so dark mode gets an
    # opaque navy-to-blue scale with white numbers instead of the translucent one.
    dark = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
    scale = ([[0, "#141f35"], [0.25, "#1f4a86"], [1, "#3987e5"]] if dark else
             [[0, "rgba(57,135,229,0.04)"], [0.25, "rgba(57,135,229,0.45)"], [1, "#184f95"]])
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=d["team"], text=text, texttemplate="%{text}", zmin=0, zmax=1, showscale=False, xgap=2, ygap=2,
        colorscale=scale, textfont=dict(color="#ffffff") if dark else None,
        hovertemplate="%{y} · %{x}: %{z:.1%}<extra></extra>"))
    style(fig, height=30 * len(d) + 70, x_title="Seed (1–4 = division winners, 5–7 = wild cards)")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(type="category", side="top", title_standoff=6)
    return fig


def score_bugs(week_games, stakes, odds):
    rec = odds.set_index("team").apply(record, axis=1)
    cards = []
    for _, g in week_games.iterrows():
        p_home = float(g["model_p"])
        day = pd.Timestamp(g["gameday"])
        when = f"{day:%a %b} {day.day}"
        rows = []
        for team, p in ((g["away_team"], 1 - p_home), (g["home_team"], p_home)):
            rows.append(f'<div class="row{" fav" if p > 0.5 else ""}"><span class="abbr">{html.escape(team)}</span>'
                        f'<span class="rec">{html.escape(rec.get(team, ""))}</span>'
                        f'<span class="pct">{p:.0%}</span></div>')
        line = pick_text(g["home_team"], g["away_team"], g["spread_line"]) if pd.notna(g["spread_line"]) else "n/a"
        bfoot = (f'Model: <b>{html.escape(pick_text(g["home_team"], g["away_team"], g["model_margin"]))}</b>'
                f' · Betting line: {html.escape(line)}')
        s = stakes[stakes["game_id"] == g["game_id"]]
        if len(s):
            s = s.iloc[0]
            bfoot += (f'<br>Playoff odds with a win / loss: {g["away_team"]} {s["away_p_playoffs_win"]:.0%} / '
                      f'{s["away_p_playoffs_loss"]:.0%} · {g["home_team"]} {s["home_p_playoffs_win"]:.0%} / '
                      f'{s["home_p_playoffs_loss"]:.0%}')
        cards.append(f'<div class="bug"><div class="when">{html.escape(when)} · {html.escape(g["away_team"])} at '
                     f'{html.escape(g["home_team"])}</div>{"".join(rows)}<div class="bfoot">{bfoot}</div></div>')
    return '<div class="bugs">' + "".join(cards) + "</div>"


def shap_chart(g):
    """What moved the predicted margin: SHAP values in points, blue = towards home, red = towards away."""
    contrib = pd.Series({f: g[f"shap_{f}"] for f in gm.FEATURES})
    contrib = contrib[contrib.abs() >= 0.05].sort_values(key=abs).tail(8)

    def shown(f):
        v = g[f]
        if f in ("div_game", "neutral"):
            return "yes" if v else "no"
        if f == "elo_diff":
            return f"{v:+.0f}"
        if f in ("rest_diff", "week"):
            return f"{v:+.0f}" if f == "rest_diff" else f"{v:.0f}"
        return f"{v:+.3f}" if abs(v) < 1 else f"{v:+.1f}"

    labels = [f"{FEATURE_NAMES.get(f, f)} = {shown(f)}" for f in contrib.index]
    fig = go.Figure(go.Bar(x=contrib.values, y=labels, orientation="h",
                           marker=dict(color=[BLUE if v > 0 else RED for v in contrib.values], cornerradius=4),
                           hovertemplate="%{y}<br>%{x:+.1f} points<extra></extra>"))
    style(fig, height=360, x_title=f"← towards {g['away_team']} · points · towards {g['home_team']} →")
    fig.update_layout(showlegend=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(zeroline=True, zerolinecolor=MUTED, showgrid=True, gridcolor=GRID)
    return fig


def stakes_chart(s):
    """Dumbbell: each team's odds if it loses (orange) and if it wins (blue)."""
    rows = []
    for side in ("away", "home"):
        team = s[f"{side}_team"]
        for col, label in (("p_playoffs", "playoffs"), ("p_division", "division"), ("p_bye", "#1 seed")):
            rows.append((f"{team} · {label}", s[f"{side}_{col}_loss"], s[f"{side}_{col}_win"]))
    rows = rows[::-1]
    y = [r[0] for r in rows]
    fig = go.Figure()
    for name, lo, hi in rows:
        fig.add_scatter(x=[lo, hi], y=[name, name], mode="lines", line=dict(color=GRID, width=4),
                        showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=[r[1] for r in rows], y=y, mode="markers+text", name="If they lose",
                    marker=dict(color=ORANGE, size=12, line=dict(color="white", width=2)),
                    text=[f"{r[1]:.0%}" for r in rows], textposition="middle left",
                    hovertemplate="%{y}: %{x:.0%} if they lose<extra></extra>")
    fig.add_scatter(x=[r[2] for r in rows], y=y, mode="markers+text", name="If they win",
                    marker=dict(color=BLUE, size=12, line=dict(color="white", width=2)),
                    text=[f"{r[2]:.0%}" for r in rows], textposition="middle right",
                    hovertemplate="%{y}: %{x:.0%} if they win<extra></extra>")
    style(fig, height=360)
    fig.update_xaxes(range=[-0.12, 1.12], tickformat=".0%", showgrid=True, gridcolor=GRID, tickvals=[0, .25, .5, .75, 1])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def history_chart(hist, teams_in, col):
    """One line per team (fixed colour per team), direct labels at the end, re-created weeks shaded."""
    fig = go.Figure()
    recreated = hist.loc[hist["recreated"], "week"]
    if len(recreated):
        fig.add_vrect(x0=recreated.min() - 0.4, x1=recreated.max() + 0.4, fillcolor=MUTED, opacity=0.08,
                      line_width=0, annotation_text="re-created", annotation_position="top left")
    for team, colour in zip(teams_in, SERIES):
        h = hist[hist["team"] == team].sort_values("week")
        fig.add_scatter(x=h["week"], y=h[col], mode="lines+markers", name=team,
                        line=dict(color=colour, width=2), marker=dict(color=colour, size=8, line=dict(color="white", width=1)),
                        hovertemplate=f"{team} after week %{{x}}: %{{y:.0%}}<extra></extra>")
    ends = [(h.iloc[-1]["week"], t, h.iloc[-1][col])
            for t in teams_in if len(h := hist[hist["team"] == t].sort_values("week"))]
    end_labels(fig, ends, gap=0.055, text=lambda name, y: f"{name} {y:.0%}")
    weeks = sorted(hist["week"].unique())
    style(fig, height=400, x_title="After week (0 = before the season)", y_title=ODDS[col])
    fig.update_xaxes(tickvals=weeks, range=[min(weeks) - 0.5, max(weeks) + 1.2])
    fig.update_yaxes(tickformat=".0%", range=[-0.02, 1.05])
    fig.update_layout(hovermode="x unified")
    return fig


# ---------------------------------------------------------------- load

try:
    games, teams = get_games(), get_teams()
    model, meta = get_model()
    rep = get_reports()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

names = dict(zip(teams["team_abbr"], teams["team_name"]))
odds = rep["season_odds"]
hist = rep["odds_history"].copy()
hist["recreated"] = hist.get("recreated", False)
hist["recreated"] = hist["recreated"].fillna(False).astype(bool)
season = int(hist["season"].max())
hist = hist[hist["season"] == season]
now_week = int(hist.loc[~hist["recreated"], "week"].max()) if (~hist["recreated"]).any() else int(hist["week"].max())
reg = games[(games["season"] == season) & (games["game_type"] == "REG")]
n_played = int(reg["result"].notna().sum())
last_run = hist.loc[hist["week"] == now_week, "run_on"].max()

st.html(STYLE)
st.html(HERO.format(linkedin=LINKEDIN_URL, github=GITHUB_URL))
st.html(chips([f"{season} season", f"Through week {now_week}", f"{n_played} of {len(reg)} games played",
               f"Updated {last_run}"], hot="10,000 simulated seasons"))

tabs = st.tabs(["Playoff odds", "This week", "What if?", "Odds over time", "How good is it?", "Data"])
tab_odds, tab_week, tab_whatif, tab_time, tab_eval, tab_data = tabs

# ---------------------------------------------------------------- playoff odds

with tab_odds:
    fav = odds.loc[odds["p_champion"].idxmax()]
    top_seed = {c: odds[odds["conf"] == c].loc[lambda d: d["p_bye"].idxmax()] for c in ss.CONFS}
    leaders = odds.loc[odds.groupby("division")["p_division"].idxmax()]
    tight = leaders.loc[leaders["p_division"].idxmin()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Super Bowl favourite", fav["team"], f"{fav['p_champion']:.0%} to win it all",
              delta_color="off", delta_arrow="off", border=True)
    for col, conf in zip((c2, c3), ss.CONFS):
        t = top_seed[conf]
        col.metric(f"{conf} #1 seed favourite", t["team"], f"{t['p_bye']:.0%} chance of the bye",
                   delta_color="off", delta_arrow="off", border=True)
    c4.metric("Tightest division", tight["division"], f"leader {tight['team']} only {tight['p_division']:.0%}",
              delta_color="off", delta_arrow="off", border=True)

    with panel("odds"):
        st.subheader("Every team's chances")
        c1, c2 = st.columns([1, 3], vertical_alignment="bottom")
        conf = c1.segmented_control("Conference", ss.CONFS, default="AFC", key="conf_odds") or "AFC"
        group = c2.toggle("Group by division", value=False)
        odds_table(odds, names, conf, group)
        st.caption("From 10,000 simulations of the rest of the season and the playoffs. Expected wins are "
                   "the average final win total. Bars in the Super Bowl and Title columns are scaled to the "
                   "favourite, so the differences show.")
    with panel("seeds"):
        st.subheader(f"Where each {conf} team is likely to be seeded")
        st.plotly_chart(seed_heatmap(odds, conf), key="seed_map")
        st.caption("Teams in order of playoff chances. Darker = more likely. Only the #1 seed gets a "
                   "first-round bye; switch conference above.")

# ---------------------------------------------------------------- this week

with tab_week:
    nxt, stakes = rep["game_next"], rep["stakes"]
    if nxt.empty:
        st.write("No games left to forecast.")
    else:
        wk = int(nxt["week"].iloc[0])
        st.subheader(f"Week {wk}: the model's picks")
        st.write("Each card shows the model's win chance for both teams, its predicted margin next to the betting "
                 "line, and what the game means for each team's playoff odds.")
        order = nxt.merge(stakes[["game_id", "swing"]], on="game_id", how="left").sort_values(
            ["gameday", "swing"], ascending=[True, False])
        st.html(score_bugs(order, stakes, odds))

        with panel("game"):
            st.subheader("Inside one game")
            labels = {r["game_id"]: f"{r['away_team']} at {r['home_team']}" for _, r in order.iterrows()}
            biggest = stakes["game_id"].iloc[0] if len(stakes) and stakes["game_id"].iloc[0] in labels else order["game_id"].iloc[0]
            gid = st.selectbox("Game", list(labels), index=list(labels).index(biggest), format_func=labels.get,
                               help="Starts on the game with the most at stake.")
            g = order[order["game_id"] == gid].iloc[0]
            m = g["model_margin"]
            k1, k2, k3 = st.columns(3)
            k1.metric("Model's pick", pick_text(g["home_team"], g["away_team"], m), border=True)
            k2.metric(f"{g['home_team']} win chance", f"{g['model_p']:.0%}", border=True)
            k3.metric("Betting line", pick_text(g["home_team"], g["away_team"], g["spread_line"])
                      if pd.notna(g["spread_line"]) else "n/a",
                      f"{m - g['spread_line']:+.1f} pts vs model" if pd.notna(g["spread_line"]) else None,
                      delta_color="off", delta_arrow="off", border=True)
            a, b = st.columns(2)
            with a:
                st.markdown("**What the model sees**")
                st.plotly_chart(shap_chart(g), key="shap")
                st.caption(f"SHAP values, in points of margin. Every game starts from the average home edge "
                           f"({g['shap_base']:+.1f}); these factors add up to the model's {m:+.1f}. Blue pushes "
                           f"towards {g['home_team']}, red towards {g['away_team']}.")
            with b:
                st.markdown("**What's at stake**")
                s = stakes[stakes["game_id"] == gid]
                if s.empty:
                    st.write("Not in this week's stakes run.")
                else:
                    st.plotly_chart(stakes_chart(s.iloc[0]), key="stakes")
                    st.caption(f"Each team's odds if it wins (blue) or loses (orange) this game, everything else "
                               f"simulated. {ss.WHAT_IF_SIMS:,} seasons each, with the same random numbers, so "
                               "the gap is the game itself.")

# ---------------------------------------------------------------- what if?

with tab_whatif:
    ctx, sigma = get_context(season)
    sched = ctx["games"].assign(played=ctx["played"])
    st.subheader("What if…?")
    st.write("Pick a team, decide some of its remaining games, and see how the odds move. Everything you leave "
             f"on *Simulate* is played out as normal ({ss.WHAT_IF_SIMS:,} seasons per run).")
    default_team = rep["stakes"]["home_team"].iloc[0] if len(rep["stakes"]) else odds["team"].iloc[0]
    order_teams = sorted(teams["team_abbr"])
    team = st.selectbox("Team", order_teams, index=order_teams.index(default_team),
                        format_func=lambda t: f"{t} · {names[t]}")
    left_games = sched[~sched["played"] & ((sched["home_team"] == team) | (sched["away_team"] == team))]
    left_games = left_games.sort_values("gameday")
    forced, decided = {}, []
    if left_games.empty:
        st.info(f"{team} has no regular-season games left.")
    else:
        cols = st.columns(3)
        for i, (_, g) in enumerate(left_games.iterrows()):
            home = g["home_team"] == team
            opp = g["away_team"] if home else g["home_team"]
            label = f"Week {int(g['week'])} · {'vs' if home else 'at'} {opp}"
            with cols[i % 3]:
                choice = st.segmented_control(label, ["Simulate", "Win", "Lose"], default="Simulate",
                                              key=f"wi_{team}_{g['game_id']}")
            if choice in ("Win", "Lose"):
                forced[g["game_id"]] = int((choice == "Win") == home)
                decided.append(f"{'Beats' if choice == 'Win' else 'Loses to'} {opp} (week {int(g['week'])})")

    with st.spinner("Simulating..."):
        base = what_if(season, ())
        new = what_if(season, tuple(sorted(forced.items())))
    b, n = base.set_index("team"), new.set_index("team")
    st.html(chips(decided or ["Every game simulated"], hot=f"{team}: what if"))
    cols = st.columns(5)
    for col, (c, label) in zip(cols, ODDS.items()):
        change = (n.at[team, c] - b.at[team, c]) * 100
        col.metric(label, f"{n.at[team, c]:.0%}", f"{change:+.0f} pts" if forced else None,
                   border=True, help=f"{b.at[team, c]:.1%} with every game simulated.")
    with panel("movers"):
        st.subheader("Who else moves")
        moves = pd.DataFrame({"Team": b.index, "Before": b["p_playoffs"] * 100, "After": n["p_playoffs"] * 100})
        moves["Change"] = moves["After"] - moves["Before"]
        moves = moves[moves["Change"].abs() >= 0.5].sort_values("Change", key=abs, ascending=False).head(10)
        if forced and len(moves):
            st.dataframe(moves, hide_index=True, column_config={
                "Before": pct("Playoffs before"), "After": pct("Playoffs after"), "Change": points("Change")})
            st.caption("Playoff odds that moved by at least half a point. Rivals for the same places move the "
                       "other way.")
        else:
            st.write("Set a game to *Win* or *Lose* to see the knock-on effects.")

# ---------------------------------------------------------------- odds over time

with tab_time:
    divisions = sorted(teams["team_division"].unique())
    tightest = tight["division"]
    c1, c2 = st.columns(2)
    division = c1.selectbox("Division", divisions, index=divisions.index(tightest))
    col = c2.selectbox("Odds of", list(ODDS), format_func=ODDS.get)
    in_div = sorted(teams.loc[teams["team_division"] == division, "team_abbr"])
    with panel("history"):
        st.subheader(f"{division}: how the odds have moved")
        st.plotly_chart(history_chart(hist[hist["team"].isin(in_div)], in_div, col), key="history")
        rc = hist.loc[hist["recreated"], "week"]
        note = (f" Weeks {rc.min()}–{rc.max()} were re-created afterwards as they would have looked then: "
                "results hidden, ratings frozen and the model retrained on earlier games only." if len(rc) else "")
        st.caption("One point per week, saved each time the forecast runs." + note)

    prev = hist[hist["week"] < now_week]["week"].max()
    if pd.notna(prev):
        with panel("movers_week"):
            st.subheader(f"Biggest movers since week {int(prev)}")
            a = hist[hist["week"] == prev].set_index("team")[col]
            z = hist[hist["week"] == now_week].set_index("team")[col]
            mv = pd.DataFrame({"Team": z.index, "Then": a.reindex(z.index).values * 100, "Now": z.values * 100})
            mv["Change"] = mv["Now"] - mv["Then"]
            mv = mv.sort_values("Change", key=abs, ascending=False).head(8)
            st.dataframe(mv, hide_index=True, column_config={
                "Then": pct(f"After week {int(prev)}"), "Now": pct(f"After week {now_week}"),
                "Change": points("Change")})
            st.caption(f"{ODDS[col]}: the teams whose odds moved most in the latest week.")

# ---------------------------------------------------------------- how good is it?

with tab_eval:
    gmx = rep["game_metrics"].set_index("metric")
    row = lambda prefix: gmx.loc[[m for m in gmx.index if m.startswith(prefix)][0]]
    picked, ll = row("Picked"), row("Win log")
    n_bt = int(gmx["games"].iloc[0])
    st.subheader("Single games: does the model beat simple baselines?")
    st.write(f"A week-by-week walk-forward test over **{n_bt:,} games since 2022**: before each week, the model "
             "was trained only on earlier games. It's compared with the betting line, a simple Elo rating, and "
             "\"the home team by its usual margin\".")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Picked the winner", f"{picked['model']:.1%}", f"{(picked['model'] - picked['elo']) * 100:+.1f} pts vs Elo",
              border=True)
    c2.metric("Win log loss", f"{ll['model']:.3f}", f"{ll['model'] - ll['elo']:+.3f} vs Elo", delta_color="inverse",
              border=True, help="Lower is better: how surprised the model was by the results.")
    c3.metric("vs the betting line", f"{ll['vegas']:.3f}", f"{ll['model'] - ll['vegas']:+.3f} model behind",
              delta_color="off", delta_arrow="off", border=True,
              help="The betting market's log loss. The model doesn't beat it.")
    c4.metric("Margin error", f"{row('Margin error')['model']:.1f} pts",
              f"betting line {row('Margin error')['vegas']:.1f}", delta_color="off", delta_arrow="off", border=True)
    st.caption("Lower log loss is better, so a negative difference is a win for the model. The betting market "
               "knows about injuries, weather and news the model never sees; very few public models beat it.")

    left, right = st.columns(2)
    with left, panel("by_season"):
        st.subheader("Season by season")
        bs = rep["game_metrics_by_season"]
        bs = bs[bs["metric"].str.startswith("Win log")].sort_values("season")
        fig = go.Figure()
        for (key, label), colour in zip(gm.PREDICTORS.items(), SERIES):
            fig.add_scatter(x=bs["season"], y=bs[key], mode="lines+markers", name=label,
                            line=dict(color=colour, width=2), marker=dict(color=colour, size=9, line=dict(color="white", width=1)),
                            hovertemplate=f"{label}, %{{x}}: %{{y:.3f}}<extra></extra>")
        last = bs.iloc[-1]
        end_labels(fig, [(last["season"], label, last[key]) for key, label in gm.PREDICTORS.items()],
                   gap=0.006, text=lambda name, y: name)
        style(fig, height=360, y_title="Win log loss (lower is better)")
        fig.update_xaxes(tickvals=bs["season"], range=[bs["season"].min() - 0.3, bs["season"].max() + 1.1])
        st.plotly_chart(fig, key="by_season_chart")
        st.caption(f"{bs['season'].max()} is a part season.")
    with right, panel("calibration"):
        st.subheader("Are the win chances honest?")
        cal = rep["game_calibration"]
        cal = cal[cal["games"] >= 20]
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated",
                        line=dict(color=MUTED, width=1, dash="dash"), hoverinfo="skip")
        fig.add_scatter(x=cal["predicted"], y=cal["actual"], mode="markers+lines", name="XGBoost model",
                        line=dict(color=BLUE, width=2),
                        marker=dict(size=[max(8, min(24, c / 12)) for c in cal["games"]], color=BLUE,
                                    line=dict(color="white", width=2)),
                        customdata=cal["games"],
                        hovertemplate="Predicted %{x:.0%} · home won %{y:.0%} (%{customdata} games)<extra></extra>")
        style(fig, height=360, x_title="Predicted home win chance", y_title="Share the home team won")
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig, key="calibration_chart")
        st.caption("Games grouped by predicted chance (bigger dots, more games). On the dashed line, the "
                   "percentages can be taken at face value.")

    with panel("importance"):
        st.subheader("What the game model relies on")
        imp = gm.importance(model).head(8).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=[FEATURE_NAMES.get(f, f) for f in imp.index], orientation="h",
                               marker=dict(color=BLUE, cornerradius=4),
                               hovertemplate="%{y}: %{x:.0%} of total gain<extra></extra>"))
        style(fig, height=340, x_title="Share of total gain")
        fig.update_layout(showlegend=False)
        fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=GRID)
        fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, key="importance_chart")
        st.caption(f"Trained on {meta['trained_games']:,} games through {meta['trained_through']}. Real margins "
                   f"land about {meta['sigma']:.1f} points either side of the prediction; the simulation uses that.")

    st.subheader("Season odds: would they have been right?")
    st.write("The same test for the playoff odds. For 2020–2025, each season was forecast **as if it were week "
             "5, 9 or 13**: later results hidden, ratings frozen, model retrained on earlier games. The baseline "
             "knows the standings too, but treats every remaining game as a coin flip (plus home field).")
    sm = rep["season_metrics"]
    model_name = f"model, tau {ss.TAU:g}"
    sm = sm[sm["predictor"].isin([model_name, "home field only"])]
    wide = sm.pivot_table(index=["event", "week"], columns="predictor", values="log_loss").reset_index()
    wide["Odds of"] = wide["event"].map(EVENT_NAMES)
    wide["order"] = wide["event"].map({e: i for i, e in enumerate(EVENT_NAMES)})
    wide = wide.sort_values(["order", "week"])
    wide["Model better?"] = np.where(wide[model_name] < wide["home field only"] - 0.002, "✅ yes",
                                     np.where(wide[model_name] > wide["home field only"] + 0.002, "❌ no", "➖ level"))
    with panel("season_eval"):
        st.dataframe(wide[["Odds of", "week", model_name, "home field only", "Model better?"]].rename(
            columns={"week": "From week", model_name: "Model", "home field only": "Coin-flip baseline"}),
            hide_index=True, height=36 * len(wide) + 40, column_config={"Model": st.column_config.NumberColumn(format="%.3f"),
                                            "Coin-flip baseline": st.column_config.NumberColumn(format="%.3f")})
        st.caption("Log loss, lower is better (within 0.002 counts as level). The model earns its keep on "
                   "division, #1 seed and Super Bowl odds. Once the standings take shape, plain playoff odds "
                   "are about as good from coin flips.")

    left, right = st.columns(2)
    with left, panel("tiebreak"):
        chk = rep["seeding_check"]
        st.subheader("Tiebreakers")
        st.metric("Real playoff brackets rebuilt", f"{int(chk['match'].sum())} of {len(chk)}",
                  f"every season {chk['season'].min()}–{chk['season'].max()}", delta_color="off", delta_arrow="off",
                  border=True)
        st.write("Ties are broken with the NFL's procedure: head-to-head, division record, common games, "
                 "conference record, strength of victory and strength of schedule, with the \"start again\" "
                 "rule for three-way ties.")
        st.write("**Simplified:** the late steps that rank points scored and allowed (and net touchdowns) are "
                 "folded into one net-points step, because the simulation draws margins rather than full "
                 "scores. Ties almost never get that far.")
    with right, panel("cant_know"):
        st.subheader("What the model can't know")
        st.markdown(
            "- **Injuries**, except a change of starting quarterback once it's announced. For future games, "
            "each team's latest starter is assumed to keep playing.\n"
            "- **Weather, trades, coaching changes** and anything else in the news.\n"
            "- **Motivation late in the season**: teams resting starters, or with nothing to play for.\n"
            "- **Points.** It predicts margins, so ties in a game are left out of the simulated season "
            "(they're rare: about 1 game in 300).\n"
            "- **Team strength inside a simulated season stays put.** Uncertainty about it is handled by a "
            "random strength shock per team, sized by the backtest.")

# ---------------------------------------------------------------- data

with tab_data:
    played = games[games["result"].notna()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Games in the table", f"{len(played):,}", border=True)
    c2.metric("Seasons", f"{played['season'].min()}–{played['season'].max()}", border=True)
    c3.metric(f"{season} games played", f"{n_played} of {len(reg)}", border=True)
    c4.metric("Simulated seasons", f"{ss.N_SIMS:,}", "per weekly run", delta_color="off", delta_arrow="off",
              border=True)

    with panel("ratings"):
        st.subheader("Team ratings now")
        state = ss.team_state(games, season, list(teams["team_abbr"]))
        long = pd.concat([reg[["gameday", "result", f"{s}_team", f"{s}_qb_name"]].set_axis(
            ["gameday", "result", "team", "qb"], axis=1) for s in ("home", "away")])
        qb = long[long["result"].notna()].sort_values("gameday").groupby("team")["qb"].last()
        ratings = pd.DataFrame({
            "Team": state.index, "Elo": state["elo"].round(0),
            "Offence EPA/play": state["off_epa"], "Defence EPA/play allowed": state["def_epa"],
            "Net EPA/play": state["off_epa"] - state["def_epa"], "Recent margin": state["margin"],
            "Latest starting QB": qb.reindex(state.index).values, "QB EPA/dropback": state["qb_epa"],
        }).sort_values("Elo", ascending=False)
        num = lambda f: st.column_config.NumberColumn(format=f)
        st.dataframe(ratings, hide_index=True, height=36 * len(ratings) + 40, column_config={
            "Elo": num("%d"), "Offence EPA/play": num("%+.3f"), "Defence EPA/play allowed": num("%+.3f"),
            "Net EPA/play": num("%+.3f"), "Recent margin": num("%+.1f"), "QB EPA/dropback": num("%+.3f")})
        st.caption("Running averages that weight recent games most (see docs/game_data.md). For defence, lower "
                   "is better. A QB's rating is shrunk towards a backup's level until they've played a lot.")

    with panel("table"):
        st.subheader("Every game since 2002")
        st.write("One row per game with the model's inputs as they stood **before kickoff**: the `home_` and "
                 "`away_` columns, and the home-minus-away differences the model reads.")
        st.dataframe(played.sort_values(["gameday", "game_id"], ascending=False).head(500), hide_index=True)
        st.caption("The latest 500 games played, newest first; the download has every game, including those "
                   "still to play. Source: nflverse, via nflreadpy.")
        st.download_button("Download the full table (CSV)", games_csv(), "games.csv",
                           mime="text/csv", icon=":material/download:")

st.html(FOOTER.format(linkedin=LINKEDIN_URL, github=GITHUB_URL))
