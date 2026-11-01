"""The game model: XGBoost predicts the home team's margin, a normal curve turns it into a win chance.

One XGBRegressor, trained on every game since 2004 (2002-03 only warm up the ratings),
reads home-minus-away differences in Elo, EPA, success rate, recent margin and the
starting QB's rating, plus rest, division game and neutral site (built in game_data.py).
Its prediction is the expected margin; the actual margin scatters around it with a
spread SIGMA (about 13 points), fitted on out-of-sample games, so

    P(home wins) = Phi(predicted margin / sigma)

The season simulation samples margins from that same curve.
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import ndtr

from game_data import ELO_HFA, ELO_PER_POINT, OUT as TABLE

ROOT = TABLE.parent.parent
FOLDER = ROOT / "models" / "game"
REPORTS = ROOT / "reports"

FEATURES = [
    "elo_diff", "net_epa_diff", "pass_net_diff", "succ_net_diff", "margin_diff",
    "qb_diff", "home_qb_shift", "away_qb_shift", "rest_diff", "div_game", "neutral", "week",
]
PARAMS = dict(objective="reg:squarederror", n_estimators=500, learning_rate=0.02, max_depth=3,
              min_child_weight=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0,
              tree_method="hist", random_state=42, n_jobs=-1)

TRAIN_FROM = 2004        # first season used for training
BACKTEST_FROM = 2022     # every week from here on is forecast with only earlier games
SIGMA_DEFAULT = 13.5     # margin spread before enough out-of-sample games are in
MIN_SIGMA_GAMES = 200
PROB_CLIP = 0.005
PREDICTORS = {"model": "XGBoost model", "vegas": "betting spread", "elo": "Elo", "home": "home field only"}


# ---------------------------------------------------------------- data

def load_table(path=TABLE):
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Run `python game_data.py` first.")
    df = pd.read_csv(path)
    return df[df["season"] >= TRAIN_FROM].reset_index(drop=True)


def played(df):
    return df[df["result"].notna()]


# ---------------------------------------------------------------- model

def fit(train):
    return xgb.XGBRegressor(**PARAMS).fit(train[FEATURES], train["result"])


def predict(model, rows):
    """Expected home margin, in points."""
    return model.predict(rows[FEATURES])


def win_prob(margin, sigma):
    return ndtr(np.asarray(margin, dtype=float) / sigma)


def fit_sigma(residuals):
    """Spread of actual margins around the prediction, from out-of-sample games."""
    residuals = np.asarray(residuals, dtype=float)
    return float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) >= MIN_SIGMA_GAMES else SIGMA_DEFAULT


def contributions(model, rows):
    """SHAP values in points of margin (rows x features), plus the base value every game starts from."""
    values = model.get_booster().predict(xgb.DMatrix(rows[FEATURES]), pred_contribs=True)
    return pd.DataFrame(values[:, :-1], columns=FEATURES, index=rows.index), float(values[0, -1])


def importance(model):
    gain = pd.Series(model.get_booster().get_score(importance_type="total_gain"))
    return (gain / gain.sum()).reindex(FEATURES).fillna(0).sort_values(ascending=False)


def save(model, sigma, info):
    FOLDER.mkdir(parents=True, exist_ok=True)
    model.save_model(FOLDER / "margin.json")
    (FOLDER / "meta.json").write_text(json.dumps({"features": FEATURES, "sigma": sigma, **info}, indent=2))


def load():
    if not (FOLDER / "meta.json").exists():
        raise FileNotFoundError(f"No saved model in {FOLDER}. Run `python game_train.py` first.")
    model = xgb.XGBRegressor()
    model.load_model(FOLDER / "margin.json")
    return model, json.loads((FOLDER / "meta.json").read_text())


# ---------------------------------------------------------------- baselines

def baseline_margins(train, test):
    """Each baseline's expected home margin for the test games."""
    hfa = train.loc[train["neutral"] == 0, "result"].mean()
    return {
        "vegas": test["spread_line"].to_numpy(dtype=float),
        "elo": ((test["elo_diff"] + ELO_HFA * (1 - test["neutral"])) / ELO_PER_POINT).to_numpy(),
        "home": np.where(test["neutral"] == 1, 0.0, hfa),
    }


# ---------------------------------------------------------------- evaluation

def backtest(df, start=BACKTEST_FROM, progress=None):
    """Walk forward through every week from `start`: train on all earlier games, forecast the week.

    Each predictor's sigma comes from its own earlier out-of-sample misses, so the win
    chances are honest too.
    """
    games = played(df)
    weeks = (games[games["season"] >= start].groupby(["season", "week"])["gameday"].min()
             .sort_values().reset_index())
    results = []
    for i, w in enumerate(weeks.itertuples(index=False)):
        test = games[(games["season"] == w.season) & (games["week"] == w.week)].copy()
        train = games[games["gameday"] < w.gameday]
        margins = {"model": predict(fit(train), test), **baseline_margins(train, test)}
        done = pd.concat(results) if results else None
        for name, m in margins.items():
            sigma = fit_sigma(done["result"] - done[f"{name}_margin"]) if done is not None else SIGMA_DEFAULT
            test[f"{name}_margin"] = m
            test[f"{name}_p"] = win_prob(m, sigma)
        results.append(test)
        if progress:
            progress(i + 1, len(weeks), f"{w.season} week {w.week}")
    bt = pd.concat(results, ignore_index=True)
    bt["home_win"] = np.where(bt["result"] > 0, 1.0, np.where(bt["result"] < 0, 0.0, 0.5))  # tie = half
    return bt


def logloss(p, y):
    p = np.clip(p, PROB_CLIP, 1 - PROB_CLIP)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def score(bt, by=None):
    """Every predictor side by side. Picked-winner rate: higher is better; the rest: lower."""
    def block(b):
        decided = b[b["result"] != 0]
        rows = []
        for name in PREDICTORS:
            m, p = b[f"{name}_margin"], b[f"{name}_p"]
            rows.append({
                "predictor": name,
                "Margin error (points, lower is better)": (b["result"] - m).abs().mean(),
                "Margin RMSE (points, lower is better)": np.sqrt(((b["result"] - m) ** 2).mean()),
                "Picked the winner (higher is better)": (np.sign(decided[f"{name}_margin"]) == np.sign(decided["result"])).mean(),
                "Win log loss (lower is better)": logloss(p, b["home_win"]),
                "Win Brier score (lower is better)": ((p - b["home_win"]) ** 2).mean(),
            })
        res = pd.DataFrame(rows).set_index("predictor").T.rename_axis("metric").reset_index()
        res["games"] = len(b)
        return res

    if by is None:
        return block(bt)
    return pd.concat([block(g).assign(**{by: k}) for k, g in bt.groupby(by)], ignore_index=True)


def calibration(bt, col="model_p", bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)):
    """Predicted vs actual home win rate, binned. A calibrated model sits on the diagonal."""
    b = pd.cut(bt[col], bins, include_lowest=True)
    return (bt.groupby(b, observed=True)
            .agg(predicted=(col, "mean"), actual=("home_win", "mean"), games=("home_win", "size"))
            .reset_index(drop=True))
