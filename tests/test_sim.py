"""The season simulation and tiebreakers follow the NFL's rules."""
import numpy as np
import pytest

import game_model as gm
import season_sim as ss

SIMS = 2_000


@pytest.fixture(scope="module")
def odds(context):
    ctx, sigma = context
    return ss.simulate(ctx, sigma, n_sims=SIMS, seed_=0)[0]


def test_rebuilds_every_real_bracket(games, teams, season):
    check = ss.check_seeding(games, teams, range(gm.TRAIN_FROM, season))
    assert check["match"].all(), check.loc[~check["match"]].to_dict("records")


def test_seven_playoff_teams_per_conference(odds):
    assert odds.groupby("conf")["p_playoffs"].sum().round(6).eq(7).all()


def test_one_division_winner_and_bye_each(odds):
    assert odds.groupby("division")["p_division"].sum().round(6).eq(1).all()
    assert odds.groupby("conf")["p_bye"].sum().round(6).eq(1).all()


def test_one_champion(odds):
    assert odds["p_champion"].sum() == pytest.approx(1)


def test_odds_narrow_down_each_round(odds):
    chain = ["p_playoffs", "p_div_round", "p_conf_final", "p_super_bowl", "p_champion"]
    for a, b in zip(chain, chain[1:]):
        assert (odds[a] >= odds[b] - 1e-9).all(), (a, b)


def test_same_seed_same_answer(context):
    ctx, sigma = context
    a = ss.simulate(ctx, sigma, n_sims=500, seed_=3)[0]
    b = ss.simulate(ctx, sigma, n_sims=500, seed_=3)[0]
    assert a.equals(b)


def test_winning_a_game_never_hurts(context):
    ctx, sigma = context
    left = ctx["games"][~ctx["played"]]
    if left.empty:
        pytest.skip("no games left this season")
    g = left.iloc[0]
    run = lambda forced: ss.simulate(ctx, sigma, n_sims=SIMS, seed_=1, forced=forced)[0].set_index("team")
    win, loss = run({g["game_id"]: 1}), run({g["game_id"]: 0})
    home = g["home_team"]
    assert win.loc[home, "p_playoffs"] >= loss.loc[home, "p_playoffs"]
    assert np.isclose(win.loc[home, "wins"], loss.loc[home, "wins"])   # played wins unchanged
