"""Shared fixtures: the saved game table, teams and model, loaded once per test run."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import game_model as gm  # noqa: E402
import season_sim as ss  # noqa: E402


@pytest.fixture(scope="session")
def games():
    return pd.read_csv(gm.TABLE)


@pytest.fixture(scope="session")
def teams():
    return ss.load_teams()


@pytest.fixture(scope="session")
def model():
    return gm.load()


@pytest.fixture(scope="session")
def season(games):
    return int(games["season"].max())


@pytest.fixture(scope="session")
def context(games, teams, model, season):
    m, meta = model
    return ss.prepare(games, season, teams, m), meta["sigma"]
