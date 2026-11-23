"""The app runs without errors and picks up new data without a reboot."""
import os
import time

import pytest
from streamlit.testing.v1 import AppTest

import game_model as gm
from conftest import ROOT

TIMEOUT = 180


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT).run()


def test_runs_without_errors(app):
    assert not app.exception, [e.value for e in app.exception]
    assert len(app.metric) >= 4


def test_title_unchanged():
    # The keep-awake GitHub Action looks for this exact text.
    assert "XGBoost NFL Playoff Odds" in (ROOT / "app.py").read_text(encoding="utf-8")


def test_reloads_after_a_data_push(app):
    path = gm.REPORTS / "season_odds.csv"
    before = path.stat()
    try:
        os.utime(path, (time.time() + 5, time.time() + 5))   # what a git pull does
        app.run()
        assert not app.exception, [e.value for e in app.exception]
    finally:
        os.utime(path, (before.st_atime, before.st_mtime))
