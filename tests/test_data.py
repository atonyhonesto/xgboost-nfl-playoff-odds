"""The files the weekly routine writes are complete and sensible."""
import game_model as gm

MAX_MB = 25   # GitHub's per-file limit


def test_one_row_per_game(games):
    assert games["game_id"].is_unique


def test_features_present(games):
    missing = set(gm.FEATURES) - set(games.columns)
    assert not missing


def test_teams(teams):
    assert len(teams) == 32
    assert teams.groupby("team_division").size().eq(4).all()


def test_reports_exist():
    for name in ("season_odds", "odds_history", "stakes", "game_next", "game_metrics", "seeding_check"):
        assert (gm.REPORTS / f"{name}.csv").exists(), name


def test_files_fit_on_github():
    for path in [gm.TABLE, gm.FOLDER / "margin.json", *gm.REPORTS.glob("*.csv")]:
        assert path.stat().st_size < MAX_MB * 1e6, path.name
