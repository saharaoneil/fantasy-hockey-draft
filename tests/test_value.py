"""Projections and draft value, on data where the right answer is known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasy import projections, value


def _config(reliability, **overrides):
    return projections.ProjectionConfig(
        season_weights=overrides.get("season_weights", (5.0, 4.0, 3.0)),
        reliability=reliability,
        games_per_season=overrides.get("games_per_season", 82),
        min_games_for_age_curve=overrides.get("min_games_for_age_curve", 40),
        age_smoothing_window=overrides.get("age_smoothing_window", 3),
    )


def _history(rows):
    """rows: (playerId, season, gamesPlayed, goals, position, age)"""
    return pd.DataFrame(
        rows,
        columns=["playerId", "season", "gamesPlayed", "goals", "position", "age"],
    )


# ---------------------------------------------------------------------------
# How hard each stat regresses
# ---------------------------------------------------------------------------


def test_a_sticky_stat_barely_regresses():
    """rho 0.88 gives K = 11 games, so an 82-game season swamps it."""
    k = projections.regression_constant(0.88, games_per_season=82)
    assert 10 < k < 12


def test_a_noisy_stat_regresses_hard():
    """rho 0.35 gives K = 152 games -- more league average than player."""
    k = projections.regression_constant(0.35, games_per_season=82)
    assert k > 82


def test_a_perfectly_sticky_stat_needs_no_regression():
    assert projections.regression_constant(1.0, games_per_season=82) == 0.0


def test_regression_constant_rejects_impossible_reliability():
    for rho in (0.0, -0.2, 1.5):
        with pytest.raises(projections.ProjectionError):
            projections.regression_constant(rho, games_per_season=82)


# ---------------------------------------------------------------------------
# The projection
# ---------------------------------------------------------------------------


def _three_seasons(goals_per_game, games=82, position="C", player=1, age=27):
    return [
        (player, season, games, goals_per_game * games, position, age)
        for season in (2023, 2024, 2025)
    ]


def test_a_full_sample_stays_close_to_its_own_rate():
    """Three 82-game seasons of evidence should not be pulled far."""
    history = _history(_three_seasons(0.6) + _three_seasons(0.2, player=2))
    board = projections.project(
        history, ("goals",), target_season=2026, config=_config({"goals": 0.9})
    )
    rate = board.set_index("playerId")["goals_rate"]
    assert rate[1] == pytest.approx(0.6, abs=0.06)
    assert rate[2] == pytest.approx(0.2, abs=0.06)


def test_a_small_sample_is_pulled_toward_the_league():
    """Same rate, less evidence -- the sparse player regresses further."""
    history = _history(
        _three_seasons(1.2, games=82, player=1)
        + _three_seasons(1.2, games=6, player=2)
        + _three_seasons(0.3, games=82, player=3)
    )
    board = projections.project(
        history, ("goals",), target_season=2026, config=_config({"goals": 0.8})
    ).set_index("playerId")
    assert board.loc[2, "goals_rate"] < board.loc[1, "goals_rate"]


def test_a_noisier_stat_is_pulled_further_than_a_stickier_one():
    history = _history(_three_seasons(1.2) + _three_seasons(0.2, player=2))
    sticky = projections.project(
        history, ("goals",), target_season=2026, config=_config({"goals": 0.95})
    ).set_index("playerId").loc[1, "goals_rate"]
    noisy = projections.project(
        history, ("goals",), target_season=2026, config=_config({"goals": 0.2})
    ).set_index("playerId").loc[1, "goals_rate"]
    assert noisy < sticky


def test_regression_is_toward_the_position_not_the_whole_league():
    """A defenceman pulled toward the all-skater mean would be projected to
    score like a forward."""
    history = _history(
        _three_seasons(1.0, position="C", player=1)
        + _three_seasons(1.0, position="C", player=2)
        + _three_seasons(0.2, position="D", player=3)
        + _three_seasons(0.2, position="D", player=4)
    )
    board = projections.project(
        history, ("goals",), target_season=2026, config=_config({"goals": 0.5})
    ).set_index("playerId")
    assert board.loc[3, "goals_rate"] < 0.4


def test_a_stat_with_no_measured_reliability_is_refused():
    history = _history(_three_seasons(0.5))
    with pytest.raises(projections.ProjectionError, match="no measured reliability"):
        projections.project(
            history, ("goals",), target_season=2026, config=_config({"assists": 0.8})
        )


def test_projecting_a_season_with_no_history_is_an_error():
    history = _history(_three_seasons(0.5))
    with pytest.raises(projections.ProjectionError, match="no seasons"):
        projections.project(
            history, ("goals",), target_season=2040, config=_config({"goals": 0.8})
        )


# ---------------------------------------------------------------------------
# Aging
# ---------------------------------------------------------------------------


def test_a_measured_decline_shows_up_as_a_multiplier_below_one():
    rows = []
    for player in range(120):
        rows.append((player, 2023, 82, 41.0, "C", 33))
        rows.append((player, 2024, 82, 32.8, "C", 34))  # 20% drop
    history = _history(rows)
    curve = projections.measure_age_curve(
        history, config=_config({"goals": 0.8}), stat="goals"
    )
    assert curve.loc[33] == pytest.approx(0.8, abs=0.02)


def test_ages_outside_the_measured_range_are_clamped_not_extrapolated():
    """Extrapolating a decline curve produces confident nonsense about 40-year-olds."""
    curve = pd.Series({28: 0.98, 29: 0.96, 30: 0.94})
    assert projections.age_multiplier_for(45, curve) == 0.94
    assert projections.age_multiplier_for(20, curve) == 0.98


def test_an_age_curve_needs_enough_players_per_bucket():
    rows = [(p, s, 82, 40.0, "C", 30) for p in range(5) for s in (2023, 2024)]
    with pytest.raises(projections.ProjectionError, match="enough players"):
        projections.measure_age_curve(
            _history(rows), config=_config({"goals": 0.8}), stat="goals"
        )


# ---------------------------------------------------------------------------
# Scoring and replacement level
# ---------------------------------------------------------------------------


def _board(rows):
    """rows: (name, position, goals_rate, projected_games)"""
    frame = pd.DataFrame(
        rows, columns=["name", "position", "goals_rate", "projected_games"]
    )
    frame["playerId"] = range(len(frame))
    return frame


SCORING = value.ScoringConfig(weights={"goals": 3.0})


def test_points_are_rate_times_games():
    board = _board([("a", "C", 0.5, 80), ("b", "C", 0.5, 40)])
    points = value.fantasy_points(board, SCORING, games_column="projected_games")
    assert list(points) == [120.0, 60.0]


def test_mixing_position_groups_before_scoring_is_refused():
    """The bug this guard exists for: concatenating skaters and goalies leaves
    NaN in each other's columns, and one NaN poisons every total."""
    board = _board([("skater", "C", 0.5, 80), ("goalie", "G", np.nan, 60)])
    with pytest.raises(value.ValueError_, match="separately"):
        value.fantasy_points(board, SCORING, games_column="projected_games")


def test_replacement_level_is_the_last_starter_at_that_position():
    league = value.LeagueConfig(teams=2, starters={"C": 1})
    board = _board([(f"c{i}", "C", 1.0 - i * 0.1, 82) for i in range(5)])
    board["projected_points"] = value.fantasy_points(
        board, SCORING, games_column="projected_games"
    )
    levels = value.replacement_levels(board, league)
    # 2 teams x 1 starter = the 2nd best centre sets the level.
    assert levels["C"] == pytest.approx(board["projected_points"].nlargest(2).iloc[-1])


def test_a_scarcer_position_makes_its_players_more_valuable():
    """The whole reason to compute VORP: fewer alternatives, more marginal value."""
    board = _board(
        [(f"c{i}", "C", 0.9 - i * 0.02, 82) for i in range(20)]
        + [(f"d{i}", "D", 0.9 - i * 0.10, 82) for i in range(20)]
    )
    board["projected_points"] = value.fantasy_points(
        board, SCORING, games_column="projected_games"
    )
    league = value.LeagueConfig(teams=4, starters={"C": 1, "D": 1})
    ranked = value.value_over_replacement(board, league)

    top_c = ranked[ranked["position"] == "C"].iloc[0]
    top_d = ranked[ranked["position"] == "D"].iloc[0]
    # Equal projected points, but defence falls off a cliff -- so D is worth more.
    assert top_c["projected_points"] == pytest.approx(top_d["projected_points"])
    assert top_d["vorp"] > top_c["vorp"]


def test_replacement_level_moves_as_the_pool_drains():
    """Called on the undrafted pool mid-draft, this is what re-ranks the board."""
    board = _board([(f"c{i}", "C", 1.0 - i * 0.1, 82) for i in range(6)])
    board["projected_points"] = value.fantasy_points(
        board, SCORING, games_column="projected_games"
    )
    league = value.LeagueConfig(teams=2, starters={"C": 1})

    before = value.replacement_levels(board, league)["C"]
    after = value.replacement_levels(board.iloc[3:], league)["C"]
    assert after < before  # the good centres are gone; replacement got worse


def test_a_position_with_no_slot_is_an_error():
    board = _board([("x", "LW", 0.5, 82)])
    board["projected_points"] = 100.0
    with pytest.raises(value.ValueError_, match="no slot"):
        value.value_over_replacement(board, value.LeagueConfig(teams=2, starters={"C": 1}))


def test_scarcity_report_ranks_by_dropoff():
    board = _board(
        [(f"c{i}", "C", 0.9 - i * 0.01, 82) for i in range(10)]
        + [(f"d{i}", "D", 0.9 - i * 0.15, 82) for i in range(10)]
    )
    board["projected_points"] = value.fantasy_points(
        board, SCORING, games_column="projected_games"
    )
    league = value.LeagueConfig(teams=3, starters={"C": 1, "D": 1})
    report = value.positional_scarcity(board, league)
    assert report.iloc[0]["position"] == "D"


def test_an_empty_league_config_is_rejected():
    with pytest.raises(value.ValueError_):
        value.LeagueConfig(teams=12, starters={})
    with pytest.raises(value.ValueError_):
        value.LeagueConfig(teams=1, starters={"C": 2})
