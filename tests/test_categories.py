"""Z-score valuation for categories leagues."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasy import categories as cats

CONFIG = cats.CategoriesConfig(
    pool_size=40, pool_iterations=4, min_pool=20, bench_multiplier=1.5,
    scale_by_roster_share=False,
)


def _skaters(goal_rates, hit_rates=None, games=82):
    """A frame of projected skaters, one row each."""
    n = len(goal_rates)
    hits = hit_rates if hit_rates is not None else [1.0] * n
    return pd.DataFrame({
        "name": [f"p{i}" for i in range(n)],
        "goals_rate": goal_rates,
        "hits_rate": hits,
        "projected_games": [games] * n,
    })


GOALS = (cats.Category("goals"),)
GOALS_AND_HITS = (cats.Category("goals"), cats.Category("hits"))


# ---------------------------------------------------------------------------
# Standardising
# ---------------------------------------------------------------------------


def test_a_better_player_scores_higher():
    frame = _skaters(list(np.linspace(0.1, 0.9, 40)))
    scored = cats.category_values(frame, GOALS, config=CONFIG)
    assert scored["category_value"].iloc[-1] > scored["category_value"].iloc[0]


def test_an_average_player_scores_about_zero():
    frame = _skaters(list(np.linspace(0.1, 0.9, 41)))
    scored = cats.category_values(frame, GOALS, config=CONFIG)
    middle = scored["category_value"].iloc[20]
    assert abs(middle) < 0.35


def test_a_category_nobody_differs_in_contributes_nothing():
    """A column that separates no one should not divide by zero, and should not
    add value either."""
    frame = _skaters([0.5] * 40, hit_rates=list(np.linspace(0.1, 3.0, 40)))
    scored = cats.category_values(frame, GOALS_AND_HITS, config=CONFIG)
    assert (scored["z_goals"] == 0).all()
    assert scored["z_hits"].std() > 0


def test_a_specialist_and_a_generalist_can_tie_on_total():
    """The thing z-scores do that a points total cannot: two identical totals
    built from completely different columns."""
    rates = list(np.linspace(0.1, 0.9, 38))
    frame = _skaters(rates + [0.9, 0.5], hit_rates=[1.0] * 38 + [0.2, 2.6])
    scored = cats.category_values(frame, GOALS_AND_HITS, config=CONFIG)
    specialist, generalist = scored.iloc[38], scored.iloc[39]
    assert specialist["z_goals"] > generalist["z_goals"]
    assert generalist["z_hits"] > specialist["z_hits"]


# ---------------------------------------------------------------------------
# Rate categories -- the trap
# ---------------------------------------------------------------------------


def _goalies(save_pcts, volumes):
    return pd.DataFrame({
        "name": [f"g{i}" for i in range(len(save_pcts))],
        "savePct_rate": save_pcts,
        "shots": volumes,
        "projected_games": [40] * len(save_pcts),
    })


SAVE_PCT = (cats.Category("savePct", rate_over="shots"),)


def test_the_same_rate_over_more_volume_is_worth_more():
    """A backup at .930 over 200 shots barely moves a team rate; a starter at
    .930 over 2,000 shots decides the column. Z-scoring the rate alone would
    call them equal."""
    rng = np.random.default_rng(0)
    frame = _goalies(
        list(rng.normal(0.905, 0.012, 28)) + [0.930, 0.930],
        list(rng.integers(400, 1800, 28)) + [200, 2000],
    )
    scored = cats.category_values(frame, SAVE_PCT, config=CONFIG)
    backup, starter = scored.iloc[28], scored.iloc[29]
    assert starter["category_value"] > backup["category_value"]


def test_a_lower_is_better_category_is_flipped():
    frame = pd.DataFrame({
        "name": [f"g{i}" for i in range(30)],
        "goalsAgainstAverage_rate": list(np.linspace(2.0, 4.0, 30)),
        "projected_games": [40] * 30,
    })
    gaa = (cats.Category("goalsAgainstAverage", higher_is_better=False,
                         rate_over="projected_games"),)
    scored = cats.category_values(frame, gaa, config=CONFIG)
    # Lowest GAA is the best player, so it must score highest.
    assert scored["category_value"].iloc[0] > scored["category_value"].iloc[-1]


def test_a_rate_category_without_its_volume_column_is_refused():
    frame = _goalies([0.9] * 30, [1000] * 30).drop(columns=["shots"])
    with pytest.raises(cats.CategoryError, match="volume column"):
        cats.category_values(frame, SAVE_PCT, config=CONFIG)


def test_an_unprojected_category_is_refused():
    frame = _skaters([0.5] * 30)
    with pytest.raises(cats.CategoryError, match="project 'assists' first"):
        cats.category_values(frame, (cats.Category("assists"),), config=CONFIG)


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


def test_the_pool_narrows_to_the_players_worth_drafting():
    frame = _skaters(list(np.linspace(0.05, 0.9, 200)))
    scored = cats.category_values(frame, GOALS, pool_size=40, config=CONFIG)
    assert int(scored["in_pool"].sum()) == 40
    # The pool is the top of the board, not an arbitrary slice.
    assert scored[scored["in_pool"]]["category_value"].min() >= \
           scored[~scored["in_pool"]]["category_value"].max()


def test_a_wider_pool_compresses_the_top():
    """Standardising against everyone drags the mean to replacement level and
    squashes the players you are actually choosing between."""
    frame = _skaters(list(np.linspace(0.05, 0.9, 200)))
    narrow = cats.category_values(frame, GOALS, pool_size=40, config=CONFIG)
    wide = cats.category_values(frame, GOALS, pool_size=200, config=CONFIG)
    assert wide["category_value"].max() > narrow["category_value"].max()


def test_a_pool_too_small_to_standardise_is_refused():
    frame = _skaters([0.5 + i * 0.01 for i in range(30)])
    with pytest.raises(cats.CategoryError, match="below the 20 needed"):
        cats.category_values(frame, GOALS, pool_size=5, config=CONFIG)


def test_too_few_players_to_value_at_all():
    with pytest.raises(cats.CategoryError, match="too few"):
        cats.category_values(_skaters([0.5] * 5), GOALS, config=CONFIG)


# ---------------------------------------------------------------------------
# Putting groups on one scale
# ---------------------------------------------------------------------------


def test_group_scale_rewards_a_group_that_punches_above_its_roster_share():
    """Goalies: a third of the standings from a sixth of the roster."""
    goalie = cats.group_scale(categories_in_group=4, total_categories=12,
                              slots_in_group=24, total_slots=144)
    skater = cats.group_scale(categories_in_group=8, total_categories=12,
                              slots_in_group=120, total_slots=144)
    assert goalie > skater
    assert goalie == pytest.approx(2.0)


def test_group_scale_rejects_impossible_shares():
    with pytest.raises(cats.CategoryError):
        cats.group_scale(categories_in_group=0, total_categories=12,
                         slots_in_group=24, total_slots=144)
    with pytest.raises(cats.CategoryError, match="cannot exceed"):
        cats.group_scale(categories_in_group=20, total_categories=12,
                         slots_in_group=24, total_slots=144)


def _two_groups():
    rng = np.random.default_rng(1)
    skaters = _skaters(list(rng.normal(0.4, 0.15, 120).clip(0.02)))
    goalies = pd.DataFrame({
        "name": [f"g{i}" for i in range(60)],
        "wins_rate": rng.normal(0.45, 0.1, 60).clip(0.05),
        "projected_games": [45] * 60,
    })
    return skaters, goalies


def test_groups_come_back_on_one_comparable_scale():
    skaters, goalies = _two_groups()
    board = cats.value_groups(
        {"skaters": skaters, "goalies": goalies},
        {"skaters": GOALS, "goalies": (cats.Category("wins"),)},
        {"skaters": 120, "goalies": 24},
        config=CONFIG,
    )
    assert set(board["value_group"]) == {"skaters", "goalies"}
    assert board["category_value"].notna().all()
    assert (board["group_scale"] == 1.0).all()  # off by default


def test_roster_share_scaling_is_available_but_not_the_default():
    """Measured, turning it on put goalies in every top place -- so it stays
    opt-in rather than being deleted or made the default."""
    skaters, goalies = _two_groups()
    scaled_config = cats.CategoriesConfig(
        pool_size=40, pool_iterations=4, min_pool=20, bench_multiplier=1.5,
        scale_by_roster_share=True,
    )
    board = cats.value_groups(
        {"skaters": skaters, "goalies": goalies},
        {"skaters": GOALS, "goalies": (cats.Category("wins"),)},
        {"skaters": 120, "goalies": 24},
        config=scaled_config,
    )
    goalie_scale = board[board["value_group"] == "goalies"]["group_scale"].iloc[0]
    skater_scale = board[board["value_group"] == "skaters"]["group_scale"].iloc[0]
    assert goalie_scale > skater_scale


def test_a_group_without_categories_is_refused():
    skaters, goalies = _two_groups()
    with pytest.raises(cats.CategoryError, match="no categories defined"):
        cats.value_groups(
            {"skaters": skaters, "goalies": goalies},
            {"skaters": GOALS},
            {"skaters": 120, "goalies": 24},
            config=CONFIG,
        )


def test_a_group_without_a_drafted_count_is_refused():
    skaters, goalies = _two_groups()
    with pytest.raises(cats.CategoryError, match="no drafted count"):
        cats.value_groups(
            {"skaters": skaters, "goalies": goalies},
            {"skaters": GOALS, "goalies": (cats.Category("wins"),)},
            {"skaters": 120},
            config=CONFIG,
        )


def test_contributions_show_which_columns_a_player_wins():
    skaters, goalies = _two_groups()
    board = cats.value_groups(
        {"skaters": skaters, "goalies": goalies},
        {"skaters": GOALS, "goalies": (cats.Category("wins"),)},
        {"skaters": 120, "goalies": 24},
        config=CONFIG,
    )
    board["position"] = "C"
    table = cats.category_contributions(
        board, GOALS + (cats.Category("wins"),), limit=5,
        rank_by="category_value",
    )
    assert "z_goals" in table.columns and "z_wins" in table.columns
    assert len(table) == 5
