"""The measurement logic, on data where the right answer is known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasy import nhl, predictability as pred


def _seasons(rows):
    """rows: (playerId, season, gamesPlayed, points, position)"""
    return pd.DataFrame(
        rows, columns=["playerId", "season", "gamesPlayed", "points", "position"]
    )


# ---------------------------------------------------------------------------
# Season identifiers
# ---------------------------------------------------------------------------


def test_season_id_joins_both_years():
    assert nhl.season_id(2023) == 20232024
    assert nhl.season_id(1999) == 19992000


def test_an_implausible_season_is_rejected():
    with pytest.raises(ValueError):
        nhl.season_id(1800)


# ---------------------------------------------------------------------------
# Pairing seasons
# ---------------------------------------------------------------------------


def test_only_consecutive_seasons_are_paired():
    """A player who misses a year and returns says nothing about year-over-year
    persistence, and folding them in would understate every stat."""
    frame = _seasons([
        (1, 2020, 82, 50, "C"),
        (1, 2021, 82, 55, "C"),   # consecutive with 2020 -> a pair
        (1, 2023, 82, 60, "C"),   # gap year -> not paired with 2021
    ])
    rates = pred._rate_frame(frame, ("points",), id_column="playerId")
    paired = pred.consecutive_pairs(rates, id_column="playerId", min_games=1)

    assert len(paired) == 1
    assert paired.iloc[0]["season"] == 2020


def test_players_below_the_games_threshold_are_dropped():
    frame = _seasons([
        (1, 2020, 5, 3, "C"), (1, 2021, 82, 60, "C"),
        (2, 2020, 82, 50, "C"), (2, 2021, 82, 55, "C"),
    ])
    rates = pred._rate_frame(frame, ("points",), id_column="playerId")
    paired = pred.consecutive_pairs(rates, id_column="playerId", min_games=20)

    assert list(paired["playerId"]) == [2]


def test_no_pairs_at_all_is_an_error_not_an_empty_answer():
    frame = _seasons([(1, 2020, 82, 50, "C"), (2, 2022, 82, 50, "C")])
    rates = pred._rate_frame(frame, ("points",), id_column="playerId")
    with pytest.raises(pred.PredictabilityError, match="no consecutive"):
        pred.consecutive_pairs(rates, id_column="playerId", min_games=1)


# ---------------------------------------------------------------------------
# Per-game conversion
# ---------------------------------------------------------------------------


def test_counting_stats_become_per_game_rates():
    """Totals confound production with availability, and the COVID seasons were
    56 and 68 games -- raw totals are not comparable across that boundary."""
    frame = _seasons([(1, 2020, 40, 40, "C"), (1, 2021, 80, 40, "C")])
    rates = pred._rate_frame(frame, ("points",), id_column="playerId")

    assert list(rates["points"]) == [1.0, 0.5]


def test_rate_stats_are_left_alone():
    frame = pd.DataFrame({
        "playerId": [1], "season": [2020], "gamesPlayed": [40],
        "savePct": [0.915],
    })
    rates = pred._rate_frame(frame, ("savePct",), id_column="playerId")
    assert rates["savePct"].iloc[0] == pytest.approx(0.915)


def test_zero_games_does_not_divide_by_zero():
    frame = _seasons([(1, 2020, 0, 0, "C"), (1, 2021, 82, 50, "C")])
    rates = pred._rate_frame(frame, ("points",), id_column="playerId")
    assert np.isnan(rates["points"].iloc[0])


def test_an_unknown_stat_is_an_error():
    frame = _seasons([(1, 2020, 82, 50, "C")])
    with pytest.raises(pred.PredictabilityError, match="not in the data"):
        pred._rate_frame(frame, ("nonsense",), id_column="playerId")


# ---------------------------------------------------------------------------
# The measurement itself
# ---------------------------------------------------------------------------


def _synthetic(persistence, *, players=200, seed=0):
    """Build two seasons where next-year rate = persistence * this year + noise."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.6, 0.25, players).clip(0.02, None)
    noise = rng.normal(0, 0.25, players)
    nxt = (persistence * base + (1 - persistence) * noise).clip(0.02, None)

    rows = []
    for index in range(players):
        rows.append((index, 2020, 82, base[index] * 82, "C"))
        rows.append((index, 2021, 82, nxt[index] * 82, "C"))
    return _seasons(rows)


def test_a_perfectly_persistent_stat_measures_near_one():
    results = pred.measure(_synthetic(1.0), ("points",), id_column="playerId")
    assert results[0].spearman > 0.99
    assert results[0].verdict == "very sticky"


def test_a_pure_noise_stat_measures_near_zero():
    results = pred.measure(_synthetic(0.0), ("points",), id_column="playerId")
    assert abs(results[0].spearman) < 0.2
    assert results[0].verdict == "close to noise"


def test_persistence_is_ordered_as_constructed():
    rhos = [
        pred.measure(_synthetic(p), ("points",), id_column="playerId")[0].spearman
        for p in (0.2, 0.5, 0.9)
    ]
    assert rhos[0] < rhos[1] < rhos[2]


def test_results_can_be_grouped_by_position():
    frame = _seasons(
        [(i, 2020, 82, 40 + i, "C" if i % 2 else "D") for i in range(120)]
        + [(i, 2021, 82, 40 + i, "C" if i % 2 else "D") for i in range(120)]
    )
    results = pred.measure(frame, ("points",), id_column="playerId", by_position=True)
    assert {r.group for r in results} == {"C", "D"}


def test_a_tiny_sample_is_skipped_rather_than_reported():
    """Fewer than 30 pairs is not a measurement, so no number is produced."""
    frame = _seasons(
        [(i, 2020, 82, 40 + i, "C") for i in range(5)]
        + [(i, 2021, 82, 40 + i, "C") for i in range(5)]
    )
    assert pred.measure(frame, ("points",), id_column="playerId") == []


def test_the_table_is_sorted_most_predictable_first():
    frame = pd.concat([_synthetic(0.9), _synthetic(0.1, seed=7)], ignore_index=True)
    frame["assists"] = frame["points"]
    results = pred.measure(frame, ("points", "assists"), id_column="playerId")
    table = pred.as_frame(results)
    assert list(table["spearman"]) == sorted(table["spearman"], reverse=True)
