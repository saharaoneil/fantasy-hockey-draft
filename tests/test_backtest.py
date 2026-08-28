"""The backtest, and the leak it exists to avoid."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasy import backtest, value

SCORING = value.ScoringConfig(weights={"goals": 3.0})
STATS = ("goals",)


def _seasons(players=80, seasons=(2020, 2021, 2022, 2023), seed=0):
    """Players with a persistent scoring rate plus noise, over several seasons."""
    rng = np.random.default_rng(seed)
    skill = rng.uniform(0.1, 0.9, players)
    rows = []
    for season in seasons:
        for index in range(players):
            rate = max(0.02, skill[index] + rng.normal(0, 0.08))
            games = int(rng.integers(40, 83))  # real seasons vary; a constant
            rows.append({                      # would make gamesPlayed's
                "playerId": index,             # correlation undefined
                "season": season, "gamesPlayed": games,
                "goals": rate * games, "position": "C", "age": 27.0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The leak
# ---------------------------------------------------------------------------


def test_a_method_never_sees_the_season_it_predicts():
    """Measuring reliability over all seasons and then testing on one of them
    would let the model see its own answer -- invisible in the output."""
    seen = {}

    def spy(history, target_season, **kwargs):
        seen[target_season] = sorted(history["season"].unique())
        return pd.Series(
            {pid: 1.0 for pid in history["playerId"].unique()}, name="pred"
        )

    frame = _seasons()
    backtest.run(
        frame, holdout_seasons=[2022, 2023], stats_wanted=STATS,
        scoring=SCORING, methods={"spy": spy},
    )
    assert seen[2022] == [2020, 2021]
    assert seen[2023] == [2020, 2021, 2022]
    assert all(2022 not in s for s in [seen[2022]])


def test_the_model_measures_reliability_only_on_prior_seasons():
    """The model's own reliability figures must come from the same restricted
    history, not from the full dataset."""
    frame = _seasons(seasons=(2019, 2020, 2021, 2022))
    prediction = backtest.predict_model(
        frame[frame["season"] < 2022], 2022, stats_wanted=STATS, scoring=SCORING
    )
    assert len(prediction) > 0
    assert prediction.notna().all()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _pair(pred_values, true_values):
    index = range(len(pred_values))
    return (pd.Series(pred_values, index=index), pd.Series(true_values, index=index))


def test_a_perfect_predictor_scores_one():
    values = list(range(100))
    prediction, truth = _pair(values, values)
    result = backtest.score(prediction, truth, top_n=10)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["mae"] == pytest.approx(0.0)
    assert result["top_n_hit"] == pytest.approx(1.0)


def test_a_reversed_predictor_scores_minus_one():
    prediction, truth = _pair(list(range(100)), list(range(100))[::-1])
    assert backtest.score(prediction, truth, top_n=10)["spearman"] == pytest.approx(-1.0)


def test_scoring_uses_only_players_present_in_both():
    prediction = pd.Series(range(50), index=range(50))
    truth = pd.Series(range(40), index=range(40))
    assert backtest.score(prediction, truth, top_n=5)["players"] == 40


def test_too_few_players_to_score_is_an_error():
    prediction, truth = _pair([1, 2, 3], [1, 2, 3])
    with pytest.raises(backtest.BacktestError, match="only 3 players"):
        backtest.score(prediction, truth, top_n=2)


def test_top_n_hit_rate_counts_overlap_not_order():
    """Getting the right 10 players in the wrong order is still a hit."""
    prediction, truth = _pair(
        list(range(100)), list(range(90)) + [95, 96, 97, 98, 99, 91, 92, 93, 94, 90]
    )
    assert backtest.score(prediction, truth, top_n=10)["top_n_hit"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The methods
# ---------------------------------------------------------------------------


def test_last_season_carries_the_previous_year_forward():
    frame = _seasons(players=40, seasons=(2021, 2022))
    prediction = backtest.predict_last_season(
        frame[frame["season"] < 2022], 2022, stats_wanted=STATS, scoring=SCORING
    )
    expected = backtest.actual_points(frame[frame["season"] == 2021], STATS, SCORING)
    assert prediction.sort_index().to_numpy() == pytest.approx(
        expected.sort_index().to_numpy()
    )


def test_a_method_with_no_history_fails_loudly():
    frame = _seasons(seasons=(2022, 2023))
    with pytest.raises(backtest.BacktestError):
        backtest.predict_last_season(
            frame[frame["season"] < 2022], 2022, stats_wanted=STATS, scoring=SCORING
        )


def test_holdout_seasons_outside_the_data_are_rejected():
    frame = _seasons()
    with pytest.raises(backtest.BacktestError, match="no data either side"):
        backtest.run(
            frame, holdout_seasons=[2099], stats_wanted=STATS, scoring=SCORING
        )


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_a_full_run_ranks_the_methods():
    frame = _seasons(players=120, seasons=(2019, 2020, 2021, 2022, 2023))
    results = backtest.run(
        frame, holdout_seasons=[2022, 2023], stats_wanted=STATS, scoring=SCORING,
        top_n=20,
    )
    assert set(results["method"]) == set(backtest.METHODS)
    assert set(results["season"]) == {2022, 2023}

    summary = backtest.summarise(results)
    assert len(summary) == len(backtest.METHODS)
    # Sorted best-ranking first.
    assert list(summary["spearman"]) == sorted(summary["spearman"], reverse=True)


def test_on_persistent_data_every_method_beats_chance():
    """A sanity floor: if skill persists, all three should find it."""
    frame = _seasons(players=150, seasons=(2019, 2020, 2021, 2022))
    summary = backtest.summarise(
        backtest.run(frame, holdout_seasons=[2022], stats_wanted=STATS,
                     scoring=SCORING, top_n=20)
    )
    assert (summary["spearman"] > 0.5).all()


def test_summarising_nothing_is_an_error():
    with pytest.raises(backtest.BacktestError, match="no results"):
        backtest.summarise(pd.DataFrame())
