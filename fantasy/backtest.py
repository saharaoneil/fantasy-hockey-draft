"""Does any of this actually beat guessing from last season's totals?

The project asserts that regressing each stat by its measured stickiness
produces better projections. That is a testable claim and it had not been
tested, so this tests it -- against the baseline it is implicitly competing
with, which is what most people actually draft on.

## Three methods, same target

Every method predicts the same thing: a player's fantasy points in a season it
was not allowed to see.

* `last_season` -- what they scored last year. The naive baseline, and the one
  the whole project argues with.
* `three_year` -- a weighted average of the last three seasons, no regression.
  Isolates the value of *regressing* from the value of merely using more
  history.
* `model` -- the full projection: weighted history, regressed by measured
  stickiness, adjusted by the measured age curve.

The gap between `three_year` and `model` is the one that matters. If it is
zero, the regression machinery earns nothing and the extra complexity is not
worth carrying.

## The leak this avoids

The reliability figures and the age curve are themselves measured from data.
Measuring them over all twelve seasons and then testing on one of those
seasons would let the model see its own answer -- a subtle leak that inflates
every number and is invisible in the output.

So for a holdout season T, both are re-measured using seasons **before T
only**. That makes the backtest slower and the numbers lower, which is the
point of running it.

## Why the headline is rank correlation

A draft is an ordering. Being wrong about everyone's absolute totals by the
same factor costs nothing, while getting the order wrong costs picks. Absolute
error is reported alongside, and it is worth reading with the known caveat that
the model's projected games are compressed, which penalises it on totals
without affecting the ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import predictability as pred
from . import projections, value


class BacktestError(Exception):
    """The backtest could not be run honestly."""


@dataclass(frozen=True)
class SeasonResult:
    """One method's performance on one holdout season."""

    season: int
    method: str
    spearman: float
    mae: float
    top_n_hit_rate: float
    players: int

    def as_row(self) -> Dict[str, object]:
        return {
            "season": self.season,
            "method": self.method,
            "spearman": round(self.spearman, 3),
            "mae": round(self.mae, 1),
            "top_n_hit": round(self.top_n_hit_rate, 3),
            "players": self.players,
        }


def actual_points(
    frame: pd.DataFrame,
    stats_wanted: Sequence[str],
    scoring: value.ScoringConfig,
) -> pd.Series:
    """What a player actually scored, under the league's scoring settings."""
    scored = frame.copy()
    for stat in stats_wanted:
        if stat not in scored.columns:
            raise BacktestError(f"{stat!r} is not in the season data")
        scored[f"{stat}_rate"] = scored[stat] / scored["gamesPlayed"].replace(0, np.nan)
    scored["projected_games"] = scored["gamesPlayed"]
    points = value.fantasy_points(
        scored.dropna(subset=[f"{s}_rate" for s in stats_wanted]),
        scoring, games_column="projected_games",
    )
    return points.rename("actual_points")


# ---------------------------------------------------------------------------
# The methods being compared
# ---------------------------------------------------------------------------


def predict_last_season(
    history: pd.DataFrame, target_season: int, *, stats_wanted, scoring, **_
) -> pd.Series:
    """The naive baseline: last season's total, carried forward unchanged."""
    previous = history[history["season"] == target_season - 1]
    if previous.empty:
        raise BacktestError(f"no {target_season - 1} season to carry forward")
    points = actual_points(previous, stats_wanted, scoring)
    return points.groupby(previous.loc[points.index, "playerId"]).first()


def predict_three_year(
    history: pd.DataFrame, target_season: int, *, stats_wanted, scoring,
    weights=(5.0, 4.0, 3.0), **_
) -> pd.Series:
    """Weighted three-season average with no regression toward the mean.

    The control for the model: same history, same weights, no shrinkage. Any
    gain the model shows over this is the regression earning its place.
    """
    lookup = {target_season - i - 1: w for i, w in enumerate(weights)}
    recent = history[history["season"].isin(lookup)].copy()
    if recent.empty:
        raise BacktestError(f"no history to average for {target_season}")

    recent["weight"] = recent["season"].map(lookup)
    points = actual_points(recent, stats_wanted, scoring)
    recent = recent.loc[points.index]
    recent["points"] = points

    grouped = recent.groupby("playerId")
    return grouped.apply(
        lambda g: np.average(g["points"], weights=g["weight"]), include_groups=False
    )


def predict_model(
    history: pd.DataFrame, target_season: int, *, stats_wanted, scoring,
    use_age_curve: bool = True, **_
) -> pd.Series:
    """The full projection, with reliability and aging measured before T only."""
    reliability = {
        r.stat: max(r.spearman, 0.05)
        for r in pred.measure(
            history, tuple(stats_wanted) + ("gamesPlayed",),
            id_column="playerId", min_games=20,
        )
    }
    missing = [s for s in stats_wanted if s not in reliability]
    if missing:
        raise BacktestError(
            f"could not measure reliability for {missing} from seasons before "
            f"{target_season}"
        )

    config = projections.ProjectionConfig(
        season_weights=projections.DEFAULT_SEASON_WEIGHTS,
        reliability=reliability,
        games_per_season=82,
        min_games_for_age_curve=40,
        age_smoothing_window=3,
    )

    curve = None
    if use_age_curve and "age" in history.columns:
        try:
            curve = projections.measure_age_curve(history, config=config)
        except projections.ProjectionError:
            curve = None

    board = projections.project(
        history, stats_wanted, target_season=target_season,
        config=config, age_curve=curve,
    )
    board["projected_games"] = projections.project_games(
        history, target_season=target_season, config=config,
        reliability=reliability["gamesPlayed"],
    ).reindex(board["playerId"]).to_numpy()

    board["prediction"] = value.fantasy_points(
        board, scoring, games_column="projected_games"
    )
    return board.set_index("playerId")["prediction"]


METHODS: Dict[str, Callable[..., pd.Series]] = {
    "last_season": predict_last_season,
    "three_year": predict_three_year,
    "model": predict_model,
}


# ---------------------------------------------------------------------------
# Scoring the predictions
# ---------------------------------------------------------------------------


def score(
    prediction: pd.Series, truth: pd.Series, *, top_n: int
) -> Dict[str, float]:
    """Rank correlation, absolute error, and how many of the top N were right."""
    joined = pd.concat(
        [prediction.rename("pred"), truth.rename("true")], axis=1, join="inner"
    ).dropna()
    if len(joined) < 30:
        raise BacktestError(f"only {len(joined)} players to score against")

    rho = stats.spearmanr(joined["pred"], joined["true"]).statistic
    mae = float((joined["pred"] - joined["true"]).abs().mean())

    predicted_top = set(joined.nlargest(top_n, "pred").index)
    actual_top = set(joined.nlargest(top_n, "true").index)
    hit = len(predicted_top & actual_top) / top_n

    return {"spearman": float(rho), "mae": mae, "top_n_hit": hit,
            "players": len(joined)}


def run(
    seasons: pd.DataFrame,
    *,
    holdout_seasons: Sequence[int],
    stats_wanted: Sequence[str],
    scoring: value.ScoringConfig,
    min_games: int = 20,
    top_n: int = 100,
    methods: Optional[Mapping[str, Callable]] = None,
) -> pd.DataFrame:
    """Backtest every method across every holdout season.

    For each season T, methods see only seasons strictly before T -- including
    when measuring reliability and the age curve, which is where the leak would
    otherwise be.

    Evaluated on players who actually played in T. A player who was injured all
    year is not evidence about projection quality; he is evidence about
    availability, which the games model handles separately.
    """
    methods = dict(methods or METHODS)
    rows: List[SeasonResult] = []

    for target in sorted(holdout_seasons):
        history = seasons[seasons["season"] < target]
        future = seasons[seasons["season"] == target]
        if history.empty or future.empty:
            raise BacktestError(f"no data either side of holdout season {target}")

        played = future[future["gamesPlayed"] >= min_games]
        truth = actual_points(played, stats_wanted, scoring)
        truth.index = played.loc[truth.index, "playerId"]

        for name, method in methods.items():
            try:
                prediction = method(
                    history, target, stats_wanted=stats_wanted, scoring=scoring
                )
            except (BacktestError, projections.ProjectionError) as exc:
                raise BacktestError(f"{name} failed on {target}: {exc}") from exc

            measured = score(prediction, truth, top_n=top_n)
            rows.append(
                SeasonResult(
                    season=target, method=name,
                    spearman=measured["spearman"], mae=measured["mae"],
                    top_n_hit_rate=measured["top_n_hit"],
                    players=int(measured["players"]),
                )
            )

    return pd.DataFrame([r.as_row() for r in rows])


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Average each method across holdout seasons, best ranking first."""
    if results.empty:
        raise BacktestError("no results to summarise")
    summary = results.groupby("method").agg(
        seasons=("season", "nunique"),
        spearman=("spearman", "mean"),
        mae=("mae", "mean"),
        top_n_hit=("top_n_hit", "mean"),
    )
    return summary.sort_values("spearman", ascending=False).round(3)
