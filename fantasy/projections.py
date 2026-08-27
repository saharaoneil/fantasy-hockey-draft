"""Project next season from history, regressing each stat by how sticky it is.

The projection is deliberately simple -- a weighted average of recent seasons,
regressed toward the league mean, then adjusted for age. There is no gradient
boosting here, and that is a choice rather than a shortcut: this family of
model is famously hard to beat on baseball projections, and a model you can
explain in a paragraph is a model whose failures you can diagnose.

What makes it more than a weighted average is that **the regression is driven
by the measured year-over-year correlation**, so the analysis in
`predictability` is load-bearing rather than decorative.

## How much to regress

For each stat, a player's rate is pulled toward the league rate by an amount
set by a regression constant `K`, expressed in games:

    projected_rate = (weighted_total + league_rate * K) / (weighted_games + K)

with `K = games_per_season * (1 - rho) / rho`.

That single expression does two jobs at once. A **sticky** stat gets a small
K and barely moves: hits at rho 0.88 gives K = 11 games, so an 82-game season
of evidence swamps it. A **noisy** stat gets a large K and is pulled hard:
games played at rho 0.35 gives K = 152, so a player's own durability record
counts for about a third of the projection and the league average for the rest.

And it handles sample size for free. The same K regresses a 15-game season far
more than an 82-game one, because the player's evidence is weighed in games
rather than assumed to be a full season.

## Age

The curve is measured from the data rather than assumed -- the average change
in points per game from each age to the next, across every pair of consecutive
40-game seasons. It comes out as the shape you would expect and is worth
stating in numbers: growth through 24, a plateau from 25 to 27, then decline
that accelerates from about -2% a year at 28 to -12% by the mid-30s.

The known caveat is survivor bias: a player who declines badly at 34 may not
play 40 games at 35, so he leaves the sample and the measured decline is
gentler than the truth. The curve is therefore optimistic about old players,
which is the safer direction for a *draft* -- it understates the case for
fading them rather than overstating it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

MIN_AGE_SAMPLE = 40
DEFAULT_SEASON_WEIGHTS = (5.0, 4.0, 3.0)


class ProjectionError(Exception):
    """The projection could not be made from what was supplied."""


@dataclass(frozen=True)
class ProjectionConfig:
    """Everything that shapes a projection. No hidden defaults."""

    season_weights: Sequence[float]
    reliability: Mapping[str, float]
    games_per_season: int
    min_games_for_age_curve: int
    age_smoothing_window: int

    def __post_init__(self) -> None:
        if not self.season_weights:
            raise ProjectionError("need at least one season weight")
        if any(w <= 0 for w in self.season_weights):
            raise ProjectionError("season weights must be positive")
        if not 1 <= self.games_per_season <= 82:
            raise ProjectionError("games_per_season must be in 1..82")
        for stat, rho in self.reliability.items():
            if not 0.0 < rho <= 1.0:
                raise ProjectionError(
                    f"reliability for {stat!r} must be in (0, 1], got {rho}"
                )


def regression_constant(rho: float, *, games_per_season: int) -> float:
    """Games of league-average evidence to blend in, given a stat's stickiness.

    `K = games * (1 - rho) / rho`. A perfectly sticky stat needs none; a noisy
    one needs more league average than the player's own record.
    """
    if not 0.0 < rho <= 1.0:
        raise ProjectionError(f"rho must be in (0, 1], got {rho}")
    return games_per_season * (1.0 - rho) / rho


# ---------------------------------------------------------------------------
# Age
# ---------------------------------------------------------------------------


def measure_age_curve(
    history: pd.DataFrame, *, config: ProjectionConfig, stat: str = "points"
) -> pd.Series:
    """Multiplier to apply going from each age to the next, measured from data.

    The delta method: pair every player's consecutive seasons, average the
    change in per-game rate within each starting-age bucket, and express it as
    a multiplier. Smoothed across neighbouring ages because single-year buckets
    are noisy and the underlying curve is not.
    """
    required = {"playerId", "season", "age", stat, "gamesPlayed"}
    missing = required - set(history.columns)
    if missing:
        raise ProjectionError(f"history is missing {sorted(missing)}")

    frame = history[history["gamesPlayed"] >= config.min_games_for_age_curve].copy()
    frame["rate"] = frame[stat] / frame["gamesPlayed"]

    current = frame[["playerId", "season", "age", "rate"]]
    following = current.copy()
    following["season"] = following["season"] - 1
    paired = current.merge(
        following, on=["playerId", "season"], suffixes=("_from", "_to")
    )
    if paired.empty:
        raise ProjectionError("no consecutive seasons to measure aging from")

    paired["bucket"] = paired["age_from"].round().astype(int)
    grouped = paired.groupby("bucket").agg(
        n=("rate_to", "size"),
        rate_from=("rate_from", "mean"),
        rate_to=("rate_to", "mean"),
    )
    grouped = grouped[grouped["n"] >= MIN_AGE_SAMPLE]
    if grouped.empty:
        raise ProjectionError("no age bucket has enough players to measure")

    multiplier = grouped["rate_to"] / grouped["rate_from"]
    return (
        multiplier.rolling(config.age_smoothing_window, center=True, min_periods=1)
        .mean()
        .rename("age_multiplier")
    )


def age_multiplier_for(age: float, curve: pd.Series) -> float:
    """Look up the multiplier for a player's most recent age.

    Ages outside the measured range are clamped to its ends rather than
    extrapolated: there is no evidence out there, and extrapolating a decline
    curve produces confident nonsense about 40-year-olds.
    """
    if curve.empty:
        raise ProjectionError("age curve is empty")
    if not np.isfinite(age):
        return 1.0
    bucket = int(round(age))
    return float(curve.loc[min(max(bucket, curve.index.min()), curve.index.max())])


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def _weighted_history(
    history: pd.DataFrame, stats: Sequence[str], *, config: ProjectionConfig,
    target_season: int,
) -> pd.DataFrame:
    """Collapse each player's recent seasons into weighted totals and games."""
    lookback = len(config.season_weights)
    weights = {
        target_season - offset - 1: weight
        for offset, weight in enumerate(config.season_weights)
    }
    recent = history[history["season"].isin(weights)].copy()
    if recent.empty:
        raise ProjectionError(
            f"no seasons in {sorted(weights)} to project {target_season} from"
        )
    recent["weight"] = recent["season"].map(weights)

    out = {}
    for stat in stats:
        if stat not in recent.columns:
            raise ProjectionError(f"{stat!r} is not in the history")
        out[stat] = recent["weight"] * recent[stat]
    weighted = pd.DataFrame(out)
    weighted["playerId"] = recent["playerId"].values
    weighted["weighted_games"] = (recent["weight"] * recent["gamesPlayed"]).values
    weighted["seasons_used"] = 1

    grouped = weighted.groupby("playerId").sum()
    grouped["latest_season"] = recent.groupby("playerId")["season"].max()
    return grouped


def project(
    history: pd.DataFrame,
    stats: Sequence[str],
    *,
    target_season: int,
    config: ProjectionConfig,
    age_curve: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Per-game rate projections for `target_season`, one row per player.

    Regression is toward the league rate **within a position**, because a
    defenceman regressed toward the all-skater scoring mean would be projected
    to score like a forward.
    """
    required = {"playerId", "season", "gamesPlayed", "position"}
    missing = required - set(history.columns)
    if missing:
        raise ProjectionError(f"history is missing {sorted(missing)}")

    unknown = [s for s in stats if s not in config.reliability]
    if unknown:
        raise ProjectionError(
            f"no measured reliability for {unknown}; run the predictability "
            "analysis first rather than guessing a value"
        )

    weighted = _weighted_history(history, stats, config=config,
                                 target_season=target_season)

    latest = (
        history.sort_values("season")
        .groupby("playerId")
        .last()[["position", "age"] if "age" in history.columns else ["position"]]
    )
    frame = weighted.join(latest, how="inner")

    league_rates = {}
    for stat in stats:
        totals = history.groupby("position")[stat].sum()
        games = history.groupby("position")["gamesPlayed"].sum()
        league_rates[stat] = (totals / games).to_dict()

    for stat in stats:
        constant = regression_constant(
            config.reliability[stat], games_per_season=config.games_per_season
        )
        league = frame["position"].map(league_rates[stat])
        frame[f"{stat}_rate"] = (frame[stat] + league * constant) / (
            frame["weighted_games"] + constant
        )

    if age_curve is not None and "age" in frame.columns:
        multiplier = frame["age"].map(
            lambda a: age_multiplier_for(a, age_curve)
        )
        frame["age_multiplier"] = multiplier
        for stat in stats:
            frame[f"{stat}_rate"] = frame[f"{stat}_rate"] * multiplier
    else:
        frame["age_multiplier"] = 1.0

    return frame.reset_index()


def project_games(
    history: pd.DataFrame,
    *,
    target_season: int,
    config: ProjectionConfig,
    reliability: float,
) -> pd.Series:
    """Projected games played, regressed hard because durability barely persists.

    Measured stickiness is about 0.35, so roughly two thirds of this projection
    is the league average. That is the honest answer, and it is also the useful
    one: it stops a draft board rewarding a player merely for having been
    healthy once.
    """
    weights = {
        target_season - offset - 1: weight
        for offset, weight in enumerate(config.season_weights)
    }
    recent = history[history["season"].isin(weights)].copy()
    if recent.empty:
        raise ProjectionError(f"no seasons to project games for {target_season}")
    recent["weight"] = recent["season"].map(weights)

    weighted = recent.groupby("playerId").apply(
        lambda g: np.average(g["gamesPlayed"], weights=g["weight"]),
        include_groups=False,
    )
    league_mean = float(recent["gamesPlayed"].mean())
    projected = league_mean + reliability * (weighted - league_mean)
    return projected.clip(0, config.games_per_season).rename("projected_games")
