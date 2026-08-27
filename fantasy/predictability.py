"""How much of a stat carries from one season to the next.

This is the question that should decide a draft board, and it is asked far
less often than "who scored the most points last year". A stat you cannot
predict is one you should not pay for, however impressive last season looked.

Four methodological choices, each of which changes the answer:

**Per-game rates, not totals.** Totals confound production with availability,
and the 2019-20 and 2020-21 seasons were 68 and 56 games, so raw totals across
that boundary are not comparable at all. Games played is separately
interesting -- durability is a real fantasy skill -- so it is measured on its
own rather than baked into everything else.

**Spearman, not Pearson.** Drafting is an ordering problem: what matters is
whether last season's ranking predicts next season's, not whether the
relationship is linear. Rank correlation also survives the outliers that a
handful of elite scorers otherwise dominate.

**A minimum-games threshold in both seasons.** Without one, a player with four
games of fluke shooting sits beside an 82-game regular and the noise swamps
the signal.

**Pooled across every consecutive pair.** More sample, and it averages over
era effects -- scoring has drifted upward over the last decade, and a single
pair of seasons would read that drift as instability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Rate stats are divided by games played; these are not.
NOT_PER_GAME = frozenset({"gamesPlayed", "toiPerGameMinutes", "savePct",
                          "goalsAgainstAverage", "shootingPct", "pointsPerGame",
                          "faceoffWinPct"})

SKATER_STATS = (
    "points", "goals", "assists", "ppPoints", "shots", "hits", "blockedShots",
    "penaltyMinutes", "plusMinus", "giveaways", "toiPerGameMinutes",
    "shootingPct", "gamesPlayed",
)
GOALIE_STATS = ("wins", "savePct", "goalsAgainstAverage", "shutouts", "saves",
                "gamesPlayed")


class PredictabilityError(Exception):
    """The measurement could not be made honestly."""


@dataclass(frozen=True)
class StatResult:
    """Year-over-year persistence of one stat, for one group of players."""

    stat: str
    group: str
    spearman: float
    pairs: int
    per_game: bool

    @property
    def verdict(self) -> str:
        """Plain-language reading, so the number is not left to interpret alone."""
        if self.spearman >= 0.70:
            return "very sticky"
        if self.spearman >= 0.50:
            return "sticky"
        if self.spearman >= 0.30:
            return "weak"
        return "close to noise"


def _rate_frame(
    frame: pd.DataFrame, stats_wanted: Sequence[str], *, id_column: str
) -> pd.DataFrame:
    """Reduce a season frame to per-game rates plus the identifiers."""
    out = frame[[id_column, "season", "gamesPlayed"]].copy()
    for stat in stats_wanted:
        if stat not in frame.columns:
            raise PredictabilityError(f"{stat!r} is not in the data")
        if stat in NOT_PER_GAME:
            out[stat] = frame[stat]
        else:
            out[stat] = frame[stat] / frame["gamesPlayed"].replace(0, np.nan)
    if "position" in frame.columns:
        out["position"] = frame["position"]
    return out


def consecutive_pairs(
    frame: pd.DataFrame, *, id_column: str, min_games: int
) -> pd.DataFrame:
    """Join each player's season N to their season N+1.

    Only genuinely consecutive seasons are paired. A player who misses a full
    year and returns is not evidence about year-over-year persistence -- it is
    evidence about something else, and quietly folding it in would understate
    every stat's stickiness.
    """
    if min_games < 1:
        raise ValueError("min_games must be at least 1")

    eligible = frame[frame["gamesPlayed"] >= min_games].copy()
    later = eligible.copy()
    later["season"] = later["season"] - 1

    paired = eligible.merge(
        later, on=[id_column, "season"], suffixes=("_now", "_next"), how="inner"
    )
    if paired.empty:
        raise PredictabilityError(
            f"no consecutive season pairs at min_games={min_games}"
        )
    return paired


def measure(
    frame: pd.DataFrame,
    stats_wanted: Sequence[str],
    *,
    id_column: str,
    min_games: int = 20,
    by_position: bool = False,
) -> List[StatResult]:
    """Spearman correlation between season N and N+1, per stat.

    Returns one result per stat (or per stat and position). Pairs with a
    missing value on either side are dropped for that stat only, so one stat's
    gaps cannot shrink another's sample.
    """
    rates = _rate_frame(frame, stats_wanted, id_column=id_column)
    paired = consecutive_pairs(rates, id_column=id_column, min_games=min_games)

    groups: List[Tuple[str, pd.DataFrame]] = [("all", paired)]
    if by_position:
        if "position_now" not in paired.columns:
            raise PredictabilityError("no position column to group by")
        groups = [
            (str(name), subset)
            for name, subset in paired.groupby("position_now")
        ]

    results: List[StatResult] = []
    for group_name, subset in groups:
        for stat in stats_wanted:
            now = subset[f"{stat}_now"]
            nxt = subset[f"{stat}_next"]
            usable = now.notna() & nxt.notna()
            if usable.sum() < 30:
                continue
            rho = stats.spearmanr(now[usable], nxt[usable]).statistic
            results.append(
                StatResult(
                    stat=stat,
                    group=group_name,
                    spearman=float(rho),
                    pairs=int(usable.sum()),
                    per_game=stat not in NOT_PER_GAME,
                )
            )
    return results


def as_frame(results: Iterable[StatResult]) -> pd.DataFrame:
    """Results as a table, sorted most predictable first."""
    frame = pd.DataFrame(
        [
            {
                "stat": r.stat,
                "group": r.group,
                "spearman": round(r.spearman, 3),
                "pairs": r.pairs,
                "per_game": r.per_game,
                "verdict": r.verdict,
            }
            for r in results
        ]
    )
    if frame.empty:
        raise PredictabilityError("no results to tabulate")
    return frame.sort_values(["group", "spearman"], ascending=[True, False])
