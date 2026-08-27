"""Turn projected stats into draft value, which is not the same as projected points.

The central idea, and the reason a points ranking is the wrong draft board:
**you are not choosing between players, you are choosing between upgrades over
the player you could have had instead.** Every team must fill a centre slot, so
the value of a centre is what he adds over the worst centre who will still be
starting somewhere at the end of the draft -- his replacement level.

That makes value league-specific in a way raw projections are not. A 12-team
league that starts four defencemen drains the defence pool far deeper than an
8-team league starting two, so the replacement-level defenceman is much worse,
and every defenceman is correspondingly more valuable. The same projections
produce a different board for a different league.

## Replacement level moves during the draft

Replacement level is usually computed once from league settings, which is fine
before the draft and wrong during it. If a run empties the centres, the
replacement centre gets worse and every remaining centre gains value; if
everyone ignores goalies, goalie value falls. `replacement_levels` therefore
takes the pool of *available* players, so recomputing it after each pick
re-ranks the board on what is actually left.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import pandas as pd

SKATER_POSITIONS = ("C", "W", "D")
GOALIE_POSITION = "G"


class ValueError_(Exception):
    """The value model was given something it cannot price."""


@dataclass(frozen=True)
class ScoringConfig:
    """Points awarded per unit of each stat.

    Defaults are a common points-league setup, but the whole point is that this
    is yours to change -- a league that counts hits and blocks produces a very
    different board from one that counts only scoring.
    """

    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError_("scoring config has no categories")


@dataclass(frozen=True)
class LeagueConfig:
    """League shape, which is what sets replacement level."""

    teams: int
    starters: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.teams < 2:
            raise ValueError_("a league needs at least two teams")
        if not self.starters:
            raise ValueError_("no starting slots defined")
        for position, count in self.starters.items():
            if count < 1:
                raise ValueError_(f"{position} has {count} starting slots")

    def drafted_at(self, position: str) -> int:
        """How many of a position will be starting across the whole league."""
        if position not in self.starters:
            raise ValueError_(f"{position!r} is not a slot in this league")
        return self.teams * self.starters[position]


DEFAULT_SCORING = ScoringConfig(
    weights={
        "goals": 3.0,
        "assists": 2.0,
        "shots": 0.2,
        "hits": 0.2,
        "blockedShots": 0.2,
        "ppPoints": 0.5,
        "penaltyMinutes": 0.1,
        # Goalies
        "wins": 4.0,
        "saves": 0.2,
        "shutouts": 3.0,
    }
)

DEFAULT_LEAGUE = LeagueConfig(
    teams=12, starters={"C": 2, "W": 4, "D": 4, "G": 2}
)


def fantasy_points(
    projections: pd.DataFrame, scoring: ScoringConfig, *, games_column: str
) -> pd.Series:
    """Season fantasy points: per-game rates scored, then multiplied by games.

    Rates and availability are kept separate right up to this line, because
    they are predictable to very different degrees -- roughly 0.8 for scoring
    rate against 0.35 for games played. Multiplying them earlier would hide
    that a durable-but-mediocre player and a fragile star arrive at the same
    total for entirely different reasons.
    """
    if games_column not in projections.columns:
        raise ValueError_(f"no {games_column!r} column to scale by")

    priced = [stat for stat in scoring.weights if f"{stat}_rate" in projections.columns]
    if not priced:
        raise ValueError_(
            "none of the scoring categories have a projected rate; expected "
            f"columns like {next(iter(scoring.weights))}_rate"
        )

    # Score one position group at a time. Concatenating skaters and goalies
    # first leaves each with NaN in the other's columns, and a NaN anywhere in
    # the sum silently poisons every total -- which is exactly the bug this
    # guard was written for. Raising here is much cheaper than a board of NaN
    # that still sorts and still prints.
    incomplete = [stat for stat in priced if projections[f"{stat}_rate"].isna().any()]
    if incomplete:
        raise ValueError_(
            f"missing projected rates for {incomplete}. Score skaters and "
            "goalies separately -- a skater has no wins column, and summing "
            "across both groups produces NaN for everyone."
        )

    per_game = sum(
        projections[f"{stat}_rate"] * scoring.weights[stat] for stat in priced
    )
    return (per_game * projections[games_column]).rename("projected_points")


def replacement_levels(
    board: pd.DataFrame, league: LeagueConfig, *, points_column: str = "projected_points"
) -> Dict[str, float]:
    """The points total of the last starter at each position, given who is available.

    Called with the full pool before a draft, this is the usual static
    replacement level. Called with the *undrafted* pool mid-draft, it is what
    re-ranks the board as positions get thin.

    A position with fewer available players than starting slots has effectively
    run dry; its replacement level is the worst player left, which correctly
    makes everyone remaining look valuable.
    """
    levels: Dict[str, float] = {}
    for position in league.starters:
        pool = board[board["position"] == position][points_column]
        if pool.empty:
            raise ValueError_(f"no players available at {position!r}")
        ranked = pool.sort_values(ascending=False).to_numpy()
        index = min(league.drafted_at(position), len(ranked)) - 1
        levels[position] = float(ranked[index])
    return levels


def value_over_replacement(
    board: pd.DataFrame,
    league: LeagueConfig,
    *,
    points_column: str = "projected_points",
) -> pd.DataFrame:
    """Add `replacement` and `vorp` columns, sorted by draft value.

    VORP, not points, is the draft order. A defenceman projected for fewer
    points than a winger can still be the better pick when defence is scarcer,
    and that inversion is the entire reason to compute this.
    """
    unknown = set(board["position"]) - set(league.starters)
    if unknown:
        raise ValueError_(
            f"positions with no slot in this league: {sorted(unknown)}"
        )

    levels = replacement_levels(board, league, points_column=points_column)
    out = board.copy()
    out["replacement"] = out["position"].map(levels)
    out["vorp"] = out[points_column] - out["replacement"]
    return out.sort_values("vorp", ascending=False).reset_index(drop=True)


def positional_scarcity(board: pd.DataFrame, league: LeagueConfig) -> pd.DataFrame:
    """How steeply value falls off at each position.

    The drop from the best starter to the last one is what "scarcity" actually
    means for a draft. A position where the top and bottom starters are close
    can be waited on; a position with a cliff cannot, whatever the raw totals
    look like.
    """
    rows = []
    for position in league.starters:
        pool = board[board["position"] == position]["projected_points"]
        ranked = pool.sort_values(ascending=False).to_numpy()
        slots = min(league.drafted_at(position), len(ranked))
        rows.append(
            {
                "position": position,
                "starters_drafted": slots,
                "best": round(float(ranked[0]), 1),
                "last_starter": round(float(ranked[slots - 1]), 1),
                "dropoff": round(float(ranked[0] - ranked[slots - 1]), 1),
                "available": len(ranked),
            }
        )
    return pd.DataFrame(rows).sort_values("dropoff", ascending=False)
