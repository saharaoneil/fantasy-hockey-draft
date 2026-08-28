"""Valuation for categories leagues, where a weighted points total is wrong.

Most fantasy hockey is played in categories -- head-to-head or roto over G, A,
PIM, PPP, SOG, HIT, BLK and a few goalie columns -- and a points-league model
misprices it badly. You do not win a categories league by accumulating a total;
you win it column by column, so what matters is how far above the field a
player puts you in each one.

The standard answer is z-scores: standardise each category against the player
pool, sum the standardised values, and rank on that. It changes the board
substantially. A 300-hit grinder who scores 30 points is close to worthless in
a points league and a genuine asset in a hits column; a balanced scorer is
worth less than his point total suggests because he wins nothing outright.

Three things a naive z-score implementation gets wrong, all of them handled
here.

## The pool has to be the players who get drafted

Standardising against all 1,000-odd players in the league drags the mean down
to replacement level, which compresses everyone worth rostering into a narrow
band at the top and flattens the distinctions that matter. The pool should be
roughly the players who will actually be drafted -- but that set depends on the
values, which depend on the pool.

So it iterates: value everyone, take the top N, re-standardise against those,
repeat until the membership stops moving. Usually two or three passes.

## Rate categories are not counting categories

Save percentage is the trap. A goalie at .930 over 200 shots and one at .930
over 2,000 shots are not equally valuable -- the second drags your team rate
where the first barely moves it. Z-scoring the rate itself ignores volume
entirely and massively overrates backups.

A rate category is therefore scored on its *impact*: how far the player's rate
sits from the pool's, multiplied by the volume they do it over. That is what
actually moves a team's season-long number.

## Skaters and goalies are valued in different columns

A skater has no save percentage, so the two groups are standardised separately
-- and their totals are then not comparable, because eight categories sum to a
bigger number than four. Left unscaled, skaters would dominate the board purely
by having more columns.

`group_scale` fixes it by asking what a roster slot is worth: a group's share
of the standings divided by its share of the roster. With eight skater columns,
four goalie columns, ten skater slots and two goalie slots, a goalie slot
controls about 2.5x the standings impact of a skater slot -- which is the real
reason goalies go earlier in categories leagues than a points-league board
would suggest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class CategoryError(Exception):
    """A categories valuation that cannot be computed honestly."""


@dataclass(frozen=True)
class Category:
    """One scoring column.

    `rate_over` names the volume column for a rate stat -- shots faced for save
    percentage, projected games for goals-against average. It refers to a
    column on the *projected* board, not the historical one, since that is the
    volume the player is expected to do it over. Leaving it None marks a
    counting stat, where the total already carries its own volume.
    """

    stat: str
    higher_is_better: bool = True
    rate_over: Optional[str] = None

    @property
    def is_rate(self) -> bool:
        return self.rate_over is not None


# Yahoo's default hockey categories, which is what most leagues actually use.
DEFAULT_SKATER_CATEGORIES = (
    Category("goals"),
    Category("assists"),
    Category("ppPoints"),
    Category("shots"),
    Category("hits"),
    Category("blockedShots"),
    Category("penaltyMinutes"),
    Category("plusMinus"),
)

DEFAULT_GOALIE_CATEGORIES = (
    Category("wins"),
    Category("saves"),
    Category("shutouts"),
    Category("savePct", rate_over="shotsAgainst"),
    Category("goalsAgainstAverage", higher_is_better=False,
             rate_over="projected_games"),
)


@dataclass(frozen=True)
class CategoriesConfig:
    """Tunables for the z-score model. No hidden defaults."""

    pool_size: int
    pool_iterations: int
    min_pool: int
    bench_multiplier: float
    scale_by_roster_share: bool

    def __post_init__(self) -> None:
        if self.pool_size < 10:
            raise CategoryError("pool_size must be at least 10 to standardise against")
        if self.pool_iterations < 1:
            raise CategoryError("need at least one pass")
        if self.bench_multiplier < 1.0:
            raise CategoryError("bench_multiplier cannot shrink the pool below starters")


# bench_multiplier 1.5: real rosters carry bench spots, so the players worth
# standardising against are more than the starters alone. It also keeps the
# goalie pool statistically usable -- 24 starting goalies is a thin sample for
# a mean and a spread, 36 is workable.
# scale_by_roster_share defaults off. Summed z is already "SDs of edge across
# the columns this group scores in", and every column counts the same in the
# standings, so a goalie's 9 and a skater's 12 compare directly. Dividing by
# each group's share of the roster on top of that is an opinionated extra --
# see `group_scale` for the argument and for why it is not the default.
DEFAULT_CATEGORIES_CONFIG = CategoriesConfig(
    pool_size=180, pool_iterations=4, min_pool=20, bench_multiplier=1.5,
    scale_by_roster_share=False,
)


def _column_for(frame: pd.DataFrame, category: Category) -> pd.Series:
    """The season-long value of a category, before standardising.

    Counting stats are projected rate times projected games. Rate stats become
    an *impact*: distance from the pool's rate, multiplied by the volume the
    player does it over -- which is what moves a team's season number, and what
    a plain z-score of the rate would throw away.
    """
    rate_column = f"{category.stat}_rate"
    if rate_column not in frame.columns:
        raise CategoryError(
            f"no {rate_column!r} column; project {category.stat!r} first"
        )

    if not category.is_rate:
        if "projected_games" not in frame.columns:
            raise CategoryError("no 'projected_games' column to scale counting stats")
        values = frame[rate_column] * frame["projected_games"]
    else:
        volume_column = category.rate_over
        if volume_column not in frame.columns:
            raise CategoryError(
                f"rate category {category.stat!r} needs its volume column "
                f"{volume_column!r}, so a backup's .930 is not priced like a "
                "starter's"
            )
        volume = frame[volume_column]
        weighted_mean = np.average(
            frame[rate_column].to_numpy(),
            weights=volume.clip(lower=0).to_numpy() + 1e-9,
        )
        values = (frame[rate_column] - weighted_mean) * volume

    return values if category.higher_is_better else -values


def _standardise(values: pd.Series, pool_mask: pd.Series) -> pd.Series:
    """Z-score everyone against the pool's mean and spread."""
    pool = values[pool_mask]
    spread = float(pool.std(ddof=0))
    if not np.isfinite(spread) or spread == 0.0:
        # Every player identical in this column: it separates nobody, so it
        # contributes nothing rather than dividing by zero.
        return pd.Series(0.0, index=values.index)
    return (values - float(pool.mean())) / spread


def category_values(
    frame: pd.DataFrame,
    categories: Sequence[Category],
    *,
    pool_size: Optional[int] = None,
    config: CategoriesConfig = DEFAULT_CATEGORIES_CONFIG,
) -> pd.DataFrame:
    """Per-category z-scores and their total, for one position group.

    The pool is found by iteration: value everyone against a provisional pool,
    take the top `pool_size`, re-standardise, repeat. Standardising against all
    players instead would drag the mean to replacement level and squash the
    players you are actually choosing between into a narrow band.

    **`pool_size` must be set per group.** A league carries about 940 skaters
    and 98 goalies, so one shared figure standardises skaters against their top
    tier and goalies against every backup in the league -- which drags the
    goalie mean down and inflates every starter's z-score enormously. Measured,
    that alone put goalies in the top eight places on the board. The pool for a
    group is the number of that group who actually get drafted.
    """
    if not categories:
        raise CategoryError("no categories to value")
    if len(frame) < config.min_pool:
        raise CategoryError(
            f"{len(frame)} players is too few to standardise against "
            f"(need {config.min_pool})"
        )

    raw = pd.DataFrame(
        {c.stat: _column_for(frame, c) for c in categories}, index=frame.index
    )

    size = config.pool_size if pool_size is None else pool_size
    if size < config.min_pool:
        raise CategoryError(
            f"pool of {size} is below the {config.min_pool} needed to "
            "standardise against"
        )

    pool_mask = pd.Series(True, index=frame.index)
    previous: Optional[set] = None

    for _ in range(config.pool_iterations):
        scored = pd.DataFrame(
            {c.stat: _standardise(raw[c.stat], pool_mask) for c in categories}
        )
        total = scored.sum(axis=1)
        top = set(total.nlargest(min(size, len(total))).index)
        if top == previous:
            break
        previous = top
        pool_mask = pd.Series(frame.index.isin(list(top)), index=frame.index)

    scored = pd.DataFrame(
        {c.stat: _standardise(raw[c.stat], pool_mask) for c in categories}
    )
    scored.columns = [f"z_{name}" for name in scored.columns]
    scored["category_value"] = scored.sum(axis=1)
    scored["in_pool"] = pool_mask
    return scored


def group_scale(
    *, categories_in_group: int, total_categories: int,
    slots_in_group: int, total_slots: int,
) -> float:
    """How much one roster slot in this group is worth, relative to the roster.

    A group's share of the standings divided by its share of the roster. Goalies
    typically control a third of the categories from a sixth of the roster, so
    a goalie slot carries roughly 2.5x the standings impact of a skater slot --
    which is why goalies go earlier in categories leagues than a points-league
    board would ever suggest.

    Without this, summed z-scores are not comparable across groups at all: eight
    categories simply add up to more than four, and skaters would top the board
    for having more columns rather than more value.
    """
    for name, number in (
        ("categories_in_group", categories_in_group),
        ("total_categories", total_categories),
        ("slots_in_group", slots_in_group),
        ("total_slots", total_slots),
    ):
        if number <= 0:
            raise CategoryError(f"{name} must be positive, got {number}")
    if categories_in_group > total_categories or slots_in_group > total_slots:
        raise CategoryError("a group cannot exceed the whole roster or category set")

    standings_share = categories_in_group / total_categories
    roster_share = slots_in_group / total_slots
    return standings_share / roster_share


def value_groups(
    groups: Mapping[str, pd.DataFrame],
    categories: Mapping[str, Sequence[Category]],
    drafted: Mapping[str, int],
    *,
    config: CategoriesConfig = DEFAULT_CATEGORIES_CONFIG,
) -> pd.DataFrame:
    """Value several position groups and put them on one comparable scale.

    `groups` maps a group name to its projected players, `categories` to the
    columns it is scored in, and `drafted` to how many of that group get taken
    across the whole league -- teams times starting slots.

    That count does two jobs: it sizes each group's standardisation pool, and
    it sets the group's share of the roster for the cross-group scaling. Both
    want the same number, which is the number of players at that group anyone
    is actually choosing between.
    """
    missing = set(groups) - set(categories)
    if missing:
        raise CategoryError(f"no categories defined for groups {sorted(missing)}")
    missing_slots = set(groups) - set(drafted)
    if missing_slots:
        raise CategoryError(f"no drafted count given for groups {sorted(missing_slots)}")

    total_categories = sum(len(categories[name]) for name in groups)
    total_slots = sum(drafted[name] for name in groups)

    valued = []
    for name, frame in groups.items():
        scored = category_values(
            frame, categories[name],
            pool_size=int(round(drafted[name] * config.bench_multiplier)),
            config=config,
        )
        scale = (
            group_scale(
                categories_in_group=len(categories[name]),
                total_categories=total_categories,
                slots_in_group=drafted[name],
                total_slots=total_slots,
            )
            if config.scale_by_roster_share
            else 1.0
        )
        out = frame.copy()
        for column in scored.columns:
            out[column] = scored[column]
        out["group_scale"] = scale
        out["category_value"] = scored["category_value"] * scale
        out["value_group"] = name
        valued.append(out)

    return pd.concat(valued, ignore_index=True)


def category_contributions(
    board: pd.DataFrame, categories: Sequence[Category], *, limit: int = 10,
    rank_by: str = "vorp",
) -> pd.DataFrame:
    """Which columns a player actually wins you, for the top of the board.

    This is the thing z-scores do that a points total cannot: tell a specialist
    apart from a generalist. Two players on the same total can be completely
    different picks -- one wins you three columns outright, the other is
    slightly above average in eight.

    Ranked by draft value rather than raw category value by default. Sorting on
    the raw total puts every goalie at the top, because goalies carry a large
    per-slot weight and a correspondingly high replacement level; it is the
    difference between them that decides a pick.
    """
    columns = [f"z_{c.stat}" for c in categories if f"z_{c.stat}" in board.columns]
    if not columns:
        raise CategoryError("no z-score columns on the board")
    if rank_by not in board.columns:
        raise CategoryError(f"{rank_by!r} is not on the board")
    top = board.nlargest(limit, rank_by)
    return top[["name", "position", rank_by] + columns].round(2)
