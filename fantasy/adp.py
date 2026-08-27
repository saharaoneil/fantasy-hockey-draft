"""Where players actually get drafted, and where that disagrees with their value.

A ranking on its own tells you who is good. The *gap* between your ranking and
where the room is drafting tells you who to actually target -- and that gap is
the only part of this project anyone else can use directly, because it says
"take him two rounds later than everyone thinks" rather than "he is good."

## There is no free NHL ADP feed, and this does not pretend otherwise

Checked at build time: Sleeper's public player API carries NHL rosters but
leaves `search_rank` null for every hockey player (it is populated for NFL
only), and FantasyPros' `robots.txt` disallows `/api/`, `/json/` and `/ajax/`,
which is where their numbers live. Yahoo needs OAuth. So there is nothing to
pull, and inventing a number here would be worse than having none.

Two honest routes instead:

**Bring your own.** `load_adp_csv` reads a two-column CSV of name and ADP.
Every major platform shows its own ADP, and it is your league's ADP that
matters anyway -- a Yahoo points league drafts nothing like an ESPN roto one,
so a single "consensus" number would have been the wrong one for most people.

**Or use the proxy.** `last_season_rank` ranks players by what they actually
scored last season under your scoring settings. That is not ADP and is never
labelled as such, but it is a fair model of how the room behaves: drafting off
last year's totals is precisely the habit this project exists to argue with.
Measuring against it answers the question that matters -- who does a
predictability-aware model like more than last season's box score does.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


class ADPError(Exception):
    """ADP data that cannot be trusted or matched."""


def normalize_name(name: str) -> str:
    """Fold a player name to something two sources can agree on.

    ADP exports and the NHL API disagree constantly about accents, periods and
    suffixes -- `T.J. Oshie`, `TJ Oshie`, `Tim Stützle`, `Tim Stutzle`. This
    strips the differences that are never meaningful and keeps the rest.
    """
    folded = unicodedata.normalize("NFKD", str(name))
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.lower().replace("-", " ")
    folded = re.sub(r"[^a-z\s]", "", folded)
    parts = [p for p in folded.split() if p and p not in SUFFIXES]
    return " ".join(parts)


@dataclass(frozen=True)
class ADPMatch:
    """The result of joining ADP to the board, including what failed to join."""

    frame: pd.DataFrame
    matched: int
    unmatched_names: Tuple[str, ...]

    @property
    def match_rate(self) -> float:
        total = self.matched + len(self.unmatched_names)
        return self.matched / total if total else 0.0


def load_adp_csv(
    path: Path, *, name_column: str = "name", adp_column: str = "adp"
) -> pd.DataFrame:
    """Read a name/ADP CSV exported from whichever platform your league uses."""
    frame = pd.read_csv(path)
    missing = {name_column, adp_column} - set(frame.columns)
    if missing:
        raise ADPError(
            f"{path} has no {sorted(missing)} column; found {list(frame.columns)}"
        )

    out = frame[[name_column, adp_column]].rename(
        columns={name_column: "name", adp_column: "adp"}
    )
    out["adp"] = pd.to_numeric(out["adp"], errors="coerce")
    if out["adp"].isna().all():
        raise ADPError(f"no numeric values in the {adp_column!r} column of {path}")

    out = out.dropna(subset=["adp"])
    out["match_key"] = out["name"].map(normalize_name)
    return out.sort_values("adp").reset_index(drop=True)


def last_season_rank(
    board: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    points_column: str,
    slots_per_position: Mapping[str, int],
) -> pd.DataFrame:
    """A stand-in for consensus: last season's finish, interleaved by position.

    Deliberately never called ADP where it is used. It models the drafting
    habit rather than the draft -- but that habit is what this project argues
    with, so the gap against it is interesting even once real ADP exists.

    **Ranking on raw points alone does not work**, and the first version of this
    did. Defencemen score fewer points than forwards, so a flat points ranking
    buries every one of them, and the value gap then reports the entire defence
    position as underdrafted. That is a positional artifact of the proxy, not a
    finding about any player.

    So positions are ranked separately and then interleaved in proportion to
    how many of each get drafted -- which is what a real draft board does,
    because the room does know defencemen are scarce.
    """
    if points_column not in actuals.columns:
        raise ADPError(f"{points_column!r} is not in the actuals")
    if not slots_per_position:
        raise ADPError("need the league's slots per position to interleave")

    scored = actuals[["playerId", points_column]].dropna().merge(
        board[["playerId", "position"]].drop_duplicates(), on="playerId", how="inner"
    )
    if scored.empty:
        raise ADPError("no players in common between the board and the actuals")

    total_slots = sum(slots_per_position.values())
    queues: Dict[str, List[int]] = {}
    for position, group in scored.groupby("position"):
        queues[str(position)] = list(
            group.sort_values(points_column, ascending=False)["playerId"]
        )

    # Draw from each position's queue at the rate that position is drafted.
    progress = {position: 0.0 for position in queues}
    order: List[int] = []
    while any(queues[p] for p in queues):
        for position in queues:
            share = slots_per_position.get(position, 0) / total_slots
            progress[position] += share
            while progress[position] >= 1.0 and queues[position]:
                order.append(queues[position].pop(0))
                progress[position] -= 1.0
        if all(not queues[p] for p in queues):
            break
        if not any(
            slots_per_position.get(p, 0) for p in queues if queues[p]
        ):  # positions with no slots would loop forever
            for position in queues:
                order.extend(queues[position])
                queues[position] = []

    return pd.DataFrame({"playerId": order, "adp": range(1, len(order) + 1)})


def join_adp(board: pd.DataFrame, adp: pd.DataFrame) -> ADPMatch:
    """Attach ADP to the board by name, reporting whatever failed to match.

    Unmatched names are returned rather than dropped. A silent 60% match rate
    would leave most of the board with no ADP and quietly turn the value-gap
    column into a ranking of who happened to join.
    """
    if "playerId" in adp.columns:
        merged = board.merge(adp[["playerId", "adp"]], on="playerId", how="left")
        return ADPMatch(
            frame=merged,
            matched=int(merged["adp"].notna().sum()),
            unmatched_names=(),
        )

    if "match_key" not in adp.columns:
        raise ADPError("ADP needs either a playerId column or a match_key")

    keyed = board.copy()
    keyed["match_key"] = keyed["name"].map(normalize_name)

    duplicates = adp["match_key"].duplicated().sum()
    if duplicates:
        raise ADPError(
            f"{duplicates} duplicate names in the ADP file after normalising; "
            "two players cannot share one ADP row"
        )

    merged = keyed.merge(adp[["match_key", "adp"]], on="match_key", how="left")
    matched_keys = set(keyed["match_key"]) & set(adp["match_key"])
    unmatched = tuple(
        sorted(adp[~adp["match_key"].isin(matched_keys)]["name"].astype(str))
    )
    return ADPMatch(
        frame=merged.drop(columns=["match_key"]),
        matched=len(matched_keys),
        unmatched_names=unmatched,
    )


def value_gap(
    board: pd.DataFrame, *, vorp_column: str = "vorp", adp_column: str = "adp"
) -> pd.DataFrame:
    """Rank by value, rank by where the room drafts, and difference the two.

    `gap` is positive for a player the model likes more than the room does --
    a target you can wait on. Negative is the reverse: someone going earlier
    than their value supports.

    Players with no ADP get no gap rather than a gap of zero. Treating an
    unmatched player as "undrafted, therefore a huge sleeper" would fill the
    top of the list with names the ADP source simply spelled differently.
    """
    for column in (vorp_column, adp_column):
        if column not in board.columns:
            raise ADPError(f"{column!r} is not on the board")

    out = board.copy()
    out["value_rank"] = out[vorp_column].rank(ascending=False, method="min")
    out["adp_rank"] = out[adp_column].rank(ascending=True, method="min")
    out["gap"] = out["adp_rank"] - out["value_rank"]
    out.loc[out[adp_column].isna(), "gap"] = pd.NA
    return out


def sleepers(
    board: pd.DataFrame, *, draftable: int, limit: int = 15, min_vorp: float = 0.0
) -> pd.DataFrame:
    """Players the model wants earlier than the room takes them.

    Two filters, both load-bearing. `min_vorp` keeps out replacement-level
    players who go undrafted for the excellent reason that nobody should draft
    them. `draftable` keeps the comparison inside the pool that actually gets
    picked -- a player at ADP 400 is not "being slept on", he is simply not in
    anyone's draft.
    """
    usable = board[
        board["gap"].notna()
        & (board["vorp"] > min_vorp)
        & (board["value_rank"] <= draftable)
    ]
    return usable.nlargest(limit, "gap")


def reaches(board: pd.DataFrame, *, draftable: int, limit: int = 15) -> pd.DataFrame:
    """Players going earlier than their value supports.

    Bounded on the ADP *value* rather than its rank within whatever frame was
    passed in. ADP already is a draft position, so comparing it to the size of
    the draftable pool is the question being asked -- was this player actually
    being taken? Ranking first would let a player at ADP 400 through simply
    because few others were in the frame.
    """
    usable = board[board["gap"].notna() & (board["adp"] <= draftable)]
    return usable.nsmallest(limit, "gap")
