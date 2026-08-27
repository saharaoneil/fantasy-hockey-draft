"""Pull season stats from the NHL's public stats API, and cache them.

Three reports are needed to cover standard fantasy categories, because no
single one carries them all:

    skater/summary    goals, assists, points, PP points, shots, +/-, PIM, TOI
    skater/realtime   hits, blocked shots, giveaways
    goalie/summary    wins, save %, GAA, shutouts, saves

Everything is cached to disk on first fetch. That is not politeness about rate
limits -- it is what makes the analysis reproducible. A stat that shifts under
you between runs turns a measurement into an anecdote.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

BASE = "https://api.nhle.com/stats/rest/en"
REGULAR_SEASON = 2
PAGE_SIZE = 100

SKATER_REPORTS = ("summary", "realtime")
GOALIE_REPORTS = ("summary",)


class NHLError(Exception):
    """The API could not be read, or returned something unusable."""


def season_id(start_year: int) -> int:
    """`2023` -> `20232024`. The API's season identifier is both years joined."""
    if not 1917 <= start_year <= 2100:
        raise ValueError(f"implausible season start year: {start_year}")
    return int(f"{start_year}{start_year + 1}")


def _get(url: str, *, retries: int = 3, pause_seconds: float = 1.0) -> dict:
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "fantasy-hockey-draft/0.1"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # network, timeout, or malformed JSON
            last = exc
            if attempt < retries - 1:
                time.sleep(pause_seconds * (attempt + 1))
    raise NHLError(f"failed to fetch {url}: {last}")


def fetch_report(
    kind: str, report: str, start_year: int, *, game_type: int = REGULAR_SEASON
) -> List[dict]:
    """Fetch every row of one report for one season, following pagination.

    The API caps a page at 100 rows and reports the true total, so the loop
    runs until the total is reached rather than until a short page arrives --
    a short page can also mean a truncated response.

    **The explicit sort is load-bearing.** Offset pagination over an unstable
    ordering silently returns some rows twice and drops others: measured on
    2023-24 without it, 924 rows came back carrying only 917 distinct players,
    so seven were duplicated and seven lost. Sorting by `playerId` makes the
    ordering total and the paging exact, which the row-count check below then
    confirms.
    """
    rows: List[dict] = []
    start = 0
    total = None
    sort = json.dumps([{"property": "playerId", "direction": "ASC"}])

    while True:
        query = urllib.parse.urlencode(
            {
                "isAggregate": "false",
                "isGame": "false",
                "limit": PAGE_SIZE,
                "start": start,
                "sort": sort,
                "cayenneExp": f"gameTypeId={game_type} and seasonId={season_id(start_year)}",
            }
        )
        payload = _get(f"{BASE}/{kind}/{report}?{query}")
        page = payload.get("data", [])
        if total is None:
            total = payload.get("total")
            if total is None:
                raise NHLError(f"{kind}/{report} {start_year}: no total in response")
        rows.extend(page)
        start += PAGE_SIZE
        if start >= total or not page:
            break

    if total is not None and len(rows) != total:
        raise NHLError(
            f"{kind}/{report} {start_year}: got {len(rows)} rows, expected {total}"
        )

    # Guards the pagination itself rather than the payload: a stable sort should
    # make every row distinct, so a repeat means the ordering slipped and rows
    # were lost as well as doubled.
    identifiers = [row["playerId"] for row in rows if "playerId" in row]
    if len(identifiers) != len(set(identifiers)):
        raise NHLError(
            f"{kind}/{report} {start_year}: {len(identifiers) - len(set(identifiers))} "
            "duplicate player rows; pagination is not stable, so rows are also missing"
        )
    return rows


def _cache_path(cache_dir: Path, kind: str, report: str, start_year: int) -> Path:
    return Path(cache_dir) / f"{kind}_{report}_{start_year}.json"


def cached_report(
    kind: str,
    report: str,
    start_year: int,
    *,
    cache_dir: Path,
    refresh: bool = False,
) -> List[dict]:
    """Fetch a report, or read it back from disk if already pulled."""
    path = _cache_path(cache_dir, kind, report, start_year)
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    rows = fetch_report(kind, report, start_year)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return rows


def skaters(start_year: int, *, cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """One row per skater for one season, joining the summary and realtime reports.

    Joined on `playerId` with an inner join on purpose: a player present in one
    report and missing from the other is a data problem, and silently keeping
    the half-populated row would put zeros into a predictability measurement.
    """
    frames = {
        report: pd.DataFrame(
            cached_report("skater", report, start_year, cache_dir=cache_dir, refresh=refresh)
        )
        for report in SKATER_REPORTS
    }

    summary = frames["summary"]
    realtime = frames["realtime"][
        ["playerId", "hits", "blockedShots", "giveaways"]
    ]
    merged = summary.merge(realtime, on="playerId", how="inner", validate="one_to_one")

    merged["season"] = start_year
    merged["position"] = merged["positionCode"].map(
        {"C": "C", "L": "W", "R": "W", "D": "D"}
    )
    merged["toiPerGameMinutes"] = merged["timeOnIcePerGame"] / 60.0
    return merged


def bios(start_year: int, *, cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Birth date, draft position and current team, one row per skater.

    Kept separate from the stat reports because it is nearly static: a bio
    changes once a year, so pulling it alongside every stat query would be
    waste. Age is what the projection actually needs.
    """
    frame = pd.DataFrame(
        cached_report("skater", "bios", start_year, cache_dir=cache_dir, refresh=refresh)
    )
    frame["season"] = start_year
    frame["birthDate"] = pd.to_datetime(frame["birthDate"], errors="coerce")

    # Age on 1 February of the season, roughly its midpoint -- a fairer single
    # number than age on opening night for a season that spans a birthday.
    midpoint = pd.Timestamp(year=start_year + 1, month=2, day=1)
    frame["age"] = (midpoint - frame["birthDate"]).dt.days / 365.25
    return frame[["playerId", "season", "birthDate", "age", "currentTeamAbbrev",
                  "draftYear", "draftOverall"]]


def goalies(start_year: int, *, cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """One row per goalie for one season."""
    frame = pd.DataFrame(
        cached_report("goalie", "summary", start_year, cache_dir=cache_dir, refresh=refresh)
    )
    frame["season"] = start_year
    frame["position"] = "G"
    return frame


def load_seasons(
    start_years: Sequence[int], *, cache_dir: Path, refresh: bool = False
) -> Dict[str, pd.DataFrame]:
    """Pull several seasons and stack them. Returns `{"skaters": ..., "goalies": ...}`."""
    if not start_years:
        raise ValueError("need at least one season")

    return {
        "skaters": pd.concat(
            [skaters(y, cache_dir=cache_dir, refresh=refresh) for y in start_years],
            ignore_index=True,
        ),
        "goalies": pd.concat(
            [goalies(y, cache_dir=cache_dir, refresh=refresh) for y in start_years],
            ignore_index=True,
        ),
    }
