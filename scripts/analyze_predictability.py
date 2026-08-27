"""Measure how much of each stat carries from one season to the next.

    python3 scripts/analyze_predictability.py --out out/

Writes a table and a chart. The chart is the shareable artifact; the table is
what the draft board is built on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fantasy import nhl, predictability as pred  # noqa: E402

FIRST_SEASON = 2014
LAST_SEASON = 2025

PRETTY = {
    "blockedShots": "Blocked shots", "toiPerGameMinutes": "Time on ice",
    "hits": "Hits", "shots": "Shots", "ppPoints": "Power-play points",
    "points": "Points", "goals": "Goals", "assists": "Assists",
    "giveaways": "Giveaways", "penaltyMinutes": "Penalty minutes",
    "shootingPct": "Shooting %", "gamesPlayed": "Games played",
    "plusMinus": "Plus/minus", "wins": "Wins", "savePct": "Save %",
    "goalsAgainstAverage": "GAA", "shutouts": "Shutouts", "saves": "Saves",
}


def chart(skater_frame, goalie_frame, path: Path) -> None:
    rows = [
        (PRETTY.get(r.stat, r.stat), r.spearman, "skater")
        for r in skater_frame.itertuples()
    ] + [
        (f"{PRETTY.get(r.stat, r.stat)} (G)", r.spearman, "goalie")
        for r in goalie_frame.itertuples()
    ]
    rows.sort(key=lambda r: r[1])

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colours = ["#c94f4f" if r[2] == "goalie" else "#3c6e9f" for r in rows]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.barh(labels, values, color=colours)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Year-over-year rank correlation (Spearman)")
    ax.set_title(
        "What actually carries from one NHL season to the next\n"
        f"{FIRST_SEASON}-{str(FIRST_SEASON+1)[2:]} to "
        f"{LAST_SEASON}-{str(LAST_SEASON+1)[2:]}, per-game rates",
        loc="left", fontsize=12,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.axvline(0.5, color="#999", lw=0.8, ls="--")
    ax.text(0.505, -0.6, "coin-flip territory below", fontsize=8, color="#666")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#3c6e9f"),
        plt.Rectangle((0, 0), 1, 1, color="#c94f4f"),
    ]
    ax.legend(handles, ["Skaters", "Goalies"], loc="lower right", frameon=False)
    for index, value in enumerate(values):
        ax.text(value + 0.012, index, f"{value:.2f}", va="center", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--cache", type=Path, default=Path("data/raw"))
    parser.add_argument("--min-skater-games", type=int, default=20)
    parser.add_argument("--min-goalie-games", type=int, default=15)
    args = parser.parse_args()

    years = list(range(FIRST_SEASON, LAST_SEASON + 1))
    data = nhl.load_seasons(years, cache_dir=args.cache)
    skaters, goalies = data["skaters"], data["goalies"]
    print(
        f"{len(years)} seasons | {len(skaters):,} skater-seasons | "
        f"{len(goalies):,} goalie-seasons"
    )

    skater_results = pred.as_frame(
        pred.measure(skaters, pred.SKATER_STATS, id_column="playerId",
                     min_games=args.min_skater_games)
    )
    goalie_results = pred.as_frame(
        pred.measure(goalies, pred.GOALIE_STATS, id_column="playerId",
                     min_games=args.min_goalie_games)
    )

    print("\nSKATERS")
    print(skater_results.to_string(index=False))
    print("\nGOALIES")
    print(goalie_results.to_string(index=False))

    by_position = pred.as_frame(
        pred.measure(skaters, ("points", "shots", "hits", "blockedShots"),
                     id_column="playerId", min_games=args.min_skater_games,
                     by_position=True)
    )
    print("\nBY POSITION")
    print(by_position.to_string(index=False))

    # The objection to the goalie result is sample size, so answer it inline.
    print("\nDoes the goalie result survive restricting to starters?")
    print(f"{'min games':>10}{'pairs':>8}{'save % rho':>13}")
    for minimum in (10, 20, 30, 40, 50):
        found = [
            r for r in pred.measure(goalies, ("savePct",), id_column="playerId",
                                    min_games=minimum)
            if r.stat == "savePct"
        ]
        if found:
            print(f"{minimum:>10}{found[0].pairs:>8}{found[0].spearman:>13.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    skater_results.to_csv(args.out / "predictability_skaters.csv", index=False)
    goalie_results.to_csv(args.out / "predictability_goalies.csv", index=False)
    by_position.to_csv(args.out / "predictability_by_position.csv", index=False)
    chart(skater_results, goalie_results, args.out / "predictability.png")
    print(f"\nwrote tables and chart to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
