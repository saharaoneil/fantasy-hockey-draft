"""Build a draft board: history -> projections -> fantasy points -> VORP.

    python3 scripts/build_draft_board.py --target-season 2026 --out out/

Writes `draft_board.csv` and `draft_board.json`. The JSON is what the draft-day
tool will read, so the model is a build artifact and the tool stays a view over
it -- which is what keeps the tool fast enough to use during a live draft.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fantasy import nhl, predictability as pred, projections, value  # noqa: E402

FIRST_SEASON = 2014
LAST_SEASON = 2025

SKATER_STATS = ("goals", "assists", "shots", "hits", "blockedShots",
                "ppPoints", "penaltyMinutes")
GOALIE_STATS = ("wins", "saves", "shutouts")


def _measured_reliability(frame, stats, *, id_column, min_games):
    """Take each stat's regression strength from the measurement, not a guess."""
    results = pred.measure(frame, tuple(stats) + ("gamesPlayed",),
                           id_column=id_column, min_games=min_games)
    return {r.stat: max(r.spearman, 0.05) for r in results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-season", type=int, default=LAST_SEASON + 1)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--cache", type=Path, default=Path("data/raw"))
    parser.add_argument("--teams", type=int, default=12)
    parser.add_argument(
        "--require-recent-season", action="store_true", default=True,
        help="drop players with no game in the most recent season; without it "
             "the board fills with players who have since retired",
    )
    args = parser.parse_args()

    years = list(range(FIRST_SEASON, LAST_SEASON + 1))
    data = nhl.load_seasons(years, cache_dir=args.cache)
    skaters, goalies = data["skaters"], data["goalies"]

    bios = pd.concat(
        [nhl.bios(y, cache_dir=args.cache) for y in years], ignore_index=True
    )
    skaters = skaters.merge(
        bios[["playerId", "season", "age"]], on=["playerId", "season"], how="left"
    )

    skater_reliability = _measured_reliability(
        skaters, SKATER_STATS, id_column="playerId", min_games=20
    )
    goalie_reliability = _measured_reliability(
        goalies, GOALIE_STATS, id_column="playerId", min_games=15
    )
    print("measured stickiness (drives how hard each stat regresses)")
    for stat, rho in sorted(skater_reliability.items(), key=lambda kv: -kv[1]):
        k = projections.regression_constant(rho, games_per_season=82)
        print(f"  {stat:<16} rho {rho:.2f}   K = {k:5.1f} games")

    config = projections.ProjectionConfig(
        season_weights=projections.DEFAULT_SEASON_WEIGHTS,
        reliability={**skater_reliability, **goalie_reliability},
        games_per_season=82,
        min_games_for_age_curve=40,
        age_smoothing_window=3,
    )

    curve = projections.measure_age_curve(skaters, config=config)
    print(f"\nage curve measured over ages {curve.index.min()}-{curve.index.max()}")
    for age in (22, 25, 28, 32, 35):
        if age in curve.index:
            print(f"  age {age} -> {curve.loc[age]:.3f}x")

    skater_board = projections.project(
        skaters, SKATER_STATS, target_season=args.target_season,
        config=config, age_curve=curve,
    )
    skater_board["projected_games"] = projections.project_games(
        skaters, target_season=args.target_season, config=config,
        reliability=skater_reliability["gamesPlayed"],
    ).reindex(skater_board["playerId"]).to_numpy()

    goalie_board = projections.project(
        goalies, GOALIE_STATS, target_season=args.target_season, config=config
    )
    goalie_board["projected_games"] = projections.project_games(
        goalies, target_season=args.target_season, config=config,
        reliability=goalie_reliability["gamesPlayed"],
    ).reindex(goalie_board["playerId"]).to_numpy()

    # Scored per position group, then concatenated. The other order leaves each
    # group with NaN in the other's stat columns and poisons every total.
    for group in (skater_board, goalie_board):
        group["projected_points"] = value.fantasy_points(
            group, value.DEFAULT_SCORING, games_column="projected_games"
        )
    board = pd.concat([skater_board, goalie_board], ignore_index=True)

    names = pd.concat([
        skaters[["playerId", "skaterFullName"]].rename(
            columns={"skaterFullName": "name"}),
        goalies[["playerId", "goalieFullName"]].rename(
            columns={"goalieFullName": "name"}),
    ]).drop_duplicates("playerId")
    board = board.merge(names, on="playerId", how="left")

    if args.require_recent_season:
        active = set(
            skaters[skaters["season"] == LAST_SEASON]["playerId"]
        ) | set(goalies[goalies["season"] == LAST_SEASON]["playerId"])
        before = len(board)
        board = board[board["playerId"].isin(active)].copy()
        print(f"\ndropped {before - len(board)} players with no "
              f"{LAST_SEASON}-{str(LAST_SEASON+1)[2:]} games (retired or departed)")

    league = value.LeagueConfig(
        teams=args.teams, starters=value.DEFAULT_LEAGUE.starters
    )
    board = value.value_over_replacement(board, league)

    print(f"\nPOSITIONAL SCARCITY ({args.teams}-team league)")
    print(value.positional_scarcity(board, league).to_string(index=False))

    columns = ["name", "position", "age", "projected_games", "projected_points",
               "replacement", "vorp"]
    print(f"\nTOP 20 BY VORP for {args.target_season}-{str(args.target_season+1)[2:]}")
    top = board[columns].head(20).copy()
    for col in ("age", "projected_games", "projected_points", "replacement", "vorp"):
        top[col] = top[col].round(1)
    print(top.to_string(index=False))

    print("\nWhere VORP disagrees with raw projected points:")
    by_points = board.sort_values("projected_points", ascending=False).head(40)
    movers = board.head(40).merge(
        by_points[["playerId"]].assign(points_rank=range(1, len(by_points) + 1)),
        on="playerId", how="left",
    )
    movers["vorp_rank"] = range(1, len(movers) + 1)
    movers["shift"] = movers["points_rank"] - movers["vorp_rank"]
    biggest = movers.reindex(movers["shift"].abs().sort_values(ascending=False).index)
    print(biggest[["name", "position", "points_rank", "vorp_rank", "shift"]]
          .dropna().head(8).to_string(index=False))

    args.out.mkdir(parents=True, exist_ok=True)
    board.to_csv(args.out / "draft_board.csv", index=False)
    (args.out / "draft_board.json").write_text(
        json.dumps(
            {
                "target_season": args.target_season,
                "league": {"teams": args.teams,
                           "starters": dict(league.starters)},
                "scoring": dict(value.DEFAULT_SCORING.weights),
                "players": json.loads(
                    board[["playerId"] + columns].to_json(orient="records")
                ),
            },
            indent=1,
        )
    )
    print(f"\nwrote {args.out}/draft_board.csv and draft_board.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
