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

from fantasy import adp as adp_mod  # noqa: E402
from fantasy import categories as cats  # noqa: E402
from fantasy import nhl, predictability as pred, projections, value  # noqa: E402

FIRST_SEASON = 2014
LAST_SEASON = 2025

SKATER_STATS = ("goals", "assists", "shots", "hits", "blockedShots",
                "ppPoints", "penaltyMinutes", "plusMinus")
GOALIE_STATS = ("wins", "saves", "shutouts", "savePct", "goalsAgainstAverage")


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
        "--format", choices=("points", "categories"), default="points",
        help="points sums weighted stats; categories z-scores each column "
             "against the draftable pool, which is how most leagues are "
             "actually scored",
    )
    parser.add_argument(
        "--adp-csv", type=Path, default=None,
        help="two-column CSV of name,adp from your platform. Without it the "
             "board falls back to last season's finish as a consensus proxy, "
             "which is labelled as a proxy and never as ADP",
    )
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

    if args.format == "categories":
        # Goalie rate categories need their volume columns, or a backup's .930
        # gets priced like a starter's.
        goalie_board["shotsAgainst"] = (
            goalie_board["saves_rate"].fillna(0) * goalie_board["projected_games"]
            / goalie_board["savePct_rate"].replace(0, pd.NA)
        ).fillna(0)

        # League-wide drafted counts, which size the standardisation pools as
        # well as the cross-group scaling.
        skater_drafted = args.teams * sum(
            v for k, v in value.DEFAULT_LEAGUE.starters.items() if k != "G"
        )
        goalie_drafted = args.teams * value.DEFAULT_LEAGUE.starters["G"]
        board = cats.value_groups(
            {"skaters": skater_board, "goalies": goalie_board},
            {"skaters": cats.DEFAULT_SKATER_CATEGORIES,
             "goalies": cats.DEFAULT_GOALIE_CATEGORIES},
            {"skaters": skater_drafted, "goalies": goalie_drafted},
        )
        board["projected_points"] = board["category_value"]
        print(f"\ncategories format: "
              f"{len(cats.DEFAULT_SKATER_CATEGORIES)} skater columns, "
              f"{len(cats.DEFAULT_GOALIE_CATEGORIES)} goalie columns")
        for name, group in board.groupby("value_group"):
            print(f"  {name:<9} roster-slot weight "
                  f"{group['group_scale'].iloc[0]:.2f}x")
    else:
        # Scored per position group, then concatenated. The other order leaves
        # each group with NaN in the other's stat columns and poisons every total.
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

    # --- ADP, or an honest stand-in for it -------------------------------
    league_shape = value.LeagueConfig(
        teams=args.teams, starters=value.DEFAULT_LEAGUE.starters
    )
    draftable = args.teams * sum(value.DEFAULT_LEAGUE.starters.values())
    if args.adp_csv:
        source = f"ADP from {args.adp_csv}"
        adp_source = str(args.adp_csv)
        match = adp_mod.join_adp(board, adp_mod.load_adp_csv(args.adp_csv))
        print(f"\nmatched {match.matched} of "
              f"{match.matched + len(match.unmatched_names)} ADP names "
              f"({match.match_rate:.0%})")
        if match.unmatched_names:
            print(f"  unmatched: {', '.join(match.unmatched_names[:8])}"
                  + (" ..." if len(match.unmatched_names) > 8 else ""))
        board = match.frame
    else:
        source = "last season's finish (a proxy for consensus, NOT real ADP)"
        adp_source = None  # the page says so plainly rather than implying data
        actuals = skaters[skaters["season"] == LAST_SEASON].copy()
        goalie_actuals = goalies[goalies["season"] == LAST_SEASON].copy()
        for group, group_stats in ((actuals, SKATER_STATS), (goalie_actuals, GOALIE_STATS)):
            for stat in group_stats:
                group[f"{stat}_rate"] = group[stat] / group["gamesPlayed"]
            group["projected_games"] = group["gamesPlayed"]
            group["last_points"] = value.fantasy_points(
                group, value.DEFAULT_SCORING, games_column="projected_games")
        finished = pd.concat([actuals, goalie_actuals], ignore_index=True)
        board = adp_mod.join_adp(
            board,
            adp_mod.last_season_rank(
                board, finished, points_column="last_points",
                slots_per_position=league_shape.starters,
            ),
        ).frame

    board = adp_mod.value_gap(board)
    print(f"\nvalue gap measured against: {source}")

    print(f"\nPOSITIONAL SCARCITY ({args.teams}-team league)")
    print(value.positional_scarcity(board, league).to_string(index=False))

    columns = ["name", "position", "age", "projected_games", "projected_points",
               "replacement", "vorp", "adp", "gap"]
    # In categories the per-column z-scores are the whole point -- they are what
    # tells a specialist apart from a generalist, which a single total cannot.
    z_columns = sorted(c for c in board.columns if c.startswith("z_"))
    export_columns = columns + (z_columns if args.format == "categories" else [])
    print(f"\nTOP 20 BY VORP for {args.target_season}-{str(args.target_season+1)[2:]}")
    top = board[columns].head(20).copy()
    for col in ("age", "projected_games", "projected_points", "replacement", "vorp"):
        top[col] = top[col].round(1)
    print(top.to_string(index=False))

    if args.format == "categories":
        print("\nWHAT THE TOP OF THE BOARD ACTUALLY WINS YOU")
        every = (tuple(cats.DEFAULT_SKATER_CATEGORIES)
                 + tuple(cats.DEFAULT_GOALIE_CATEGORIES))
        print(cats.category_contributions(board, every, limit=8)
              .to_string(index=False))

    print("\nTARGETS - the model likes them more than the room does")
    show = ["name", "position", "vorp", "adp", "value_rank", "gap"]
    tgt = adp_mod.sleepers(board, draftable=draftable, limit=10).copy()
    for col in ("vorp", "adp", "value_rank", "gap"):
        tgt[col] = tgt[col].astype(float).round(0)
    print(tgt[show].to_string(index=False))

    print("\nREACHES - going earlier than their value supports")
    rch = adp_mod.reaches(board, draftable=draftable, limit=10).copy()
    for col in ("vorp", "adp", "value_rank", "gap"):
        rch[col] = rch[col].astype(float).round(0)
    print(rch[show].to_string(index=False))

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
                "format": args.format,
                "adp_source": adp_source,
                "value_label": "Z" if args.format == "categories" else "Pts",
                "categories": (
                    [c.stat for c in cats.DEFAULT_SKATER_CATEGORIES]
                    + [c.stat for c in cats.DEFAULT_GOALIE_CATEGORIES]
                    if args.format == "categories" else []
                ),
                "league": {"teams": args.teams,
                           "starters": dict(league.starters)},
                "scoring": dict(value.DEFAULT_SCORING.weights),
                "players": json.loads(
                    board[["playerId"] + export_columns].to_json(orient="records")
                ),
            },
            indent=1,
        )
    )
    print(f"\nwrote {args.out}/draft_board.csv and draft_board.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
