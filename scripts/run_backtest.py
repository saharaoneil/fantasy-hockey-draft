"""Test the projections against the baseline they claim to beat.

    python3 scripts/run_backtest.py --out out/

Every method predicts a season it was not allowed to see, including when the
reliability figures and age curve were measured. If `model` does not beat
`three_year`, the regression machinery is not earning its keep and should be
cut rather than defended.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fantasy import backtest, nhl, value  # noqa: E402

FIRST_SEASON = 2014
LAST_SEASON = 2025
SKATER_STATS = ("goals", "assists", "shots", "hits", "blockedShots",
                "ppPoints", "penaltyMinutes")

LABELS = {
    "last_season": "Last season's totals",
    "three_year": "3-season average (no regression)",
    "model_no_age": "Regressed, no age curve",
    "model": "Model (regressed + aged)",
}


def chart(results: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, group in results.groupby("method"):
        group = group.sort_values("season")
        ax.plot(group["season"], group["spearman"], marker="o",
                label=LABELS.get(method, method), linewidth=2)
    ax.set_xlabel("Holdout season (projected using only prior seasons)")
    ax.set_ylabel("Rank correlation with actual fantasy points")
    ax.set_title("Does regressing by measured stickiness beat guessing "
                 "from last year?", loc="left", fontsize=12)
    ax.grid(alpha=.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--cache", type=Path, default=Path("data/raw"))
    parser.add_argument("--first-holdout", type=int, default=2018)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    years = list(range(FIRST_SEASON, LAST_SEASON + 1))
    skaters = nhl.load_seasons(years, cache_dir=args.cache)["skaters"]
    bios = pd.concat([nhl.bios(y, cache_dir=args.cache) for y in years],
                     ignore_index=True)
    skaters = skaters.merge(bios[["playerId", "season", "age"]],
                            on=["playerId", "season"], how="left")

    holdouts = list(range(args.first_holdout, LAST_SEASON + 1))
    print(f"holdout seasons: {holdouts[0]}-{holdouts[-1]} ({len(holdouts)} seasons)")
    print("each projected using only seasons before it, including when measuring")
    print("reliability and the age curve\n")

    # The ablation runs by default. Reporting only the final model would leave
    # every component unjustified, and one of them turns out to be negative.
    methods = dict(backtest.METHODS)
    methods["model_no_age"] = partial(backtest.predict_model, use_age_curve=False)

    results = backtest.run(
        skaters, holdout_seasons=holdouts, stats_wanted=SKATER_STATS,
        scoring=value.DEFAULT_SCORING, top_n=args.top_n, methods=methods,
    )
    print("PER SEASON")
    print(results.pivot(index="season", columns="method", values="spearman")
          .to_string())

    summary = backtest.summarise(results)
    print("\nAVERAGED ACROSS HOLDOUT SEASONS")
    print(summary.to_string())

    rho = summary["spearman"]
    print("\nWHAT EACH PIECE CONTRIBUTES")
    print(f"  more history alone     {rho['three_year'] - rho['last_season']:+.3f}"
          "   (3-season average vs last season)")
    print(f"  regression             {rho['model_no_age'] - rho['three_year']:+.3f}"
          "   (regressed vs unregressed, same history)")
    print(f"  age curve              {rho['model'] - rho['model_no_age']:+.3f}"
          "   (aged vs not)")
    print(f"  ---")
    print(f"  model vs last season   {rho['model'] - rho['last_season']:+.3f}")

    if rho["three_year"] < rho["last_season"]:
        print("\n  Note: more history WITHOUT regression is worse than no extra")
        print("  history at all. The regression is not a refinement on top of a")
        print("  longer average -- it is what makes the longer average useful.")
    if rho["model"] - rho["last_season"] <= 0.005:
        print("\n  The model is not beating the naive baseline. Say so.")

    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "backtest_by_season.csv", index=False)
    summary.to_csv(args.out / "backtest_summary.csv")
    chart(results, args.out / "backtest.png")
    print(f"\nwrote tables and chart to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
