# fantasy-hockey-draft

A draft board built on what's actually predictable, not on what happened last year.

```bash
python3 -m pip install -r requirements.txt
python3 scripts/analyze_predictability.py --out out/   # the finding
python3 scripts/build_draft_board.py --out out/        # the board
```

Pulls 12 NHL seasons from the league's public stats API, measures how much of
each stat carries from one season to the next, and uses that measurement to
build a draft board.

## The finding

![Year-over-year predictability of NHL stats](out/predictability.png)

Twelve seasons (2014-15 through 2025-26), 10,999 skater-seasons and 1,177
goalie-seasons, per-game rates, Spearman rank correlation between consecutive
seasons.

### Every goalie stat is less predictable than almost every skater stat

| stat | year-over-year ρ |
|---|---|
| Skater points | **0.81** |
| Goalie save % | **0.30** |
| Goalie wins | 0.34 |
| Goalie shutouts | 0.12 |

Skater points are about **2.7× more predictable** than goalie save percentage.
The *most* predictable thing about a goalie — how many games they play, at 0.48
— is still less predictable than the *tenth* most predictable skater stat.

"Wait on goalies" is folk wisdom. This is the number behind it.

**The obvious objection is sample size**, since a lot of those goalies are
backups with 15 games. It survives:

| min games | pairs | save % ρ |
|---|---|---|
| 10 | 638 | 0.313 |
| 30 | 341 | 0.365 |
| 50 | 105 | 0.425 |

Restricting to workhorse starters lifts it from 0.30 to 0.43 — so sample size
explains part of the gap, but not most of it. Skater points at the same
threshold sit at 0.835.

### The "boring" stats are the reliable ones

| stat | ρ |
|---|---|
| Blocked shots | 0.88 |
| Time on ice | 0.88 |
| Hits | 0.88 |
| Shots | 0.86 |
| Power-play points | 0.82 |
| **Points** | **0.81** |
| Goals | 0.78 |
| Assists | 0.74 |

Hits, blocks and shots are **more** predictable than points. They're driven by
role and deployment, which change slowly, while scoring carries finishing luck.
In a categories league the peripheral columns are the ones you can actually
bank on.

### Two things nobody should pay for

**Plus/minus (0.27)** is close to noise. It depends on teammates, goaltending
and on-ice shooting percentage — almost none of which is the player.

**Games played (0.35).** Durability is far less predictable than draft-day
conversation assumes. "He's injury prone" carries much less signal than "he
plays 20 minutes a night" (time on ice, 0.88).

### One nuance worth stating honestly

Shooting percentage lands at **0.61**, higher than the usual "it all regresses"
line suggests. That isn't a contradiction: this measures whether *rankings*
persist, and genuine snipers really do out-shoot the field year after year.
The regression-to-mean claim is about a player's deviation from *their own*
baseline, which is a different question than the one measured here.

## From the finding to a draft board

A stat you can't predict is one you shouldn't pay for. So the measurement
isn't decorative — it sets how hard each stat regresses in the projection.

### The projection

A weighted average of the last three seasons, pulled toward the league rate by
an amount derived from that stat's measured stickiness:

```
projected_rate = (weighted_total + league_rate × K) / (weighted_games + K)
K = games_per_season × (1 − ρ) / ρ
```

One expression, two jobs. **Hits** (ρ 0.88) get K = 11 games, so a full season
of evidence swamps the prior. **Games played** (ρ 0.35) gets K = 152 — more
league average than player. And it handles sample size for free: the same K
regresses a 15-game season much harder than an 82-game one.

No gradient boosting, deliberately. This family of model is hard to beat as a
baseline, and one you can explain in a paragraph is one whose failures you can
diagnose.

### The age curve is measured, not assumed

Average change in points-per-game from each age to the next, across every pair
of consecutive 40-game seasons:

| ages | change per year |
|---|---|
| 20–24 | +3% to +13% |
| 25–27 | flat (peak) |
| 28–31 | −2% to −6% |
| 33–37 | −7% to −13% |

Known caveat: survivor bias. A player who falls apart at 34 doesn't play 40
games at 35, so he leaves the sample and the measured decline is gentler than
reality. The curve is therefore **optimistic about old players** — which is the
safer direction for a draft, since it understates the case for fading them
rather than overstating it.

### Value, not points

Draft order is **value over replacement**, not projected points, because you're
choosing between upgrades over the player you'd have had otherwise. The 2026-27
board makes the difference concrete:

| player | points rank | VORP rank | shift |
|---|---|---|---|
| Filip Gustavsson (G) | 9 | 40 | **−31** |
| Jeremy Swayman (G) | 8 | 36 | **−28** |
| Igor Shesterkin (G) | 6 | 25 | −19 |
| Brady Tkachuk (W) | 38 | 14 | **+24** |
| David Pastrnak (W) | 26 | 5 | +21 |

Goalies pile up raw points from wins and saves, but every league starts them
and the replacement-level goalie is nearly as good — so their marginal value
collapses. That's the same conclusion the predictability analysis reached from
a completely different direction: goalies are both **unpredictable and
low-VORP**. Two independent reasons to wait.

Replacement level is computed from the *available* pool, not fixed from league
settings, so recomputing it after each pick re-ranks the board as positions
drain. That's what the draft-day tool will do.

### Known limitations

- **Projected games are compressed** (~60 for everyone). With ρ = 0.35, the
  model regresses hard toward a league mean that includes fringe players. It
  barely affects *ordering* — VORP is a difference, so a roughly uniform
  compression cancels — but absolute totals read low.
- **Goalie ages are missing**; the bios endpoint pulled here is skater-only, so
  no age curve is applied to goalies.
- **No ADP**, so the board can't yet show where value diverges from where
  players actually go.

## Method

Four choices, each of which changes the answer:

- **Per-game rates, not totals.** Totals confound production with
  availability, and 2019-20 and 2020-21 were 68 and 56 games — raw totals
  aren't comparable across that boundary. Games played is measured separately
  instead, since durability is its own fantasy skill.
- **Spearman, not Pearson.** Drafting is an ordering problem, and rank
  correlation survives the outliers that a few elite scorers would otherwise
  dominate.
- **A minimum-games threshold on both seasons.** Without one, four games of
  fluke shooting sits beside an 82-game regular.
- **Only genuinely consecutive seasons.** A player who misses a year and
  returns is evidence about something else; folding them in would understate
  every stat.

### Two bugs worth recording

The first version of the data pull returned 924 rows for a season carrying only
917 distinct players. Offset pagination over an **unstable sort** silently
returns some rows twice and drops others — seven duplicated, seven lost. Fixed
by sorting on `playerId`, and `fetch_report` now checks for repeats explicitly,
because the duplicate is the visible half of a bug whose other half is missing
data.

Second: concatenating skaters and goalies *before* scoring left each group with
NaN in the other's stat columns, and one NaN in the sum silently produced a
board of NaN for everyone — which still sorted, and still printed. Now
`fantasy_points` raises if any priced column has a gap, because a loud failure
is much cheaper than a plausible-looking board that is entirely empty.

## Layout

```
fantasy/nhl.py             pull and cache season stats from the NHL API
fantasy/predictability.py  year-over-year persistence, measured honestly
fantasy/projections.py     regress each stat by how sticky it actually is
fantasy/value.py           scoring, replacement level, VORP
scripts/analyze_predictability.py   the finding: tables and the chart
scripts/build_draft_board.py        the board: CSV and JSON
```

`draft_board.json` is the build artifact the draft-day tool will read, so the
model stays offline and the tool stays a fast view over it.

Data is cached to `data/raw/` on first run so the analysis is reproducible —
a stat that shifts between runs is an anecdote, not a measurement.

## Source

NHL public stats API (`api.nhle.com/stats/rest/en`). No key required.
