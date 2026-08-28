# fantasy-hockey-draft

A draft board built on what's actually predictable, not on what happened last
year — and backtested to check that the distinction is worth anything.

**[Open the draft board →](https://saharaoneil.github.io/fantasy-hockey-draft/)**

```bash
python3 -m pip install -r requirements.txt
python3 scripts/analyze_predictability.py --out out/   # the finding
python3 scripts/build_draft_board.py --out out/        # the board
python3 scripts/build_draft_page.py                    # the draft-day tool
python3 scripts/run_backtest.py --out out/             # does it actually work?
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

### Replacement level during the draft, and a correction

Replacement level is computed from the pool still available and the slots still
to fill, rather than fixed once from league settings. An earlier version of this
README said that "as a position drains, its replacement level drops and everyone
left at it gains value." Measured in the live tool, that is wrong in both
direction and cause:

| scenario | D replacement level |
|---|---|
| pre-draft | 111 |
| top 40 defencemen taken **in value order** | 111 (unchanged) |
| 40 *weak* defencemen taken (ranks 60–100) | **164** |
| a run on centres instead | 111 (unchanged) |

Draining a position in value order changes nothing, because the 8th-best
*available* defenceman is still the 48th-best overall — the same player the
static calculation already pointed at. What moves the number is picks
**deviating from value order**, and when they do it moves *up*: if the league
reaches for weak defencemen, your floor at defence improves, so chasing defence
becomes *less* urgent rather than more.

That is the opposite of the usual draft-room instinct, and it falls out of what
VORP actually measures — an upgrade over the player you would otherwise end up
with. If everyone left at a position is equivalent, it does not matter which one
you get.

### Known limitations

- **Projected games are compressed** (~60 for everyone). With ρ = 0.35, the
  model regresses hard toward a league mean that includes fringe players. It
  barely affects *ordering* — VORP is a difference, so a roughly uniform
  compression cancels — but absolute totals read low.
- **Goalie ages are missing**; the bios endpoint pulled here is skater-only, so
  no age curve is applied to goalies.
- **ADP is a proxy unless you supply real numbers** — see below.

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
web/template.html          the draft-day page; data is baked in at build time
docs/index.html            the built page, served by GitHub Pages
fantasy/nhl.py             pull and cache season stats from the NHL API
fantasy/predictability.py  year-over-year persistence, measured honestly
fantasy/projections.py     regress each stat by how sticky it actually is
fantasy/value.py           scoring, replacement level, VORP
fantasy/adp.py             draft position, name matching, and the value gap
fantasy/backtest.py        holdout testing against the naive baseline
scripts/analyze_predictability.py   the finding: tables and the chart
scripts/build_draft_board.py        the board: CSV and JSON
scripts/build_draft_page.py         bake the board into one HTML file
scripts/run_backtest.py             holdout test and component ablation
```

`draft_board.json` is the build artifact the page is baked from, so the model
stays offline and the tool stays a fast view over it.

## Does it actually work?

Backtested against eight holdout seasons (2018-19 through 2025-26). Each season
is projected using **only the seasons before it** — including when measuring
the reliability figures and the age curve, since measuring those over all
twelve seasons and then testing on one of them would let the model see its own
answer.

![Backtest: rank correlation by holdout season](out/backtest.png)

| method | rank correlation | MAE | top-100 hit rate |
|---|---|---|---|
| **Model (regressed + aged)** | **0.757** | 40.1 | 0.684 |
| Regressed, no age curve | 0.739 | 40.5 | 0.666 |
| Last season's totals | 0.723 | 41.7 | 0.678 |
| 3-season average, no regression | 0.712 | 41.5 | 0.656 |

The model beats the naive baseline in **8 of 8** seasons. The margin is real
but modest — **+0.034** rank correlation. Hockey is noisy and this family of
model is hard to improve on; anyone claiming a large edge here is measuring
something wrong.

### What each piece actually contributes

| component | contribution |
|---|---|
| more history alone | **−0.011** |
| the regression | **+0.027** |
| the age curve | **+0.018** |

**More history on its own makes it worse.** A three-season average without
regression (0.712) is *below* just using last season (0.723) — extra seasons
add stale information about players whose role has changed, and averaging it in
blurs the signal.

The regression is what makes the extra history useful. It isn't a refinement on
top of a longer average; it's the thing that turns a longer average from a
liability into an improvement. That's the project's central claim, and it's the
one the backtest was built to be able to falsify.

The ablation runs by default rather than on request, because reporting only the
final model would leave every component unjustified — and one of them turns out
to be negative.

## ADP, and why there isn't a feed for it

A ranking says who's good. The **gap** between your ranking and where the room
drafts says who to actually target — that's the part anyone else can use.

There is no free NHL ADP feed, and this doesn't pretend otherwise. Checked at
build time:

- **Sleeper's** public API carries NHL rosters but leaves `search_rank` null for
  every hockey player (it's populated for NFL only).
- **FantasyPros'** `robots.txt` disallows `/api/`, `/json/` and `/ajax/` — which
  is where the numbers live.
- **Yahoo** requires OAuth.

So there are two honest routes instead of an invented one.

**Bring your own.** `--adp-csv` takes a two-column `name,adp` file from
whatever platform your league uses. That's better than a "consensus" number
anyway: a Yahoo points league drafts nothing like an ESPN roto one, and it's
*your* league's ADP that decides who's actually available. Name matching folds
accents, periods and suffixes (`Tim Stützle` = `Tim Stutzle`, `T.J. Oshie` =
`TJ Oshie`), and unmatched names are reported rather than dropped — a silent
60% match rate would turn the gap column into a ranking of who happened to join.

**Or use the proxy.** Without a file, the board ranks players by what they
actually scored last season under your scoring settings. It is never labelled
ADP. It models the *habit* — drafting off last year's totals — which is exactly
what this project argues with.

### The proxy needed fixing before it was useful

The first version ranked on raw points, and the target list came back as
"every defenceman." That's a positional artifact, not a finding: defencemen
score fewer points, so a flat ranking buries all of them, and VORP then
correctly says they're underdrafted — but it says it about the whole position
at once.

Now positions are ranked separately and interleaved in proportion to how many
of each get drafted, which is what a real board does, because the room does
know defencemen are scarce. The targets that survive are individuals:

| player | VORP | proxy ADP | gap |
|---|---|---|---|
| Auston Matthews (C) | 55 | 221 | **+193** |
| Brady Tkachuk (W) | 73 | 88 | +74 |
| Connor Hellebuyck (G) | 80 | 84 | +73 |
| Mikko Rantanen (W) | 64 | 64 | +42 |

The signal is legible: these are largely players who missed time last season, so
a last-season ranking buries them while a three-season projection doesn't.

## The draft-day tool

One self-contained HTML file, ~240 KB, no dependencies and no server. The data
is **embedded rather than fetched**, because fetching a sibling JSON is blocked
by CORS from `file://` — and a draft tool that only works online is one that
fails in a basement on bad wifi.

Styled after a 1970s NHL programme: newsprint ground, sweater-stripe bands,
arena-signage capitals and scoreboard numerals. **No web fonts** — the display
face is built from stacks that ship with the OS, because loading a font would
break the offline promise above for the sake of a typeface.

- One click to mark a player taken, one to claim them; undo for misclicks
- Instant search and position filters
- Roster panel showing which slots you have filled
- **Targets** filter: available players the model rates well above where
  they're going
- ADP and gap columns, with the gap greyed out for players whose value is
  negative — a big gap on a replacement-level player is noise, not a bargain
- CSV export of whatever is still available
- Picks persist in `localStorage`, so a refresh mid-draft loses nothing

**The page re-implements the value calculation in JavaScript**, which risks
drifting from `value.py`. So it checks itself: on load it recomputes VORP for
an untouched board and compares against the numbers Python wrote. If they
disagree by more than half a point it shows a warning banner rather than
quietly ranking on the wrong arithmetic.

Data is cached to `data/raw/` on first run so the analysis is reproducible —
a stat that shifts between runs is an anecdote, not a measurement.

## Source

NHL public stats API (`api.nhle.com/stats/rest/en`). No key required.
