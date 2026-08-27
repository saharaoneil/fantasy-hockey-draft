"""ADP joining, the consensus proxy, and the value gap."""

from __future__ import annotations

import pandas as pd
import pytest

from fantasy import adp


def _board(rows):
    """rows: (playerId, name, position, vorp)"""
    return pd.DataFrame(rows, columns=["playerId", "name", "position", "vorp"])


# ---------------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------------


def test_names_match_across_the_ways_sources_spell_them():
    """Exports and the NHL API disagree about accents, periods and suffixes."""
    assert adp.normalize_name("T.J. Oshie") == adp.normalize_name("TJ Oshie")
    assert adp.normalize_name("Tim Stützle") == adp.normalize_name("Tim Stutzle")
    assert adp.normalize_name("Alex Wennberg Jr.") == adp.normalize_name("Alex Wennberg")
    assert adp.normalize_name("Pierre-Luc Dubois") == adp.normalize_name("Pierre Luc Dubois")


def test_different_players_do_not_collide():
    assert adp.normalize_name("Cale Makar") != adp.normalize_name("Taylor Makar")


# ---------------------------------------------------------------------------
# Loading a CSV
# ---------------------------------------------------------------------------


def test_a_csv_loads_and_sorts_by_adp(tmp_path):
    path = tmp_path / "adp.csv"
    path.write_text("name,adp\nB Player,12\nA Player,3\n")
    frame = adp.load_adp_csv(path)
    assert list(frame["name"]) == ["A Player", "B Player"]
    assert list(frame["adp"]) == [3, 12]


def test_a_csv_without_the_expected_columns_is_refused(tmp_path):
    path = tmp_path / "adp.csv"
    path.write_text("player,rank\nA,1\n")
    with pytest.raises(adp.ADPError, match="no \\['adp', 'name'\\] column"):
        adp.load_adp_csv(path)


def test_a_csv_with_no_numeric_adp_is_refused(tmp_path):
    path = tmp_path / "adp.csv"
    path.write_text("name,adp\nA,early\nB,late\n")
    with pytest.raises(adp.ADPError, match="no numeric values"):
        adp.load_adp_csv(path)


# ---------------------------------------------------------------------------
# Joining
# ---------------------------------------------------------------------------


def test_unmatched_names_are_reported_rather_than_dropped():
    """A silent 60% match rate would turn the gap column into a ranking of
    who happened to join."""
    board = _board([(1, "Cale Makar", "D", 100.0), (2, "Quinn Hughes", "D", 90.0)])
    source = pd.DataFrame({
        "name": ["Cale Makar", "Someone Elses Guy"],
        "adp": [3, 40],
    })
    source["match_key"] = source["name"].map(adp.normalize_name)

    match = adp.join_adp(board, source)
    assert match.matched == 1
    assert match.unmatched_names == ("Someone Elses Guy",)
    assert match.match_rate == pytest.approx(0.5)
    assert match.frame.set_index("playerId").loc[2, "adp"] != match.frame.set_index("playerId").loc[2, "adp"]  # NaN


def test_accented_names_still_join():
    board = _board([(1, "Tim Stützle", "C", 80.0)])
    source = pd.DataFrame({"name": ["Tim Stutzle"], "adp": [14]})
    source["match_key"] = source["name"].map(adp.normalize_name)
    assert adp.join_adp(board, source).matched == 1


def test_duplicate_names_in_the_adp_file_are_refused():
    board = _board([(1, "A Player", "C", 10.0)])
    source = pd.DataFrame({"name": ["A Player", "A. Player"], "adp": [1, 2]})
    source["match_key"] = source["name"].map(adp.normalize_name)
    with pytest.raises(adp.ADPError, match="duplicate names"):
        adp.join_adp(board, source)


def test_an_adp_frame_keyed_by_playerid_skips_name_matching():
    board = _board([(1, "Anyone", "C", 10.0)])
    source = pd.DataFrame({"playerId": [1], "adp": [5]})
    match = adp.join_adp(board, source)
    assert match.matched == 1
    assert match.unmatched_names == ()


# ---------------------------------------------------------------------------
# The consensus proxy
# ---------------------------------------------------------------------------


def _actuals(rows):
    return pd.DataFrame(rows, columns=["playerId", "last_points"])


def test_the_proxy_interleaves_positions_instead_of_ranking_on_raw_points():
    """The bug the first version had: defencemen score fewer points, so a flat
    ranking buries all of them and the gap reports the whole position as
    underdrafted -- an artifact of the proxy, not a finding."""
    board = _board(
        [(i, f"f{i}", "W", 0.0) for i in range(6)]
        + [(100 + i, f"d{i}", "D", 0.0) for i in range(6)]
    )
    # Forwards outscore every defenceman.
    actuals = _actuals(
        [(i, 200 - i) for i in range(6)] + [(100 + i, 100 - i) for i in range(6)]
    )
    ranked = adp.last_season_rank(
        board, actuals, points_column="last_points",
        slots_per_position={"W": 1, "D": 1},
    ).merge(board[["playerId", "position"]], on="playerId")

    top_six = ranked.nsmallest(6, "adp")["position"].tolist()
    assert "D" in top_six  # a flat points ranking would have none


def test_the_best_player_at_a_position_goes_before_the_worst():
    board = _board([(i, f"p{i}", "C", 0.0) for i in range(4)])
    actuals = _actuals([(i, 100 - i * 10) for i in range(4)])
    ranked = adp.last_season_rank(
        board, actuals, points_column="last_points", slots_per_position={"C": 1}
    )
    assert list(ranked.sort_values("adp")["playerId"]) == [0, 1, 2, 3]


def test_the_proxy_needs_league_shape():
    board = _board([(1, "a", "C", 0.0)])
    with pytest.raises(adp.ADPError, match="slots per position"):
        adp.last_season_rank(
            board, _actuals([(1, 10)]), points_column="last_points",
            slots_per_position={},
        )


def test_the_proxy_needs_overlapping_players():
    board = _board([(1, "a", "C", 0.0)])
    with pytest.raises(adp.ADPError, match="no players in common"):
        adp.last_season_rank(
            board, _actuals([(999, 10)]), points_column="last_points",
            slots_per_position={"C": 1},
        )


# ---------------------------------------------------------------------------
# The gap
# ---------------------------------------------------------------------------


def _gapped():
    board = _board([
        (1, "target", "C", 90.0),    # high value, drafted late
        (2, "fair", "C", 80.0),
        (3, "reach", "C", 10.0),     # low value, drafted early
    ])
    board["adp"] = [30, 2, 1]
    return adp.value_gap(board)


def test_a_late_pick_with_high_value_reads_as_a_target():
    frame = _gapped().set_index("name")
    assert frame.loc["target", "gap"] > 0
    assert frame.loc["reach", "gap"] < 0


def test_a_player_without_adp_gets_no_gap_rather_than_a_flattering_one():
    """Treating an unmatched player as undrafted would fill the target list
    with names the ADP source merely spelled differently."""
    board = _board([(1, "known", "C", 50.0), (2, "unmatched", "C", 49.0)])
    board["adp"] = [10, None]
    frame = adp.value_gap(board).set_index("name")
    assert pd.isna(frame.loc["unmatched", "gap"])


def test_targets_exclude_players_nobody_should_draft():
    board = _board([(i, f"p{i}", "C", -50.0) for i in range(5)])
    board["adp"] = [500, 501, 502, 503, 504]
    frame = adp.value_gap(board)
    assert adp.sleepers(frame, draftable=100).empty


def test_targets_stay_inside_the_draftable_pool():
    board = _board([(1, "deep", "C", 5.0), (2, "early", "C", 100.0)])
    board["adp"] = [900, 3]
    frame = adp.value_gap(board)
    found = adp.sleepers(frame, draftable=1, limit=5)
    assert "deep" not in list(found["name"])


def test_reaches_ignore_players_nobody_was_taking():
    """A fringe player at ADP 400 cannot be reached for."""
    board = _board([(1, "fringe", "G", -100.0), (2, "real", "C", 5.0)])
    board["adp"] = [400, 4]
    frame = adp.value_gap(board)
    found = adp.reaches(frame, draftable=50, limit=5)
    assert list(found["name"]) == ["real"]


def test_the_gap_needs_both_columns():
    board = _board([(1, "a", "C", 5.0)])
    with pytest.raises(adp.ADPError, match="'adp' is not on the board"):
        adp.value_gap(board)
