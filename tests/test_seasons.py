"""season_to_months: general contiguous-month season codes (v3 item F)."""
import pytest

from acmaddl.fetch import SEASON_MONTHS, season_to_months
from acmaddl import parse_target


def test_classic_codes_unchanged():
    for code, expected in SEASON_MONTHS.items():
        assert season_to_months(code) == expected
        # the general parser agrees with the table for every classic code
        assert season_to_months(code.lower()) == expected


def test_four_month_seasons():
    assert season_to_months("JJAS") == (6, 9)     # the GHA/Sahel 4-month season
    assert season_to_months("MAMJ") == (3, 6)
    assert season_to_months("ONDJ") == (10, 1)    # wraparound


def test_two_month_and_long_runs():
    assert season_to_months("ND") == (11, 12)
    assert season_to_months("JF") == (1, 2)
    assert season_to_months("NDJF") == (11, 2)    # wraparound
    assert season_to_months("JFMAMJJASOND") == (1, 12)


def test_unknown_codes_raise():
    with pytest.raises(ValueError):
        season_to_months("J")          # single letters are ambiguous
    with pytest.raises(ValueError):
        season_to_months("MAMA")       # not a contiguous run
    with pytest.raises(ValueError):
        season_to_months("XYZ")


def test_parse_target_accepts_general_codes():
    start, end = parse_target("JJAS", year=2026)
    assert (start.year, start.month, start.day) == (2026, 6, 1)
    assert (end.year, end.month, end.day) == (2026, 9, 30)
    # wraparound crosses the year boundary
    start, end = parse_target("NDJF", year=2026)
    assert (start.year, start.month) == (2026, 11)
    assert (end.year, end.month, end.day) == (2027, 2, 28)
