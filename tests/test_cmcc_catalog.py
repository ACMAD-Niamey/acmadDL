"""CMCC SPS4 (c3s/cmcc-sps4, c3s/cmcc-sps4-daily) catalog entries: offline unit tests.

SPS4 (C3S CDS system=4, model CMCC-CM3) is CMCC's current operational seasonal
system, live since 2025-08-01, superseding SPS3.5 (system=35, c3s/cmcc[-daily]).
Ranges and member counts here were verified against the live CDS on 2026-06-25
(hindcast 1993-2024 @ 30 members, real-time 2025+ @ 50 members; precip+sst served
in both the monthly and daily collections).

Live CDS fetches for these products live in tests/test_integration.py.
"""
import warnings

import pytest

from acmaddl import catalog

MONTHLY = "c3s/cmcc-sps4"
DAILY = "c3s/cmcc-sps4-daily"


# ── Entry shape ──────────────────────────────────────────────────────────────

def test_monthly_entry_present_and_cds():
    e = catalog.info(MONTHLY)
    assert e["adapter"] == "cds"
    assert e["cds_dataset"] == "seasonal-monthly-single-levels"
    assert e["cds_model"] == "cmcc"
    assert e["system"] == "4"
    assert e["product_type"] == "monthly_mean"
    assert e["leadtime_month"] == [1, 2, 3, 4, 5, 6]


def test_daily_entry_present_and_cds():
    e = catalog.info(DAILY)
    assert e["adapter"] == "cds"
    assert e["cds_dataset"] == "seasonal-original-single-levels"
    assert e["cds_model"] == "cmcc"
    assert e["system"] == "4"
    assert "product_type" not in e


# ── Variables ────────────────────────────────────────────────────────────────

def test_monthly_variables():
    v = catalog.info(MONTHLY)["variables"]
    assert set(v) == {"precip", "temp", "sst"}
    assert v["precip"]["native_name"] == "total_precipitation"
    assert v["precip"]["short_name"] == "tprate"
    assert v["precip"]["units"] == "m s-1"
    assert v["precip"]["target_units"] == "mm/day"
    assert v["temp"]["native_name"] == "2m_temperature"
    assert v["temp"]["short_name"] == "t2m"
    assert v["temp"]["target_units"] == "C"
    assert v["sst"]["native_name"] == "sea_surface_temperature"
    assert v["sst"]["units"] == "K"
    assert v["sst"]["target_units"] == "K"


def test_daily_precip_is_accumulated_tp():
    v = catalog.info(DAILY)["variables"]
    assert set(v) == {"precip", "temp", "sst"}
    assert v["precip"]["native_name"] == "total_precipitation"
    assert v["precip"]["short_name"] == "tp"
    assert v["precip"]["units"] == "m"
    assert v["precip"]["target_units"] == "mm/day"
    assert v["precip"]["accumulated"] is True
    assert v["sst"]["native_name"] == "sea_surface_temperature"
    assert v["sst"]["target_units"] == "K"


# ── Ensemble + ranges (verified vs live CDS 2026-06-25) ──────────────────────

@pytest.mark.parametrize("product", [MONTHLY, DAILY])
def test_member_counts_and_ranges(product):
    g = catalog.info(product)["grid"]
    assert g["hindcast_members"] == 30
    assert g["forecast_members"] == 50
    assert g["hindcast_range"] == [1993, 2024]
    assert g["forecast_range"] == [2025, None]


# ── Live-list membership ─────────────────────────────────────────────────────

def test_sps4_entries_in_live_product_list():
    live = catalog.list_products(include_deprecated=False)
    assert MONTHLY in live
    assert DAILY in live


def test_sps4_not_deprecated():
    assert catalog.info(MONTHLY).get("deprecated", False) is False
    assert catalog.info(DAILY).get("deprecated", False) is False


# ── SPS3.5 deprecation (set when SPS4 superseded it on 2025-08-01) ────────────

def test_sps35_monthly_deprecated_points_at_sps4():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        info = catalog.info("c3s/cmcc")
    assert info["deprecated"] is True
    assert info["successor"] == "c3s/cmcc-sps4"
    assert any(issubclass(x.category, DeprecationWarning) for x in w), \
        f"expected a DeprecationWarning, got {[str(x.message) for x in w]}"


def test_sps35_daily_deprecated_points_at_sps4():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        info = catalog.info("c3s/cmcc-daily")
    assert info["deprecated"] is True
    assert info["successor"] == "c3s/cmcc-sps4-daily"
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_sps35_still_exposes_own_variables_and_system():
    """SPS3.5 must keep its own config (the PyCPT SPSv3p5 -> c3s/cmcc reference),
    NOT be repointed/aliased to SPS4."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = catalog.info("c3s/cmcc")
    assert {"precip", "sst"} <= set(m["variables"])
    assert m["system"] == "35"
    assert m["successor"] in catalog.list_products()


def test_sps35_excluded_from_live_list():
    live = catalog.list_products(include_deprecated=False)
    assert "c3s/cmcc" not in live
    assert "c3s/cmcc-daily" not in live
