"""Issuance-keyed forecast archives: URL templating, assembly, and fetch wiring.

An observational archive is keyed by the time it describes; a forecast archive
by *two* times (when it was issued, what it is about). These tests cover the
`issuance` catalog block that lets one HTTP adapter serve both.

No network: the remote opens are monkeypatched, and the real CHIRPS-GEFS pull
lives in the network-marked integration suite.
"""
import importlib
from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from acmaddl.adapters._issuance import (
    enumerate_files,
    issuance_config,
    lead_timedelta,
    parse_init_dates,
)
from acmaddl.adapters.http import HTTPAdapter
from acmaddl.catalog import _catalog

BASE = "https://example.org/archive"

DAILY = {
    "path_pattern": "{init:%Y}/{init:%m}/{init:%d}",
    "file_pattern": "c3g_{valid:%Y}.{valid:%m}.{valid:%d}.tif",
    "leads": [0, 2],
    "lead_units": "days",
}


def _cfg(**overrides):
    return issuance_config({"issuance": {**DAILY, **overrides}})


# --- config validation -----------------------------------------------------


def test_a_product_without_an_issuance_block_is_a_plain_time_series():
    assert issuance_config({"source_url": "x"}) is None


def test_issuance_expands_the_inclusive_lead_range():
    assert _cfg()["leads"] == [0, 1, 2]
    assert _cfg(leads=[0, 0])["leads"] == [0]
    assert _cfg(leads=[3, 5])["leads"] == [3, 4, 5]


def test_leads_default_to_a_single_zero_lead():
    cfg = issuance_config({"issuance": {"file_pattern": "x_{init:%Y}.tif"}})
    assert cfg["leads"] == [0]
    assert cfg["path_pattern"] == ""


def test_issuance_requires_a_file_pattern():
    with pytest.raises(ValueError, match="requires 'file_pattern'"):
        issuance_config({"issuance": {"leads": [0, 1]}})


def test_inverted_or_malformed_leads_are_rejected():
    with pytest.raises(ValueError, match="inclusive \\[min, max\\] pair"):
        _cfg(leads=[5, 3])
    with pytest.raises(ValueError, match="inclusive \\[min, max\\] pair"):
        _cfg(leads=[1, 2, 3])


def test_unknown_lead_units_are_rejected():
    with pytest.raises(ValueError, match="lead_units"):
        _cfg(lead_units="fortnights")


# --- date parsing ----------------------------------------------------------


def test_parse_init_dates_sorts_and_deduplicates():
    got = parse_init_dates(["2026-07-05", "2026-07-01", "2026-07-05"])
    assert got == [datetime(2026, 7, 1), datetime(2026, 7, 5)]


def test_parse_init_dates_accepts_datetimes():
    assert parse_init_dates([datetime(2026, 7, 5)]) == [datetime(2026, 7, 5)]


def test_a_month_keyed_init_is_rejected_for_an_issuance_product():
    with pytest.raises(ValueError, match="full YYYY-MM-DD dates"):
        parse_init_dates(["2026-07"])


# --- URL enumeration -------------------------------------------------------


def test_enumerate_files_puts_the_init_in_the_path_and_the_valid_date_in_the_name():
    files = enumerate_files(BASE, _cfg(), ["2026-07-05"])
    assert [f.url for f in files] == [
        f"{BASE}/2026/07/05/c3g_2026.07.05.tif",
        f"{BASE}/2026/07/05/c3g_2026.07.06.tif",
        f"{BASE}/2026/07/05/c3g_2026.07.07.tif",
    ]
    assert [f.lead for f in files] == [0, 1, 2]


def test_enumerate_files_crosses_the_month_boundary_correctly():
    files = enumerate_files(BASE, _cfg(leads=[0, 1]), ["2026-07-31"])
    assert files[1].url.endswith("2026/07/31/c3g_2026.08.01.tif")


def test_enumerate_files_crosses_the_year_boundary_correctly():
    files = enumerate_files(BASE, _cfg(leads=[0, 1]), ["2026-12-31"])
    assert files[1].url.endswith("2026/12/31/c3g_2027.01.01.tif")


def test_enumerate_files_handles_a_leap_day():
    files = enumerate_files(BASE, _cfg(leads=[0, 1]), ["2024-02-28"])
    assert files[1].url.endswith("c3g_2024.02.29.tif")


def test_enumerate_files_supports_an_init_keyed_filename_with_no_lead():
    cfg = issuance_config({"issuance": {
        "path_pattern": "{init:%Y}",
        "file_pattern": "c3g_{init:%Y}.{init:%m}.{init:%d}.tif",
        "leads": [0, 0],
    }})
    files = enumerate_files(BASE, cfg, ["2026-07-05"])
    assert [f.url for f in files] == [f"{BASE}/2026/c3g_2026.07.05.tif"]


def test_enumerate_files_supports_a_lead_numbered_filename():
    cfg = issuance_config({"issuance": {
        "file_pattern": "fcst_{init:%Y%m%d}_lead{lead:02d}.nc",
        "leads": [0, 1],
    }})
    files = enumerate_files(BASE, cfg, ["2026-07-05"])
    assert [f.url for f in files] == [
        f"{BASE}/fcst_20260705_lead00.nc",
        f"{BASE}/fcst_20260705_lead01.nc",
    ]


def test_enumerate_files_orders_issuances_then_leads():
    files = enumerate_files(BASE, _cfg(leads=[0, 1]), ["2026-07-06", "2026-07-05"])
    assert [(f.init.day, f.lead) for f in files] == [(5, 0), (5, 1), (6, 0), (6, 1)]


def test_enumerate_files_needs_at_least_one_issuance():
    with pytest.raises(ValueError, match="no issuance dates"):
        enumerate_files(BASE, _cfg(), [])


def test_hourly_leads_advance_by_hours_not_days():
    cfg = issuance_config({"issuance": {
        "file_pattern": "{valid:%Y%m%d%H}.nc", "leads": [0, 24], "lead_units": "hours",
    }})
    files = enumerate_files(BASE, cfg, ["2026-07-05"])
    assert files[0].url.endswith("2026070500.nc")
    assert files[-1].url.endswith("2026070600.nc")


def test_lead_timedelta_carries_its_own_units():
    """An integer lead_time is ambiguous the moment two products disagree on
    whether it counts days or hours."""
    days = lead_timedelta([0, 1], "days")
    hours = lead_timedelta([0, 1], "hours")
    assert days[1] == np.timedelta64(1, "D")
    assert hours[1] == np.timedelta64(1, "h")
    assert days[1] != hours[1]


# --- adapter assembly ------------------------------------------------------


def _raster(value):
    return xr.Dataset(
        {"precip": (("latitude", "longitude"), np.full((2, 2), float(value)))},
        coords={"latitude": [0.0, 1.0], "longitude": [30.0, 31.0]},
    )


@pytest.fixture
def stub_opens(monkeypatch):
    """Serve a raster per URL, valued by the day-of-month in its filename."""
    calls = []

    def fake_open(url, region, variable=None, fill_value=None, timestamp=None):
        calls.append(url)
        day = int(url.split(".")[-2])
        return _raster(day)

    monkeypatch.setattr("acmaddl.adapters.http._open_raster", fake_open)
    return calls


def _fetch(config, **kwargs):
    return HTTPAdapter().fetch_data(
        {"source_url": BASE, "format": "tif", "_verbose": False, "_progress": False,
         "variables": {"precip": {"native_name": "precip"}}, **config},
        "precip", **kwargs,
    )


def test_adapter_builds_an_init_time_by_lead_time_cube(stub_opens):
    got = _fetch({"issuance": DAILY, "_init_dates": ["2026-07-05", "2026-07-08"]})
    assert got.sizes == {"init_time": 2, "lead_time": 3, "latitude": 2, "longitude": 2}
    assert list(got.init_time.values.astype("datetime64[D]").astype(str)) == [
        "2026-07-05", "2026-07-08"
    ]
    np.testing.assert_array_equal(
        got.lead_time.values, lead_timedelta([0, 1, 2], "days")
    )


def test_adapter_places_each_file_at_its_own_init_and_lead(stub_opens):
    got = _fetch({"issuance": DAILY, "_init_dates": ["2026-07-05"]})
    # The stub values each raster with its valid day-of-month: 5, 6, 7.
    values = got.precip.isel(init_time=0, latitude=0, longitude=0).values
    np.testing.assert_array_equal(values, [5.0, 6.0, 7.0])


def test_adapter_derives_valid_time_from_init_plus_lead(stub_opens):
    got = _fetch({"issuance": DAILY, "_init_dates": ["2026-07-05"]})
    valid = got.valid_time.isel(init_time=0).values.astype("datetime64[D]").astype(str)
    assert list(valid) == ["2026-07-05", "2026-07-06", "2026-07-07"]


def test_adapter_refuses_an_issuance_product_without_an_init(stub_opens):
    with pytest.raises(ValueError, match="issuance-keyed"):
        _fetch({"issuance": DAILY})


def test_adapter_raises_on_a_missing_file_by_default(monkeypatch):
    def fake_open(url, region, variable=None, fill_value=None, timestamp=None):
        if url.endswith("07.tif"):
            raise RuntimeError("404")
        return _raster(1)

    monkeypatch.setattr("acmaddl.adapters.http._open_raster", fake_open)
    with pytest.raises(RuntimeError, match="refusing to return partial data"):
        _fetch({"issuance": DAILY, "_init_dates": ["2026-07-05"]})


def test_allow_partial_keeps_the_lead_axis_aligned(monkeypatch):
    """A missing lead must become NaN in place, not shift every later lead down
    by one — which would silently mislabel a 2-day forecast as a 1-day one."""
    def fake_open(url, region, variable=None, fill_value=None, timestamp=None):
        if url.endswith("06.tif"):
            raise RuntimeError("404")
        return _raster(int(url.split(".")[-2]))

    monkeypatch.setattr("acmaddl.adapters.http._open_raster", fake_open)
    got = _fetch({"issuance": DAILY, "_init_dates": ["2026-07-05"],
                  "_allow_partial": True})
    assert got.sizes["lead_time"] == 3
    values = got.precip.isel(init_time=0, latitude=0, longitude=0).values
    assert values[0] == 5.0 and np.isnan(values[1]) and values[2] == 7.0


def test_adapter_deduplicates_repeated_issuance_dates(stub_opens):
    got = _fetch({"issuance": DAILY, "_init_dates": ["2026-07-05", "2026-07-05"]})
    assert got.sizes["init_time"] == 1
    assert len(stub_opens) == 3  # not 6


def test_plain_time_series_products_are_untouched_by_the_issuance_branch(monkeypatch):
    """The absence of an `issuance` block must leave the original year/month
    file-pattern path exactly as it was."""
    opened = []

    def fake_open(url, region, variable=None, fill_value=None, timestamp=None):
        opened.append(url)
        return _raster(1).expand_dims(time=[np.datetime64("2020-01-01")])

    monkeypatch.setattr("acmaddl.adapters.http._open_cog_subset", fake_open)
    got = _fetch({"file_pattern": "chirps.{year}.tif"}, date_range=(2020, 2020))
    assert opened == [f"{BASE}/chirps.2020.tif"]
    assert "time" in got.dims


# --- fetch() wiring --------------------------------------------------------


def _fake_env(monkeypatch, raw, config):
    fetch_mod = importlib.import_module("acmaddl.fetch")
    seen = {}

    class FakeAdapter:
        def fetch_data(self, cfg, variable, date_range=None, region=None):
            seen["config"] = cfg
            seen["date_range"] = date_range
            return raw

    monkeypatch.setattr(fetch_mod, "get_adapter", lambda name: FakeAdapter())
    monkeypatch.setattr(fetch_mod.catalog, "get", lambda name: config)
    return fetch_mod, seen


ISSUANCE_ENTRY = {
    "adapter": "http",
    "source_url": BASE,
    "format": "tif",
    "issuance": DAILY,
    "variables": {"precip": {"native_name": "precip", "units": "mm/day",
                             "target_units": "mm/day"}},
    "grid": {"lat_res": 0.05, "lon_res": 0.05},
}


def _raw_cube():
    """What the issuance branch of the HTTP adapter really hands to normalize."""
    cube = xr.Dataset(
        {"precip": (("init_time", "lead_time", "latitude", "longitude"),
                    np.ones((1, 3, 2, 2)))},
        coords={
            "init_time": [np.datetime64("2026-07-05")],
            "lead_time": lead_timedelta([0, 1, 2], "days"),
            "latitude": [0.0, 1.0], "longitude": [30.0, 31.0],
        },
    )
    return cube.assign_coords(valid_time=cube.init_time + cube.lead_time)


def test_fetch_threads_a_single_issuance_date_to_the_adapter(monkeypatch):
    fetch_mod, seen = _fake_env(monkeypatch, _raw_cube(), ISSUANCE_ENTRY)
    fetch_mod.fetch("chc/chirps-gefs-daily", "precip", init="2026-07-05",
                    cache=False, verbose=False, progress=False)
    assert seen["config"]["_init_dates"] == ["2026-07-05"]


def test_fetch_accepts_a_sequence_of_issuance_dates(monkeypatch):
    fetch_mod, seen = _fake_env(monkeypatch, _raw_cube(), ISSUANCE_ENTRY)
    inits = [f"{y}-06-30" for y in range(2001, 2005)]
    fetch_mod.fetch("chc/chirps-gefs-daily", "precip", init=inits,
                    cache=False, verbose=False, progress=False)
    assert seen["config"]["_init_dates"] == inits
    assert seen["config"]["init_months"] == [6]
    assert seen["date_range"] == (2001, 2004)


def test_fetch_rejects_a_sequence_for_a_non_issuance_product(monkeypatch):
    entry = {k: v for k, v in ISSUANCE_ENTRY.items() if k != "issuance"}
    fetch_mod, _ = _fake_env(monkeypatch, _raw_cube(), entry)
    with pytest.raises(ValueError, match="accepts a single init"):
        fetch_mod.fetch("obs/whatever", "precip", init=["2026-07-05", "2026-07-06"],
                        cache=False, verbose=False)


def test_fetch_rejects_a_month_keyed_init_for_an_issuance_product(monkeypatch):
    fetch_mod, _ = _fake_env(monkeypatch, _raw_cube(), ISSUANCE_ENTRY)
    with pytest.raises(ValueError, match="full 'YYYY-MM-DD' init date"):
        fetch_mod.fetch("chc/chirps-gefs-daily", "precip", init="2026-07",
                        cache=False, verbose=False)


def test_fetch_rejects_target_combined_with_an_issuance_sequence(monkeypatch):
    fetch_mod, _ = _fake_env(monkeypatch, _raw_cube(), ISSUANCE_ENTRY)
    with pytest.raises(ValueError, match="cannot be combined with a sequence"):
        fetch_mod.fetch("chc/chirps-gefs-daily", "precip",
                        init=["2026-07-05", "2026-07-06"], target="JJA",
                        cache=False, verbose=False)


def test_fetch_rejects_an_empty_issuance_sequence(monkeypatch):
    fetch_mod, _ = _fake_env(monkeypatch, _raw_cube(), ISSUANCE_ENTRY)
    with pytest.raises(ValueError, match="empty sequence"):
        fetch_mod.fetch("chc/chirps-gefs-daily", "precip", init=[],
                        cache=False, verbose=False)


def test_normalize_preserves_the_forecast_coordinates(monkeypatch):
    fetch_mod, _ = _fake_env(monkeypatch, _raw_cube(), ISSUANCE_ENTRY)
    got = fetch_mod.fetch("chc/chirps-gefs-daily", "precip", init="2026-07-05",
                          cache=False, verbose=False, progress=False)
    assert got.sizes["init_time"] == 1 and got.sizes["lead_time"] == 3
    assert "lat" in got.dims and "lon" in got.dims
    # normalize maps `valid_time` onto the canonical observational name.
    assert "time" in got.coords
    assert got.time.dims == ("init_time", "lead_time")


# --- the catalog entries ---------------------------------------------------


@pytest.mark.parametrize("product", ["chc/chirps-gefs-daily", "chc/chirps-gefs-15day"])
def test_chirps_gefs_entries_declare_a_valid_issuance_block(product):
    entry = _catalog[product]
    assert entry["adapter"] == "http"
    assert entry["format"] == "tif"
    cfg = issuance_config(entry)
    assert cfg is not None and cfg["leads"]
    assert entry["variables"]["precip"]["fill_value"] == -9999


def test_chirps_gefs_daily_spans_sixteen_leads():
    cfg = issuance_config(_catalog["chc/chirps-gefs-daily"])
    assert cfg["leads"] == list(range(16))


def test_chirps_gefs_daily_urls_match_the_live_layout():
    """Pinned against a real listing of the CHC server (probed 2026-07-09):
    .../v3/daily/global/2026/07/05/c3g_2026.07.05.tif ... c3g_2026.07.20.tif"""
    entry = _catalog["chc/chirps-gefs-daily"]
    files = enumerate_files(entry["source_url"], issuance_config(entry), ["2026-07-05"])
    assert files[0].url == (
        "https://data.chc.ucsb.edu/products/CHIRPS-GEFS/v3/daily/global/"
        "2026/07/05/c3g_2026.07.05.tif"
    )
    assert files[-1].url.endswith("2026/07/05/c3g_2026.07.20.tif")


def test_chirps_gefs_15day_is_one_file_per_issuance():
    entry = _catalog["chc/chirps-gefs-15day"]
    files = enumerate_files(entry["source_url"], issuance_config(entry), ["2026-01-04"])
    assert len(files) == 1
    assert files[0].url.endswith("15_day/global/data/2026/c3g_2026.01.04.tif")


def test_chirps_gefs_coverage_excludes_2020():
    """GEFSv12's operational stream starts September 2020, so CHC publishes no
    complete 2020. The catalog must not claim otherwise."""
    grid = _catalog["chc/chirps-gefs-daily"]["grid"]
    assert grid["hindcast_range"] == [2001, 2019]
    assert grid["forecast_range"] == [2021, None]


# --- month pruning for plain time-series products --------------------------
# Not an issuance feature, but the same file-enumeration code path.


def test_month_pruning_downloads_only_the_requested_months(monkeypatch):
    """A JJAS analysis needs four months of every year, not twelve. Downloading
    the other eight and discarding them triples the requests against a
    rate-limited server."""
    opened = []

    def fake_open(url, region, variable=None, fill_value=None, timestamp=None):
        opened.append(url)
        return _raster(1).expand_dims(time=[np.datetime64("2020-01-01")])

    monkeypatch.setattr("acmaddl.adapters.http._open_cog_subset", fake_open)
    _fetch({"file_pattern": "p.{year}.{month:02d}.tif", "init_months": [6, 7, 8, 9]},
           date_range=(2001, 2002))
    assert len(opened) == 8
    assert all(url[-6:-4] in ("06", "07", "08", "09") for url in opened)


def test_without_month_pruning_every_month_is_downloaded(monkeypatch):
    opened = []

    def fake_open(url, region, variable=None, fill_value=None, timestamp=None):
        opened.append(url)
        return _raster(1).expand_dims(time=[np.datetime64("2020-01-01")])

    monkeypatch.setattr("acmaddl.adapters.http._open_cog_subset", fake_open)
    _fetch({"file_pattern": "p.{year}.{month:02d}.tif"}, date_range=(2001, 2002))
    assert len(opened) == 24


def test_month_pruning_does_not_touch_year_only_patterns(monkeypatch):
    opened = []

    def fake_open(url, region, variable=None, fill_value=None, timestamp=None):
        opened.append(url)
        return _raster(1).expand_dims(time=[np.datetime64("2020-01-01")])

    monkeypatch.setattr("acmaddl.adapters.http._open_cog_subset", fake_open)
    _fetch({"file_pattern": "p.{year}.tif", "init_months": [6]}, date_range=(2001, 2002))
    assert len(opened) == 2


OBS_ENTRY = {
    "adapter": "http",
    "source_url": BASE,
    "format": "cog",
    "file_pattern": "p.{year}.{month:02d}.cog",
    "variables": {"precip": {"native_name": "precip", "units": "mm/month",
                             "target_units": "mm/day"}},
    "grid": {"lat_res": 0.05, "lon_res": 0.05},
}


def _raw_monthly():
    stamps = [np.datetime64(f"2001-{m:02d}-01") for m in (6, 7, 8, 9)]
    return xr.Dataset(
        {"precip": (("time", "latitude", "longitude"), np.ones((4, 2, 2)))},
        coords={"time": stamps, "latitude": [0.0, 1.0], "longitude": [30.0, 31.0]},
    )


def test_fetch_months_reaches_the_adapter_as_init_months(monkeypatch):
    fetch_mod, seen = _fake_env(monkeypatch, _raw_monthly(), OBS_ENTRY)
    fetch_mod.fetch("obs/x", "precip", hindcast=(2001, 2001), months=[9, 6, 7, 8],
                    cache=False, verbose=False, progress=False)
    assert seen["config"]["init_months"] == [6, 7, 8, 9]


def test_fetch_months_rejects_out_of_range_or_empty_values(monkeypatch):
    fetch_mod, _ = _fake_env(monkeypatch, _raw_monthly(), OBS_ENTRY)
    for bad in ([0], [13], []):
        with pytest.raises(ValueError, match="calendar months in 1..12"):
            fetch_mod.fetch("obs/x", "precip", hindcast=(2001, 2001), months=bad,
                            cache=False, verbose=False)


def test_fetch_months_is_rejected_alongside_init(monkeypatch):
    fetch_mod, _ = _fake_env(monkeypatch, _raw_cube(), ISSUANCE_ENTRY)
    with pytest.raises(ValueError, match="no meaning alongside init"):
        fetch_mod.fetch("chc/chirps-gefs-daily", "precip", init="2026-07-05",
                        months=[6], cache=False, verbose=False)


def test_fetch_months_keys_the_cache_without_a_new_cache_argument():
    """`months` rides the existing `init_months` cache slot, so no fetch already
    on disk is invalidated, and two different month sets do not collide."""
    import inspect

    # `from acmaddl import fetch` gives the function; the module is one level in.
    fetch_mod = importlib.import_module("acmaddl.fetch")

    args = inspect.signature(fetch_mod._fetch_raw_cached).parameters
    assert "init_months" in args
    assert "months" not in args
