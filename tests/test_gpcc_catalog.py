"""GPCC monthly gauge analyses from DWD's open-data server (`http` adapter).

Two streams, same shape and grid:

* `obs/gpcc-monitoring-v2020` — the quality-controlled monitoring product, 1982-.
* `obs/gpcc-first-guess` — the near-real-time first guess, NetCDF from 2013-.

Both are gzipped NetCDF with a time axis that the default opener cannot use, so
both declare `time_from_pattern`. The adapter mechanics behind that live in
tests/test_http_adapter.py; here we pin the catalog entries and the
normalization contract.
"""
import fnmatch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from acmaddl import catalog
from acmaddl.adapters import get_adapter
from acmaddl.normalize import normalize

MONITORING = "obs/gpcc-monitoring-v2020"
FIRST_GUESS = "obs/gpcc-first-guess"
BOTH = (MONITORING, FIRST_GUESS)


# ── Entry shape ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("product", BOTH)
def test_served_over_http_from_dwd_open_data(product):
    e = catalog.info(product)
    assert e["adapter"] == "http"
    assert e["source_url"].startswith("https://opendata.dwd.de/climate_environment/GPCC")
    assert e["format"] == "netcdf"


@pytest.mark.parametrize("product", BOTH)
def test_files_are_gzipped_netcdf_per_year_and_month(product):
    pattern = catalog.info(product)["file_pattern"]
    assert "{year}" in pattern and "{month" in pattern
    assert pattern.endswith(".nc.gz"), pattern


@pytest.mark.parametrize("product", BOTH)
def test_time_comes_from_the_filename_not_the_file(product):
    """Both DWD encodings defeat the default opener — see _stamp_time."""
    assert catalog.info(product)["time_from_pattern"] is True


@pytest.mark.parametrize("product", BOTH)
def test_one_degree_monthly_precip_only(product):
    e = catalog.info(product)
    assert set(e["variables"]) == {"precip"}
    assert e["grid"]["lat_res"] == 1.0 and e["grid"]["lon_res"] == 1.0
    assert e["grid"]["temporal"] == "monthly"


@pytest.mark.parametrize("product", BOTH)
def test_native_totals_are_kept_as_mm_per_month(product):
    """Follows obs/tamsat: native monthly totals are not divided into a rate.

    The ("mm/month", "mm/day") conversion in normalize divides by a flat 30,
    which no calendar month has; keeping the source's own totals avoids
    inventing precision the data doesn't have.
    """
    v = catalog.info(product)["variables"]["precip"]
    assert v["native_name"] == "p"
    assert v["units"] == "mm/month" and v["target_units"] == "mm/month"


def test_monitoring_covers_the_full_record_from_1982():
    assert catalog.info(MONITORING)["grid"]["hindcast_range"][0] == 1982


def test_first_guess_starts_in_2013_when_the_netcdf_stream_starts():
    """Probed 2026-09-02: the year directories exist from 2004, but 2004-2012
    hold only legacy GrADS binaries (gpcc_first_guess_MM_YYYY.gz) — the
    .nc.gz files this entry fetches begin in 2013. Declaring 2004 would
    promise nine years the entry cannot deliver."""
    assert catalog.info(FIRST_GUESS)["grid"]["hindcast_range"][0] == 2013


@pytest.mark.parametrize("product", BOTH)
def test_config_health_ok(product):
    e = catalog.info(product)
    result = get_adapter(e["adapter"]).health_check(e, probe_remote=False)
    assert result["healthy"] is True and result["kind"] == "config"


def test_the_two_streams_do_not_share_a_source_url():
    assert (catalog.info(MONITORING)["source_url"]
            != catalog.info(FIRST_GUESS)["source_url"])


def test_the_patterns_match_the_filenames_dwd_actually_serves():
    """Probed filenames, so a pattern typo fails here instead of at fetch time."""
    real = {
        MONITORING: "monitoring_v2020_10_2024_03.nc.gz",
        FIRST_GUESS: "first_guess_monthly_2024_03.nc.gz",
    }
    for product, name in real.items():
        pattern = catalog.info(product)["file_pattern"]
        built = pattern.format(year=2024, month=3)
        assert built == f"2024/{name}", built


# ── Through normalize on a GPCC-shaped dataset ───────────────────────────────

def _synthetic_gpcc():
    """DWD's layout: var `p`, lat descending (N->S), lon in -180..180."""
    times = pd.date_range("2020-01-01", "2020-12-01", freq="MS")
    lat = np.arange(89.5, -90.0, -1.0)
    lon = np.arange(-179.5, 180.0, 1.0)
    data = np.full((len(times), len(lat), len(lon)), 100.0, dtype="float32")
    return xr.Dataset(
        {"p": (["time", "lat", "lon"], data)},
        coords={"time": times.values.astype("datetime64[ns]"), "lat": lat, "lon": lon},
    )


@pytest.mark.parametrize("product", BOTH)
def test_normalize_renames_p_and_keeps_monthly_totals(product):
    out = normalize(_synthetic_gpcc(), catalog.info(product), "precip")
    assert "p" not in out and "precip" in out
    assert out["precip"].attrs["units"] == "mm/month"
    np.testing.assert_allclose(float(out["precip"].mean()), 100.0)


@pytest.mark.parametrize("product", BOTH)
def test_normalize_sorts_dwd_latitude_ascending(product):
    out = normalize(_synthetic_gpcc(), catalog.info(product), "precip")
    assert out["lat"].values[0] < out["lat"].values[-1]
