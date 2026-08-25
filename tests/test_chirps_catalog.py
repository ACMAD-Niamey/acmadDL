"""CHIRPS catalog naming convention + entry-shape unit tests (offline).

Naming convention: obs/chirps-{version}-{cadence}[-rhiza]
  - Native UCSB products carry version + cadence and use the http adapter.
  - Rhiza Research / Sheerwater mirrors end in -rhiza and use the sheerwater
    adapter.

Real-data fetches for these products live in tests/test_integration.py.
"""
import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from acmaddl import catalog

# The full expected CHIRPS product set after the naming rework.
NATIVE = [
    "obs/chirps-v2-monthly", "obs/chirps-v2-daily", "obs/chirps-v2-pentad",
    "obs/chirps-v2-dekad", "obs/chirps-v2-annual",
    "obs/chirps-v3-monthly", "obs/chirps-v3-daily", "obs/chirps-v3-pentad",
    "obs/chirps-v3-dekad", "obs/chirps-v3-annual",
]
MIRRORS = [
    "obs/chirps-v3-daily-rhiza", "obs/chirps-v2-dekadal-rhiza", "obs/chirps-live-rhiza",
]
ALL_CHIRPS = NATIVE + MIRRORS

# Names retired by the rework — must no longer resolve.
RETIRED = [
    "obs/chirps", "obs/chirps-direct", "obs/chirps-v2",
    "obs/chirps-dekadal", "obs/chirps-live",
]

NATIVE_FORMAT = {
    "monthly": "cog", "daily": "netcdf", "pentad": "netcdf",
    "dekad": "netcdf", "annual": "tif",
}


def test_all_expected_chirps_products_present():
    for prod in ALL_CHIRPS:
        assert catalog.info(prod) is not None, f"missing {prod}"


def test_retired_chirps_names_gone():
    for old in RETIRED:
        with pytest.raises(Exception):
            catalog.info(old)


def test_naming_convention_rhiza_iff_mirror():
    """A name ending in -rhiza must be a sheerwater mirror; otherwise native http."""
    for prod in ALL_CHIRPS:
        e = catalog.info(prod)
        if prod.endswith("-rhiza"):
            assert e["adapter"] == "sheerwater", f"{prod} should be a sheerwater mirror"
        else:
            assert e["adapter"] == "http", f"{prod} should be native UCSB (http)"


def test_native_names_match_version_cadence_pattern():
    pat = re.compile(r"^obs/chirps-v[23]-(monthly|daily|pentad|dekad|annual)$")
    for prod in NATIVE:
        assert pat.match(prod), f"{prod} breaks the version-cadence convention"


@pytest.mark.parametrize("prod", NATIVE)
def test_native_format_matches_cadence(prod):
    cadence = prod.rsplit("-", 1)[1]
    e = catalog.info(prod)
    assert e["format"] == NATIVE_FORMAT[cadence]
    assert e["source_url"].startswith("https://data.chc.ucsb.edu/")


def test_period_cadences_are_not_relabelled_as_daily_rate():
    """pentad/dekad/annual are accumulations; they must NOT claim mm/day, which
    normalize would apply without dividing (silent wrongness)."""
    for prod in ["obs/chirps-v2-pentad", "obs/chirps-v2-dekad", "obs/chirps-v2-annual",
                 "obs/chirps-v3-pentad", "obs/chirps-v3-dekad", "obs/chirps-v3-annual"]:
        v = catalog.info(prod)["variables"]["precip"]
        assert v["units"] == v["target_units"], f"{prod} should not convert units"
        assert v["target_units"] != "mm/day", f"{prod} must not claim a daily rate"


def test_monthly_converts_to_daily_rate():
    for prod in ["obs/chirps-v2-monthly", "obs/chirps-v3-monthly"]:
        v = catalog.info(prod)["variables"]["precip"]
        assert v["units"] == "mm/month" and v["target_units"] == "mm/day"


def test_v3_daily_rhiza_documents_coverage_floor():
    """The mirror only covers 2000+, unlike native v3 which starts 1981."""
    e = catalog.info("obs/chirps-v3-daily-rhiza")
    assert e["source"] == "chirps_v3"
    assert e["grid"]["hindcast_range"][0] == 2000


@pytest.mark.parametrize("prod", NATIVE)
def test_native_chirps_declare_request_interval(prod):
    """Native UCSB CHIRPS entries hit data.chc.ucsb.edu, which CrowdSec-bans an IP
    that exceeds ~2 req/s (4-hour 403). Each native entry must declare a positive
    request_interval floor so multi-file pulls are paced by default."""
    ri = catalog.info(prod).get("request_interval")
    assert isinstance(ri, (int, float)) and ri > 0, \
        f"{prod} must declare a request_interval throttle floor"


@pytest.mark.parametrize("prod", [p for p in NATIVE if NATIVE_FORMAT[p.rsplit("-", 1)[1]] == "netcdf"])
def test_native_chirps_netcdf_cap_concurrent_connections(prod):
    """The per-year NetCDF products download whole files (some many GiB) in
    parallel; they must cap concurrent connections via max_workers."""
    mw = catalog.info(prod).get("max_workers")
    assert isinstance(mw, int) and 1 <= mw <= 8, \
        f"{prod} must cap concurrent connections with max_workers in 1..8"


@pytest.mark.parametrize("prod", NATIVE)
def test_native_config_health_ok(prod):
    from acmaddl.adapters import get_adapter
    e = catalog.info(prod)
    result = get_adapter(e["adapter"]).health_check(e, probe_remote=False)
    assert result["healthy"] is True
    assert result["kind"] == "config"


# ── Adapter: annual year-only time-stamping (the new http.py code path) ──────

def _write_tiny_raster(path):
    import rioxarray  # noqa: F401  registers the .rio accessor
    da = xr.DataArray(
        np.ones((3, 3), dtype="float32"),
        dims=("y", "x"),
        coords={"y": [2.0, 1.0, 0.0], "x": [0.0, 1.0, 2.0]},
    ).rio.write_crs("EPSG:4326")
    da.rio.to_raster(path)


def test_open_raster_stamps_year_only_annual_filename(tmp_path):
    """Annual rasters (chirps-...-{year}.tif) carry no month; the adapter must
    stamp January 1 of that year rather than dropping the time coord."""
    from acmaddl.adapters.http import _open_cog_subset
    p = tmp_path / "chirps-v2.0.2017.tif"
    _write_tiny_raster(str(p))
    ds = _open_cog_subset(str(p), region=None, variable="precip")
    assert "time" in ds.dims
    assert pd.Timestamp(ds["time"].values[0]) == pd.Timestamp("2017-01-01")


def test_open_raster_still_stamps_month_for_monthly_filename(tmp_path):
    """Regression: the YYYY.MM monthly stamp must keep working unchanged."""
    from acmaddl.adapters.http import _open_cog_subset
    p = tmp_path / "chirps-v3.0.2017.05.tif"
    _write_tiny_raster(str(p))
    ds = _open_cog_subset(str(p), region=None, variable="precip")
    assert pd.Timestamp(ds["time"].values[0]) == pd.Timestamp("2017-05-01")
