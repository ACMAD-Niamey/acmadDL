"""GPCP v2.3 monthly precipitation from NOAA NCEI (`http` adapter).

NCEI stamps each filename with its processing date
(`gpcp_v02r03_monthly_d202403_c20240607.nc`), so the entry uses a `*` in
`file_pattern` and the adapter resolves it against the directory listing. The
resolution mechanics are tested in tests/test_http_adapter.py.
"""
import fnmatch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from acmaddl import catalog
from acmaddl.adapters import get_adapter
from acmaddl.normalize import normalize

PRODUCT = "obs/gpcp-v2-3"


def test_served_over_http_from_ncei():
    e = catalog.info(PRODUCT)
    assert e["adapter"] == "http"
    assert e["source_url"].startswith("https://www.ncei.noaa.gov/data/")
    assert "gpcp-monthly" in e["source_url"]
    assert e["format"] == "netcdf"


def test_the_processing_date_suffix_is_a_wildcard():
    pattern = catalog.info(PRODUCT)["file_pattern"]
    assert "*" in pattern, "the _c<processing date> suffix cannot be predicted"
    assert pattern.format(year=2024, month=3) == \
        "2024/gpcp_v02r03_monthly_d202403_*.nc"


def test_the_pattern_matches_a_real_final_filename():
    pattern = catalog.info(PRODUCT)["file_pattern"].format(year=2024, month=3)
    assert fnmatch.fnmatch("gpcp_v02r03_monthly_d202403_c20240607.nc",
                           pattern.split("/")[-1])


def test_the_pattern_excludes_the_preliminary_stream():
    """NCEI keeps gpcp_v02r03-preliminary_monthly_… in the same directory.

    This entry is the final, settled product; matching a preliminary file would
    quietly splice a provisional estimate into the record.
    """
    pattern = catalog.info(PRODUCT)["file_pattern"].format(year=2026, month=6)
    assert not fnmatch.fnmatch(
        "gpcp_v02r03-preliminary_monthly_d202606_c20260710.nc",
        pattern.split("/")[-1])


def test_two_point_five_degree_monthly_precip_only():
    e = catalog.info(PRODUCT)
    assert set(e["variables"]) == {"precip"}
    assert e["grid"]["lat_res"] == 2.5 and e["grid"]["lon_res"] == 2.5
    assert e["grid"]["temporal"] == "monthly"


def test_native_rate_is_kept_as_mm_per_day():
    """Matches obs/cmap, the other 2.5° merged-precip product."""
    v = catalog.info(PRODUCT)["variables"]["precip"]
    assert v["native_name"] == "precip"
    assert v["units"] == "mm/day" and v["target_units"] == "mm/day"


def test_record_starts_in_1979():
    assert catalog.info(PRODUCT)["grid"]["hindcast_range"][0] == 1979


def test_config_health_ok():
    e = catalog.info(PRODUCT)
    result = get_adapter(e["adapter"]).health_check(e, probe_remote=False)
    assert result["healthy"] is True and result["kind"] == "config"


def _synthetic_gpcp():
    """NCEI's layout: latitude/longitude (not lat/lon), lon in 0..360."""
    times = pd.date_range("2020-01-01", "2020-12-01", freq="MS")
    lat = np.arange(-88.75, 90.0, 2.5)
    lon = np.arange(1.25, 360.0, 2.5)
    data = np.full((len(times), len(lat), len(lon)), 3.0, dtype="float32")
    return xr.Dataset(
        {"precip": (["time", "latitude", "longitude"], data),
         "precip_error": (["time", "latitude", "longitude"], data * 0.1)},
        coords={"time": times.values.astype("datetime64[ns]"),
                "latitude": lat, "longitude": lon},
    )


def test_normalize_renames_coords_and_keeps_the_daily_rate():
    out = normalize(_synthetic_gpcp(), catalog.info(PRODUCT), "precip")
    assert "lat" in out.dims and "lon" in out.dims
    assert out["precip"].attrs["units"] == "mm/day"
    np.testing.assert_allclose(float(out["precip"].mean()), 3.0)
