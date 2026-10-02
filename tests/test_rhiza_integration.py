"""Real-data integration tests for the rhiza/* products.

Each test runs one Rhiza skill for real through acmaddl.fetch over a small
East-Africa box with the smallest sensible request, and asserts the normalized
contract. Marked integration+network; skipped in the default unit run.

    uv run pytest -m "integration and not cds" tests/test_rhiza_integration.py
    uv run pytest -m "integration and cds"     tests/test_rhiza_integration.py   # ECDS key

Earthdata-backed products (imerg-daily, smap-daily) skip unless
EARTHDATA_USERNAME is set or ~/.netrc has an urs.earthdata.nasa.gov entry.
"""
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

import acmaddl
from acmaddl.adapters import rhiza

pytestmark = [pytest.mark.integration, pytest.mark.network]

REGION = [-2, 2, 36, 40]
YESTERDAY = date.today() - timedelta(days=1)


def _earthdata():
    rc = Path.home() / ".netrc"
    return bool(os.environ.get("EARTHDATA_USERNAME")) or (rc.exists() and "urs.earthdata.nasa.gov" in rc.read_text())


def _latest(product):
    r = acmaddl.check_product(product, probe_remote=True)
    assert r["healthy"], r
    return r["latest"]


def _check_forecast(ds, var, units):
    assert set(ds[var].dims) >= {"init_time", "lead_time", "lat", "lon"}
    assert np.issubdtype(ds["lead_time"].dtype, np.timedelta64)
    assert ds[var].attrs["units"] == units
    assert np.isfinite(ds[var].values).any()
    assert float(ds.lat.min()) >= REGION[0] - 1 and float(ds.lat.max()) <= REGION[1] + 1
    for k in ("weather_skills_history", "rhiza_skill", "rhiza_skill_version", "rhiza_pin"):
        assert k in ds.attrs


def _check_obs(ds, var, units):
    assert set(ds[var].dims) == {"time", "lat", "lon"}
    assert ds[var].attrs["units"] == units
    assert np.isfinite(ds[var].values).any()
    assert "weather_skills_history" in ds.attrs and "rhiza_skill" in ds.attrs


def test_ifs_ens_15d_precip():
    init = _latest("rhiza/ifs-ens-15d")
    ds = acmaddl.fetch("rhiza/ifs-ens-15d", "precip", init=init, region=REGION, cache=False)
    _check_forecast(ds, "precip", "mm/day")
    assert ds.sizes["member"] == 51 and 0 in ds["member"].values


def test_gfs_temp_is_deterministic():
    init = _latest("rhiza/gfs")
    ds = acmaddl.fetch("rhiza/gfs", "temp", init=init, region=REGION, cache=False)
    _check_forecast(ds, "temp", "C")
    assert ds.sizes.get("member", 1) == 1


def test_gefs_35d_member_count():
    init = _latest("rhiza/gefs-35d")
    ds = acmaddl.fetch("rhiza/gefs-35d", "precip", init=init, region=REGION, cache=False)
    _check_forecast(ds, "precip", "mm/day")
    assert ds.sizes["member"] == 31


def test_gefs_analysis_trailing_window():
    # No hindcast: the trailing 10 days ending at the product's published latest day.
    ds = acmaddl.fetch("rhiza/gefs-analysis", "precip", region=REGION, cache=False)
    _check_obs(ds, "precip", "mm/day")
    assert ds.sizes["time"] >= 2 * 8          # 3-hourly, at least two full days


def test_imerg_late_half_hourly_trailing_window():
    ds = acmaddl.fetch("rhiza/imerg-late-30min", "precip", region=REGION, cache=False)
    _check_obs(ds, "precip", "mm/day")
    assert ds.sizes["time"] >= 2 * 48         # half-hourly, at least two full days


def test_subc_mme_7d_sum_and_anomaly():
    init = _latest("rhiza/subc-mme-7d")
    ds = acmaddl.fetch("rhiza/subc-mme-7d", "precip", init=init, region=REGION, cache=False)
    _check_forecast(ds, "precip", "mm")
    assert "pr_anomaly" in ds.data_vars
    assert ds.sizes["lead_time"] == 1


def test_chirps_daily_two_chunks(monkeypatch):
    end = date.fromisoformat(_latest("rhiza/chirps-daily"))
    # Days 1-2 of the latest month in 1-day chunks: two real skill runs, concatenated.
    from acmaddl import catalog
    monkeypatch.setattr(rhiza, "_today", lambda: date(end.year, end.month, 3))
    cfg = catalog.info("rhiza/chirps-daily") | {"window_days": 1}
    ds = rhiza.RhizaAdapter().fetch_data(cfg | {"init_months": [end.month]}, "precip",
                                         date_range=(end.year, end.year), region=REGION)
    assert ds.sizes["time"] == 2 and ds.attrs["rhiza_skill"] == "chirps-fetch"
    assert float(ds["latitude"].max()) <= REGION[1] + 0.1   # cropped before concat


@pytest.mark.skipif(not _earthdata(), reason="no Earthdata credentials")
def test_imerg_daily_two_days():
    ds = acmaddl.fetch("rhiza/imerg-daily", "precip", region=REGION, cache=False)   # trailing window
    _check_obs(ds, "precip", "mm/day")


def test_era5_temp_two_days():
    y = YESTERDAY - timedelta(days=10)
    ds = acmaddl.fetch("rhiza/era5", "temp", hindcast=(y.year, y.year), months=[y.month], region=REGION, cache=False)
    _check_obs(ds, "temp", "C")


def test_oisst_daily():
    y = YESTERDAY - timedelta(days=10)
    ds = acmaddl.fetch("rhiza/oisst-daily", "sst", hindcast=(y.year, y.year), months=[y.month], region=[-10, 0, 40, 50], cache=False)
    _check_obs(ds, "sst", "C")


@pytest.mark.skipif(not _earthdata(), reason="no Earthdata credentials")
def test_smap_daily():
    ds = acmaddl.fetch("rhiza/smap-daily", "soil_moisture", region=REGION, cache=False)
    _check_obs(ds, "soil_moisture", "m3/m3")


def test_cmip6_two_months_in_the_future():
    # Monthly data: the skill needs at least two time points to stamp an interval.
    ds = acmaddl.fetch("rhiza/cmip6", "temp", hindcast=(2030, 2030), months=[1, 2], region=REGION, cache=False)
    _check_obs(ds, "temp", "C")
    assert ds.sizes["time"] == 2


@pytest.mark.cds
def test_ecmwf_s2s_precip():
    init = (date.today() - timedelta(days=4)).isoformat()
    ds = acmaddl.fetch("rhiza/ecmwf-s2s", "precip", init=init, region=REGION, cache=False)
    _check_forecast(ds, "precip", "mm/day")
    assert ds.sizes["member"] == 101 and 0 in ds["member"].values
