"""Real-data integration tests for the weather-skills/* products.

Each test runs one Rhiza skill for real through acmaddl.fetch over a small
East-Africa box with the smallest sensible request, and asserts the normalized
contract. Marked integration+network; skipped in the default unit run.

    uv run pytest -m "integration and not cds" tests/test_weather_skills_integration.py
    uv run pytest -m "integration and cds"     tests/test_weather_skills_integration.py   # ECDS key

Earthdata-backed products (imerg-daily, smap-daily) skip unless an Earthdata
Login token is set (EARTHDATA_TOKEN or ~/.earthdatarc).
"""
import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

import acmaddl
from acmaddl.adapters import weather_skills

pytestmark = [pytest.mark.integration, pytest.mark.network]

REGION = [-2, 2, 36, 40]
YESTERDAY = date.today() - timedelta(days=1)


def _earthdata():
    return bool(os.environ.get("EARTHDATA_TOKEN")) or (Path.home() / ".earthdatarc").exists()


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
    for k in ("weather_skills_history", "weather_skills_name", "weather_skills_version", "weather_skills_pin"):
        assert k in ds.attrs


def _check_obs(ds, var, units):
    assert set(ds[var].dims) == {"time", "lat", "lon"}
    assert ds[var].attrs["units"] == units
    assert np.isfinite(ds[var].values).any()
    assert "weather_skills_history" in ds.attrs and "weather_skills_name" in ds.attrs


def test_ifs_ens_15d_precip():
    init = _latest("weather-skills/ifs-ens-15d")
    ds = acmaddl.fetch("weather-skills/ifs-ens-15d", "precip", init=init, region=REGION, cache=False)
    _check_forecast(ds, "precip", "mm/day")
    assert ds.sizes["member"] == 51 and 0 in ds["member"].values


def test_gfs_temp_is_deterministic():
    init = _latest("weather-skills/gfs")
    ds = acmaddl.fetch("weather-skills/gfs", "temp", init=init, region=REGION, cache=False)
    _check_forecast(ds, "temp", "C")
    assert ds.sizes.get("member", 1) == 1


def test_gefs_35d_member_count():
    init = _latest("weather-skills/gefs-35d")
    ds = acmaddl.fetch("weather-skills/gefs-35d", "precip", init=init, region=REGION, cache=False)
    _check_forecast(ds, "precip", "mm/day")
    assert ds.sizes["member"] == 31


def test_gefs_analysis_trailing_window():
    # No hindcast: the trailing 10 days ending at the product's published latest day.
    ds = acmaddl.fetch("weather-skills/gefs-analysis", "precip", region=REGION, cache=False)
    _check_obs(ds, "precip", "mm/day")
    assert ds.sizes["time"] >= 2 * 8          # 3-hourly, at least two full days


def test_imerg_late_half_hourly_trailing_window():
    ds = acmaddl.fetch("weather-skills/imerg-late-30min", "precip", region=REGION, cache=False)
    _check_obs(ds, "precip", "mm/day")
    assert ds.sizes["time"] >= 2 * 48         # half-hourly, at least two full days


def test_subc_mme_7d_sum_and_anomaly():
    init = _latest("weather-skills/subc-mme-7d")
    ds = acmaddl.fetch("weather-skills/subc-mme-7d", "precip", init=init, region=REGION, cache=False)
    _check_forecast(ds, "precip", "mm")
    assert "pr_anomaly" in ds.data_vars
    assert ds.sizes["lead_time"] == 1


def test_chirps_daily_two_chunks(monkeypatch):
    end = date.fromisoformat(_latest("weather-skills/chirps-daily"))
    # Days 1-2 of the latest month in 1-day chunks: two real skill runs, concatenated.
    from acmaddl import catalog
    monkeypatch.setattr(weather_skills, "_today", lambda: date(end.year, end.month, 3))
    cfg = catalog.info("weather-skills/chirps-daily") | {"window_days": 1}
    ds = weather_skills.WeatherSkillsAdapter().fetch_data(cfg | {"init_months": [end.month]}, "precip",
                                         date_range=(end.year, end.year), region=REGION)
    assert ds.sizes["time"] == 2 and ds.attrs["weather_skills_name"] == "chirps-fetch"
    assert float(ds["latitude"].max()) <= REGION[1] + 0.1   # cropped before concat


@pytest.mark.skipif(not _earthdata(), reason="no Earthdata credentials")
def test_imerg_daily_two_days():
    ds = acmaddl.fetch("weather-skills/imerg-daily", "precip", region=REGION, cache=False)   # trailing window
    _check_obs(ds, "precip", "mm/day")


def test_era5_temp_two_days():
    y = YESTERDAY - timedelta(days=10)
    ds = acmaddl.fetch("weather-skills/era5", "temp", hindcast=(y.year, y.year), months=[y.month], region=REGION, cache=False)
    _check_obs(ds, "temp", "C")


def test_oisst_daily():
    y = YESTERDAY - timedelta(days=10)
    ds = acmaddl.fetch("weather-skills/oisst-daily", "sst", hindcast=(y.year, y.year), months=[y.month], region=[-10, 0, 40, 50], cache=False)
    _check_obs(ds, "sst", "C")


@pytest.mark.skipif(not _earthdata(), reason="no Earthdata credentials")
def test_smap_daily():
    ds = acmaddl.fetch("weather-skills/smap-daily", "soil_moisture", region=REGION, cache=False)
    _check_obs(ds, "soil_moisture", "m3/m3")


def test_cmip6_two_months_in_the_future():
    # Monthly data: the skill needs at least two time points to stamp an interval.
    ds = acmaddl.fetch("weather-skills/cmip6", "temp", hindcast=(2030, 2030), months=[1, 2], region=REGION, cache=False)
    _check_obs(ds, "temp", "C")
    assert ds.sizes["time"] == 2


@pytest.mark.cds
def test_ecmwf_s2s_precip():
    init = (date.today() - timedelta(days=4)).isoformat()
    ds = acmaddl.fetch("weather-skills/ecmwf-s2s", "precip", init=init, region=REGION, cache=False)
    _check_forecast(ds, "precip", "mm/day")
    assert ds.sizes["member"] == 101 and 0 in ds["member"].values


def test_runner_round_trip_on_a_live_product():
    """acmadDL fetch -> their clip-region via the runner == acmadDL's own crop; and
    their provenance tool reads the chain fetch -> acmaddl -> clip-region."""
    from acmaddl import weather_skills as ws
    init = _latest("weather-skills/ifs-ens-15d")
    ds = acmaddl.fetch("weather-skills/ifs-ens-15d", "precip", init=init, region=[-4, 4, 34, 42], cache=False)
    clipped = ws.run("clip-region", ds, bbox=[-2, 2, 36, 40])
    ours = ds.sel(lat=slice(-2, 2), lon=slice(36, 40))
    assert clipped.sizes["lat"] == ours.sizes["lat"] and clipped.sizes["lon"] == ours.sizes["lon"]
    np.testing.assert_allclose(clipped["precip"].transpose(*ours["precip"].dims).values, ours["precip"].values)
    chain = json.loads(ws.run("provenance", clipped, format="json"))
    chain = chain if isinstance(chain, list) else next(iter(chain.values()))
    assert [e["skill"] for e in chain] == ["dynamical-fetch", "acmaddl", "clip-region"]
