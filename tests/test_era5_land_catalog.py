"""obs/era5-land-monthly catalog entry: offline unit tests.

ERA5-Land monthly is the higher-resolution (0.1°) land sibling of obs/era5
(single-levels, 0.25°), used as the reference/truth for scoring downscaled
precip (e.g. the SaWaM SEAS5-BCSD product). It rides the existing CDS adapter
(product_type monthly_averaged_reanalysis) with no adapter changes.

Live CDS fetches for this product live in tests/test_integration.py.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from acmaddl import catalog
from acmaddl.adapters import get_adapter
from acmaddl.normalize import normalize

PRODUCT = "obs/era5-land-monthly"


# ── Entry shape ──────────────────────────────────────────────────────────────

def test_entry_present_and_cds():
    e = catalog.info(PRODUCT)
    assert e["adapter"] == "cds"
    assert e["cds_dataset"] == "reanalysis-era5-land-monthly-means"
    assert e["product_type"] == "monthly_averaged_reanalysis"


def test_grid_is_point_one_degree():
    g = catalog.info(PRODUCT)["grid"]
    assert g["lat_res"] == 0.1 and g["lon_res"] == 0.1


def test_higher_resolution_than_single_level_era5():
    """The land product must be finer than obs/era5 (0.25°) — its reason to exist."""
    land = catalog.info(PRODUCT)["grid"]
    base = catalog.info("obs/era5")["grid"]
    assert land["lat_res"] < base["lat_res"]
    assert land["lon_res"] < base["lon_res"]


def test_precip_and_temp_declare_their_cds_names():
    """The variable SET is pinned by test_pev_is_a_third_variable_… below."""
    v = catalog.info(PRODUCT)["variables"]
    assert v["precip"]["native_name"] == "total_precipitation"
    assert v["precip"]["short_name"] == "tp"
    assert v["temp"]["native_name"] == "2m_temperature"
    assert v["temp"]["short_name"] == "t2m"


def test_land_only_no_sst():
    """ERA5-Land is land-only; unlike obs/era5 it must not advertise sst."""
    assert "sst" not in catalog.info(PRODUCT)["variables"]


def test_precip_declares_metres_to_daily_rate():
    v = catalog.info(PRODUCT)["variables"]["precip"]
    assert v["units"] == "m" and v["target_units"] == "mm/day"


def test_temp_declares_kelvin_to_celsius():
    v = catalog.info(PRODUCT)["variables"]["temp"]
    assert v["units"] == "K" and v["target_units"] == "C"


def test_config_health_ok():
    e = catalog.info(PRODUCT)
    result = get_adapter(e["adapter"]).health_check(e, probe_remote=False)
    assert result["healthy"] is True
    assert result["kind"] == "config"


def test_obs_era5_left_untouched():
    """Regression: adding the land entry must not disturb the legacy obs/era5."""
    e = catalog.info("obs/era5")
    assert e["cds_dataset"] == "reanalysis-era5-single-levels-monthly-means"
    assert e["grid"]["lat_res"] == 0.25 and e["grid"]["lon_res"] == 0.25


# ── End-to-end through normalize on a synthetic ERA5-Land-shaped dataset ─────
# CDS delivers variables under their short_name (tp, t2m), so the synthetic
# fixture uses those — exercising normalize's short_name fallback together with
# the real catalog unit conversions (m→mm/day ×1000, K→C).

def _synthetic_era5_land():
    times = pd.date_range("2019-01-01", "2019-12-01", freq="MS")
    lat = np.round(np.arange(6.1, 21.0, 0.1), 1)
    lon = np.round(np.arange(31.0, 41.0, 0.1), 1)
    shape = (len(times), len(lat), len(lon))
    return xr.Dataset(
        {
            "tp": (["time", "latitude", "longitude"],
                   np.full(shape, 0.005, dtype="float32")),   # 0.005 m → 5 mm/day
            "t2m": (["time", "latitude", "longitude"],
                    np.full(shape, 300.0, dtype="float32")),   # 300 K → 26.85 C
        },
        coords={"time": times.values.astype("datetime64[ns]"),
                "latitude": lat, "longitude": lon},
    )


def test_normalize_precip_metres_to_mm_per_day():
    out = normalize(_synthetic_era5_land(), catalog.info(PRODUCT), "precip")
    assert out["precip"].attrs["units"] == "mm/day"
    assert "tp" not in out
    np.testing.assert_allclose(float(out["precip"].mean()), 5.0, atol=1e-4)
    assert "lat" in out.dims and "lon" in out.dims


def test_normalize_temp_kelvin_to_celsius():
    out = normalize(_synthetic_era5_land(), catalog.info(PRODUCT), "temp")
    assert out["temp"].attrs["units"] == "C"
    # The converted field is uniform 26.85 C (min == max). Aggregate in float64:
    # under numpy 2.x scalar promotion the K->C result stays float32, and a
    # float32 mean over the ~180k-cell grid accumulates enough rounding error to
    # drift past a 0.01 tolerance even though every cell is exact.
    assert float(out["temp"].min()) == pytest.approx(26.85, abs=0.01)
    assert float(out["temp"].max()) == pytest.approx(26.85, abs=0.01)
    np.testing.assert_allclose(float(out["temp"].astype("float64").mean()), 26.85, atol=0.01)


# ── Potential evaporation ────────────────────────────────────────────────────
# CDS serves ERA5-Land pev in metres with a NEGATIVE sign (ERA5's flux
# convention: negative = upward = evaporation). Verified live 2026-09-02 for
# July 2020 over 5-20 N / 10 W-10 E: min -0.0230, mean -0.0090, max -0.0025 m.

def test_pev_is_a_third_variable_alongside_precip_and_temp():
    assert set(catalog.info(PRODUCT)["variables"]) == {"precip", "temp", "pev"}


def test_pev_declares_the_cds_name_and_short_name():
    v = catalog.info(PRODUCT)["variables"]["pev"]
    assert v["native_name"] == "potential_evaporation"
    assert v["short_name"] == "pev"


def test_pev_converts_metres_to_a_positive_daily_rate():
    """Same m -> mm/day convention as precip, plus the sign flip."""
    v = catalog.info(PRODUCT)["variables"]["pev"]
    assert v["units"] == "m" and v["target_units"] == "mm/day"
    assert v["scale"] == -1.0


def _synthetic_era5_land_pev():
    times = pd.date_range("2020-07-01", periods=1, freq="MS")
    lat = np.round(np.arange(5.0, 20.0, 0.1), 1)
    lon = np.round(np.arange(-10.0, 10.0, 0.1), 1)
    shape = (len(times), len(lat), len(lon))
    return xr.Dataset(
        {"pev": (["valid_time", "latitude", "longitude"],
                 np.full(shape, -0.009, dtype="float32"))},
        coords={"valid_time": times.values.astype("datetime64[ns]"),
                "latitude": lat, "longitude": lon},
    )


def test_normalize_returns_pev_as_a_positive_mm_per_day_rate():
    out = normalize(_synthetic_era5_land_pev(), catalog.info(PRODUCT), "pev")
    assert out["pev"].attrs["units"] == "mm/day"
    val = float(out["pev"].astype("float64").mean())
    assert val > 0, f"pev should be a positive evaporative demand, got {val}"
    np.testing.assert_allclose(val, 9.0, atol=1e-3)


def test_pev_does_not_disturb_the_precip_conversion():
    """Regression: the scale knob is per-variable, not per-product."""
    out = normalize(_synthetic_era5_land(), catalog.info(PRODUCT), "precip")
    np.testing.assert_allclose(float(out["precip"].mean()), 5.0, atol=1e-4)
