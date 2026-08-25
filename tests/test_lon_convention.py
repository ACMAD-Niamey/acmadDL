"""Offline tests for longitude-convention-robust selection and netCDF round-trip.

Covers two footguns fixed together:
  * select_lon: a bbox in one longitude convention (-180..180 vs 0..360) against a source
    in the other silently under-selected (e.g. slice(-180,180) on a 0..358 grid returned
    only 0..180). Now translated + wrap-safe.
  * sanitize_for_netcdf: OPeNDAP/CF datasets carried bounds vars + stale encoding that made
    to_netcdf raise "NetCDF: String match to name in use". Now rebuilt to round-trip.
"""
import numpy as np
import pytest
import xarray as xr

from acmaddl.normalize import select_lon, sanitize_for_netcdf


def _ds(lons, name="lon"):
    lons = np.asarray(lons, dtype=float)
    return xr.Dataset({"v": (name, np.arange(lons.size, dtype=float))},
                      coords={name: (name, lons)})


@pytest.fixture
def ds360():
    return _ds(np.arange(0, 360, 2.0))


@pytest.fixture
def ds180():
    return _ds(np.arange(-180, 180, 2.0))


def _span(ds, name="lon"):
    return float(ds[name].min()), float(ds[name].max()), ds.sizes[name]


def test_global_request_keeps_all_360(ds360):
    lo, hi, n = _span(select_lon(ds360, -180, 180))
    assert (lo, hi, n) == (0.0, 358.0, 180)


def test_global_request_keeps_all_180(ds180):
    lo, hi, n = _span(select_lon(ds180, 0, 360))
    assert n == 180 and lo == -180.0


def test_negative_box_on_360_source_translates(ds360):
    # Nino3.4 requested as -170..-120 -> 190..240 on a 0..360 grid
    lo, hi, _ = _span(select_lon(ds360, -170, -120))
    assert (lo, hi) == (190.0, 240.0)


def test_over180_box_on_180_source_translates(ds180):
    lo, hi, _ = _span(select_lon(ds180, 190, 240))
    assert (lo, hi) == (-170.0, -120.0)


def test_seam_wrapping_box_360(ds360):
    # Atlantic -70..20 wraps the 0/360 seam -> both sides present
    out = select_lon(ds360, -70, 20)
    lons = out.lon.values
    assert (lons < 30).any() and (lons > 280).any()


def test_same_convention_unchanged(ds360):
    lo, hi, _ = _span(select_lon(ds360, 120, 160))
    assert (lo, hi) == (120.0, 160.0)


def test_empty_lon_is_safe(ds360):
    empty = ds360.isel(lon=slice(0, 0))
    assert select_lon(empty, 10, 20).sizes["lon"] == 0


def test_sanitize_roundtrips_with_bounds_and_encoding(tmp_path):
    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), np.random.rand(3, 4, 5)),
         "time_bnds": (("time", "nbnds"), np.zeros((3, 2)))},
        coords={"time": np.arange(3), "lat": np.arange(4.0), "lon": np.arange(5.0)},
    )
    ds["sst"].encoding = {"source": "remote", "original_shape": (9, 9, 9)}
    ds["time"].attrs["bounds"] = "time_bnds"
    clean = sanitize_for_netcdf(ds)
    assert "time_bnds" not in clean.variables and "nbnds" not in clean.dims
    p = tmp_path / "rt.nc"
    clean.to_netcdf(p)  # must not raise
    reopened = xr.open_dataset(p)
    assert np.allclose(reopened["sst"].values, ds["sst"].values)


def test_sanitize_preserves_forecast_dims(tmp_path):
    fc = xr.Dataset(
        {"precip": (("year", "member", "lat", "lon"), np.random.rand(3, 2, 4, 5))},
        coords={"year": [2000, 2001, 2002], "member": [0, 1],
                "lat": np.arange(4.0), "lon": np.arange(5.0)},
    )
    clean = sanitize_for_netcdf(fc)
    assert dict(clean.sizes) == {"year": 3, "member": 2, "lat": 4, "lon": 5}
    p = tmp_path / "fc.nc"
    clean.to_netcdf(p)
    assert np.allclose(xr.open_dataset(p)["precip"].values, fc["precip"].values)
