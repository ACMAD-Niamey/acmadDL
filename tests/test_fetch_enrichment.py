import numpy as np, xarray as xr, pytest, acmaddl


def _toy_monthly():  # 3 years of monthly precip on a coarse grid
    t = xr.cftime_range("1993-01", periods=36, freq="MS", calendar="standard")
    lat, lon = np.arange(-2, 3.0, 1.0), np.arange(20, 25.0, 1.0)
    data = np.ones((36, lat.size, lon.size))
    return xr.DataArray(data, dims=("time", "lat", "lon"),
                        coords={"time": t, "lat": lat, "lon": lon})


def _patch_fetch_internals(monkeypatch):
    # `acmaddl.fetch` is shadowed by the re-exported `fetch()` function (see
    # acmaddl/__init__.py: `from .fetch import fetch`), so reach the actual
    # submodule via sys.modules to monkeypatch its real module-level names.
    import sys
    import acmaddl.fetch as _  # ensure imported
    fetchmod = sys.modules["acmaddl.fetch"]
    monkeypatch.setattr(fetchmod, "_fetch_raw_cached",
                        lambda *a, **k: xr.Dataset({"precip": _toy_monthly()}))
    monkeypatch.setattr(fetchmod, "normalize", lambda ds, *a, **k: ds)
    # cache=False in these toy tests takes the direct-adapter path (not
    # _fetch_raw_cached), so also stub get_adapter(...).fetch_data(...).
    fake_adapter = type("FakeAdapter", (), {
        "fetch_data": staticmethod(lambda *a, **k: xr.Dataset({"precip": _toy_monthly()})),
    })()
    monkeypatch.setattr(fetchmod, "get_adapter", lambda name: fake_adapter)
    return fetchmod


def test_seasonal_mean_collapses_months_to_year(monkeypatch):
    _patch_fetch_internals(monkeypatch)
    ds = acmaddl.fetch("obs/chirps-v2-monthly", "precip", target="MAM",
                       region=[-2, 2, 20, 24], seasonal="mean", cache=False, verbose=False)
    assert "year" in ds["precip"].dims and "time" not in ds["precip"].dims
    assert list(ds["precip"].year.values) == [1993, 1994, 1995]


def test_seasonal_mean_wraparound_shifts_year_and_drops_incomplete(monkeypatch):
    # Monthly data Jan 1993 .. Dec 1995; value encodes year + month/100 so the test
    # verifies BOTH the month selection and the season-year labelling.
    import sys
    t = xr.cftime_range("1993-01", periods=36, freq="MS", calendar="standard")
    lat, lon = np.arange(-2, 3.0, 1.0), np.arange(20, 25.0, 1.0)
    vals = (np.array([tt.year + tt.month / 100.0 for tt in t])[:, None, None]
            * np.ones((1, lat.size, lon.size)))
    toy = xr.DataArray(vals, dims=("time", "lat", "lon"),
                       coords={"time": t, "lat": lat, "lon": lon})
    import acmaddl.fetch as _  # ensure imported
    fetchmod = sys.modules["acmaddl.fetch"]
    monkeypatch.setattr(fetchmod, "_fetch_raw_cached",
                        lambda *a, **k: xr.Dataset({"precip": toy}))
    monkeypatch.setattr(fetchmod, "normalize", lambda ds, *a, **k: ds)
    fake = type("FakeAdapter", (), {
        "fetch_data": staticmethod(lambda *a, **k: xr.Dataset({"precip": toy}))})()
    monkeypatch.setattr(fetchmod, "get_adapter", lambda name: fake)

    ds = acmaddl.fetch("obs/chirps-v2-monthly", "precip", target="NDJ",
                       region=[-2, 2, 20, 24], seasonal="mean", cache=False, verbose=False)
    da = ds["precip"]
    # NDJ = Nov, Dec, Jan; the season is labelled by its Nov/Dec year, so the Jan
    # value shifts back one year. Within Jan1993..Dec1995 only 1993 and 1994 are
    # complete seasons (1992 has only Jan1993; 1995 lacks Jan1996) -> both dropped.
    assert "year" in da.dims and "time" not in da.dims
    assert da.year.dtype.kind == "i"                 # season year stays integer
    assert list(da.year.values) == [1993, 1994]
    exp_1993 = (1993.11 + 1993.12 + 1994.01) / 3.0   # Nov1993, Dec1993, Jan1994
    exp_1994 = (1994.11 + 1994.12 + 1995.01) / 3.0
    np.testing.assert_allclose(float(da.sel(year=1993).mean()), exp_1993, atol=1e-6)
    np.testing.assert_allclose(float(da.sel(year=1994).mean()), exp_1994, atol=1e-6)


def test_grid_res_and_regrid_to_are_mutually_exclusive(monkeypatch):
    _patch_fetch_internals(monkeypatch)
    with pytest.raises(ValueError, match="grid_res.*regrid_to|mutually exclusive"):
        acmaddl.fetch("obs/chirps-v2-monthly", "precip", target="MAM",
                      grid_res=0.5, regrid_to=_toy_monthly(), cache=False, verbose=False)


def test_cover_buffer_clamps_longitude_to_pm180(monkeypatch):
    """A full-globe-longitude region (e.g. the CCA box lon=(-180,180)) in
    cover mode must not buffer past +/-180: fetch.py clamped lat but not
    lon, so the padded bbox (-181.5, 181.5) reached the adapter/cache key
    and CDS returned a degenerate 3-column sliver instead of the full grid.
    """
    import sys
    import acmaddl.fetch as _  # ensure imported
    fetchmod = sys.modules["acmaddl.fetch"]

    captured = {}

    def fake_raw(product, variable, config, date_range, region, *a, **k):
        captured["region"] = region  # this is fetch_bbox
        lat = np.arange(-31, 32.0, 1.0)
        lon = np.arange(-180, 180.0, 1.0)
        t = xr.cftime_range("1993-01", periods=12, freq="MS")
        return xr.Dataset({"precip": xr.DataArray(
            np.ones((12, lat.size, lon.size)), dims=("time", "lat", "lon"),
            coords={"time": t, "lat": lat, "lon": lon})})

    monkeypatch.setattr(fetchmod, "_fetch_raw_cached", fake_raw)
    monkeypatch.setattr(fetchmod, "normalize", lambda ds, *a, **k: ds)

    acmaddl.fetch("c3s/cmcc-sps4", "precip", target="MAM",
                  region=[-30, 30, -180, 180], boundary="cover", region_buffer=1.5,
                  cache=True, verbose=False)

    lon_w, lon_e = captured["region"][2], captured["region"][3]
    assert lon_w >= -180.0 and lon_e <= 180.0, captured["region"]
