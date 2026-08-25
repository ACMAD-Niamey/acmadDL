"""acmaddl.obs_predictor: observed field as a seasonal CCA predictor.

Added on the `acmad` branch. Uses a monkeypatched fetch (no network) to check the
canonical (year, member, lat, lon) shaping and the hindcast/forecast-year split.
"""
import importlib
import numpy as np
import xarray as xr
import acmaddl

assemble_mod = importlib.import_module("acmaddl.assemble")  # the module, not the fn


def _fake_fetch(product, variable, *, hindcast, **kw):
    """Stand in for fetch(seasonal='mean'): a (year, lat, lon) obs season field."""
    y0, y1 = hindcast
    years = np.arange(y0, y1 + 1)
    lat = np.array([-2.0, 0.0, 2.0])
    lon = np.array([10.0, 12.0])
    data = np.random.default_rng(0).random((len(years), len(lat), len(lon)))
    da = xr.DataArray(data, dims=("year", "lat", "lon"),
                      coords={"year": years, "lat": lat, "lon": lon}, name=variable)
    return xr.Dataset({variable: da})


def test_obs_predictor_shapes_and_forecast_split(monkeypatch):
    monkeypatch.setattr(assemble_mod, "fetch", _fake_fetch)
    hcst, fcst = acmaddl.obs_predictor(
        "obs/ersst-v5", "sst", target="ASO",
        hindcast=(1991, 2020), forecast_year=2026, region=[-35, 35, 0, 360])

    # Canonical predictor shape, ready for deepscale.seasonal_mme tracks.
    assert hcst.dims == ("year", "member", "lat", "lon")
    assert fcst.dims == ("year", "member", "lat", "lon")
    # Hindcast spans the training window; forecast is the single forecast year.
    assert list(hcst.year.values) == list(range(1991, 2021))
    assert list(fcst.year.values) == [2026]
    assert hcst.sizes["member"] == 1 and fcst.sizes["member"] == 1


def test_obs_predictor_accepts_months_form(monkeypatch):
    # ACMAD Exp-1: a single initialisation month as predictor, months=[6], no season code.
    captured = {}

    def _fake(product, variable, *, hindcast, **kw):
        captured.update(kw)
        return _fake_fetch(product, variable, hindcast=hindcast, **kw)

    monkeypatch.setattr(assemble_mod, "fetch", _fake)
    hcst, fcst = acmaddl.obs_predictor(
        "obs/ersst-v5", "sst", months=[6],
        hindcast=(1991, 2020), forecast_year=2026, region=[-35, 35, 0, 360])
    assert captured.get("months") == [6] and captured.get("target") is None
    assert hcst.dims == ("year", "member", "lat", "lon")
    assert list(fcst.year.values) == [2026]


def test_obs_predictor_requires_exactly_one_of_target_or_months():
    import pytest
    with pytest.raises(ValueError, match="exactly one of"):
        acmaddl.obs_predictor("obs/ersst-v5", "sst", hindcast=(1991, 2020), forecast_year=2026)
    with pytest.raises(ValueError, match="exactly one of"):
        acmaddl.obs_predictor("obs/ersst-v5", "sst", target="ASO", months=[6],
                              hindcast=(1991, 2020), forecast_year=2026)
