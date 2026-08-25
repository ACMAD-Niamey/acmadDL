import importlib

import numpy as np
import xarray as xr

import acmaddl


def _ym(years):  # (year, member, lat, lon) toy
    lat, lon = np.arange(-1, 2.0, 1.0), np.arange(20, 23.0, 1.0)
    return xr.DataArray(np.ones((len(years), 1, lat.size, lon.size)),
        dims=("year", "member", "lat", "lon"),
        coords={"year": years, "member": [0], "lat": lat, "lon": lon})


def test_assemble_pairs_hindcast_and_forecast(monkeypatch):
    # acmaddl.assemble is shadowed by the re-exported `assemble()` function
    # (see acmaddl/__init__.py: `from .assemble import assemble`), so reach
    # the actual submodule via importlib to monkeypatch its real
    # module-level `fetch` name.
    amod = importlib.import_module("acmaddl.assemble")

    calls = []

    def fake_fetch(product, variable, **kw):
        calls.append(kw)
        years = list(range(*[kw["hindcast"][0], kw["hindcast"][1] + 1])) if kw.get("hindcast") else [2026]
        return xr.Dataset({variable: _ym(years)})

    monkeypatch.setattr(amod, "fetch", fake_fetch)
    roster = [("ModelA", "nmme/a", (1993, 1995), (1993, 1997))]
    out = acmaddl.assemble(roster, "precip", init="2026-01", target="MAM", range_index=2)
    assert set(out) == {"ModelA"}
    hcst, fcst = out["ModelA"]
    assert list(hcst.year.values) == [1993, 1994, 1995]
    assert list(fcst.year.values) == [2026]
    # range_index=2 picked the (1993,1995) column, not (1993,1997)
    assert any(c.get("hindcast") == (1993, 1995) for c in calls)


def test_assemble_adds_member_dim_when_fetch_lacks_one(monkeypatch):
    """Regression test: fetch results without native ensemble members (e.g.
    CFSv2) must still come out of assemble() with a `member` dim, since
    downstream consumers (deepscale/methods/cca.py) unconditionally do
    hindcast.mean("member")."""
    amod = importlib.import_module("acmaddl.assemble")

    def _no_member(years):  # (year, lat, lon) toy -- no member dim at all
        lat, lon = np.arange(-1, 2.0, 1.0), np.arange(20, 23.0, 1.0)
        return xr.DataArray(np.ones((len(years), lat.size, lon.size)),
            dims=("year", "lat", "lon"),
            coords={"year": years, "lat": lat, "lon": lon})

    def fake_fetch(product, variable, **kw):
        years = list(range(*[kw["hindcast"][0], kw["hindcast"][1] + 1])) if kw.get("hindcast") else [2026]
        return xr.Dataset({variable: _no_member(years)})

    monkeypatch.setattr(amod, "fetch", fake_fetch)
    roster = [("ModelA", "nmme/a", (1993, 1995), (1993, 1997))]
    out = acmaddl.assemble(roster, "precip", init="2026-01", target="MAM", range_index=2)
    hcst, fcst = out["ModelA"]
    assert "member" in hcst.dims and hcst.sizes["member"] == 1
    assert "member" in fcst.dims and fcst.sizes["member"] == 1
