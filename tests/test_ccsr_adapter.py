"""CCSR successor OPeNDAP adapter (acmaddl #14 / §6.2).

The Columbia CCSR site replaced the sunset IRI Data Library. It serves the
NMME models (SPEAR, CanSIPS-IC4, ...) over plain OPeNDAP with a different
convention from IRI:
  - the variable is part of the path (no Ingrid `/.<var>/dods` suffix)
  - S (init) and target are a CF "<unit> since 1960-01-01" axis; most models
    use "hours since", but some (SPEAR, CanSIPS-IC4) use "days since"
  - L is an integer 0..11 lead (L=0 == the init month), with a self-describing
    `target[S,L]` variable giving each (init, lead)'s target month
  - longitude is 0..360

These tests exercise the adapter's coordinate decoding + `target`-driven season
selection on a synthetic CCSR-shaped dataset (no network).
"""
import numpy as np
import pytest
import xarray as xr


def _hours_since_1960(dt64):
    return ((dt64.astype("datetime64[h]") - np.datetime64("1960-01-01", "h"))
            / np.timedelta64(1, "h")).astype("int64")


def _synthetic_ccsr(init_month=2, init_years=(1991, 1992, 1993), n_leads=6,
                    members=3, native="prcp"):
    """Mimic a CCSR hindcast variable dataset: prcp[S, M, L, Y, X] + target[S, L]."""
    S_dt = np.array([np.datetime64(f"{y:04d}-{init_month:02d}-01") for y in init_years])
    S = _hours_since_1960(S_dt)
    L = np.arange(n_leads)
    M = np.arange(1, members + 1)
    Y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])           # ascending lat
    X = np.array([28.0, 30.0, 35.0, 40.0, 200.0, 359.0])  # 0..360 lon (incl. >180)

    # target[S, L] = init month + L months (L=0 is the init month).
    target = np.empty((len(S), n_leads), dtype="int64")
    for i, y in enumerate(init_years):
        for l in range(n_leads):
            mo = init_month + l
            yy = y + (mo - 1) // 12
            mm = (mo - 1) % 12 + 1
            target[i, l] = _hours_since_1960(np.datetime64(f"{yy:04d}-{mm:02d}-01"))

    rng = np.random.default_rng(0)
    data = rng.standard_normal((len(S), members, n_leads, len(Y), len(X))).astype("float32")
    ds = xr.Dataset(
        {native: (["S", "M", "L", "Y", "X"], data),
         "target": (["S", "L"], target)},
        coords={"S": S, "M": M, "L": L, "Y": Y, "X": X},
    )
    ds["S"].attrs["units"] = "hours since 1960-01-01"
    ds["target"].attrs["units"] = "hours since 1960-01-01"
    return ds


def _mam_config(native="prcp", variable="precip"):
    from datetime import datetime
    return {
        "adapter": "ccsr",
        "source_url": "https://example/NMME/NOAA-GFDL/SPEAR/hindcast",
        "variables": {variable: {"native_name": native, "units": "mm/day",
                                 "target_units": "mm/day"}},
        "init_months": [2],                                  # Feb init
        "target_range": (datetime(1991, 3, 1), datetime(1991, 5, 31)),  # MAM
        "_verbose": False,
    }


def test_adapter_registered():
    from acmaddl.adapters import get_adapter
    from acmaddl.adapters.ccsr import CCSRAdapter
    assert isinstance(get_adapter("ccsr"), CCSRAdapter)


def test_target_driven_season_selection_means_mam_leads():
    """For a Feb init, MAM = target months {3,4,5} = leads L∈{1,2,3}; the
    adapter must select exactly those leads and average them."""
    from acmaddl.adapters.ccsr import CCSRAdapter
    ds = _synthetic_ccsr(init_month=2, n_leads=6)
    out = CCSRAdapter()._process(ds.copy(deep=True), "prcp", _mam_config(),
                                 date_range=(1991, 1993), region=None)
    # L collapsed (season-averaged), S/M/Y/X retained.
    assert "L" not in out.dims
    # Expected mean over leads 1,2,3 of the original data. Compare reduced over
    # the spatial+member axes so the comparison is independent of the adapter's
    # longitude re-ordering (0..360 -> ±180 + sortby), which has its own test.
    expected = ds["prcp"].isel(L=[1, 2, 3]).mean("L").mean(["M", "Y", "X"])
    got = out["prcp"].mean(["M", "Y", "X"])
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-5)
    # Selecting the wrong leads (e.g. all 6) would give a different mean:
    wrong = ds["prcp"].mean("L").mean(["M", "Y", "X"])
    assert not np.allclose(got.values, wrong.values)


def test_S_decoded_to_datetime_and_year_filtered():
    from acmaddl.adapters.ccsr import CCSRAdapter
    ds = _synthetic_ccsr(init_years=(1991, 1992, 1993))
    out = CCSRAdapter()._process(ds.copy(deep=True), "prcp", _mam_config(),
                                 date_range=(1992, 1993), region=None)
    assert np.issubdtype(out["S"].dtype, np.datetime64)
    assert list(out["S"].dt.year.values) == [1992, 1993]


def test_longitude_wrapped_to_pm180_and_region_cropped():
    from acmaddl.adapters.ccsr import CCSRAdapter
    ds = _synthetic_ccsr()
    out = CCSRAdapter()._process(ds.copy(deep=True), "prcp", _mam_config(),
                                 date_range=(1991, 1993), region=[-1, 1, 29, 41])
    assert float(out["X"].min()) >= 29.0 and float(out["X"].max()) <= 41.0
    assert float(out["Y"].min()) >= -1.0 and float(out["Y"].max()) <= 1.0
    # the 200/359 (->-160/-1) lons are cropped out of the 29..41 window
    assert float(out["X"].max()) <= 41.0


def test_health_check_config_and_missing_url():
    from acmaddl.adapters.ccsr import CCSRAdapter
    a = CCSRAdapter()
    ok = a.health_check({"source_url": "https://example/x", "variables": {}}, probe_remote=False)
    assert ok["healthy"] is True and ok["kind"] == "config"
    bad = a.health_check({"variables": {}}, probe_remote=False)
    assert bad["healthy"] is False


def test_decode_time_since_handles_hours_days_and_months():
    """Regression: the CCSR server serves different models with different time
    units. Most use 'hours since 1960-01-01' (IRI convention), but SPEAR and
    CanSIPS-IC4 use 'days since 1960-01-01'. The decoder must handle both (plus
    months) rather than assuming hours."""
    from acmaddl.adapters.ccsr import _decode_time_since

    hours = _decode_time_since("hours since 1960-01-01", np.array([0, 24, 48]))
    assert hours.astype("datetime64[D]")[2] == np.datetime64("1960-01-03")

    days = _decode_time_since("days since 1960-01-01", np.array([0, 31, 60]))
    assert days.astype("datetime64[D]")[1] == np.datetime64("1960-02-01")

    months = _decode_time_since("months since 1960-01-01", np.array([0, 1, 12]))
    assert months.astype("datetime64[M]")[2] == np.datetime64("1961-01")

    with pytest.raises(ValueError, match="since"):
        _decode_time_since("parsecs since 1960-01-01", np.array([0]))


def _capture_url(monkeypatch):
    """Patch the adapter's xr.open_dataset to record the opened URL and return a
    synthetic CCSR dataset, so URL routing can be tested without network."""
    from acmaddl.adapters import ccsr as ccsr_mod
    captured = {}

    def fake_open(url, **kwargs):
        captured["url"] = url
        return _synthetic_ccsr(native="prcp")

    monkeypatch.setattr(ccsr_mod.xr, "open_dataset", fake_open)
    return captured


def test_split_streams_routes_to_forecast_or_hindcast(monkeypatch):
    """split_streams entries pick the forecast/ vs hindcast/ subdir from the
    requested years (past the hindcast range -> forecast)."""
    from acmaddl.adapters.ccsr import CCSRAdapter
    captured = _capture_url(monkeypatch)
    cfg = {
        "adapter": "ccsr", "split_streams": True,
        "source_url": "https://x/NMME/NOAA-GFDL/SPEAR",
        "variables": {"precip": {"native_name": "prcp", "units": "mm", "target_units": "mm"}},
        "grid": {"hindcast_range": [1991, 2020]},
        "init_months": [2], "_verbose": False,
    }
    CCSRAdapter().fetch_data(cfg, "precip", date_range=(2012, 2013), region=None)
    assert captured["url"].endswith("/NOAA-GFDL/SPEAR/hindcast/prcp")
    CCSRAdapter().fetch_data(cfg, "precip", date_range=(2026, 2026), region=None)
    assert captured["url"].endswith("/NOAA-GFDL/SPEAR/forecast/prcp")


def test_split_streams_support_stream_specific_variable_paths(monkeypatch):
    """CanSIPS precipitation uses /hindcast/prcp but /forecast/pr."""
    from acmaddl.adapters import ccsr as ccsr_mod
    from acmaddl.adapters.ccsr import CCSRAdapter
    captured = {}

    def fake_open(url, **kwargs):
        captured["url"] = url
        return _synthetic_ccsr(native="pr")

    monkeypatch.setattr(ccsr_mod.xr, "open_dataset", fake_open)
    cfg = {
        "adapter": "ccsr", "split_streams": True,
        "source_url": "https://x/NMME/ECCC/CanSIPS-IC4",
        "variables": {
            "precip": {
                "native_name": "pr",
                "path_name": {"hindcast": "prcp", "forecast": "pr"},
                "units": "mm", "target_units": "mm",
            }
        },
        "grid": {"hindcast_range": [1990, 2024]},
        "init_months": [9], "_verbose": False,
    }
    CCSRAdapter().fetch_data(cfg, "precip", date_range=(2016, 2016), region=None)
    assert captured["url"].endswith("/ECCC/CanSIPS-IC4/hindcast/prcp")
    CCSRAdapter().fetch_data(cfg, "precip", date_range=(2025, 2025), region=None)
    assert captured["url"].endswith("/ECCC/CanSIPS-IC4/forecast/pr")


def test_combined_stream_has_no_subdir(monkeypatch):
    """Without split_streams (CCSM4), the variable hangs directly off the base url."""
    from acmaddl.adapters.ccsr import CCSRAdapter
    captured = _capture_url(monkeypatch)
    cfg = {
        "adapter": "ccsr",
        "source_url": "https://x/NMME/COLA-RSMAS/CCSM4",
        "variables": {"precip": {"native_name": "prcp", "units": "mm", "target_units": "mm"}},
        "grid": {"hindcast_range": [1982, 2026]},
        "init_months": [2], "_verbose": False,
    }
    CCSRAdapter().fetch_data(cfg, "precip", date_range=(2012, 2013), region=None)
    assert captured["url"].endswith("/COLA-RSMAS/CCSM4/prcp")


def test_single_year_fetch_opens_once_per_year(monkeypatch):
    from acmaddl.adapters import ccsr as ccsr_mod
    from acmaddl.adapters.ccsr import CCSRAdapter
    opened = []

    def fake_open(url, **kw):
        opened.append(url)
        # full multi-year dataset each call; adapter must slice per year
        return _synthetic_ccsr(init_month=2, init_years=(1991, 1992, 1993), native="prcp")

    monkeypatch.setattr(ccsr_mod.xr, "open_dataset", fake_open)
    cfg = {
        "adapter": "ccsr", "single_year_fetch": True,
        "source_url": "https://x/NMME/NOAA-GFDL/SPEAR",
        "variables": {"precip": {"native_name": "prcp", "units": "mm", "target_units": "mm"}},
        "grid": {"hindcast_range": [1991, 2020]},
        "init_months": [2], "_verbose": False,
    }
    out = CCSRAdapter().fetch_data(cfg, "precip", date_range=(1991, 1993), region=None)
    assert len(opened) == 3                     # one open per year
    assert out.sizes["S"] == 3                  # 3 init times concatenated


def test_single_year_fetch_routes_each_year_to_its_own_stream(monkeypatch):
    """A split_streams + single_year_fetch request spanning the hindcast/forecast
    boundary must route each year to its own subdir (year > hindcast_range[1] ->
    forecast), not pick one subdir from date_range[0] for the whole loop. The
    boundary year's other-stream years must not be silently dropped."""
    from acmaddl.adapters import ccsr as ccsr_mod
    from acmaddl.adapters.ccsr import CCSRAdapter
    opened = []

    def fake_open(url, **kw):
        opened.append(url)
        # Each stream only serves its own years: hindcast has 2020, forecast 2021.
        year = (2020,) if "hindcast" in url else (2021,)
        return _synthetic_ccsr(init_month=2, init_years=year, native="prcp")

    monkeypatch.setattr(ccsr_mod.xr, "open_dataset", fake_open)
    cfg = {
        "adapter": "ccsr", "split_streams": True, "single_year_fetch": True,
        "source_url": "https://x/NMME/NOAA-GFDL/SPEAR",
        "variables": {"precip": {"native_name": "prcp", "units": "mm", "target_units": "mm"}},
        "grid": {"hindcast_range": [1991, 2020]},
        "init_months": [2], "_verbose": False,
    }
    out = CCSRAdapter().fetch_data(cfg, "precip", date_range=(2020, 2021), region=None)
    assert any(u.endswith("/SPEAR/hindcast/prcp") for u in opened)   # 2020 -> hindcast
    assert any(u.endswith("/SPEAR/forecast/prcp") for u in opened)   # 2021 -> forecast
    assert out.sizes["S"] == 2                                       # neither year dropped


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.parametrize("product,variable,members,region", [
    ("nmme/spear-hindcast", "precip", 15, [-2, 2, 34, 40]),
    ("nmme/spearb-hindcast", "sst", 15, [-5, 5, 55, 70]),      # ocean bbox for SST
    ("nmme/cansipsic4-hindcast", "precip", 40, [-2, 2, 34, 40]),
])
def test_ccsr_hindcast_entries_live_fetch(product, variable, members, region):
    """End-to-end against the real CCSR server for each wired hindcast entry:
    decode + Feb-init filter + target-driven MAM lead selection + normalize."""
    import acmaddl
    ds = acmaddl.fetch(product, variable, init="2016-02", target="MAM",
                       hindcast=(2014, 2016), region=region, cache=False, verbose=False)
    da = ds[variable] if variable in getattr(ds, "data_vars", {}) else ds
    assert ds.sizes.get("member") == members
    assert "L" not in da.dims                       # season leads collapsed
    assert float(np.isfinite(da).mean()) > 0.5


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.parametrize("product", ["nmme/spear", "nmme/spearb", "nmme/cansipsic4"])
def test_ccsr_forecast_paths_reachable(product):
    """The CCSR forecast endpoints are live (catalog source_url opens)."""
    from acmaddl import catalog
    from acmaddl.adapters import get_adapter
    entry = catalog.info(product)
    result = get_adapter(entry["adapter"]).health_check(entry, probe_remote=True)
    assert result["healthy"] is True, result


@pytest.mark.parametrize(
    "product",
    ["nmme/ccsm4", "nmme/cesm1", "nmme/geoss2s", "nmme/spear", "nmme/cansipsic4"],
)
def test_ccsr_temp_units_celsius_not_double_converted(product):
    """Regression (temp units bug): CCSR serves NMME 2 m temperature in Celsius
    (variable renamed t2m -> tas in the 2026-07 CMIP renaming). The catalog must
    declare units=C so normalize() does not subtract 273.15 a second time and
    produce sub-absolute-zero values (the bug produced mean ~ -280 C for
    physical ~ -7 C).
    """
    from acmaddl import catalog
    from acmaddl.normalize import normalize

    cfg = catalog.info(product)
    assert cfg["adapter"] == "ccsr"
    # A raw CCSR temp field: physical Celsius values, no units attribute.
    raw = xr.Dataset(
        {"tas": (("lat", "lon"), np.array([[-7.0, -5.0], [0.0, 3.0]], dtype=float))},
        coords={"lat": [54.0, 55.0], "lon": [-106.0, -105.0]},
    )
    out = normalize(raw, cfg, "temp")
    vals = out["temp"].values
    assert out["temp"].attrs["units"] == "C"
    # No spurious Kelvin subtraction: values stay physical and unchanged.
    assert float(np.nanmin(vals)) > -60.0 and float(np.nanmax(vals)) < 60.0
    np.testing.assert_allclose(
        np.sort(vals.ravel()), [-7.0, -5.0, 0.0, 3.0], atol=1e-9
    )
