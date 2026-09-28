"""Unit tests for the CDSAdapter's s2s-forecasts dataset branch.

These tests do not hit the network. They patch cdsapi.Client.retrieve and
assert the request dict and dataset name that the adapter sends. The real
network call is exercised in tests/test_integration.py.
"""

import pytest
import xarray as xr

from acmaddl.adapters.cds import CDSAdapter


def _s2s_product_config(**overrides):
    config = {
        "adapter": "cds",
        "cds_dataset": "s2s-forecasts",
        "cds_model": "ecmwf",
        "forecast_type": "perturbed_forecast",
        "variables": {
            "precip": {
                "native_name": "total_precipitation",
                "short_name": "tp",
                "units": "m",
                "target_units": "mm/day",
                "accumulated": True,
            },
        },
        "grid": {"lat_res": 1.5, "lon_res": 1.5},
        "_verbose": False,
    }
    config.update(overrides)
    return config


class _FakeCDSClient:
    """Captures the dataset name and request dict; writes a stub netCDF."""

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.calls = []

    def retrieve(self, dataset, request, target):
        self.calls.append((dataset, dict(request), target))
        # Write a minimal netCDF so the adapter's xr.open_dataset succeeds.
        import numpy as np
        ds = xr.Dataset(
            {"tp": (["latitude", "longitude"], np.zeros((2, 2), dtype="float32"))},
            coords={"latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
        )
        ds.to_netcdf(target)


def test_s2s_request_uses_origin_not_originating_centre(monkeypatch):
    """The s2s-forecasts dataset takes `origin` (not `originating_centre`)."""
    fake = _FakeCDSClient()
    monkeypatch.setattr("cdsapi.Client", lambda *a, **kw: fake)

    adapter = CDSAdapter()
    config = _s2s_product_config()
    config["_init_date"] = "2026-05-12"  # set by fetch() when init is YYYY-MM-DD
    adapter.fetch_data(config, "precip", region=[-2, 2, 36, 40])

    assert len(fake.calls) == 1
    dataset, request, _ = fake.calls[0]
    assert dataset == "s2s-forecasts"
    assert request["origin"] == "ecmwf"
    assert "originating_centre" not in request


def test_s2s_request_passes_single_day_year_month(monkeypatch):
    """Year, month, day are single-valued strings derived from _init_date."""
    fake = _FakeCDSClient()
    monkeypatch.setattr("cdsapi.Client", lambda *a, **kw: fake)

    adapter = CDSAdapter()
    config = _s2s_product_config()
    config["_init_date"] = "2026-05-12"
    adapter.fetch_data(config, "precip", region=[-2, 2, 36, 40])

    _, request, _ = fake.calls[0]
    assert request["year"] == "2026"
    assert request["month"] == "05"
    assert request["day"] == "12"


def test_s2s_request_includes_forecast_type_and_time(monkeypatch):
    """forecast_type comes from product config; time defaults to 00:00."""
    fake = _FakeCDSClient()
    monkeypatch.setattr("cdsapi.Client", lambda *a, **kw: fake)

    adapter = CDSAdapter()
    config = _s2s_product_config(forecast_type="control_forecast")
    config["_init_date"] = "2026-05-12"
    adapter.fetch_data(config, "precip", region=[-2, 2, 36, 40])

    _, request, _ = fake.calls[0]
    assert request["forecast_type"] == "control_forecast"
    assert request["time"] == "00:00"


def test_s2s_request_uses_mars_range_leadtime_by_default(monkeypatch):
    """leadtime_hour defaults to the Mars-style range '0/to/1104/by/24' (=46 days)."""
    fake = _FakeCDSClient()
    monkeypatch.setattr("cdsapi.Client", lambda *a, **kw: fake)

    adapter = CDSAdapter()
    config = _s2s_product_config()
    config["_init_date"] = "2026-05-12"
    adapter.fetch_data(config, "precip", region=[-2, 2, 36, 40])

    _, request, _ = fake.calls[0]
    assert request["leadtime_hour"] == ["0/to/1104/by/24"]


def test_s2s_request_honors_explicit_leadtime_override(monkeypatch):
    """Explicit leadtime_hour in config overrides the default."""
    fake = _FakeCDSClient()
    monkeypatch.setattr("cdsapi.Client", lambda *a, **kw: fake)

    adapter = CDSAdapter()
    config = _s2s_product_config(leadtime_hour=["24", "48", "72"])
    config["_init_date"] = "2026-05-12"
    adapter.fetch_data(config, "precip", region=[-2, 2, 36, 40])

    _, request, _ = fake.calls[0]
    assert request["leadtime_hour"] == ["24", "48", "72"]


def test_s2s_per_variable_leadtime_beats_product_default(monkeypatch):
    """A variable-level leadtime_hour wins over the product-level default.

    sst is instantaneous: ECDS silently returns only step 24 for the MARS range
    shorthand that the accumulated variables require, so the catalog gives sst an
    explicit list. Regression guard — the failure mode is silent short data, not
    an error.
    """
    fake = _FakeCDSClient()
    monkeypatch.setattr("cdsapi.Client", lambda *a, **kw: fake)

    adapter = CDSAdapter()
    config = _s2s_product_config()
    config["variables"]["sst"] = {
        "native_name": "sea_surface_temperature",
        "units": "K",
        "leadtime_hour": ["0_24", "24_48"],
    }
    config["_init_date"] = "2026-05-12"
    adapter.fetch_data(config, "sst", region=[-2, 2, 36, 40])

    _, request, _ = fake.calls[0]
    assert request["leadtime_hour"] == ["0_24", "24_48"]

    # ...and the accumulated variable in the same product keeps the shorthand.
    fake.calls.clear()
    adapter.fetch_data(config, "precip", region=[-2, 2, 36, 40])
    _, request, _ = fake.calls[0]
    assert request["leadtime_hour"] == ["0/to/1104/by/24"]


def test_s2s_catalog_sst_enumerates_all_46_leads():
    """The shipped catalog gives c3s/ecmwf-s2s sst the full explicit lead list."""
    from acmaddl.catalog import info

    sst = info("c3s/ecmwf-s2s")["variables"]["sst"]
    assert len(sst["leadtime_hour"]) == 46
    assert sst["leadtime_hour"][0] == "0_24"
    assert sst["leadtime_hour"][-1] == "1080_1104"


def test_s2s_client_uses_url_override_when_present(monkeypatch):
    """If cds_url is set in product config, cdsapi.Client receives it."""
    captured = {}

    class _Capture(_FakeCDSClient):
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            super().__init__()

    monkeypatch.setattr("cdsapi.Client", _Capture)

    adapter = CDSAdapter()
    config = _s2s_product_config(cds_url="https://ecds.ecmwf.int/api")
    config["_init_date"] = "2026-05-12"
    adapter.fetch_data(config, "precip", region=[-2, 2, 36, 40])

    assert captured["kwargs"].get("url") == "https://ecds.ecmwf.int/api"


def test_s2s_client_omits_url_when_not_set(monkeypatch):
    """When cds_url is absent, cdsapi.Client is constructed without a url kwarg
    so it falls back to ~/.cdsapirc."""
    captured = {}

    class _Capture(_FakeCDSClient):
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            super().__init__()

    monkeypatch.setattr("cdsapi.Client", _Capture)

    adapter = CDSAdapter()
    config = _s2s_product_config()
    config["_init_date"] = "2026-05-12"
    adapter.fetch_data(config, "precip", region=[-2, 2, 36, 40])

    assert "url" not in captured["kwargs"]


def test_s2s_request_rejects_malformed_init_date(monkeypatch):
    """Malformed _init_date raises a clear ValueError before hitting the network."""
    monkeypatch.setattr("cdsapi.Client", _FakeCDSClient)

    adapter = CDSAdapter()
    for bad in ["2026-05", "2026-5-12", "05/12/2026", "not-a-date", "2026-05-12T00:00"]:
        config = _s2s_product_config()
        config["_init_date"] = bad
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            adapter.fetch_data(config, "precip", region=[-2, 2, 36, 40])


def test_fetch_routes_date_string_init_to_adapter(monkeypatch):
    """fetch(init='YYYY-MM-DD', product='c3s/ecmwf-s2s', ...) sets
    config['_init_date'] before invoking the adapter."""
    import acmaddl

    fake = _FakeCDSClient()
    monkeypatch.setattr("cdsapi.Client", lambda *a, **kw: fake)
    # Skip the nuthatch cache wrapper to make assertions deterministic.
    # The cache stores results by key; bypassing it keeps the test focused
    # on request shape.
    ds = acmaddl.fetch(
        product="c3s/ecmwf-s2s",
        variable="precip",
        init="2026-05-12",
        region=[-2, 2, 36, 40],
        cache=False,
        verbose=False,
    )
    assert ds is not None
    assert len(fake.calls) == 1
    _, request, _ = fake.calls[0]
    # The full date should have been parsed and routed into the request.
    assert (request["year"], request["month"], request["day"]) == ("2026", "05", "12")


def test_parse_init_accepts_ymd_and_ym():
    """parse_init returns a date-like for both YYYY-MM and YYYY-MM-DD."""
    from acmaddl.fetch import parse_init

    ym = parse_init("2026-05")
    ymd = parse_init("2026-05-12")
    assert ym.year == 2026 and ym.month == 5
    assert ymd.year == 2026 and ymd.month == 5 and ymd.day == 12


def test_two_s2s_fetches_with_different_init_dates_do_not_collide_in_cache(
    monkeypatch, tmp_path
):
    """Two fetch() calls with different init dates must result in two
    distinct adapter calls (no cache shadowing)."""
    import acmaddl

    fake = _FakeCDSClient()
    monkeypatch.setattr("cdsapi.Client", lambda *a, **kw: fake)

    # Run the fetch with cache OFF first to establish a baseline call count.
    acmaddl.fetch(
        product="c3s/ecmwf-s2s", variable="precip",
        init="2026-05-12", region=[-2, 2, 36, 40], cache=False, verbose=False,
    )
    acmaddl.fetch(
        product="c3s/ecmwf-s2s", variable="precip",
        init="2026-05-15", region=[-2, 2, 36, 40], cache=False, verbose=False,
    )
    assert len(fake.calls) == 2, (
        f"Expected 2 adapter calls (cache off), got {len(fake.calls)}. "
        f"Dates seen: {[c[1].get('day') for c in fake.calls]}"
    )
    days = [c[1]["day"] for c in fake.calls]
    assert days == ["12", "15"], days


def test_two_s2s_fetches_with_different_init_dates_distinct_in_cache(
    monkeypatch, tmp_path
):
    """Same as the previous test, but with cache=True. The two calls MUST
    still hit the adapter twice — otherwise the second call would silently
    return the first call's data, which is a correctness bug."""
    import acmaddl
    from nuthatch.config import NuthatchConfig

    fake = _FakeCDSClient()
    monkeypatch.setattr("cdsapi.Client", lambda *a, **kw: fake)
    monkeypatch.setenv("NUTHATCH_ROOT_FILESYSTEM", str(tmp_path))
    monkeypatch.setenv("NUTHATCH_LOCAL_FILESYSTEM", str(tmp_path))
    # Suppress the pyproject lookup so env vars take effect.
    monkeypatch.setattr(NuthatchConfig, "_find_nuthatch_config",
                        lambda self, p: None)

    acmaddl.fetch(
        product="c3s/ecmwf-s2s", variable="precip",
        init="2026-05-12", region=[-2, 2, 36, 40], cache=True, verbose=False,
    )
    acmaddl.fetch(
        product="c3s/ecmwf-s2s", variable="precip",
        init="2026-05-15", region=[-2, 2, 36, 40], cache=True, verbose=False,
    )
    assert len(fake.calls) == 2, (
        f"Cache collision: 2 distinct inits should produce 2 adapter calls, "
        f"got {len(fake.calls)}. The second fetch was silently served from cache."
    )


# ---------------------------------------------------------------------------
# Reforecast normalization (hdate → year)
# ---------------------------------------------------------------------------

def _make_reforecast_like_dataset():
    """A stub xr.Dataset whose dims/coords mimic what CDS returns for a
    reforecast request: hdate (datetime), number (member), step (lead),
    latitude, longitude."""
    import numpy as np
    import pandas as pd
    import xarray as xr
    hdates = pd.to_datetime([
        "2006-05-15", "2007-05-15", "2008-05-15", "2009-05-15", "2010-05-15",
    ])
    return xr.Dataset(
        {
            "tp": (
                ["hdate", "number", "step", "latitude", "longitude"],
                np.zeros((5, 3, 4, 2, 2), dtype="float32"),
            ),
        },
        coords={
            "hdate": hdates,
            "number": [0, 1, 2],
            "step": pd.to_timedelta([24, 48, 72, 96], unit="h"),
            "latitude": [0.0, 1.0],
            "longitude": [0.0, 1.0],
        },
    )


def test_normalize_renames_hdate_to_year_with_integer_years():
    """hdate (datetime64) becomes year (int) after normalization."""
    from acmaddl.normalize import normalize
    config = _s2s_product_config()
    ds = _make_reforecast_like_dataset()
    out = normalize(ds, config, "precip")
    assert "year" in out.dims
    assert "hdate" not in out.dims
    assert out["year"].dtype.kind == "i"
    assert list(out["year"].values) == [2006, 2007, 2008, 2009, 2010]


def test_normalize_preserves_other_s2s_dims():
    """After hdate→year, the rest of the renames (number→member, step→lead_time) still work."""
    from acmaddl.normalize import normalize
    config = _s2s_product_config()
    ds = _make_reforecast_like_dataset()
    out = normalize(ds, config, "precip")
    assert "member" in out.dims
    assert "lead_time" in out.dims
    assert "lat" in out.dims
    assert "lon" in out.dims
    assert set(out["precip"].dims) >= {"year", "member", "lead_time", "lat", "lon"}



def test_s2s_reforecast_pins_hyear_from_date_range(monkeypatch):
    """The training period must not be an unpinned ECDS default.

    Without hyear/hmonth/hday the collection happens to return the full suite, so
    the omission is invisible -- until the server-side default changes and every
    downstream calibration is silently fitted on a different set of years.
    """
    fake = _FakeCDSClient()
    monkeypatch.setattr("cdsapi.Client", lambda *a, **kw: fake)

    adapter = CDSAdapter()
    config = _s2s_product_config()
    config["_init_date"] = "2026-08-31"
    config["_reforecast"] = True
    adapter.fetch_data(config, "precip", date_range=(2006, 2025), region=[-2, 2, 36, 40])

    dataset, request, _ = fake.calls[0]
    assert dataset == "s2s-reforecasts"
    assert request["hyear"] == [str(y) for y in range(2006, 2026)]
    assert request["hmonth"] == "08"
    assert request["hday"] == "31"
    # the real-time issuance date is still the forecast date, not a hindcast one
    assert (request["year"], request["month"], request["day"]) == ("2026", "08", "31")


def test_s2s_realtime_does_not_send_hyear(monkeypatch):
    fake = _FakeCDSClient()
    monkeypatch.setattr("cdsapi.Client", lambda *a, **kw: fake)
    adapter = CDSAdapter()
    config = _s2s_product_config()
    config["_init_date"] = "2026-08-31"
    adapter.fetch_data(config, "precip", date_range=(2026, 2026), region=[-2, 2, 36, 40])
    _, request, _ = fake.calls[0]
    assert "hyear" not in request
