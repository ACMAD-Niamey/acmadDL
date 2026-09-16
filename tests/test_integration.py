"""Integration tests that fetch a small sample from each data product.

These require network access and, for CDS products, valid credentials in ~/.cdsapirc.

Run all:       pytest -m integration tests/test_integration.py
Run non-CDS:   pytest -m "integration and not cds" tests/test_integration.py
"""

import pytest
import acmaddl

# Small region (East Africa) to keep downloads minimal
REGION = [-2, 2, 36, 40]


def _check_dataset(ds, variable, region=None):
    """Common assertions for any fetched dataset."""
    assert variable in ds, f"Expected variable '{variable}' in dataset, got {list(ds.data_vars)}"
    assert ds[variable].size > 0, "Variable has no data"
    assert "units" in ds[variable].attrs, "Variable missing units attribute"
    assert "lat" in ds.coords, "Missing 'lat' coordinate after normalization"
    assert "lon" in ds.coords, "Missing 'lon' coordinate after normalization"
    if region:
        lat_s, lat_n, lon_w, lon_e = region
        assert float(ds.lat.min()) >= lat_s - 1, "Latitude below requested region"
        assert float(ds.lat.max()) <= lat_n + 1, "Latitude above requested region"


# ── OPeNDAP ─────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.network
def test_fetch_nmme_cfsv2_precip():
    # nmme/cfsv2 is the hindcast entry (1982 to 2011-03); use an in-range init.
    ds = acmaddl.fetch(
        product="nmme/cfsv2",
        variable="precip",
        init="2010-01",
        region=REGION,
        verbose=True,
    )
    _check_dataset(ds, "precip", REGION)
    assert ds["precip"].attrs["units"] == "mm/day"


@pytest.mark.integration
@pytest.mark.network
def test_fetch_nmme_cfsv2_temp():
    # nmme/cfsv2 is the hindcast entry (1982 to 2011-03); use an in-range init.
    ds = acmaddl.fetch(
        product="nmme/cfsv2",
        variable="temp",
        init="2010-01",
        region=REGION,
        verbose=True,
    )
    _check_dataset(ds, "temp", REGION)
    assert ds["temp"].attrs["units"] == "C"


@pytest.mark.integration
@pytest.mark.network
def test_fetch_nmme_cfsv2_forecast_precip():
    # nmme/cfsv2-forecast is the live real-time stream (2011-04 to present), so a
    # recent init like 2024-01 is in range. Covers the forecast endpoint, which
    # the hindcast tests above do not exercise.
    ds = acmaddl.fetch(
        product="nmme/cfsv2-forecast",
        variable="precip",
        init="2024-01",
        region=REGION,
        verbose=True,
    )
    _check_dataset(ds, "precip", REGION)
    assert ds["precip"].attrs["units"] == "mm/day"


@pytest.mark.integration
@pytest.mark.network
def test_fetch_nmme_cfsv2_routes_streams_and_forecast_sst():
    """The unified nmme/cfsv2 (split_streams) routes by init year against the real
    IRI endpoints: a hindcast-range init resolves to .HINDCAST, a post-2010 init
    to .FORECAST -- including sst, which both streams serve. End-to-end guard for
    the opendap split-stream routing (URL-level routing is covered no-network in
    tests/test_opendap_adapter.py)."""
    OCEAN = [-5, 5, 55, 70]   # sst needs an ocean bbox
    # Hindcast-year init -> .HINDCAST stream.
    h = acmaddl.fetch("nmme/cfsv2", "precip", init="2010-01", target="MAM",
                      region=REGION, cache=False, verbose=False)
    _check_dataset(h, "precip", REGION)
    assert h["precip"].attrs["units"] == "mm"
    assert h.sizes["member"] == 24
    # Forecast-year init via the SAME id -> .FORECAST stream.
    f = acmaddl.fetch("nmme/cfsv2", "precip", init="2024-01", target="MAM",
                      region=REGION, cache=False, verbose=False)
    _check_dataset(f, "precip", REGION)
    assert f["precip"].attrs["units"] == "mm"
    assert f.sizes["member"] == 24
    # Forecast sst -- previously unavailable; symmetric across streams now.
    s = acmaddl.fetch("nmme/cfsv2", "sst", init="2024-01", target="MAM",
                      region=OCEAN, cache=False, verbose=False)
    assert "sst" in s
    assert "member" in s.dims
    assert float(s["sst"].notnull().mean()) > 0.3, "expected finite sst over ocean"


@pytest.mark.integration
@pytest.mark.network
def test_cfsv2_jan_init_has_consecutive_years_incl_2011():
    """Task A5 regression guard: the IRI cfsv2 HINDCAST/FORECAST streams split by
    MONTH (not year) at 2011-03/2011-04, so a Jan-init request spanning 2011 used
    to silently drop the year (routed entirely to FORECAST, which has no Jan-2011
    init). The catalog's hindcast_range/forecast_range now overlap at 2011 so
    _resolve_streams fetches that year from BOTH streams; this end-to-end fetch
    must return consecutive years including a real (non-NaN) 2011."""
    import numpy as np
    da = acmaddl.fetch("nmme/cfsv2", "precip", init="2026-01", target="MAM",
                       hindcast=(2009, 2013), region=[-30, 30, -180, 180],
                       year_index=True, boundary="center", cache=False, verbose=False)["precip"]
    yrs = [int(y) for y in da.year.values]
    assert yrs == [2009, 2010, 2011, 2012, 2013], yrs           # consecutive, 2011 present
    y11 = da.sel(year=2011)
    assert float(np.isfinite(y11).mean()) > 0.99                 # empty member slots removed
    assert 90.0 < float(np.nanmean(y11)) < 900.0                # sane MAM total in mm


# ── NCEI ─────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.network
def test_fetch_nmme_ccsm4_precip():
    ds = acmaddl.fetch(
        product="nmme/ccsm4",
        variable="precip",
        init="2024-01",
        target="MAM",
        region=REGION,
        hindcast=(2024, 2024),
        verbose=True,
    )
    _check_dataset(ds, "precip", REGION)
    assert ds["precip"].attrs["units"] == "mm"
    _mean = float(ds["precip"].mean())
    assert 5.0 < _mean < 500.0, f"monthly-total magnitude off: {_mean}"
    assert "member" in ds.dims


@pytest.mark.integration
@pytest.mark.network
def test_fetch_nmme_ccsm4_temp():
    ds = acmaddl.fetch(
        product="nmme/ccsm4",
        variable="temp",
        init="2024-01",
        target="MAM",
        region=REGION,
        hindcast=(2024, 2024),
        verbose=True,
    )
    _check_dataset(ds, "temp", REGION)
    assert ds["temp"].attrs["units"] == "C"
    assert "member" in ds.dims


@pytest.mark.integration
@pytest.mark.network
def test_fetch_nmme_geoss2s_precip():
    ds = acmaddl.fetch(
        product="nmme/geoss2s",
        variable="precip",
        init="2024-01",
        target="MAM",
        region=REGION,
        hindcast=(2024, 2024),
        verbose=True,
    )
    _check_dataset(ds, "precip", REGION)
    assert ds["precip"].attrs["units"] == "mm"
    _mean = float(ds["precip"].mean())
    assert 5.0 < _mean < 500.0, f"monthly-total magnitude off: {_mean}"
    assert "member" in ds.dims


@pytest.mark.integration
@pytest.mark.network
def test_fetch_nmme_geoss2s_temp():
    ds = acmaddl.fetch(
        product="nmme/geoss2s",
        variable="temp",
        init="2024-01",
        target="MAM",
        region=REGION,
        hindcast=(2024, 2024),
        verbose=True,
    )
    _check_dataset(ds, "temp", REGION)
    assert ds["temp"].attrs["units"] == "C"
    assert "member" in ds.dims


# ── HTTP ─────────────────────────────────────────────────────────────────────

# Native UCSB CHIRPS. Monthly is COG and annual is GeoTIFF — both range-read,
# so they're light. The Sheerwater mirror routing is covered (mocked) by
# tests/test_sheerwater_catalog.py.
_CHIRPS_NATIVE_RASTER = [
    pytest.param("obs/chirps-v3-monthly", dict(hindcast=(2020, 2020), region=REGION), "mm/month", id="v3-monthly"),
    pytest.param("obs/chirps-v2-monthly", dict(hindcast=(2020, 2020), region=REGION), "mm/month", id="v2-monthly"),
    pytest.param("obs/chirps-v3-annual", dict(hindcast=(2020, 2020), region=REGION), "mm", id="v3-annual"),
    pytest.param("obs/chirps-v2-annual", dict(hindcast=(2020, 2020), region=REGION), "mm", id="v2-annual"),
]


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.parametrize("product, kwargs, units", _CHIRPS_NATIVE_RASTER)
def test_fetch_chirps_native_raster(product, kwargs, units):
    ds = acmaddl.fetch(product=product, variable="precip", verbose=True, **kwargs)
    _check_dataset(ds, "precip", REGION)
    assert ds["precip"].attrs["units"] == units


@pytest.mark.integration
@pytest.mark.network
def test_fetch_chirps_native_dekad_netcdf():
    # Sub-monthly native cadences are one NetCDF per year, downloaded whole then
    # cropped. v2 dekad is the smallest (~160 MiB/yr) and exercises the netcdf
    # cadence path end to end. v3-daily is deliberately NOT tested here — its
    # per-year NetCDF is ~23.5 GiB; use obs/chirps-v3-daily-rhiza for daily.
    ds = acmaddl.fetch(
        product="obs/chirps-v2-dekad",
        variable="precip",
        hindcast=(2020, 2020),
        region=REGION,
        verbose=True,
    )
    _check_dataset(ds, "precip", REGION)
    assert ds["precip"].attrs["units"] == "mm"   # 10-day totals, not a daily rate
    assert ds.sizes["time"] == 36                 # 36 calendar dekads per year


# ── HTTP adapter: retries + rate limiting (local flaky NetCDF server) ───────
# An in-process HTTP server with controllable failures verifies retry and
# request_interval behavior end-to-end through the real urllib stack. We use
# the NetCDF path here (not COG) because urllib opens each file as a single
# request, which makes "fail N times then succeed" semantics unambiguous;
# GDAL/vsicurl issues many range reads per COG and ties them to a per-process
# response cache, both of which obscure what a test is actually asserting.
# The COG path shares the same _with_retry / _RateLimiter plumbing, so this
# test exercises the contract that matters.


def _make_tiny_netcdf(path, year, month):
    """Write a 2x2 netcdf the adapter will accept and reshape via _subset_region."""
    import numpy as np
    import xarray as xr
    arr = np.full((2, 2), float(year * 100 + month), dtype="float32")
    ds = xr.Dataset(
        {"precip": (["latitude", "longitude"], arr)},
        coords={"latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
    )
    ds.to_netcdf(path)


class _FlakyServer:
    """In-process HTTP server that optionally fails the first `fail_first_n`
    requests per URL with a 503. Tracks per-URL hit counts.
    """

    def __init__(self, files_root, fail_first_n=0):
        import os
        import threading
        from http.server import HTTPServer, BaseHTTPRequestHandler

        self.files_root = files_root
        self.fail_first_n = fail_first_n
        self.hits = {}
        self._lock = threading.Lock()
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass  # silence stderr

            def do_GET(self):
                with outer._lock:
                    outer.hits[self.path] = outer.hits.get(self.path, 0) + 1
                    attempt = outer.hits[self.path]
                if attempt <= outer.fail_first_n:
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                full = os.path.join(outer.files_root, self.path.lstrip("/"))
                if not os.path.isfile(full):
                    self.send_response(404)
                    self.end_headers()
                    return
                with open(full, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                self.wfile.write(body)

        self._server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/"

    def stop(self):
        self._server.shutdown()
        self._thread.join(timeout=2)


def _local_netcdf_config(base_url, **overrides):
    config = {
        "adapter": "http",
        "source_url": base_url,
        "file_pattern": "fake-{year}.{month:02d}.nc",
        "format": "netcdf",
        "variables": {
            "precip": {"native_name": "precip", "units": "mm/month",
                       "target_units": "mm/day"},
        },
        "_verbose": False,
        "_progress": False,
    }
    config.update(overrides)
    return config


@pytest.mark.integration
def test_http_adapter_retries_recover_against_real_flaky_server(tmp_path):
    """End-to-end: real HTTP requests hit a 503-then-200 server, retries
    absorb the failures, all 12 months land cleanly."""
    from acmaddl.adapters.http import HTTPAdapter
    for month in range(1, 13):
        _make_tiny_netcdf(str(tmp_path / f"fake-2010.{month:02d}.nc"), 2010, month)
    server = _FlakyServer(str(tmp_path), fail_first_n=2)
    try:
        config = _local_netcdf_config(
            server.base_url,
            _max_retries=3,
            _retry_backoff=0.05,
        )
        adapter = HTTPAdapter()
        result = adapter.fetch_data(config, "precip", date_range=(2010, 2010))
        assert result.sizes["time"] == 12
        # Each URL was hit (2 forced failures + 1 success) = 3 times.
        assert all(count == 3 for count in server.hits.values())
    finally:
        server.stop()


@pytest.mark.integration
def test_http_adapter_retries_exhaust_against_persistent_failure(tmp_path):
    """When the retry budget is exhausted, strict mode raises rather than
    quietly returning a partial dataset."""
    from acmaddl.adapters.http import HTTPAdapter
    for month in range(1, 13):
        _make_tiny_netcdf(str(tmp_path / f"fake-2010.{month:02d}.nc"), 2010, month)
    server = _FlakyServer(str(tmp_path), fail_first_n=10)
    try:
        config = _local_netcdf_config(
            server.base_url,
            _max_retries=2,
            _retry_backoff=0.01,
        )
        adapter = HTTPAdapter()
        with pytest.raises(RuntimeError, match="12/12 file"):
            adapter.fetch_data(config, "precip", date_range=(2010, 2010))
    finally:
        server.stop()


@pytest.mark.integration
def test_http_adapter_rate_limiter_paces_real_requests(tmp_path):
    """request_interval enforces a real wallclock floor between file opens
    even when worker threads would otherwise fire them concurrently."""
    import time as _time
    from acmaddl.adapters.http import HTTPAdapter
    for month in range(1, 13):
        _make_tiny_netcdf(str(tmp_path / f"fake-2010.{month:02d}.nc"), 2010, month)
    server = _FlakyServer(str(tmp_path), fail_first_n=0)
    try:
        config = _local_netcdf_config(
            server.base_url,
            _request_interval=0.15,
        )
        adapter = HTTPAdapter()
        t0 = _time.monotonic()
        result = adapter.fetch_data(config, "precip", date_range=(2010, 2010))
        elapsed = _time.monotonic() - t0
        # 12 files at 0.15s spacing ≈ 1.8s lower bound. Allow CI slack —
        # the point is that pacing dominates the 8-worker pool's concurrency.
        assert elapsed >= 11 * 0.15 * 0.9
        assert result.sizes["time"] == 12
    finally:
        server.stop()


# ── Observational SST (ERSST v5, NOAA PSL OPeNDAP) ──────────────────────────
# Tests cover: two geographies, two non-adjacent years, multi-year series.
# These were written against a planned `sst/ersst-v5` product that was never
# implemented; the product now exists as `obs/ersst-v5` (the catalog's `obs/`
# channel convention, as with obs/era5, which also carries sst).

@pytest.mark.integration
@pytest.mark.network
def test_fetch_ersst_v5_east_africa_recent():
    ds = acmaddl.fetch(
        product="obs/ersst-v5",
        variable="sst",
        hindcast=(2020, 2020),
        region=[-20, 20, 20, 55],
        verbose=False, progress=False, cache=False,
    )
    _check_dataset(ds, "sst", [-20, 20, 20, 55])
    assert ds["sst"].attrs["units"] == "C"
    assert ds.sizes["time"] == 12
    vmin, vmax = float(ds["sst"].min(skipna=True)), float(ds["sst"].max(skipna=True))
    assert -3.0 <= vmin and vmax <= 45.0, f"sst out of range: {vmin}..{vmax}"


@pytest.mark.integration
@pytest.mark.network
def test_fetch_ersst_v5_west_pacific_historical():
    # Warm-pool region, well inside 0..360 longitude convention
    region = [-10, 10, 140, 180]
    ds = acmaddl.fetch(
        product="obs/ersst-v5",
        variable="sst",
        hindcast=(2000, 2000),
        region=region,
        verbose=False, progress=False, cache=False,
    )
    _check_dataset(ds, "sst", region)
    assert ds.sizes["time"] == 12
    # Warm pool SSTs are typically 27-30 C
    mean_sst = float(ds["sst"].mean(skipna=True))
    assert 25.0 < mean_sst < 32.0, f"warm pool mean sst implausible: {mean_sst}"


@pytest.mark.integration
@pytest.mark.network
def test_fetch_ersst_v5_multiyear_timeseries():
    ds = acmaddl.fetch(
        product="obs/ersst-v5",
        variable="sst",
        hindcast=(2010, 2012),
        region=[-20, 20, 20, 55],
        verbose=False, progress=False, cache=False,
    )
    _check_dataset(ds, "sst", None)
    assert ds.sizes["time"] == 36, f"expected 36 months, got {ds.sizes['time']}"
    # Time should be strictly increasing
    import numpy as np
    t = ds["time"].values
    assert (np.diff(t) > np.timedelta64(0, "ns")).all(), "time not monotonic"


# ── CDS (require credentials) ───────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_c3s_ecmwf_precip():
    ds = acmaddl.fetch(
        product="c3s/ecmwf",
        variable="precip",
        init="2000-01",
        target="MAM",
        region=REGION,
        hindcast=(2000, 2000),
        verbose=True,
    )
    _check_dataset(ds, "precip", REGION)
    assert ds["precip"].attrs["units"] == "mm/day"
    assert "lead_time" in ds.dims


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
@pytest.mark.parametrize("product", ["c3s/ecmwf-monthly", "c3s/ecmwf"])
def test_c3s_collapsed_seasonal_precip_is_mm(product):
    """Monthly-rate and daily-deaccumulated CDS paths share the mm contract."""
    ds = acmaddl.fetch(
        product=product,
        variable="precip",
        init="2000-01",
        target="MAM",
        region=[-1, 1, 36, 38],
        hindcast=(2000, 2000),
        year_index=True,
        cache=False,
        verbose=False,
        progress=False,
    )
    da = ds["precip"]
    assert da.attrs["units"] == "mm"
    assert "year" in da.dims and "lead_time" not in da.dims
    assert 10.0 < float(da.mean()) < 2000.0


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_c3s_ecmwf_temp():
    ds = acmaddl.fetch(
        product="c3s/ecmwf",
        variable="temp",
        init="2000-01",
        target="MAM",
        region=REGION,
        hindcast=(2000, 2000),
        verbose=True,
    )
    _check_dataset(ds, "temp", REGION)
    assert ds["temp"].attrs["units"] == "C"
    assert "lead_time" in ds.dims


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_c3s_ecmwf_precip_no_target_through_cache():
    """Reproduce issue #24: daily CDS fetch hangs in the nuthatch cache layer.

    Emmett's exact reproduction:
        acmaddl.fetch('c3s/ecmwf', 'precip', init='2026-02', region=[-13, 23, 21, 52])

    Without `target`, fetch.py leaves `leadtime_hour` unset, so the CDS adapter
    (cds.py:92-93) falls back to the default 214-day range × 51 members — a much
    larger payload than the targeted daily tests above. Reported symptom: 0% CPU
    indefinitely after the `[acmaddl:cds] download complete` log, with the cdsapi
    TCP socket in CLOSE_WAIT.

    cache=True (default) so we exercise the nuthatch write path. A SIGALRM-based
    timeout converts the hang into a hard test failure instead of blocking the
    suite indefinitely.
    """
    import signal

    region = [-13, 23, 21, 52]

    def _on_alarm(signum, frame):
        raise TimeoutError(
            "acmaddl.fetch hung past 1800s after CDS download — see issue #24"
        )

    old_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(1800)
    try:
        ds = acmaddl.fetch(
            product="c3s/ecmwf",
            variable="precip",
            init="2026-02",
            region=region,
            verbose=True,
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    _check_dataset(ds, "precip", region)
    assert ds["precip"].attrs["units"] == "mm/day"
    assert "lead_time" in ds.dims


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_c3s_ecmwf_monthly_temp():
    ds = acmaddl.fetch(
        product="c3s/ecmwf-monthly",
        variable="temp",
        init="2024-01",
        region=REGION,
        verbose=True,
    )
    _check_dataset(ds, "temp", REGION)
    assert ds["temp"].attrs["units"] == "C"


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_c3s_ecmwf_s2s_precip():
    """End-to-end fetch of one ECMWF S2S issuance for a small East Africa box.

    Requires ECDS credentials in ~/.cdsapirc (the ecds.ecmwf.int endpoint,
    not the Copernicus CDS one). The catalog entry overrides cds_url to point
    at the right endpoint; the user's key in ~/.cdsapirc must match.
    """
    ds = acmaddl.fetch(
        product="c3s/ecmwf-s2s",
        variable="precip",
        init="2026-05-18",       # a recent Monday; bump if archive cycles this out
        region=REGION,
        verbose=True,
    )
    _check_dataset(ds, "precip", REGION)
    assert ds["precip"].attrs["units"] == "mm/day"
    # S2S forecasts out to ~46 days at 24h steps. After deaccumulation
    # acmaddl drops the first step, so we expect ~46 lead_time entries.
    assert ds.sizes["lead_time"] >= 20
    assert ds.sizes["member"] >= 2          # perturbed ensemble has 50+
    assert "init_time" in ds.coords          # scalar issuance date
    assert "time" in ds.coords               # 1-D valid forecast time


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_eccc_cansips_temp():
    ds = acmaddl.fetch(
        product="c3s/eccc-cansips",
        variable="temp",
        init="2000-01",
        target="MAM",
        region=REGION,
        hindcast=(2000, 2000),
        verbose=True,
    )
    _check_dataset(ds, "temp", REGION)
    assert ds["temp"].attrs["units"] == "C"
    assert "member" in ds.dims


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_eccc_cansipsv3_temp():
    ds = acmaddl.fetch(
        product="c3s/eccc-cansipsv3",
        variable="temp",
        init="2000-01",
        target="MAM",
        region=REGION,
        hindcast=(2000, 2000),
        verbose=True,
    )
    _check_dataset(ds, "temp", REGION)
    assert ds["temp"].attrs["units"] == "C"
    assert "member" in ds.dims


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_meteofrance_temp():
    ds = acmaddl.fetch(
        product="c3s/meteofrance",
        variable="temp",
        init="2000-01",
        target="MAM",
        region=REGION,
        hindcast=(2000, 2000),
        verbose=True,
    )
    _check_dataset(ds, "temp", REGION)
    assert ds["temp"].attrs["units"] == "C"
    assert "member" in ds.dims


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_cmcc_temp():
    ds = acmaddl.fetch(
        product="c3s/cmcc",
        variable="temp",
        init="2000-01",
        target="MAM",
        region=REGION,
        hindcast=(2000, 2000),
        verbose=True,
    )
    _check_dataset(ds, "temp", REGION)
    assert ds["temp"].attrs["units"] == "C"
    assert "member" in ds.dims


# ── CMCC SPS4 (system 4) — current operational system, verified 2026-06-25 ────
# Hindcast init 2003-11 (Nov) and the 30/50-member split were confirmed via a
# live CDS probe. sst needs an ocean bbox.

SST_OCEAN = [-5, 5, 55, 70]   # Arabian Sea


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_cmcc_sps4_precip_hindcast():
    ds = acmaddl.fetch(
        product="c3s/cmcc-sps4",
        variable="precip",
        init="2003-11",
        target="MAM",
        region=REGION,
        hindcast=(2003, 2003),
        verbose=True,
    )
    _check_dataset(ds, "precip", REGION)
    assert ds["precip"].attrs["units"] == "mm/day"
    assert ds.sizes["member"] == 30


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_cmcc_sps4_sst_hindcast():
    ds = acmaddl.fetch(
        product="c3s/cmcc-sps4",
        variable="sst",
        init="2003-11",
        target="MAM",
        region=SST_OCEAN,
        hindcast=(2003, 2003),
        verbose=True,
    )
    _check_dataset(ds, "sst", SST_OCEAN)
    assert ds["sst"].attrs["units"] == "K"
    assert ds.sizes["member"] == 30


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_cmcc_sps4_precip_realtime_forecast():
    # 2025-11 is in forecast_range [2025, null]; real-time SPS4 runs 50 members.
    ds = acmaddl.fetch(
        product="c3s/cmcc-sps4",
        variable="precip",
        init="2025-11",
        target="MAM",
        region=REGION,
        verbose=True,
    )
    _check_dataset(ds, "precip", REGION)
    assert ds["precip"].attrs["units"] == "mm/day"
    assert ds.sizes["member"] == 50


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_cmcc_sps4_daily_precip_and_sst():
    p = acmaddl.fetch(
        product="c3s/cmcc-sps4-daily",
        variable="precip",
        init="2003-11",
        target="MAM",
        region=REGION,
        hindcast=(2003, 2003),
        verbose=True,
    )
    _check_dataset(p, "precip", REGION)
    assert p["precip"].attrs["units"] == "mm/day"
    assert "member" in p.dims

    s = acmaddl.fetch(
        product="c3s/cmcc-sps4-daily",
        variable="sst",
        init="2003-11",
        target="MAM",
        region=SST_OCEAN,
        hindcast=(2003, 2003),
        verbose=True,
    )
    _check_dataset(s, "sst", SST_OCEAN)
    assert s["sst"].attrs["units"] == "K"


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_dwd_temp():
    ds = acmaddl.fetch(
        product="c3s/dwd",
        variable="temp",
        init="2000-01",
        target="MAM",
        region=REGION,
        hindcast=(2000, 2000),
        verbose=True,
    )
    _check_dataset(ds, "temp", REGION)
    assert ds["temp"].attrs["units"] == "C"
    assert "member" in ds.dims


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_ukmo_temp():
    ds = acmaddl.fetch(
        product="c3s/ukmo",
        variable="temp",
        init="2000-01",
        target="MAM",
        region=REGION,
        hindcast=(2000, 2000),
        verbose=True,
    )
    _check_dataset(ds, "temp", REGION)
    assert ds["temp"].attrs["units"] == "C"
    assert "member" in ds.dims


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_jma_temp():
    ds = acmaddl.fetch(
        product="c3s/jma",
        variable="temp",
        init="2000-01",
        target="MAM",
        region=REGION,
        hindcast=(2000, 2000),
        verbose=True,
    )
    _check_dataset(ds, "temp", REGION)
    assert ds["temp"].attrs["units"] == "C"
    assert "member" in ds.dims
    assert "init_time" in ds.coords


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_era5_temp():
    ds = acmaddl.fetch(
        product="obs/era5",
        variable="temp",
        init="2024-01",
        region=REGION,
        verbose=True,
    )
    _check_dataset(ds, "temp", REGION)
    assert ds["temp"].attrs["units"] == "C"


# ── ERA5-Land monthly (CDS, 0.1° land sibling of obs/era5) ───────────────────

@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_era5_land_monthly_precip():
    ds = acmaddl.fetch(
        product="obs/era5-land-monthly",
        variable="precip",
        init="2024-01",
        region=REGION,
        verbose=True,
    )
    _check_dataset(ds, "precip", REGION)
    assert ds["precip"].attrs["units"] == "mm/day"
    # Distinguishing feature vs obs/era5 (0.25°): the land product is 0.1°.
    assert round(float(abs(ds.lon[1] - ds.lon[0])), 2) == 0.1
    assert round(float(abs(ds.lat[1] - ds.lat[0])), 2) == 0.1


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_era5_land_monthly_temp():
    ds = acmaddl.fetch(
        product="obs/era5-land-monthly",
        variable="temp",
        init="2024-01",
        region=REGION,
        verbose=True,
    )
    _check_dataset(ds, "temp", REGION)
    assert ds["temp"].attrs["units"] == "C"


# ── Shapefile region input, per data-source system (acmaddl-plan §5, #12) ────
# Each adapter slices the bbox differently — OPeNDAP/NCEI/CCSR by xarray dims,
# CDS server-side via `area`, HTTP via COG windows, Sheerwater via bbox→global —
# so every system gets a shapefile case to prove the bbox→adapter→polygon-clip
# path end to end. Params mirror each adapter's existing integration test (the
# known-good live calls); the masking step itself is adapter-agnostic.
#
# Not covered: s3 (NMME hindcast storage — same in-memory slice as OPeNDAP, but
# needs AWS creds), iridl (sunset, replaced by ccsr), mars (no catalog product).

def _kenya_shapefile():
    from pathlib import Path
    shp = Path(__file__).resolve().parents[1] / "data" / "shapefiles" / "kenya.shp"
    if not shp.exists():
        pytest.skip("kenya.shp missing; run scripts/fetch_country_shapefiles.py")
    return str(shp)


def _assert_masked_to_kenya(ds, variable):
    """The result must be a Kenya-bbox grid with data inside the border and
    NaN outside it (Kenya is non-rectangular, so a polygon mask must show both)."""
    import numpy as np
    _check_dataset(ds, variable)
    field = ds[variable]
    for dim in ("member", "M", "lead_time", "init_time", "time"):
        if dim in field.dims:
            field = field.isel({dim: 0})
    assert bool(field.notnull().any()), "no data inside Kenya"
    assert bool(field.isnull().any()), "expected cells masked outside the polygon"
    nairobi = field.sel(lat=-1.3, lon=36.8, method="nearest")
    assert not bool(np.isnan(nairobi)), "expected data at Nairobi"


# (product, variable, fetch kwargs) per system; id = adapter name.
_SHAPEFILE_SYSTEMS = [
    pytest.param("nmme/cfsv2", "precip",
                 dict(init="2010-02", target="MAM", hindcast=(2010, 2010)),
                 id="opendap"),
    pytest.param("nmme/ccsm4", "precip",
                 dict(init="2024-01", target="MAM", hindcast=(2024, 2024)),
                 id="ncei"),
    pytest.param("nmme/spear", "precip",
                 dict(init="2024-01", target="MAM", hindcast=(2024, 2024)),
                 id="ccsr"),
    pytest.param("obs/chirps-v3-monthly", "precip",
                 dict(init="2024-03"),
                 id="http"),
    pytest.param("obs/chirps-v3-daily-rhiza", "precip",
                 dict(init="2010-03", target="MAM"),
                 id="sheerwater"),
    pytest.param("c3s/ecmwf", "precip",
                 dict(init="2000-01", target="MAM", hindcast=(2000, 2000)),
                 marks=pytest.mark.cds, id="cds"),
]


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.parametrize("product, variable, kwargs", _SHAPEFILE_SYSTEMS)
def test_fetch_shapefile_region_per_system(product, variable, kwargs):
    """Every data-source system honours a Kenya shapefile region. Needs the
    `geo` extra (geopandas) + the kenya.shp fixture."""
    pytest.importorskip("geopandas")
    shp = _kenya_shapefile()
    ds = acmaddl.fetch(product, variable, region=shp, verbose=True, **kwargs)
    _assert_masked_to_kenya(ds, variable)


@pytest.mark.integration
@pytest.mark.network
def test_cache_keys_on_region_not_just_product():
    """Two different shapefiles (different bbox) for the SAME product must not
    collide in the cache: requesting Nigeria must never return Kenya's cached
    data. Guards that the region/bbox is part of the raw cache key."""
    from pathlib import Path
    pytest.importorskip("geopandas")
    shp_dir = Path(__file__).resolve().parents[1] / "data" / "shapefiles"
    kenya, nigeria = shp_dir / "kenya.shp", shp_dir / "nigeria.shp"
    if not (kenya.exists() and nigeria.exists()):
        pytest.skip("country shapefiles missing; run scripts/fetch_country_shapefiles.py")

    common = dict(product="nmme/cfsv2", variable="precip",
                  init="2010-02", target="MAM", hindcast=(2010, 2010))
    k = acmaddl.fetch(region=str(kenya), verbose=False, **common)["precip"]
    n = acmaddl.fetch(region=str(nigeria), verbose=False, **common)["precip"]

    # Nigeria (West Africa, lon ~3–14) must not come back as Kenya (lon ~34–42).
    assert float(k.lon.min()) > 30, "Kenya extent unexpected"
    assert float(n.lon.max()) < 20, \
        "Nigeria returned Kenya's cached data — region missing from cache key"


# ── CHC issuance-keyed forecasts (CHIRPS-GEFS) ─────────────────────────────


@pytest.mark.integration
@pytest.mark.network
def test_fetch_chirps_gefs_daily_single_issuance():
    """One real issuance from the live CHC server: 16 daily leads as a
    (init_time, lead_time, lat, lon) cube. Guards the catalog's URL templates
    against an upstream layout change."""
    ds = acmaddl.fetch(
        product="chc/chirps-gefs-daily", variable="precip",
        init="2026-07-05", region=REGION, verbose=True,
    )
    _check_dataset(ds, "precip", REGION)
    assert ds.sizes["init_time"] == 1
    assert ds.sizes["lead_time"] == 16
    assert ds["precip"].attrs["units"] == "mm/day"
    # valid_time = init + lead; normalize maps it onto the canonical `time`.
    assert str(ds.time.isel(init_time=0, lead_time=15).values)[:10] == "2026-07-20"
    assert float(ds["precip"].min()) >= 0.0


@pytest.mark.integration
@pytest.mark.network
def test_fetch_chirps_gefs_across_many_issuances():
    """The hindcast-skill fetch: the same calendar issuance of several years,
    stacked on init_time. Uses the 15-day accumulation product (one raster per
    issuance) to keep the pull small."""
    inits = [f"{year}-06-30" for year in (2015, 2016, 2017)]
    ds = acmaddl.fetch(
        product="chc/chirps-gefs-15day", variable="precip",
        init=inits, region=REGION, verbose=True,
    )
    _check_dataset(ds, "precip", REGION)
    assert ds.sizes["init_time"] == 3
    assert ds.sizes["lead_time"] == 1
    assert ds["precip"].attrs["units"] == "mm"
    years = [str(v)[:4] for v in ds.init_time.values]
    assert years == ["2015", "2016", "2017"]


# ── GPCC (DWD) / GPCP (NCEI) / OISST (NOAA PSL) / ERA5-Land pev ─────────────
# A Sahel box that CROSSES THE PRIME MERIDIAN on purpose: it is what caught the
# http adapter's convention-blind longitude crop against GPCP's 0..360 grid.
SAHEL = [8, 18, -8, 8]


@pytest.mark.integration
@pytest.mark.network
def test_fetch_gpcc_monitoring_precip():
    ds = acmaddl.fetch("obs/gpcc-monitoring-v2020", "precip",
                       hindcast=(2020, 2020), months=[7, 8, 9],
                       region=SAHEL, verbose=True)
    _check_dataset(ds, "precip", SAHEL)
    assert ds["precip"].attrs["units"] == "mm/month"
    # time_from_pattern: three requested months, three distinct stamps, in order
    assert [str(t)[:7] for t in ds.time.values] == ["2020-07", "2020-08", "2020-09"]
    mean = float(ds["precip"].mean())
    assert 20 < mean < 600, f"implausible Sahel JAS monthly total: {mean}"


@pytest.mark.integration
@pytest.mark.network
def test_fetch_gpcc_first_guess_precip():
    """The near-real-time stream. 2013 is its first NetCDF year."""
    ds = acmaddl.fetch("obs/gpcc-first-guess", "precip",
                       hindcast=(2013, 2013), months=[8],
                       region=SAHEL, verbose=True)
    _check_dataset(ds, "precip", SAHEL)
    assert ds["precip"].attrs["units"] == "mm/month"
    assert str(ds.time.values[0])[:7] == "2013-08"


@pytest.mark.integration
@pytest.mark.network
def test_fetch_gpcp_resolves_the_processing_date_suffix():
    ds = acmaddl.fetch("obs/gpcp-v2-3", "precip", hindcast=(2020, 2020),
                       months=[8], region=SAHEL, verbose=True)
    _check_dataset(ds, "precip", SAHEL)
    assert ds["precip"].attrs["units"] == "mm/day"
    mean = float(ds["precip"].mean())
    assert 0.5 < mean < 20, f"implausible Sahel August rate: {mean}"


@pytest.mark.integration
@pytest.mark.network
def test_gpcp_keeps_both_sides_of_a_prime_meridian_bbox():
    """GPCP is served on a 0..360 grid; obs/cmap (same convention, different
    adapter) is the reference for what the crop must return."""
    kw = dict(hindcast=(2020, 2020), months=[8], region=SAHEL, verbose=False)
    gpcp = acmaddl.fetch("obs/gpcp-v2-3", "precip", **kw)
    cmap = acmaddl.fetch("obs/cmap", "precip", **kw)
    assert list(gpcp.lon.values) == list(cmap.lon.values), (
        f"gpcp {gpcp.lon.values} != cmap {cmap.lon.values}")
    assert (gpcp.lon.values > 350).any() and (gpcp.lon.values < 10).any()


@pytest.mark.integration
@pytest.mark.network
def test_fetch_oisst_highres_sst_across_chunks():
    """Four years at max_request_years=2 exercises the chunk loop; the
    always-on obs guard would raise on a zero-filled chunk."""
    ocean = [-5, 15, -30, -10]   # tropical Atlantic, no land-mask trap
    ds = acmaddl.fetch("obs/oisst-v2-highres", "sst", hindcast=(2017, 2020),
                       region=ocean, verbose=True)
    _check_dataset(ds, "sst", ocean)
    assert ds["sst"].attrs["units"] == "C"
    assert ds.sizes["time"] == 48, ds.sizes
    mean = float(ds["sst"].mean())
    assert 20 < mean < 32, f"implausible tropical Atlantic SST: {mean}"


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_fetch_era5_land_pev_is_a_positive_rate():
    """ERA5 serves pev as a NEGATIVE upward flux; the catalog flips it."""
    ds = acmaddl.fetch("obs/era5-land-monthly", "pev", hindcast=(2020, 2020),
                       months=[7], region=SAHEL, verbose=True)
    _check_dataset(ds, "pev", SAHEL)
    assert ds["pev"].attrs["units"] == "mm/day"
    import numpy as np
    assert float(np.nanmin(ds["pev"].values)) > 0, "pev must be positive demand"
    mean = float(np.nanmean(ds["pev"].values))
    assert 2 < mean < 20, f"implausible Sahel July PET: {mean}"


@pytest.mark.integration
@pytest.mark.network
def test_fetch_oisst_across_the_longitude_seam():
    """The regression that hid behind a contiguous test region.

    Every other OISST test here uses a box west of the prime meridian
    ([-30, -10] -> 330..350, contiguous), so none of them exercised a
    seam-crossing selection. Lazily concatenated, that is a single malformed DAP
    request which NOAA PSL answers with zeros at ANY size — an 80x90 deg box
    truncated even at 230k values. The adapter must issue one contiguous request
    per segment instead.
    """
    import numpy as np
    africa = [-40, 40, -30, 60]     # spans the seam, ~115k cells per month
    ds = acmaddl.fetch("obs/oisst-v2-highres", "sst", hindcast=(2019, 2019),
                       region=africa, verbose=True)
    _check_dataset(ds, "sst")
    lons = ds.lon.values
    assert (lons > 330).any() and (lons < 60).any(), f"seam not covered: {lons}"
    sst = ds["sst"].values
    # Truncation is a zero-filled response: real SST varies within every month.
    per_month = [float(np.nanstd(sst[i])) for i in range(sst.shape[0])]
    assert min(per_month) > 1.0, f"a month came back near-constant: {per_month}"
    assert 15 < float(np.nanmean(sst)) < 30, float(np.nanmean(sst))
