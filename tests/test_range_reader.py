"""Paced range reads of remote rasters (acmaddl.adapters._range) and the HTTP
adapter's behaviour when a server blocks the client.

No external network: a local HTTP server on 127.0.0.1 serves a CHIRPS-shaped
COG, logs every request, and can be told to refuse (403 / 429) or go silent.

Why this exists: GDAL's /vsicurl/ sent ~60 unpaced requests per COG open
(mostly HEAD probes for sidecar files that do not exist), which got an address
blocked by data.chc.ucsb.edu at any per-file request_interval. CHC's published
limits are ~2 requests/second per IP and 20 connections; these tests pin the
properties that keep acmadDL inside them.
"""
import http.server
import re
import threading
import time

import numpy as np
import pytest
import xarray as xr

rasterio = pytest.importorskip("rasterio")

from acmaddl.adapters import http as http_mod
from acmaddl.adapters._range import RangeStore
from acmaddl.adapters._robust import (
    ServerBlockedError, ServerUnreachableError, _is_transient, _RateLimiter)

REGION = [-5, 15, 33, 48]          # lat_s, lat_n, lon_w, lon_e


class _Server:
    """Range-capable file server that records requests and can misbehave."""

    def __init__(self, root):
        self.requests = []         # (time, method, path, range)
        self.refuse_with = None    # HTTP status to answer every request with
        self.refuse_body = b""     # body of a refusal
        self.refuse_headers = {}   # extra headers of a refusal
        self.refuse_first = None   # refuse only this many requests, then serve
        self.ignore_range = False
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _serve(self, send_body):
                rng = self.headers.get("Range")
                outer.requests.append((time.monotonic(), self.command, self.path, rng))
                outer.user_agent = self.headers.get("User-Agent")
                path = root / self.path.lstrip("/")
                refuse = outer.refuse_with and (
                    outer.refuse_first is None or len(outer.requests) <= outer.refuse_first)
                if refuse or not path.is_file():
                    self.send_response(outer.refuse_with if refuse else 404)
                    for key, value in (outer.refuse_headers if refuse else {}).items():
                        self.send_header(key, value)
                    body = outer.refuse_body if refuse else b""
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                data = path.read_bytes()
                lo, hi = 0, len(data) - 1
                if rng and not outer.ignore_range:
                    m = re.match(r"bytes=(\d+)-(\d*)", rng)
                    lo = int(m.group(1))
                    hi = min(int(m.group(2)) if m.group(2) else hi, hi)
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(data)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(hi - lo + 1))
                self.end_headers()
                if send_body:
                    self.wfile.write(data[lo:hi + 1])

            def do_GET(self):
                self._serve(True)

            def do_HEAD(self):
                self._serve(False)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(scope="module")
def cog_dir(tmp_path_factory):
    """Three monthly COGs on the CHIRPS grid (0.05 deg, 50S-50N, 7200 x 2000)."""
    from rasterio.transform import from_origin

    root = tmp_path_factory.mktemp("cogs")
    rng = np.random.default_rng(0)
    for month in (1, 2, 3):
        data = rng.gamma(0.5, 60, size=(2000, 7200)).astype("float32")
        with rasterio.open(root / f"fake-2000.{month:02d}.cog", "w", driver="COG",
                           height=2000, width=7200, count=1, dtype="float32",
                           crs="EPSG:4326", transform=from_origin(-180, 50, 0.05, 0.05),
                           BLOCKSIZE=512, COMPRESS="DEFLATE", OVERVIEWS="NONE") as dst:
            dst.write(data, 1)
    return root


@pytest.fixture
def server(cog_dir):
    srv = _Server(cog_dir)
    yield srv
    srv.close()


def _config(server, **overrides):
    config = {
        "adapter": "http", "source_url": server.base + "/",
        "file_pattern": "fake-{year}.{month:02d}.cog", "format": "cog",
        "variables": {"precip": {"native_name": "precip", "units": "mm/month",
                                 "target_units": "mm/month"}},
        "_verbose": False, "_progress": False, "init_months": [1, 2, 3],
        "_max_retries": 0,
    }
    config.update(overrides)
    return config


def test_open_raster_sends_only_ranged_gets_for_the_file_itself(server, cog_dir):
    """No HEAD, no directory listing, no sidecar probes: a handful of ranged
    GETs of the one URL, and the same pixels a local read gives."""
    url = f"{server.base}/fake-2000.01.cog"
    ds = http_mod._open_raster(url, REGION, variable="precip")

    assert 1 <= len(server.requests) <= 8          # vsicurl sent ~60
    for _, method, path, rng in server.requests:
        assert method == "GET" and path == "/fake-2000.01.cog"
        assert rng and rng.startswith("bytes=")
    assert server.user_agent.startswith("acmadDL/")

    local = xr.open_dataset(cog_dir / "fake-2000.01.cog", engine="rasterio")
    local = local.sel(y=slice(REGION[1] + 1, REGION[0] - 1),
                      x=slice(REGION[2] - 1, REGION[3] + 1))["band_data"].squeeze()
    np.testing.assert_array_equal(ds["precip"].values, local.values)
    assert set(ds["precip"].dims) == {"latitude", "longitude"}


def test_no_byte_range_is_requested_twice(server):
    http_mod._open_raster(f"{server.base}/fake-2000.01.cog", REGION, variable="precip")
    ranges = [rng for *_, rng in server.requests]
    assert len(ranges) == len(set(ranges))


def test_limiter_is_waited_on_before_every_request(server):
    """request_interval spaces HTTP requests, not just files."""
    waits = []

    class Limiter:
        def wait(self):
            waits.append(len(server.requests))

    token = http_mod._REQUEST_LIMITER.set(Limiter())
    try:
        http_mod._open_raster(f"{server.base}/fake-2000.01.cog", REGION, variable="precip")
    finally:
        http_mod._REQUEST_LIMITER.reset(token)
    # one wait per request, each taken before that request was sent
    assert waits == list(range(len(server.requests)))


def test_fetch_never_exceeds_the_request_interval_on_the_wire(server):
    """End to end through fetch_data: the gap between any two requests the
    server sees is at least request_interval."""
    interval = 0.15
    ds = http_mod.HTTPAdapter().fetch_data(
        _config(server, request_interval=interval), "precip",
        date_range=(2000, 2000), region=REGION)
    assert ds.sizes["time"] == 3
    times = [t for t, *_ in server.requests]
    assert len(times) >= 3
    assert min(np.diff(times)) >= interval * 0.9
    assert http_mod._REQUEST_LIMITER.get() is None      # limiter not leaked


def test_server_that_ignores_range_still_works(server):
    server.ignore_range = True
    ds = http_mod._open_raster(f"{server.base}/fake-2000.01.cog", REGION, variable="precip")
    assert len(server.requests) == 1
    assert np.isfinite(ds["precip"].values).all()


@pytest.fixture
def sleeps(monkeypatch):
    """Record (and skip) the back-off sleeps after a refusal."""
    import acmaddl.adapters._range as range_mod

    recorded = []
    monkeypatch.setattr(range_mod.time, "sleep", recorded.append)
    return recorded


@pytest.mark.parametrize("status, body, n_requests", [
    (403, b"", 1),                                           # never retried
    (429, b"<h1>Automated bulk download detected</h1>", 1),  # the server says: blocked
    (429, b"", 4),                                           # 3 waited-out retries, then stop
])
def test_refusal_stops_the_pull(server, sleeps, status, body, n_requests):
    """A refusal ends the pull: the other files are never requested, the
    adapter's generic retry does not fire, with or without allow_partial."""
    server.refuse_with, server.refuse_body = status, body
    for allow_partial in (False, True):
        server.requests.clear()
        sleeps.clear()
        with pytest.raises(ServerBlockedError, match=f"HTTP Error {status}") as err:
            http_mod.HTTPAdapter().fetch_data(
                _config(server, _max_retries=3, _retry_backoff=0.0,
                        _allow_partial=allow_partial),
                "precip", date_range=(2000, 2000), region=REGION)
        assert len(server.requests) == n_requests
        # wait before each retry: 15, 30, 60 s (plus up to 1 s jitter)
        # (shorter sleeps are the limiter pacing the retries at the slowed rate)
        assert [int(s) for s in sleeps if s >= 10] == [15, 30, 60][:n_requests - 1]
        if body:
            assert "bulk download" in str(err.value)


def test_brief_429_is_waited_out(server, sleeps):
    """CHC also answers 429 for a moment when busy. Honour Retry-After, retry
    the same request, carry on."""
    server.refuse_with, server.refuse_first = 429, 2
    server.refuse_headers = {"Retry-After": "7"}
    ds = http_mod._open_raster(f"{server.base}/fake-2000.01.cog", REGION, variable="precip")
    assert np.isfinite(ds["precip"].values).all()
    assert [int(s) for s in sleeps if s >= 2] == [7, 7]
    assert [r[3] for r in server.requests[:3]] == ["bytes=0-1048575"] * 3


def test_429_slows_the_rest_of_the_pull(server, sleeps):
    """Each 429 halves the pace for every later request of the pull."""
    limiter = _RateLimiter(1.0)
    server.refuse_with, server.refuse_first = 429, 2
    token = http_mod._REQUEST_LIMITER.set(limiter)
    try:
        http_mod._open_raster(f"{server.base}/fake-2000.01.cog", REGION, variable="precip")
    finally:
        http_mod._REQUEST_LIMITER.reset(token)
    assert limiter.min_interval == 4.0            # 1 s -> 2 s -> 4 s
    assert _RateLimiter(8.0).slow_down() == 10.0   # capped
    assert _RateLimiter(20.0).slow_down() == 20.0  # never shortened


def test_blocked_error_is_never_transient():
    assert not _is_transient(ServerBlockedError("http://x/y", "HTTP Error 429"))


def test_silent_server_stops_the_pull_after_two_files(monkeypatch):
    """A block can look like timeouts. Two files in a row with no answer end
    the pull instead of working through the whole list."""
    opened = []

    def unanswered(url, region, variable=None, fill_value=None, timestamp=None):
        opened.append(url)
        raise ServerUnreachableError(f"no answer from {url}: timed out")

    monkeypatch.setattr(http_mod, "_open_cog_subset", unanswered)
    config = {
        "adapter": "http", "source_url": "https://fake/",
        "file_pattern": "fake-{year}.{month:02d}.cog", "format": "cog",
        "variables": {"precip": {"native_name": "precip"}},
        "_verbose": False, "_progress": False, "_max_retries": 0,
        "_allow_partial": True,
    }
    with pytest.raises(ServerBlockedError, match="No answer for 2 files in a row"):
        http_mod.HTTPAdapter().fetch_data(config, "precip", date_range=(2000, 2000))
    assert len(opened) == 2                           # not all 12 months


def test_netcdf_download_block_stops_the_pull(monkeypatch):
    import urllib.error

    calls = []

    def refused(url, dest):
        calls.append(url)
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)

    monkeypatch.setattr("acmaddl.adapters.http.urllib.request.urlretrieve", refused)
    waits = []
    monkeypatch.setattr(http_mod.time, "sleep", waits.append)
    config = {
        "adapter": "http", "source_url": "https://fake/",
        "file_pattern": "fake-{year}.nc", "format": "netcdf", "max_workers": 1,
        "variables": {"precip": {"native_name": "precip"}},
        "_verbose": False, "_progress": False, "_max_retries": 3, "_retry_backoff": 0.0,
    }
    with pytest.raises(ServerBlockedError):
        http_mod.HTTPAdapter().fetch_data(config, "precip", date_range=(2000, 2005))
    # the first file: one request and three backed-off retries; no other file
    assert calls == ["https://fake/fake-2000.nc"] * 4
    assert [int(w) for w in waits if w >= 1] == [15, 30, 60]


def test_range_store_counts_and_caches(server):
    store = RangeStore(f"{server.base}/fake-2000.01.cog", limiter=_RateLimiter(0),
                       chunk_size=4096)
    with store.open() as f:
        a = f.read(10)
        f.seek(0)
        assert f.read(10) == a
        f.seek(-4, 2)
        assert len(f.read()) == 4
    assert store.n_requests == 2 == len(server.requests)
    with pytest.raises(FileNotFoundError):
        store.opener(store.url + ".aux.xml")
    assert len(server.requests) == 2                  # a sidecar lookup costs nothing
