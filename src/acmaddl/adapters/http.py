import fnmatch
import gzip
import os
import re
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray as xr
from .base import AdapterBase
from ..normalize import select_lon
from ._issuance import enumerate_files, issuance_config, lead_timedelta
from ._robust import (
    _DEFAULT_MAX_RETRIES,
    _DEFAULT_RETRY_BACKOFF,
    _RateLimiter,
    _with_retry,
)


_MAX_WORKERS = 8

# Start day of each sub-monthly slice, so a filename's dekad/pentad index maps
# to a real timestamp. Dekads: day 1/11/21. Pentads: day 1/6/11/16/21/26.
_DEKAD_DAYS = (1, 11, 21)
_PENTAD_DAYS = (1, 6, 11, 16, 21, 26)


def _enumerate_timeseries(file_pattern, date_range, product_config):
    """Return ``[(filename, timestamp)]`` for a time-series product.

    ``timestamp`` is an explicit ``pd.Timestamp`` for sub-monthly cadences —
    where the filename's dekad/pentad index (``{dekad}``/``{pentad}``) cannot be
    unambiguously read back from the name — and ``None`` for monthly/yearly
    files, whose timestamp the opener infers from the filename as before.

    A product declaring ``time_from_pattern`` gets an explicit timestamp for the
    monthly and yearly cadences too, because for those the enumeration replaces
    the file's own time axis rather than merely labelling a raster (see
    ``_stamp_time``).

    Month pruning via ``init_months`` applies to every sub-monthly cadence, so a
    JJAS analysis fetches four months of dekads, not twelve.
    """
    import pandas as pd

    from_pattern = bool(product_config.get("time_from_pattern"))

    if not date_range:
        # No range: one recent file, accounting for observational lag.
        recent = datetime.now().replace(day=1) - timedelta(days=60)
        if "{dekad}" in file_pattern:
            return [(file_pattern.format(year=recent.year, month=recent.month, dekad=1),
                     pd.Timestamp(recent.year, recent.month, 1))]
        if "{pentad}" in file_pattern:
            return [(file_pattern.format(year=recent.year, month=recent.month, pentad=1),
                     pd.Timestamp(recent.year, recent.month, 1))]
        if "{month" in file_pattern:
            return [(file_pattern.format(year=recent.year, month=recent.month),
                     pd.Timestamp(recent.year, recent.month, 1) if from_pattern else None)]
        return [(file_pattern.format(year=recent.year),
                 pd.Timestamp(recent.year, 1, 1) if from_pattern else None)]

    y0, y1 = date_range
    months = product_config.get("init_months") or range(1, 13)

    if "{dekad}" in file_pattern:
        return [(file_pattern.format(year=y, month=m, dekad=d),
                 pd.Timestamp(y, m, _DEKAD_DAYS[d - 1]))
                for y in range(y0, y1 + 1) for m in months for d in (1, 2, 3)]
    if "{pentad}" in file_pattern:
        return [(file_pattern.format(year=y, month=m, pentad=p),
                 pd.Timestamp(y, m, _PENTAD_DAYS[p - 1]))
                for y in range(y0, y1 + 1) for m in months for p in range(1, 7)]
    if "{month" in file_pattern:
        return [(file_pattern.format(year=y, month=m),
                 pd.Timestamp(y, m, 1) if from_pattern else None)
                for y in range(y0, y1 + 1) for m in months]
    return [(file_pattern.format(year=y),
             pd.Timestamp(y, 1, 1) if from_pattern else None)
            for y in range(y0, y1 + 1)]


def _list_directory(url):
    """Filenames linked from an HTTP directory listing (Apache/nginx style)."""
    with urllib.request.urlopen(url, timeout=30) as resp:
        html = resp.read().decode(errors="replace")
    return [os.path.basename(h) for h in re.findall(r'href="([^"]+)"', html)]


def _resolve_wildcards(base, entries, list_dir=None):
    """Resolve a ``*`` in each enumerated relative path against the server.

    NCEI stamps its GPCP filenames with the processing date
    (``…_d202403_c20240607.nc``), which no static pattern can predict, so the
    catalog writes the predictable part and a ``*`` for the rest. Each directory
    is listed at most once per fetch — a 45-year pull costs 45 listings, not 540.

    Several matches means the month was reprocessed; the lexically greatest name
    wins, which for a fixed-width ``_c<YYYYMMDD>`` suffix is the most recent
    reprocessing. No match raises rather than skipping the month: a silently
    dropped file is indistinguishable downstream from data the source never
    published.

    The match is a plain fnmatch against the pattern, so a sibling stream in the
    same directory is excluded by the literal part of the name — NCEI keeps
    ``gpcp_v02r03-preliminary_monthly_…`` alongside the final
    ``gpcp_v02r03_monthly_…``, and only the latter matches the final product's
    pattern.
    """
    list_dir = _list_directory if list_dir is None else list_dir
    listings = {}
    resolved = []
    for rel, stamp in entries:
        if "*" not in rel:
            resolved.append((rel, stamp))
            continue
        dirpart, _, namepat = rel.rpartition("/")
        dir_url = f"{base.rstrip('/')}/{dirpart}/" if dirpart else f"{base.rstrip('/')}/"
        if dir_url not in listings:
            listings[dir_url] = list_dir(dir_url)
        hits = sorted(n for n in listings[dir_url] if fnmatch.fnmatch(n, namepat))
        if not hits:
            raise FileNotFoundError(
                f"no file matching {namepat!r} in {dir_url} "
                f"({len(listings[dir_url])} name(s) listed)"
            )
        resolved.append((f"{dirpart}/{hits[-1]}" if dirpart else hits[-1], stamp))
    return resolved


def _resolve_max_workers(product_config, n_urls):
    """Effective NetCDF download concurrency for this product.

    A per-product ``max_workers`` in the catalog entry caps concurrent
    connections (e.g. so a multi-year pull of multi-GiB CHIRPS NetCDFs doesn't
    open eight huge downloads at once). It can only lower concurrency, never
    raise it above the global ``_MAX_WORKERS`` ceiling, and is bounded by the
    number of files actually being fetched. Always at least one worker.
    """
    cap = product_config.get("max_workers")
    workers = _MAX_WORKERS if cap is None else min(_MAX_WORKERS, int(cap))
    return max(1, min(workers, n_urls))


def _iter(items, desc, enabled=True):
    if not enabled:
        return items
    try:
        from tqdm.auto import tqdm
        return tqdm(items, desc=desc)
    except Exception:
        return items


def _subset_region(ds, region):
    """Subset a dataset to a region [lat_s, lat_n, lon_w, lon_e] immediately after loading.

    Longitude goes through ``normalize.select_lon`` rather than a plain slice,
    because the crop happens here — before normalize sees the data — and the
    source's longitude convention is only knowable once a file is open. A bbox
    written in -180..180 against a 0..360 grid (obs/gpcp-v2-3) would otherwise
    lose everything west of the prime meridian, and normalize cannot recover
    cells that were already dropped. This is the same helper, for the same
    reason, that the OPeNDAP adapter uses on its pre-normalize crop.
    """
    if region is None:
        return ds
    lat_s, lat_n, lon_w, lon_e = region
    buf = 1.0
    lat_dim = "latitude" if "latitude" in ds.dims else "lat" if "lat" in ds.dims else None
    lon_dim = "longitude" if "longitude" in ds.dims else "lon" if "lon" in ds.dims else None
    if lat_dim and lon_dim:
        ds = ds.sortby(lat_dim)
        ds = ds.sel({lat_dim: slice(lat_s - buf, lat_n + buf)})
        ds = select_lon(ds, lon_w - buf, lon_e + buf, lon_name=lon_dim)
    return ds.load()


def _open_raster(url, region, variable=None, fill_value=None):
    """Open a COG via HTTP range reads, subsetting to region without downloading the whole file.

    Returns a 2-D ``(latitude, longitude)`` dataset carrying no time coordinate:
    the caller decides what time the raster represents. `_open_cog_subset`
    infers it from the filename (the observational convention); the issuance
    path stamps init/lead explicitly.

    CHIRPS and similar climate COGs often store a sentinel fill (e.g. -9999) without
    declaring it in the TIFF nodata tag — rasterio then passes it through as data.
    If `fill_value` is provided, mask those cells as NaN before returning.

    GDAL caches vsicurl responses (including transient HTTP errors) keyed by
    URL at the process level. That cache replays a previous 503 on retry, so
    we disable it for the open via CPL_VSIL_CURL_NON_CACHED. Cost is a few
    extra range reads per file; benefit is that the adapter's retry loop
    actually re-hits the server instead of looping on a cached failure.
    """
    import rasterio
    import rioxarray  # noqa: F401
    with rasterio.Env(CPL_VSIL_CURL_NON_CACHED="/vsicurl/"):
        ds = xr.open_dataset(url, engine="rasterio")
    if region:
        lat_s, lat_n, lon_w, lon_e = region
        buf = 1.0
        # COGs use x/y coords (lon/lat), y is descending
        ds = ds.sel(y=slice(lat_n + buf, lat_s - buf), x=slice(lon_w - buf, lon_e + buf))
    ds = ds.load()
    if fill_value is not None and "band_data" in ds:
        ds["band_data"] = ds["band_data"].where(ds["band_data"] != fill_value)
    # Rename COG default coords/vars to standard names
    renames = {}
    if "x" in ds.dims:
        renames["x"] = "longitude"
    if "y" in ds.dims:
        renames["y"] = "latitude"
    if "band_data" in ds and variable:
        renames["band_data"] = variable
    if renames:
        ds = ds.rename(renames)
    # Drop band dim if it's size 1
    if "band" in ds.dims and ds.sizes["band"] == 1:
        ds = ds.squeeze("band", drop=True)
    return ds


def _retrieve(url, dest):
    """Download ``url`` to ``dest``, transparently gunzipping a gzipped payload.

    DWD serves the GPCC products as ``.nc.gz``. netCDF4 reads a file's internal
    zlib chunk compression but not a gzip *wrapper*, so the payload is expanded
    in place before the opener ever sees it. Keyed off the URL suffix rather
    than a catalog knob: a ``.gz`` URL is unambiguous, and a product whose
    files are gzipped upstream has no reason to declare it twice.
    """
    urllib.request.urlretrieve(url, dest)
    if url.endswith(".gz"):
        with gzip.open(dest, "rb") as fh:
            payload = fh.read()
        with open(dest, "wb") as fh:
            fh.write(payload)


def _stamp_time(ds, stamp, url):
    """Replace a file's time axis with the timestamp its filename implies.

    Opt-in per product via ``time_from_pattern``, for sources whose in-file time
    encoding is either undecodable or not self-contained. Both DWD GPCC streams
    are: monitoring_v2020 writes ``20240301.0`` with units ``day as %Y%m%d.%f``
    (a GrADS convention, not a CF ``<unit> since <epoch>``, so nothing decodes
    it), and first_guess writes ``0`` with units ``months since 2024-3-1``
    (unpadded, so decoding raises — and since every file says ``0`` against its
    own reference month, a concat keeps the first file's units and collapses the
    whole record onto one date).

    The enumeration that built the URL is the only thing that knows what a
    single-slice file holds, so it — not the file — is the authority here. This
    is the same principle the issuance path already follows. A multi-slice file
    has no single answer, so it raises rather than mislabel silently.
    """
    when = [_ns_stamp(stamp)]
    if "time" not in ds.dims:
        return ds.expand_dims(time=when)
    if ds.sizes["time"] != 1:
        raise ValueError(
            f"time_from_pattern expects one time step per file, but {url} holds "
            f"{ds.sizes['time']}; a filename can only name a single step. Drop "
            f"time_from_pattern for this product and let its own axis decode."
        )
    return ds.assign_coords(time=when)


def _ns_stamp(value):
    """Coerce a timestamp to nanosecond-precision datetime64. Building a time
    coordinate from a coarser resolution makes xarray/pandas emit a
    non-nanosecond UserWarning; with one raster opened per file, that otherwise
    floods a multi-file download. Forcing ns here silences it at the source."""
    return np.datetime64(pd.Timestamp(value), "ns")


def _open_cog_subset(url, region, variable=None, fill_value=None, timestamp=None):
    """`_open_raster` plus a time coordinate.

    ``timestamp`` stamps the raster explicitly — needed for sub-monthly cadences
    (a dekad/pentad index in the filename can't be read back unambiguously). When
    it is ``None``, the timestamp is inferred from the filename: monthly/COG
    names carry YYYY.MM (e.g. chirps-v3.0.2020.01.cog); annual rasters carry only
    the year (e.g. chirps-v2.0.2020.tif), stamped January 1.
    """
    ds = _open_raster(url, region, variable=variable, fill_value=fill_value)
    if "time" in ds.dims:
        return ds
    if timestamp is not None:
        return ds.expand_dims(time=[_ns_stamp(timestamp)])
    m = re.search(r'(\d{4})\.(\d{2})', url)
    if m:
        ds = ds.expand_dims(time=[_ns_stamp(f"{m.group(1)}-{m.group(2)}-01")])
    else:
        ym = re.search(r'\.(\d{4})\.(?:cog|tif)$', url)
        if ym:
            ds = ds.expand_dims(time=[_ns_stamp(f"{ym.group(1)}-01-01")])
    return ds


class HTTPAdapter(AdapterBase):
    def health_check(self, product_config, probe_remote=False):
        url = product_config.get("source_url")
        if not url:
            return {
                "healthy": False,
                "kind": "config",
                "message": "Missing source_url in product config.",
                "probe_remote": bool(probe_remote),
            }

        if not (url.startswith("http://") or url.startswith("https://")):
            return {
                "healthy": False,
                "kind": "config",
                "message": f"Unsupported source_url protocol: {url}",
                "probe_remote": bool(probe_remote),
            }

        if not probe_remote:
            return {
                "healthy": True,
                "kind": "config",
                "message": "HTTP adapter config is valid.",
                "probe_remote": False,
            }

        try:
            request = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(request, timeout=10):
                pass
            return {
                "healthy": True,
                "kind": "remote",
                "message": "HTTP source reachable.",
                "probe_remote": True,
            }
        except Exception as e:
            return {
                "healthy": False,
                "kind": "remote",
                "message": f"HTTP source unreachable: {e}",
                "probe_remote": True,
            }

    def fetch_data(self, product_config, variable, date_range=None, region=None):
        verbose = product_config.get("_verbose", True)
        progress = product_config.get("_progress", True)
        # Strict by default: any per-file fetch error aborts the whole pull.
        # Partial results have silently poisoned downstream caches and skill
        # metrics in the past (e.g. CHIRPS rate-limit losing two decades of
        # months but pipeline still claiming success). Callers that genuinely
        # want best-effort fetches can opt in via fetch(..., allow_partial=True).
        allow_partial = product_config.get("_allow_partial", False)
        max_retries = int(product_config.get("_max_retries", _DEFAULT_MAX_RETRIES))
        retry_backoff = float(product_config.get("_retry_backoff", _DEFAULT_RETRY_BACKOFF))
        # request_interval is resolved as a *floor*: a per-product catalog value
        # (plain "request_interval") sets a safe minimum pace that a caller's
        # fetch(request_interval=...) — injected here as "_request_interval" —
        # may raise but not silently undercut. Native CHIRPS entries declare one
        # so multi-file pulls stay under the UCSB CrowdSec rate ban (~2 req/s).
        _caller_interval = product_config.get("_request_interval")
        _catalog_interval = product_config.get("request_interval")
        request_interval = max(
            float(_caller_interval) if _caller_interval is not None else 0.0,
            float(_catalog_interval) if _catalog_interval is not None else 0.0,
        )
        rate_limiter = _RateLimiter(request_interval)
        base_url = product_config["source_url"]
        fmt = product_config.get("format", "netcdf")

        issuance = issuance_config(product_config)
        if issuance is not None:
            return self._fetch_issuance(
                product_config, variable, issuance, base_url, fmt, region,
                rate_limiter, max_retries, retry_backoff, verbose, progress,
                allow_partial,
            )

        file_pattern = product_config.get("file_pattern")
        if not file_pattern:
            raise ValueError("HTTP adapter requires 'file_pattern' in product config")

        # (filename, timestamp) pairs. timestamp is explicit for sub-monthly
        # cadences (dekad/pentad), else None and inferred by the opener.
        entries = _enumerate_timeseries(file_pattern, date_range, product_config)
        # A `*` in the pattern means the filename carries something unpredictable
        # (NCEI's processing-date suffix), so ask the server what it actually has.
        if "*" in file_pattern:
            entries = _resolve_wildcards(base_url, entries)
        base = base_url.rstrip("/")
        urls = [f"{base}/{f}" for f, _ in entries]
        stamps = [ts for _, ts in entries]

        # NetCDF downloads run in a worker pool capped per-product (COG/TIF is
        # always sequential, so its worker count is 1).
        netcdf_workers = _resolve_max_workers(product_config, len(urls))
        if verbose:
            workers = netcdf_workers if fmt not in ("cog", "tif") else 1
            print(f"[acmaddl:http] downloading {len(urls)} file(s) "
                  f"(format={fmt}, workers={workers}, "
                  f"max_retries={max_retries}, request_interval={request_interval}s)")

        if fmt in ("cog", "tif"):
            var_cfg = product_config.get("variables", {}).get(variable, {})
            native_name = var_cfg.get("native_name", variable)
            fill_value = var_cfg.get("fill_value")
            datasets = []
            failures = []
            for url, ts in _iter(list(zip(urls, stamps)), "acmadDL HTTP download",
                                 enabled=progress):
                rate_limiter.wait()
                try:
                    ds = _with_retry(
                        lambda u=url, t=ts: _open_cog_subset(
                            u, region, variable=native_name, fill_value=fill_value,
                            timestamp=t),
                        max_retries, retry_backoff,
                        label=f"COG open {url}", verbose=verbose,
                    )
                    datasets.append(ds)
                except Exception as e:
                    failures.append((url, e))
                    print(f"Error fetching {url}: {e}")
        else:
            # `time_from_pattern` products get their time axis from the
            # enumeration, so the file's own axis is neither decoded nor trusted.
            # Stamps are withheld otherwise: the dekad/pentad timestamps that
            # _enumerate_timeseries always produces are for the raster path,
            # where they label an image that carries no time coordinate at all.
            from_pattern = bool(product_config.get("time_from_pattern"))
            datasets, failures = self._fetch_netcdf_parallel(
                urls, region, progress, rate_limiter, max_retries, retry_backoff,
                verbose, netcdf_workers,
                stamps=stamps if from_pattern else None,
                decode_times=not from_pattern)

        if failures and not allow_partial:
            raise RuntimeError(
                f"HTTP adapter: {len(failures)}/{len(urls)} file(s) failed; "
                f"refusing to return partial data (pass allow_partial=True to "
                f"fetch() to override). First failure: {failures[0][0]}: {failures[0][1]}"
            )
        if not datasets:
            raise RuntimeError("No data files retrieved")
        return xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]

    def _fetch_issuance(self, product_config, variable, issuance, base_url, fmt,
                        region, rate_limiter, max_retries, retry_backoff, verbose,
                        progress, allow_partial):
        """Fetch an issuance-keyed forecast archive into (init_time, lead_time, y, x).

        One file per (issuance date, lead). Files are opened without any
        filename time-sniffing — the coordinates come from the enumeration that
        built the URL, which is the only place that knows what each file means.
        """
        init_dates = product_config.get("_init_dates")
        if not init_dates:
            raise ValueError(
                "this product is issuance-keyed (its catalog entry has an "
                "'issuance' block), so fetch() needs init=... — a 'YYYY-MM-DD' "
                "issuance date, or a sequence of them."
            )
        files = enumerate_files(base_url, issuance, init_dates)
        var_cfg = product_config.get("variables", {}).get(variable, {})
        native_name = var_cfg.get("native_name", variable)
        fill_value = var_cfg.get("fill_value")

        if verbose:
            print(f"[acmaddl:http] downloading {len(files)} issuance file(s) "
                  f"({len(init_dates)} init x {len(issuance['leads'])} lead, "
                  f"format={fmt}, request_interval={rate_limiter.min_interval}s)")

        opened, failures = {}, []
        for handle in _iter(files, "acmadDL issuance download", enabled=progress):
            rate_limiter.wait()
            try:
                opened[(handle.init, handle.lead)] = _with_retry(
                    lambda h=handle: self._open_issuance_file(
                        h.url, fmt, region, native_name, fill_value,
                        max_retries, retry_backoff, verbose),
                    max_retries, retry_backoff,
                    label=f"issuance open {handle.url}", verbose=verbose,
                )
            except Exception as e:
                failures.append((handle.url, e))
                print(f"Error fetching {handle.url}: {e}")

        if failures and not allow_partial:
            raise RuntimeError(
                f"HTTP adapter: {len(failures)}/{len(files)} issuance file(s) failed; "
                f"refusing to return partial data (pass allow_partial=True to "
                f"fetch() to override). First failure: {failures[0][0]}: {failures[0][1]}"
            )
        if not opened:
            raise RuntimeError("No data files retrieved")

        return self._assemble_issuance(opened, issuance, allow_partial)

    @staticmethod
    def _open_issuance_file(url, fmt, region, native_name, fill_value,
                            max_retries, retry_backoff, verbose):
        if fmt in ("cog", "tif"):
            return _open_raster(url, region, variable=native_name,
                                fill_value=fill_value)
        _, ds = HTTPAdapter._download_one(url, region, None, max_retries,
                                          retry_backoff, verbose)
        return ds

    @staticmethod
    def _assemble_issuance(opened, issuance, allow_partial):
        """(init, lead) -> dataset, into a (init_time, lead_time, ...) cube.

        With allow_partial the grid can be ragged, so leads are reindexed to the
        declared full set: a missing lead becomes NaN rather than silently
        shifting every later lead down by one.
        """
        leads = issuance["leads"]
        lead_coord = lead_timedelta(leads, issuance["lead_units"])
        inits = sorted({init for init, _ in opened})

        per_init = []
        for init in inits:
            present = [lead for lead in leads if (init, lead) in opened]
            if not present:
                continue
            stacked = xr.concat(
                [opened[(init, lead)] for lead in present],
                dim=xr.IndexVariable(
                    "lead_time", lead_timedelta(present, issuance["lead_units"])),
            )
            if len(present) != len(leads):
                stacked = stacked.reindex(lead_time=lead_coord)
            per_init.append(stacked)

        combined = xr.concat(
            per_init,
            dim=xr.IndexVariable("init_time", np.array(inits, dtype="datetime64[ns]")),
        )
        # valid_time is what an observation would be stamped with, so a forecast
        # can be verified without the caller re-deriving it.
        combined = combined.assign_coords(
            valid_time=combined.init_time + combined.lead_time
        )
        return combined

    @staticmethod
    def _download_one(url, region, rate_limiter=None, max_retries=0, backoff=0.0,
                      verbose=True, stamp=None, decode_times=True):
        """Download a single NetCDF file and return (url, xr.Dataset).

        Per-file retries + an optional shared rate limiter live here so they
        apply to each worker thread; the pool itself only bounds concurrency.

        ``stamp`` (with ``decode_times=False``) replaces the file's time axis
        with what its filename implies — per file, BEFORE the caller concatenates
        them, which is the only point at which a non-self-contained time
        encoding can still be read correctly.
        """
        def _do():
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                _retrieve(url, tmp.name)
                ds = xr.open_dataset(tmp.name, decode_times=decode_times)
                if stamp is not None:
                    ds = _stamp_time(ds, stamp, url)
                ds = _subset_region(ds, region)
                os.unlink(tmp.name)
            return ds

        if rate_limiter is not None:
            rate_limiter.wait()
        ds = _with_retry(_do, max_retries, backoff,
                         label=f"NetCDF download {url}", verbose=verbose)
        return url, ds

    def _fetch_netcdf_parallel(self, urls, region, progress, rate_limiter,
                               max_retries, retry_backoff, verbose, max_workers,
                               stamps=None, decode_times=True):
        results = {}
        errors = []
        stamps = [None] * len(urls) if stamps is None else stamps
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._download_one, u, region,
                            rate_limiter, max_retries, retry_backoff, verbose,
                            stamp=st, decode_times=decode_times): u
                for u, st in zip(urls, stamps)
            }
            for fut in _iter(as_completed(futures), "acmadDL HTTP download", enabled=progress):
                try:
                    url, ds = fut.result()
                    results[url] = ds
                except Exception as e:
                    errors.append((futures[fut], e))
                    print(f"Error fetching {futures[fut]}: {e}")
        datasets = [results[u] for u in urls if u in results]
        return datasets, errors
