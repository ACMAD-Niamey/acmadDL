import numpy as np
import xarray as xr
from .base import AdapterBase
from ..normalize import decode_months_since, select_lon
from ._robust import (_DEFAULT_MAX_RETRIES, _DEFAULT_RETRY_BACKOFF, _with_retry,
                      reject_if_degenerate)


# IRI Data Library's Ingrid convention: the variable is a path segment and the
# dataset is served from a `/dods` suffix. Most of the catalog is IRIDL, so this
# stays the default, but a catalog entry may override it — a THREDDS server
# (e.g. NOAA PSL) serves a whole NetCDF at one URL with no variable segment.
_DEFAULT_URL_TEMPLATE = "{base}/.{native_name}/dods"


def _build_url(product_config, native_name, base):
    template = product_config.get("url_template", _DEFAULT_URL_TEMPLATE)
    return template.format(base=base, native_name=native_name)


def _sort_ascending(ds, *coords):
    """Sort the named 1-D coords ascending, leaving already-ascending ones alone."""
    for coord in coords:
        if coord not in ds.coords or ds[coord].ndim != 1 or ds.sizes[coord] < 2:
            continue
        values = ds[coord].values
        if values[0] > values[-1]:
            ds = ds.sortby(coord)
    return ds


# A DAP2 response larger than the server's cap comes back truncated. netCDF4
# prints "DAP DATADDS packet is apparently too short" to stderr and hands us a
# zero-filled array WITHOUT raising — so a 30-year SST request silently becomes
# 30 years of 0 degC, land mask and all. Requesting in chunks keeps each
# response under any plausible cap; `_reject_degenerate` catches the rest.
_DEFAULT_MAX_REQUEST_YEARS = 5


def _reject_degenerate(ds, variable, label):
    """Reject a bitwise-constant or all-NaN multi-step OPeNDAP response (a truncated DAP packet).

    Thin wrapper over :func:`acmaddl.adapters._robust.reject_if_degenerate` (the shared detector
    used across all adapters and the fetch cache path); ``reject_all_nan=True`` keeps the stricter
    obs-chunk policy — an observed field over its valid domain is never entirely missing.
    """
    reject_if_degenerate(ds, variable, label, reject_all_nan=True)


def _load_obs_chunks(ds, variable, y0, y1, max_years, verbose, label):
    """Slice ``[y0, y1]`` in year blocks, loading and validating each in turn."""
    chunks = []
    for start in range(y0, y1 + 1, max_years):
        end = min(start + max_years - 1, y1)
        piece = ds.sel(time=slice(f"{start}-01-01", f"{end}-12-31"))
        if piece.sizes.get("time", 0) == 0:
            continue
        if verbose:
            print(f"[acmaddl:opendap] loading {start}-{end}")
        piece = piece.load()
        _reject_degenerate(piece, variable, f"{label} ({start}-{end})")
        chunks.append(piece)
    if not chunks:
        raise RuntimeError(f"OPeNDAP: {label} has no data in {y0}-{y1}")
    return xr.concat(chunks, dim="time") if len(chunks) > 1 else chunks[0]


class OPeNDAPAdapter(AdapterBase):
    def health_check(self, product_config, probe_remote=False):
        url = product_config.get("source_url")
        if not url:
            return {
                "healthy": False,
                "kind": "config",
                "message": "Missing source_url in product config.",
                "probe_remote": bool(probe_remote),
            }

        if not probe_remote:
            return {
                "healthy": True,
                "kind": "config",
                "message": "OPeNDAP adapter config is valid.",
                "probe_remote": False,
            }

        # split_streams entries carry a `{stream}` placeholder; probe the hindcast
        # endpoint (always present) so the literal braces don't reach the server.
        probe_url = url.format(stream="HINDCAST") if product_config.get("split_streams") else url
        try:
            ds = xr.open_dataset(probe_url, engine="netcdf4")
            ds.close()
            return {
                "healthy": True,
                "kind": "remote",
                "message": "OPeNDAP dataset opened successfully.",
                "probe_remote": True,
            }
        except Exception as e:
            return {
                "healthy": False,
                "kind": "remote",
                "message": f"OPeNDAP probe failed: {e}",
                "probe_remote": True,
            }

    def fetch_data(self, product_config, variable, date_range=None, region=None):
        # Resolve the request into one or more (stream, sub_range) segments. A
        # single segment reproduces today's start-year routing exactly; an
        # `append_streams` product spanning the hindcast/forecast boundary yields
        # two segments that we fetch separately and stitch on S.
        segments = self._resolve_streams(product_config, date_range)
        parts = [self._fetch_one_stream(product_config, variable, stream, rng, region)
                 for stream, rng in segments]
        if len(parts) == 1:
            return parts[0]
        # validate concat compatibility, then stitch on S. Segments are allowed to
        # be empty or smaller than their nominal year span -- e.g. under the
        # cfsv2 boundary-year overlap (see _resolve_streams), the forecast
        # segment for a Jan-init request has no Jan-2011 init and simply
        # contributes nothing for that year; this is expected, not an error.
        ref = parts[0]
        for p in parts[1:]:
            for d in ("Y", "X", "M", "L"):
                if d in ref.dims and d in p.dims and ref.sizes[d] != p.sizes[d]:
                    raise ValueError(
                        f"append_streams: stream segments disagree on dim {d!r} "
                        f"({ref.sizes[d]} vs {p.sizes[d]}); cannot concat.")
        combined = xr.concat(parts, dim="S")
        # Defensive dedup: the two streams shouldn't share an init (S value) for
        # any given month, but if a boundary-year overlap ever causes both
        # streams to return the same init (e.g. a catalog edge case), a
        # duplicate S value must never double-count downstream (mean/ensemble
        # math). Keep the first occurrence.
        if "S" in combined.coords:
            _, unique_idx = np.unique(combined["S"].values, return_index=True)
            if len(unique_idx) != combined.sizes["S"]:
                combined = combined.isel(S=np.sort(unique_idx))
        return combined

    def _fetch_one_stream(self, product_config, variable, stream, date_range, region):
        verbose = product_config.get("_verbose", True)
        max_retries = int(product_config.get("_max_retries", _DEFAULT_MAX_RETRIES))
        retry_backoff = float(product_config.get("_retry_backoff", _DEFAULT_RETRY_BACKOFF))
        var_cfg = product_config["variables"][variable]
        native_name = var_cfg["native_name"]
        base = product_config["source_url"].rstrip("/")
        # Stream routing: a split_streams entry carries a `{stream}` placeholder in
        # its source_url; the (HINDCAST/FORECAST) token is now chosen upstream by
        # `_resolve_streams` and passed in as `stream`. Mirrors the CCSR adapter's
        # split-stream routing, for the IRI NMME models that file hindcast and
        # forecast at sibling .HINDCAST/.FORECAST paths.
        if product_config.get("split_streams"):
            base = base.format(stream=stream.upper())
        url = _build_url(product_config, native_name, base)
        if verbose:
            print(f"[acmaddl:opendap] opening remote dataset: {url}")
        # NMME's "months since" time encoding is not CF-decodable by xarray, so
        # opening raw is the default and `decode_months_since` handles it below.
        # An observational source with an ordinary "days since" axis opts in, so
        # that its date_range can be sliced with real timestamps.
        decode_times = bool(product_config.get("decode_times", False))
        ds = _with_retry(
            lambda: xr.open_dataset(url, engine="netcdf4", decode_times=decode_times),
            max_retries, retry_backoff,
            label=f"OPeNDAP open {url}", verbose=verbose,
        )
        if "S" in ds.coords:
            # NMME OPeNDAP: S is encoded as "months since YYYY-MM-DD"
            units = ds["S"].attrs.get("units", "")
            if "months since" in units:
                s_years, s_months = decode_months_since(units, ds.S.values)
                mask = np.ones(len(ds.S), dtype=bool)
                if date_range:
                    y0, y1 = date_range
                    mask &= (s_years >= y0) & (s_years <= y1)
                if "init_months" in product_config:
                    mask &= np.isin(s_months, product_config["init_months"])
                ds = ds.sel(S=ds.S[mask])
                if verbose:
                    n = int(mask.sum())
                    print(f"[acmaddl:opendap] filtered S to {n} init times")
        # IRIDL observational monthly axis: T is "months since YYYY-MM-DD" (a
        # non-CF encoding xarray cannot decode, same as NMME's S). Decode it to
        # real monthly timestamps and expose as `time`, so the year-slice /
        # chunk-load / seasonal-mean path below treats it exactly like a PSL
        # "days since" obs source. Mirrors the S handling above.
        if "T" in ds.coords and "S" not in ds.coords:
            t_units = ds["T"].attrs.get("units", "")
            if "months since" in t_units:
                t_years, t_months = decode_months_since(t_units, ds["T"].values)
                times = np.array([np.datetime64(f"{y:04d}-{m:02d}")
                                  for y, m in zip(t_years, t_months)])
                ds = ds.rename({"T": "time"}).assign_coords(time=times)
                if verbose:
                    print(f"[acmaddl:opendap] decoded IRIDL T ('months since') "
                          f"to {len(times)} monthly timestamps")
        # An observational time axis is chunk-loaded after the region slice, so
        # each DAP request carries only the cells the caller asked for.
        obs_years = None
        if "S" not in ds.coords and date_range:
            y0, y1 = date_range
            if "year" in ds.dims or "year" in ds.coords:
                ds = ds.sel(year=slice(y0, y1))
            elif "time" in ds.coords:
                obs_years = (y0, y1)
        if region:
            lat_s, lat_n, lon_w, lon_e = region
            lat_name = "Y" if "Y" in ds.dims else "lat"
            lon_name = "X" if "X" in ds.dims else "lon"
            # A descending axis (NOAA PSL files lat 88N -> 88S) makes an
            # ascending `slice` select nothing at all — silently, and with no
            # error. Sort first so the slice means what it says.
            ds = _sort_ascending(ds, lat_name, lon_name)
            # Longitude selection is convention-aware (−180..180 vs 0..360) and
            # wrap-safe; a naive slice(lon_w, lon_e) silently under-selects when the
            # request and the source disagree on convention. See normalize.select_lon.
            ds = ds.sel({lat_name: slice(lat_s, lat_n)})
            ds = select_lon(ds, lon_w, lon_e, lon_name=lon_name)

        if obs_years is not None:
            max_years = int(product_config.get(
                "max_request_years", _DEFAULT_MAX_REQUEST_YEARS))
            ds = _load_obs_chunks(
                ds, native_name, obs_years[0], obs_years[1], max_years, verbose, url)

        # Select and average only the target-season lead months when specified.
        # NMME PENTAD_SAMPLES/.MONTHLY uses L = (lead_month - 0.5) half-integer
        # convention: L=0.5 → month 1 after init, L=1.5 → month 2, etc.
        # Without this, all 10 leads are returned and callers get an annual mean
        # instead of the correct seasonal mean.
        if "target_lead_months" in product_config and "L" in ds.dims:
            lead_months = product_config["target_lead_months"]
            target_L = [m - 0.5 for m in lead_months]
            avail_L = np.asarray(sorted(float(v) for v in ds.L.values))
            # Match on nearest value (tolerant of OPeNDAP float decoding noise)
            # rather than exact `in` membership, which can spuriously miss.
            sel_L = [lt for lt in target_L
                     if avail_L.size and np.min(np.abs(avail_L - lt)) < 1e-6]
            # A stream segment can legitimately have zero init times after the S
            # filter above -- e.g. under the cfsv2 boundary-year overlap (see
            # _resolve_streams / catalog.yaml), the FORECAST segment for a
            # Jan-init request has no Jan-2011 init at all, so mask.sum() == 0.
            # CONFIRMED against the live IRI endpoint: reducing L via `.isel`/
            # `.sel(...).mean("L")` over a 0-length S dim leaves the lazy
            # netCDF4/OPeNDAP-backed array's *actual* underlying data 5-D even
            # though the resulting Dataset's declared `.dims`/`.sizes` correctly
            # report L as dropped (4-D) -- a metadata/lazy-array desync specific
            # to a 0-length outer dim that only manifests when xr.concat later
            # forces materialization (`ValueError: ... 4 dimension(s) ... 5
            # dimension(s)`), not from `.sizes` alone. A numpy-backed dataset
            # does not reproduce this. Since an empty (S=0) segment carries no
            # real data regardless of transformation, sidestep the lazy backend
            # entirely: rebuild it as a genuinely empty, eager (numpy-backed)
            # Dataset with the post-reduction dims/shape, so concat sees a real
            # 4-D array rather than a mislabeled lazy one.
            if ds.sizes.get("S", 1) == 0:
                data_vars = {}
                for name, var in ds.data_vars.items():
                    dims = tuple(d for d in var.dims if d != "L")
                    shape = tuple(ds.sizes[d] for d in dims)
                    data_vars[name] = (dims, np.empty(shape, dtype=var.dtype))
                non_l_dims = {d for v in data_vars.values() for d in v[0]}
                coords = {d: ds[d].values for d in non_l_dims if d in ds.coords}
                ds = xr.Dataset(data_vars, coords=coords)
            elif sel_L:
                idx_L = [int(np.argmin(np.abs(avail_L - lt))) for lt in sel_L]
                ds = ds.isel(L=idx_L).mean("L").load()
            # If none of the target leads are available (and S is non-empty)
            # fall through unchanged (caller's existing post-processing handles it)

        return ds
