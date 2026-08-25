import logging
import numpy as np
import xarray as xr
from nuthatch import cache

from . import catalog
from .adapters import get_adapter
from .adapters._robust import reject_if_degenerate, DegenerateResponseError
from .normalize import normalize, sanitize_for_netcdf
from .region import resolve_region

from .storage import save

_CACHE_VERSION = 8  # bump when adapter logic or normalization changes


def _fetch_raw(product: str, variable: str, config: dict,
               date_range, region, reject_degenerate: bool = False) -> xr.Dataset:
    """Download raw (un-normalized) data from the adapter. Cached by _fetch_raw_cached.

    When ``reject_degenerate`` is set (opt-in, via ``fetch(degenerate_attempts>1)``), a
    bitwise-constant / zero-filled response is rejected here — before it can be memoized — so a
    truncated DAP packet never poisons the cache, for any adapter. Off by default so mechanics
    that legitimately use constant fields (and callers who don't opt in) are unaffected. all-NaN
    is tolerated (a land-masked region can legitimately be all-NaN)."""
    adapter = get_adapter(config["adapter"])
    ds = adapter.fetch_data(config, variable, date_range=date_range, region=region)
    if reject_degenerate:
        reject_if_degenerate(ds, variable, f"acmaddl.fetch({product!r})", reject_all_nan=False)
    return ds


@cache(namespace="acmaddl", version=str(_CACHE_VERSION), backend="basic",
       cache_args=["product", "variable", "date_range", "region", "init_months",
                   "init_date", "target_months"])
def _fetch_raw_cached(product: str, variable: str, config: dict,
                      date_range, region, init_months=None,
                      init_date=None, target_months=None,
                      reject_degenerate=False) -> xr.Dataset:
    """Nuthatch-cached wrapper around _fetch_raw.

    Cache key uses only the stable scalar arguments, not the full config dict,
    to avoid dict-hashing issues. init_months is included because it determines
    which seasonal slice is fetched; init_date is included because S2S
    forecasts are keyed on a single issuance date. target_months is included
    because the seasonal adapters reduce over lead INSIDE this cached call —
    ccsr averages the leads whose target month falls in the season, opendap/s3
    select `target_lead_months`, and the CDS adapter requests only the season's
    leadtimes — so two different target seasons at the same init month would
    otherwise collide and the second would silently be served the first's
    lead-averaged data. reject_degenerate is NOT a cache arg — it only governs
    whether a fresh (cache-miss) response is validated before being memoized,
    so it must not fork the key.
    """
    return _fetch_raw(product, variable, config, date_range, region,
                      reject_degenerate=reject_degenerate)

SEASON_MONTHS = {
    "DJF": (12, 2), "JFM": (1, 3), "FMA": (2, 4), "MAM": (3, 5),
    "AMJ": (4, 6), "MJJ": (5, 7), "JJA": (6, 8), "JAS": (7, 9),
    "ASO": (8, 10), "SON": (9, 11), "OND": (10, 12), "NDJ": (11, 1),
}

_MONTH_INITIALS = "JFMAMJJASOND"


def season_to_months(code):
    """(start_month, end_month) for a contiguous month-initial season code.

    Accepts the classic 3-month codes ("MAM", "OND", ...) and ANY contiguous
    run of 2-12 month initials — e.g. "JJAS" (Jun-Sep, the 4-month season the
    Sahel/GHA outlooks use), "ND" (Nov-Dec), "NDJF" (Nov-Feb, wraparound).
    The run is matched circularly against the month-initial sequence
    JFMAMJJASOND; every valid contiguous run of two or more months has exactly
    one starting position, so codes are unambiguous. Raises ``ValueError`` for
    unknown codes (single letters are ambiguous by construction: J, M, A).
    """
    code_u = str(code).upper()
    if code_u in SEASON_MONTHS:
        return SEASON_MONTHS[code_u]
    n = len(code_u)
    if not (2 <= n <= 12) or any(c not in _MONTH_INITIALS for c in code_u):
        raise ValueError(
            f"Unknown season code {code!r}: expected a contiguous run of 2-12 "
            "month initials (e.g. 'MAM', 'JJAS', 'NDJF')."
        )
    doubled = _MONTH_INITIALS * 2
    starts = [i for i in range(12) if doubled[i:i + n] == code_u]
    if len(starts) != 1:
        raise ValueError(
            f"Season code {code!r} does not name a contiguous month run "
            f"(matches {len(starts)} starting months)."
        )
    s = starts[0] + 1
    e = ((starts[0] + n - 1) % 12) + 1
    return (s, e)


def _log(verbose, msg):
    if verbose:
        print(f"[acmaddl] {msg}")


def _set_nuthatch_verbosity(verbose):
    """Make ``verbose`` govern the nuthatch cache library's logging too.

    nuthatch pins its own logger to INFO with a dedicated handler, so it prints
    a "Found cache …"/"Caching result …" line on every access regardless of the
    caller. That's the caching internals, not something a fetch caller asked to
    see — so with ``verbose=False`` we quiet it (INFO -> WARNING). The child
    ``nuthatch.nuthatch`` logger has its level set explicitly upstream, so it
    must be lowered directly; the parent covers the other submodules.
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.getLogger("nuthatch").setLevel(level)
    logging.getLogger("nuthatch.nuthatch").setLevel(level)


def parse_target(target, year=None):
    if isinstance(target, tuple) and len(target) == 2:
        return target
    if isinstance(target, str):
        from datetime import datetime
        import calendar
        s, e = season_to_months(target)   # any contiguous month run, e.g. "JJAS"
        y = year or datetime.now().year
        y_end = y + (1 if e < s else 0)
        return (datetime(y, s, 1), datetime(y_end, e, calendar.monthrange(y_end, e)[1]))
    raise ValueError(f"Unknown target: {target}")


def parse_init(init):
    if isinstance(init, str):
        from datetime import datetime
        if len(init) == 10:
            # YYYY-MM-DD (S2S-style daily issuance)
            return datetime.strptime(init, "%Y-%m-%d")
        # YYYY-MM (seasonal-style monthly init)
        return datetime.strptime(init[:7], "%Y-%m")
    return init


def _is_init_sequence(init):
    """True when ``init`` names several issuances rather than one.

    Anything iterable that is not a string or a date: a list, a tuple, a numpy
    array, a pandas DatetimeIndex.
    """
    from datetime import date, datetime

    if isinstance(init, (str, bytes, datetime, date)):
        return False
    return hasattr(init, "__iter__")


def _daily_issuance_key(init):
    """The 'YYYY-MM-DD' form of a single issuance, or None if it is month-keyed.

    Seasonal products are issued for a month; sub-seasonal and short-range ones
    for a specific day. Only the latter can be enumerated as issuance dates.
    """
    if isinstance(init, str):
        return init if len(init) == 10 else None
    if hasattr(init, "strftime"):
        return init.strftime("%Y-%m-%d")
    return None


def _match_lon_convention(obj, target_lons):
    """Roll `obj`'s longitude into the target grid's convention before interpolation.

    Interpolating a 0..360 source onto a target grid that includes negative
    longitudes (e.g. an Africa grid spanning -25..55) silently NaNs the western
    band, because the source has no coordinates at negative longitudes. Shift the
    source into the target's convention (0..360 <-> -180..180) and re-sort so the
    interpolation covers the whole target.
    """
    if "lon" not in getattr(obj, "coords", {}):
        return obj
    lon = np.asarray(obj["lon"].values, dtype=float)
    if lon.size == 0:
        return obj
    t_min, t_max = float(np.min(target_lons)), float(np.max(target_lons))
    s_min, s_max = float(np.min(lon)), float(np.max(lon))
    if t_min < 0.0 and s_max > 180.0:          # target -180..180, source 0..360
        obj = obj.assign_coords(lon=(((obj["lon"] + 180.0) % 360.0) - 180.0))
        return obj.sortby("lon")
    if t_max > 180.0 and s_min < 0.0:          # target 0..360, source -180..180
        obj = obj.assign_coords(lon=(obj["lon"] % 360.0))
        return obj.sortby("lon")
    return obj


def fetch(product, variable, init=None, target=None, region=None,
          hindcast=None, destination=None, format="netcdf", verbose=True,
          progress=True, cache=True, allow_partial=False,
          max_retries=3, retry_backoff=1.0, request_interval=0.0,
          reforecast=False, boundary="center", region_buffer=1.5,
          year_index=False, seasonal=None, grid_res=None, regrid_to=None,
          months=None, degenerate_attempts=1):
    """Fetch, normalize, and optionally save climate data.

    init names the forecast issuance. Seasonal products take a month
    ("2026-07"); sub-seasonal and short-range ones take a day ("2026-07-05").
    Issuance-keyed products (whose catalog entry has an `issuance` block) also
    accept a *sequence* of dates, which is how you fetch the same calendar
    issuance of many years for a hindcast-skill analysis:

        acmaddl.fetch("chc/chirps-gefs-daily", "precip",
                      init=[f"{y}-06-30" for y in range(2001, 2020)],
                      region="ethiopia.shp")

    The result is stacked on init_time, with lead_time and valid_time coords.

    months restricts an observational fetch to those calendar months, e.g.
    months=[6, 7, 8, 9] for a JJAS analysis. For products stored one file per
    (year, month) this prunes the *download*, not just the result: a four-month
    season over 45 years is 180 files rather than 540. Against a rate-limited
    server that is the difference between a pull and a ban.

    region accepts a bbox [lat_s, lat_n, lon_w, lon_e], a path to a .shp
    shapefile, or a shapely / geopandas geometry. For shapefiles and geometries
    the bounding box drives upstream slicing and the result is masked to the
    true polygon (cells outside it become NaN). Shapefile/geometry support
    needs the `geo` extra: pip install 'acmadDL[geo]'.

    boundary selects which grid cells count as inside the region, for both bbox
    and shapefile/geometry inputs:
      "center" (default): keep a cell only if its centre lies inside the region
        — the xarray/CDO/rasterio convention, and unbiased for area means.
      "cover": keep every cell the region touches, so it's covered to its true
        edges (matches rasterio's all_touched=True). Best for display/masking
        and small or coarse-grid regions.

    region_buffer (degrees, default 1.5): only used when boundary="cover". How
    far to pad the fetched bbox so boundary cells aren't clipped off before the
    region is covered. Must exceed half the grid spacing (1.5° covers grids up
    to ~3°); raise it for coarser products.

    cache=True (default): results are cached locally via nuthatch.
    cache=False: bypass the cache and fetch fresh from the source.

    allow_partial=False (default): adapters that fetch multiple files raise
    if any requested file fails, so partial results never reach the cache or
    downstream metrics. Set True to opt into best-effort behaviour (returns
    whatever subset succeeded). Currently honoured by the HTTP adapter.

    max_retries / retry_backoff: per-file retry budget for transient fetch
    failures (rate-limit responses, network blips). Each retry waits
    retry_backoff * 2**attempt seconds with jitter. Set max_retries=0 to
    disable. Currently honoured by the HTTP adapter.

    request_interval: minimum seconds between successive file opens (shared
    across worker threads). Use this to stay under a server's
    requests-per-second budget. Default 0 = no pacing. Currently honoured by
    the HTTP adapter. A product may declare its own request_interval (and a
    max_workers concurrency cap) in the catalog as a safety floor; this kwarg
    can raise that floor but not lower it. The native CHIRPS entries set one so
    bulk pulls stay under the UCSB CrowdSec rate ban — raise it further for
    large regional domains (rule of thumb: range-reads-per-file / 2 seconds).

    reforecast: when True, fetch the reforecast (hindcast) suite associated
    with the given issuance instead of the forecast itself. Currently
    honoured by the CDS adapter's s2s-forecasts branch.

    seasonal=None (default): no seasonal aggregation. seasonal="mean" subsets
    the `target` season's months on the `time` dim and averages them to one
    value per year (dim `time` -> `year`). No-op for data vars that have no
    `time` dim (e.g. year_index=True results, already lead-averaged). Requires
    `target`. Within-year seasons (e.g. MAM) are labelled by their calendar
    year. Wraparound seasons that cross the calendar-year boundary (NDJ, DJF)
    are labelled by the year of their starting month (so DJF Dec YYYY + Jan/Feb
    YYYY+1 -> year YYYY, matching how a wraparound forecast target is labelled
    by its initialization year); the wrapped months are shifted back a year
    before grouping, and incomplete seasons at the ends of the record (missing
    one or more of the season's months) are dropped.

    grid_res=None (default): no regridding. grid_res=<float> interpolates
    onto a regular lat/lon grid at that resolution spanning `region`
    (np.arange(lat_s, lat_n + grid_res/2, grid_res), same for lon). Requires
    `region`. Mutually exclusive with `regrid_to`.

    regrid_to=None (default): no regridding. regrid_to=<xr.DataArray>
    interpolates onto that DataArray's lat/lon coordinates. Mutually
    exclusive with `grid_res`.
    """
    _set_nuthatch_verbosity(verbose)
    _log(verbose, f"fetch start: product={product}, variable={variable}")

    config = dict(catalog.get(product))
    # Capability check first: a variable the product doesn't carry is a static
    # catalog fact, so it must fail here — typed, and before a byte crosses the
    # network — rather than as a bare KeyError from inside whichever adapter
    # runs. See acmaddl.errors.VariableNotSupported.
    catalog.require_variable(product, variable, config=config)
    config["_verbose"] = verbose
    config["_progress"] = progress
    config["_allow_partial"] = allow_partial
    config["_max_retries"] = max_retries
    config["_retry_backoff"] = retry_backoff
    config["_request_interval"] = request_interval

    # Reforecast mode is meaningful only for S2S right now, but plumb it
    # generically through the adapter config (outside the `if init:` block) so
    # the flag is set regardless of whether `init` was supplied — guarding
    # against a silent fall-through to forecast mode in mis-calls.
    config["_reforecast"] = bool(reforecast)

    # Reforecast dispatch: the CDS adapter switches its collection name to
    # `s2s-reforecasts` based on `_reforecast` (set above). The legacy MARS
    # path is retained for emergencies but no longer the default — ECMWF
    # has decommissioned legacy WEB-API access to the S2S dataset.

    date_range = hindcast
    cache_init_date = None
    # Calendar months of the requested target season. Part of the cache key: the
    # seasonal adapters collapse the lead axis for this season inside the cached
    # call, so the key has to distinguish OND from SON at the same init month.
    cache_target_months = None

    if months is not None:
        if init is not None:
            raise ValueError(
                "months prunes an observational fetch by calendar month; it has "
                "no meaning alongside init, which already names the issuance."
            )
        wanted = sorted({int(m) for m in months})
        if not wanted or not all(1 <= m <= 12 for m in wanted):
            raise ValueError(f"months must be calendar months in 1..12, got {months!r}")
        # `init_months` already means "which months of the year are in this
        # request" (the OPeNDAP adapter filters its init axis with it) and is
        # already a cache argument. Reusing it prunes the HTTP file enumeration
        # and keys the cache correctly without adding a cache arg, which would
        # invalidate every fetch already on disk.
        config["init_months"] = wanted

    if init is not None and _is_init_sequence(init):
        # Several issuances at once: the hindcast-skill case, where you want the
        # same calendar issuance of every year. Only products that declare an
        # `issuance` block know how to enumerate more than one.
        inits = list(init)
        if not inits:
            raise ValueError("init was an empty sequence")
        if target is not None:
            raise ValueError(
                "target selects lead months relative to a single init; it cannot "
                "be combined with a sequence of issuance dates. Fetch the leads "
                "and select the target window afterwards."
            )
        if config.get("issuance") is None:
            raise ValueError(
                f"product {product!r} accepts a single init, not a sequence of "
                f"{len(inits)}. Only issuance-keyed products (whose catalog entry "
                "has an 'issuance' block) can be fetched across many issuances."
            )
        keys = [_daily_issuance_key(i) for i in inits]
        if any(key is None for key in keys):
            raise ValueError(
                "issuance dates must be full 'YYYY-MM-DD' dates or datetimes; "
                f"got {inits!r}"
            )
        init_dts = [parse_init(k) for k in keys]
        config["_init_dates"] = sorted(set(keys))
        config["init_months"] = sorted({d.month for d in init_dts})
        if date_range is None:
            years = [d.year for d in init_dts]
            date_range = (min(years), max(years))
        # Reuse the existing `init_date` cache slot rather than adding a new one,
        # so every already-cached single-issuance fetch stays valid.
        cache_init_date = tuple(config["_init_dates"])
        init = None  # the single-init branch below does not apply

    if init:
        init_dt = parse_init(init)
        config["init_months"] = [init_dt.month]
        if date_range is None:
            if reforecast:
                # ECMWF S2S reforecasts span ~20 years of equivalent-week
                # historical issuances ending the year before the realtime
                # forecast. Default to that window when the caller doesn't
                # pin one explicitly.
                date_range = (init_dt.year - 20, init_dt.year - 1)
            else:
                date_range = (init_dt.year, init_dt.year)

        # For sub-daily / single-issuance datasets (S2S), thread the original
        # date string through to the adapter so it can build a single-day request.
        # The seasonal datasets ignore _init_date and use init_months instead.
        if isinstance(init, str) and len(init) == 10:
            config["_init_date"] = init
            cache_init_date = init
        if config.get("issuance") is not None:
            key = _daily_issuance_key(init)
            if key is None:
                raise ValueError(
                    f"product {product!r} is issuance-keyed and needs a full "
                    f"'YYYY-MM-DD' init date, got {init!r}"
                )
            config["_init_dates"] = [key]
            cache_init_date = key

        if target:
            target_range = parse_target(target, year=init_dt.year)
            s_month, e_month = target_range[0].month, target_range[1].month
            if s_month <= e_month:
                target_months = list(range(s_month, e_month + 1))
            else:
                target_months = list(range(s_month, 13)) + list(range(1, e_month + 1))
            lead_months = [(m - init_dt.month) % 12 + 1 for m in target_months]
            if "cds_model" in config:
                if config.get("cds_dataset") == "seasonal-original-single-levels":
                    from datetime import datetime, timedelta
                    init_date = datetime(init_dt.year, init_dt.month, 1)
                    hours = []
                    for m in lead_months:
                        m_start = init_dt.month + m - 1
                        y_start = init_dt.year + (m_start - 1) // 12
                        m_start = (m_start - 1) % 12 + 1
                        m_end = init_dt.month + m
                        y_end = init_dt.year + (m_end - 1) // 12
                        m_end = (m_end - 1) % 12 + 1
                        start = (datetime(y_start, m_start, 1) - init_date).days * 24
                        end = (datetime(y_end, m_end, 1) - init_date).days * 24
                        hours.extend(range(start, end, 24))
                    hours = sorted(set(hours))
                    var_cfg = config["variables"][variable]
                    if var_cfg.get("accumulated") and hours:
                        extra = hours[0] - 24
                        if extra > 0:
                            hours = [extra] + hours
                    config["leadtime_hour"] = hours
                else:
                    config["leadtime_month"] = lead_months
            config["target_lead_months"] = lead_months
            config["target_range"] = target_range
            cache_target_months = tuple(target_months)

    # Resolve the region once: adapters and the cache key see only the bbox
    # (stable, hashable); the optional polygon geometry is applied as the final
    # mask in normalize(). Shapefiles/geometries thus never reach the adapter or
    # the cache args — two requests with the same bbox share cached raw data.
    bbox, geometry = resolve_region(region)
    cover = boundary == "cover"

    # In "cover" mode, pad the fetched bbox by region_buffer degrees so the grid
    # extends past the region's edges. The bbox slice is centre-based, so without
    # this the boundary cells (centre just outside the bounds, but extent still
    # covering a sliver of the region) get dropped, leaving edges empty.
    # normalize() then trims back to the region. The default "center" mode keeps
    # only centre-in cells, which are within the bounds, so no padding is needed.
    fetch_bbox = bbox
    if cover and bbox is not None:
        b = region_buffer
        fetch_bbox = [max(-90.0, bbox[0] - b), min(90.0, bbox[1] + b),
                      max(-180.0, bbox[2] - b), min(180.0, bbox[3] + b)]

    _log(verbose, f"downloading via adapter={config['adapter']}")
    # degenerate_attempts>1 is the opt-in for the degenerate-response guard (a zero-filled /
    # truncated response). A fresh fetch is validated inside _fetch_raw before it can be cached; a
    # cache HIT is validated here too, so a legacy entry poisoned before this guard existed is
    # caught rather than silently reused; on failure we retry with the cache bypassed to get past a
    # poisoned entry / a transient truncation and re-hit the source. degenerate_attempts=1 (the
    # default) skips all of this, preserving the original behaviour for callers that don't opt in.
    want_reject = degenerate_attempts > 1
    attempts = max(1, int(degenerate_attempts))
    raw = None
    for attempt in range(attempts):
        use_cache = cache and attempt == 0
        try:
            if use_cache:
                # only pass reject_degenerate when opting in, so the default call signature stays
                # byte-identical (existing callers / mocks that pin the signature are untouched)
                extra = {"reject_degenerate": True} if want_reject else {}
                raw = _fetch_raw_cached(
                    product, variable, config, date_range, fetch_bbox,
                    init_months=tuple(config.get("init_months", [])),
                    init_date=cache_init_date,
                    target_months=cache_target_months, **extra,
                )
                if want_reject:
                    reject_if_degenerate(raw, variable, f"acmaddl.fetch({product!r}) [cached]",
                                         reject_all_nan=False)
            else:
                extra = {"reject_degenerate": True} if want_reject else {}
                raw = _fetch_raw(product, variable, config, date_range, fetch_bbox, **extra)
            break
        except DegenerateResponseError as e:
            if attempt == attempts - 1:
                raise
            _log(verbose, f"{e} — retrying without cache ({attempt + 2}/{attempts})")
    _log(verbose, "normalizing dataset")
    # normalize gets the original (unpadded) bbox: in cover mode it expands by
    # half a cell to cover it; in center mode it slices to it exactly.
    clean = normalize(raw, config, variable, bbox, geometry=geometry,
                      boundary=boundary, year_index=year_index)

    if grid_res is not None and regrid_to is not None:
        raise ValueError("pass grid_res or regrid_to, not both (mutually exclusive)")
    if seasonal is not None:
        if seasonal != "mean":
            raise ValueError(f"seasonal must be 'mean' or None, got {seasonal!r}")
        if target is None and months is None:
            raise ValueError(
                "seasonal='mean' needs a season to average over: pass either target= "
                "(a 3-month season code) or months= (explicit calendar months)."
            )
        if target is not None and months is not None:
            raise ValueError(
                "seasonal='mean' takes target= or months=, not both — they specify the same "
                "thing two ways. Pass one."
            )
        for name, da in list(clean.data_vars.items()):
            if "time" in da.dims:
                if target is not None:
                    s, e = season_to_months(target)
                    sel_months = [((s - 1 + k) % 12) + 1 for k in range((e - s) % 12 + 1)]
                    sub = da.sel(time=da.time.dt.month.isin(sel_months))
                    if e < s:
                        # Wraparound season (e.g. DJF, NDJ): the months that fall past
                        # December (month < s) belong to the season that began in the
                        # previous calendar year's month s, so shift their year back by one.
                        # The season is labelled by the year of its starting month, matching
                        # how the seasonal forecasts label a wraparound target by its
                        # initialization year. Only complete seasons (all months present) are
                        # kept, dropping partial seasons at the record's ends.
                        yr = sub["time"].dt.year
                        season_year = xr.where(sub["time"].dt.month < s, yr - 1, yr)
                        sub = sub.assign_coords(season_year=("time", season_year.data))
                        counts = sub["time"].groupby("season_year").count()
                        means = sub.groupby("season_year").mean("time")
                        means = means.where(counts == len(sel_months), drop=True)
                        clean[name] = means.rename({"season_year": "year"})
                    else:
                        clean[name] = sub.groupby("time.year").mean("time")
                else:
                    # no season given: average the explicitly-requested calendar months
                    # (e.g. months=[6] for a single-month June predictor).
                    sel_months = sorted({int(m) for m in months})
                    sub = da.sel(time=da.time.dt.month.isin(sel_months))
                    clean[name] = sub.groupby("time.year").mean("time")
    if grid_res is not None:
        lat_s, lat_n, lon_w, lon_e = region
        lats = np.arange(lat_s, lat_n + grid_res / 2.0, grid_res)
        lons = np.arange(lon_w, lon_e + grid_res / 2.0, grid_res)
        clean = _match_lon_convention(clean, lons)
        clean = clean.interp(lat=lats, lon=lons)
    if regrid_to is not None:
        clean = _match_lon_convention(clean, regrid_to.lon.values)
        clean = clean.interp(lat=regrid_to.lat.values, lon=regrid_to.lon.values)

    # A region was requested but the source isn't spatially griddable (e.g.
    # station/tabular data has no lat/lon grid): fail loudly rather than
    # silently handing back the full, unsubset dataset.
    if region is not None and ("lat" not in clean.dims or "lon" not in clean.dims):
        raise ValueError(
            f"region was requested for product {product!r} (adapter "
            f"{config['adapter']!r}), but its data has no lat/lon grid, so it "
            f"can't be subset to a bounding box or shapefile. Non-gridded "
            f"sources (e.g. station/tabular observations) don't support spatial "
            f"region selection — drop `region`, or use a gridded product."
        )

    # Rebuild so the result round-trips through to_netcdf: OPeNDAP/CF sources carry
    # bounds vars + stale encoding that otherwise raise "NetCDF: String match to name
    # in use" on write (in the returned object as well as the save() path below).
    clean = sanitize_for_netcdf(clean)

    if destination:
        _log(verbose, f"saving output -> {destination}")
        save(clean, destination, format)

    _log(verbose, "fetch complete")
    return clean
