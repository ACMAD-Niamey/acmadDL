import warnings

import numpy as np
import pandas as pd
import xarray as xr

_COORD_RENAMES = {
    # Spatial
    "latitude": "lat", "longitude": "lon",
    "LAT": "lat", "LON": "lon",
    "Y": "lat", "X": "lon",
    # Temporal – observational
    "forecast_time": "time", "valid_time": "time", "T": "time", "TIME": "time",
    # Temporal – forecast
    "S": "init_time", "forecast_reference_time": "init_time", "indexing_time": "init_time",
    "L": "lead_time", "forecastMonth": "lead_time", "forecast_period": "lead_time",
    "step": "lead_time",
    # Ensemble
    "M": "member", "number": "member",
}

_CONVERSIONS = {
    ("K", "C"): lambda da: da - 273.15,
    # CCSR's 2026-07 CMIP renaming also switched SST from Kelvin to Celsius;
    # this preserves acmaddl's existing "NMME sst in K" contract.
    ("C", "K"): lambda da: da + 273.15,
    ("kg m-2 s-1", "mm/day"): lambda da: da * 86400,
    ("m s-1", "mm/day"): lambda da: da * 1000 * 86400,
    ("m", "mm/day"): lambda da: da * 1000,
    ("m/s", "mm/day"): lambda da: da * 86_400_000,
    ("mm/month", "mm/day"): lambda da: da / 30.0,
    # Identity: variable is already accumulated mm; after deaccumulation
    # (diff over 24h lead_time), each value is mm per 24h = mm/day.
    ("mm", "mm/day"): lambda da: da,
}


def _season_day_counts(init_time, target_range):
    """Exact target-season length for every forecast initialization year."""
    import calendar

    start_month = target_range[0].month
    end_month = target_range[1].month
    months = (list(range(start_month, end_month + 1)) if start_month <= end_month
              else list(range(start_month, 13)) + list(range(1, end_month + 1)))
    counts = []
    for init in init_time.values:
        base_year = int(init.astype("datetime64[Y]").astype(int)) + 1970
        init_month = int(init.astype("datetime64[M]").astype(int) % 12) + 1
        target_start_year = base_year + (start_month < init_month)
        counts.append(sum(
            calendar.monthrange(target_start_year + (month < start_month), month)[1]
            for month in months
        ))
    return xr.DataArray(counts, dims=("init_time",), coords={"init_time": init_time})


def _season_month_day_counts(init_time, target_range, lead_time):
    """Per-target-month day weights for monthly lead-resolved forecasts."""
    import calendar

    start_month = target_range[0].month
    end_month = target_range[1].month
    months = (list(range(start_month, end_month + 1)) if start_month <= end_month
              else list(range(start_month, 13)) + list(range(1, end_month + 1)))
    weights = []
    for init in init_time.values:
        base_year = int(init.astype("datetime64[Y]").astype(int)) + 1970
        init_month = int(init.astype("datetime64[M]").astype(int) % 12) + 1
        target_start_year = base_year + (start_month < init_month)
        weights.append([
            calendar.monthrange(target_start_year + (month < start_month), month)[1]
            for month in months
        ])
    return xr.DataArray(
        weights,
        dims=("init_time", "lead_time"),
        coords={"init_time": init_time, "lead_time": lead_time},
    )


def decode_months_since(units_str, vals):
    """Decode 'months since YYYY-MM-DD' values to (years, months) arrays.

    Returns (years, months) as integer arrays.
    """
    ref_str = units_str.split("months since")[1].strip()
    ref = pd.Timestamp(ref_str)
    ref_total = ref.year * 12 + (ref.month - 1)
    total = ref_total + np.round(vals).astype(int)
    return total // 12, total % 12 + 1


def _decode_numeric_times(ds):
    """Decode numeric time coordinates with CF-style 'X since Y' units to datetime64."""
    for coord in list(ds.coords):
        if coord not in ds.dims:
            continue
        units = ds[coord].attrs.get("units", "")
        if "since" not in units:
            continue
        vals = ds[coord].values
        if not (np.issubdtype(vals.dtype, np.floating)
                or np.issubdtype(vals.dtype, np.integer)):
            continue

        # ns precision at the source for all three branches below: pandas'
        # inferred resolution for pd.to_datetime/pd.Timestamp arithmetic is not
        # guaranteed to be ns (it picks the coarsest resolution that losslessly
        # fits the values), and each `dates` here becomes a coordinate via
        # assign_coords — non-ns there triggers xarray's non-nanosecond
        # UserWarning.
        if "months since" in units:
            years, months = decode_months_since(units, vals)
            dates = pd.to_datetime([
                f"{y}-{m:02d}-01" for y, m in zip(years, months)
            ]).values.astype("datetime64[ns]")
            ds = ds.assign_coords({coord: dates})
        elif "days since" in units:
            ref_str = units.split("days since")[1].strip()
            ref = pd.Timestamp(ref_str)
            dates = (ref + pd.to_timedelta(vals, unit="D")).values.astype("datetime64[ns]")
            ds = ds.assign_coords({coord: dates})
        elif "hours since" in units:
            ref_str = units.split("hours since")[1].strip()
            ref = pd.Timestamp(ref_str)
            dates = (ref + pd.to_timedelta(vals, unit="h")).values.astype("datetime64[ns]")
            ds = ds.assign_coords({coord: dates})

    return ds


def clip_to_geometry(ds, geometry, all_touched=False):
    """Mask a lat/lon dataset to a polygon, NaN-ing cells outside it.

    This is the final, true-polygon clip that refines the coarse bbox crop:
    bbox slicing keeps a rectangle, this keeps only cells inside ``geometry``.
    The dataset must have ``lat``/``lon`` dims; ``geometry`` must be a shapely
    geometry in EPSG:4326 (as returned by :func:`acmaddl.region.resolve_region`).

    Uses rioxarray's geometry clip (rioxarray is a core dependency).
    ``all_touched`` selects the boundary rule (matching rasterio's parameter):

    * ``False`` (default): a cell is kept only if its centre lies inside the
      polygon — the xarray/CDO/rasterio convention, and unbiased for area means.
    * ``True``: every cell the polygon touches is kept, so the shape is covered
      to its true edges (given fetch() padded the grid past them).

    ``drop=True`` crops the output to the polygon's bounding box so there's no
    NaN margin from any padding. Cells inside the bbox but excluded by the
    boundary rule remain NaN.
    """
    import rioxarray  # noqa: F401  registers the .rio accessor

    if "lat" not in ds.dims or "lon" not in ds.dims:
        return ds

    clipped = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    clipped = clipped.rio.write_crs("EPSG:4326", inplace=False)
    clipped = clipped.rio.clip([geometry], crs="EPSG:4326",
                               drop=True, all_touched=all_touched)
    # rio.clip adds a `spatial_ref` coord and a `grid_mapping` encoding; strip
    # them so the result matches the plain (lat, lon, ...) schema callers expect.
    clipped = clipped.drop_vars("spatial_ref", errors="ignore")
    for v in clipped.data_vars:
        clipped[v].encoding.pop("grid_mapping", None)
    # rio.clip can flip lat to descending; restore the ascending convention.
    if "lat" in clipped.dims:
        clipped = clipped.sortby("lat")
    return clipped


def select_lon(ds, lon_w, lon_e, lon_name="lon"):
    """Longitude subselection that is robust to the source's longitude convention.

    A bbox given in the −180..180 convention against a 0..360 source (or vice versa)
    would otherwise silently under-select — e.g. ``slice(-180, 180)`` on a 0..358 grid
    returns only 0..180, dropping the Pacific and half the Atlantic. This:
      * returns all longitudes when a full-globe span is requested,
      * translates the requested bounds into the data's convention, and
      * handles a box that wraps the seam (west > east after translation) by selecting
        both sides and concatenating.

    Shared by the OPeNDAP adapter (pre-normalize, dim may be ``X``) and ``normalize``
    (dim ``lon``), since the source convention is only known once the data is opened.
    """
    if lon_name not in ds.dims or ds.sizes.get(lon_name, 0) == 0:
        return ds
    lon = ds[lon_name].values
    # slice(...) needs a monotonic index; a seam-crossing region selected upstream
    # (e.g. by the adapter) can arrive non-monotonic. Sort ascending first.
    if lon.size > 1 and not (np.all(np.diff(lon) > 0) or np.all(np.diff(lon) < 0)):
        ds = ds.sortby(lon_name)
        lon = ds[lon_name].values
    lo, hi = float(np.nanmin(lon)), float(np.nanmax(lon))

    # Full-globe request -> keep everything (this is the −180..180 vs 0..360 footgun).
    if (lon_e - lon_w) >= 359.0:
        return ds

    data_0_360 = hi > 180.0
    w, e = lon_w, lon_e
    if data_0_360:
        # data in 0..360: map any negative requested bound into 0..360
        if w < 0:
            w += 360.0
        if e < 0:
            e += 360.0
    else:
        # data in −180..180: map any requested bound above 180 into −180..180
        if w > 180.0:
            w -= 360.0
        if e > 180.0:
            e -= 360.0

    if w <= e:
        out = ds.sel({lon_name: slice(w, e)})
    else:
        # box wraps the seam (e.g. Atlantic −70..20 -> 290..20 in 0..360)
        out = xr.concat([ds.sel({lon_name: slice(w, hi)}),
                         ds.sel({lon_name: slice(lo, e)})], dim=lon_name)

    if out.sizes.get(lon_name, 0) == 0:
        warnings.warn(
            f"Longitude selection [{lon_w}, {lon_e}] returned no cells against a "
            f"source spanning [{lo:.1f}, {hi:.1f}]. Check the longitude convention "
            f"(the source uses {'0..360' if data_0_360 else '-180..180'}).",
            stacklevel=2,
        )
    return out


def sanitize_for_netcdf(ds):
    """Return a copy that round-trips cleanly through ``to_netcdf`` (netCDF4 engine).

    OPeNDAP/CF sources arrive with CF bounds variables (``time_bnds`` over an ``nbnds``
    dimension) and inherited variable encoding (``source``, ``original_shape``) left over
    from the remote store. Those make ``to_netcdf`` raise ``NetCDF: String match to name
    in use``, and — importantly — clearing ``.encoding`` or dropping the bounds var *in
    place* does not fix it (stale index/variable encoding survives). Rebuilding the dataset
    from fresh arrays does. Data is already materialized by the time this runs, so the
    rebuild is a cheap copy and preserves values, coords, dims, and attrs.
    """
    drop = [v for v in ds.variables
            if "bnd" in str(v).lower() or "bound" in str(v).lower()]
    ds = ds.drop_vars(drop, errors="ignore")
    coords = {c: (ds[c].dims, np.asarray(ds[c].values),
                  {k: a for k, a in ds[c].attrs.items() if k != "bounds"})
              for c in ds.coords}
    data_vars = {v: (ds[v].dims, np.asarray(ds[v].values),
                     {k: a for k, a in ds[v].attrs.items() if k != "bounds"})
                 for v in ds.data_vars}
    attrs = {k: a for k, a in ds.attrs.items() if k != "_NCProperties"}
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def _resolve_native_name(ds, var_cfg, variable):
    """The dataset's name for ``variable``, tolerant of source-side renames.

    Tries every name the catalog declares (``native_name``, ``short_name``,
    ``path_name`` — including per-stream dict forms) in order. Sources rename
    variables under us (CCSR's CMOR migration: ``prcp`` -> ``pr`` -> paths too),
    and locally CACHED raw data keeps whatever name it had when fetched, so a
    single hard-coded name breaks on either skew. If no declared name matches
    but the dataset has exactly ONE data variable, use it with a warning; only
    a genuinely ambiguous dataset raises, with a stale-cache hint.
    """
    import warnings

    candidates = []
    for key in ("native_name", "short_name", "path_name"):
        val = var_cfg.get(key)
        if isinstance(val, dict):
            candidates.extend(v for v in val.values() if v)
        elif val:
            candidates.append(val)
    for cand in candidates:
        if cand in ds:
            return cand
    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        warnings.warn(
            f"normalize: none of the declared names {candidates} found for "
            f"{variable!r}; using the dataset's only data variable "
            f"{data_vars[0]!r}. The source may have renamed its variables since "
            "the catalog (or since this data was cached).",
            RuntimeWarning, stacklevel=3,
        )
        return data_vars[0]
    raise ValueError(
        f"normalize: cannot locate {variable!r}: none of the declared names "
        f"{candidates} are in the dataset (data variables: {data_vars}). If this "
        "came from the local cache, the source may have renamed variables since "
        "it was cached — clear the stale entry under ~/.nuthatch and refetch."
    )


def normalize(ds, product_config, variable, region=None, geometry=None,
              boundary="center", year_index=False):
    ds = _decode_numeric_times(ds)

    # ECMWF S2S GRIB files use `time` as the (scalar) init issuance and
    # `valid_time` as the per-step forecast time. The rename map below sends
    # `valid_time → time`, which would collide with the existing scalar `time`.
    # If `time` is scalar (not a dim), pre-rename it to `init_time` so the
    # collision can't happen.
    if ("time" in ds.coords and "time" not in ds.dims
            and "init_time" not in ds.coords and "init_time" not in ds.dims):
        ds = ds.rename({"time": "init_time"})

    # ECMWF S2S reforecast responses carry an `hdate` dimension — the
    # calendar issuance dates of each historical reforecast. Convert to
    # integer years and rename to `year` so the output matches the
    # (year, member, lead_time, lat, lon) hindcast convention used
    # downstream. Must run before _COORD_RENAMES.
    if "hdate" in ds.dims:
        years = ds["hdate"].dt.year.values
        ds = ds.assign_coords(hdate=years).rename({"hdate": "year"})

    renames = {k: v for k, v in _COORD_RENAMES.items() if k in ds.dims or k in ds.coords}
    if renames:
        ds = ds.rename(renames)

    # Some sources expose fixed-width member coordinates with structurally
    # empty trailing slots. Drop only members that are all-NaN across every
    # variable dimension; populated ensemble spread is preserved unchanged.
    if product_config.get("drop_empty_members") and "member" in ds.dims:
        ds = ds.dropna("member", how="all")

    # Generic member-reduce knob for products whose public contract explicitly
    # requests a subset average. Catalog-driven and data-driven so it applies
    # to any product that declares it; absent -> preserve all native members.
    # Runs right after `member` exists (renamed above) and before the
    # `year_index` reshape below, so it applies uniformly to plain and
    # year_index fetches alike.
    member_reduce = product_config.get("member_reduce")
    if member_reduce and "member" in ds.dims:
        first = member_reduce["first"]
        op = member_reduce.get("op", "mean")
        if op != "mean":
            raise ValueError(f"member_reduce op must be 'mean', got {op!r}")
        reduced = ds.isel(member=slice(0, first)).mean("member", keep_attrs=True)
        ds = reduced.expand_dims(member=[0])

    var_cfg = product_config["variables"][variable]
    if variable not in ds:
        ds = ds.rename({_resolve_native_name(ds, var_cfg, variable): variable})

    # Deaccumulate variables (e.g. CDS total_precipitation is accumulated from init)
    if var_cfg.get("accumulated") and "lead_time" in ds.dims:
        diffed = ds[variable].diff(dim="lead_time")
        ds = ds.sel(lead_time=diffed.lead_time)
        ds[variable] = diffed

    src, tgt = var_cfg["units"], var_cfg["target_units"]
    if var_cfg.get("season_total_conversion") and product_config.get("target_range"):
        if "init_time" in ds.dims:
            ds[variable] = ds[variable] * _season_day_counts(
                ds["init_time"], product_config["target_range"])
        else:
            start, end = product_config["target_range"]
            ds[variable] = ds[variable] * ((end - start).days + 1)
    elif (src, tgt) in _CONVERSIONS:
        ds[variable] = _CONVERSIONS[(src, tgt)](ds[variable])

    # A season-total product fetched without target= has no accumulation
    # window, so retain its native rate and label rather than inventing a total.
    ds[variable].attrs["units"] = (
        src if var_cfg.get("season_total_conversion") and not product_config.get("target_range")
        else tgt
    )

    # Optional no-data sentinel masking. Some upstream sources (e.g.
    # CHIRPS via sheerwater's chirps_raw_live) return a numeric sentinel
    # (typically -9999) for ocean/missing cells instead of NaN. Catalog
    # can specify `fill_value` per variable to convert those to NaN so
    # downstream aggregations skip them correctly.
    fill_value = var_cfg.get("fill_value")
    if fill_value is not None:
        ds[variable] = ds[variable].where(ds[variable] != fill_value)

    if "lat" in ds.dims:
        ds = ds.sortby("lat")

    cover = boundary == "cover"

    if geometry is not None and "lat" in ds.dims and "lon" in ds.dims:
        # Polygon mask drives the spatial selection; the boundary rule (centre
        # vs all-touched) follows `boundary`. clip crops to the polygon's bbox.
        ds = clip_to_geometry(ds, geometry, all_touched=cover)
    elif region and "lat" in ds.dims and "lon" in ds.dims:
        lat_s, lat_n, lon_w, lon_e = region
        if cover:
            # Include every cell whose extent overlaps the requested box, i.e.
            # cells whose centre lies within half a grid step of the bounds.
            r_lat = (float(np.median(np.diff(ds.lat.values))) / 2
                     if ds.sizes.get("lat", 0) > 1 else 0.0)
            r_lon = (float(np.median(np.diff(ds.lon.values))) / 2
                     if ds.sizes.get("lon", 0) > 1 else 0.0)
            lat_s, lat_n = lat_s - r_lat, lat_n + r_lat
            lon_w, lon_e = lon_w - r_lon, lon_e + r_lon
        ds = ds.sel(lat=slice(lat_s, lat_n))
        ds = select_lon(ds, lon_w, lon_e)

    for coord, attr in [("lat", "Y"), ("lon", "X"), ("time", "T"), ("init_time", "T")]:
        if coord in ds.coords:
            ds[coord].attrs["axis"] = attr

    if year_index and "init_time" in ds.dims:
        # A collapsed target-season precipitation forecast has one public unit
        # contract across adapter families: accumulated mm. Lead-resolved fetches
        # retain their honest per-step units; this branch is specifically the
        # year_index/assemble reduction boundary.
        if (variable == "precip" and product_config.get("target_range")
                and ds[variable].attrs.get("units") == "mm/day"):
            if "lead_time" not in ds.dims:
                ds[variable] = ds[variable] * _season_day_counts(
                    ds["init_time"], product_config["target_range"])
            elif (product_config.get("leadtime_month")
                  and ds.sizes["lead_time"] == len(product_config["leadtime_month"])):
                weights = _season_month_day_counts(
                    ds["init_time"], product_config["target_range"], ds["lead_time"])
                ds[variable] = (ds[variable] * weights).sum("lead_time", keep_attrs=True)
            else:
                # Daily CDS leads have already been deaccumulated to one value
                # per 24 h; summing those values gives the target accumulation.
                ds[variable] = ds[variable].sum("lead_time", keep_attrs=True)
            ds[variable].attrs["units"] = "mm"

        years = ds["init_time"].dt.year.values.astype(int)
        ds = ds.assign_coords(init_time=years).rename({"init_time": "year"})
        if "lead_time" in ds.dims:
            # keep_attrs so the units attribute set above survives the reduction
            # (xarray drops data-variable attrs on mean by default); matches the
            # default init_time path and what validate.py relies on.
            ds = ds.mean("lead_time", keep_attrs=True)

    return ds
