#!/usr/bin/env python3
"""Validate acmaddl outputs against IRI Data Library.

Fetches ensemble-mean seasonal precipitation from acmaddl and from IRI,
then computes temporal and spatial Pearson correlations to reproduce the
validation table from Emmett's comparison.

Test setup: init=Feb, target=MAM (lead months 2,3,4). All collapsed seasonal
precipitation is compared in mm.

Prerequisites:
  - IRI DL auth: ~/.pycpt_dlauth  (for C3S mirrors on IRI)
  - IRI OPeNDAP: works without auth for NMME .MONTHLY endpoints
  - AWS CLI configured (for S3 hindcast products)
  - CDS API: ~/.cdsapirc (for CDS products — can be slow)

Usage:
  python scripts/validate_against_iri.py                    # all models
  python scripts/validate_against_iri.py --models CFSv2 GEOSS2S
  python scripts/validate_against_iri.py --skip-cds         # skip slow CDS fetches
  python scripts/validate_against_iri.py --region -20 20 20 55
"""

import argparse
import functools
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import xarray as xr

# Force unbuffered output so progress is visible when run in background
print = functools.partial(print, flush=True)

# ---------------------------------------------------------------------------
# IRI Data Library paths
# ---------------------------------------------------------------------------
# NMME models: direct OPeNDAP access to IRI's .MONTHLY collections
# C3S models: IRI mirrors at SOURCES/.EU/.Copernicus/.CDS/.C3S/...
#   Variable is "prcp" with server-side unit conversion to mm/day.
#   Paths inferred from Emmett's table; update if IRI changes structure.

IRIDL_BASE = "https://iridl.ldeo.columbia.edu"

# Standard IRI unit ops to convert C3S precip rate to mm/day
_C3S_UNIT_OPS = "c%3A/1000/(mm%20m-1)/%3Ac/mul/c%3A/86400/(s%20day-1)/%3Ac/mul"

THREELETTERS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


# ---------------------------------------------------------------------------
# Model definitions — matches Emmett's validation table
# ---------------------------------------------------------------------------
MODELS = {
    # ---- NMME (IRI OPeNDAP) ----
    "CFSv2": {
        "acmaddl_product": "nmme/cfsv2",
        "iri_type": "nmme",
        # acmadDL uses PENTAD_SAMPLES/.MONTHLY; pycpt uses PENTAD_SAMPLES_FULL
        # (different sub-collection, but same data). Compare against same source
        # since PENTAD_SAMPLES_FULL has a different Ingrid path structure.
        "iri_url": f"{IRIDL_BASE}/SOURCES/.Models/.NMME/.NCEP-CFSv2/.HINDCAST/.PENTAD_SAMPLES/.MONTHLY",
        "iri_var": "prec",
        "hindcast": (1982, 2010),
        "match": "exact",
    },
    "CCSM4": {
        "acmaddl_product": "nmme/ccsm4-hindcast",
        "iri_type": "nmme",
        "iri_url": f"{IRIDL_BASE}/SOURCES/.Models/.NMME/.COLA-RSMAS-CCSM4/.MONTHLY",
        "iri_var": "prec",
        "hindcast": (1982, 2010),
        "match": "exact",
    },
    "GEOSS2S": {
        "acmaddl_product": "nmme/geoss2s-hindcast",
        "iri_type": "nmme",
        # NOTE: Use NASA-GEOSS2S/.HINDCAST (current model). The legacy
        # NASA-GMAO/.MONTHLY endpoint is a different model version.
        "iri_url": f"{IRIDL_BASE}/SOURCES/.Models/.NMME/.NASA-GEOSS2S/.HINDCAST/.MONTHLY",
        "iri_var": "prec",
        "hindcast": (1982, 2010),
        "match": "exact",
    },

    # ---- C3S via CDS (acmaddl) vs IRI mirror ----
    "ECMWF": {
        "acmaddl_product": "c3s/ecmwf-monthly",
        "iri_type": "c3s",
        "iri_path": "SOURCES/.EU/.Copernicus/.CDS/.C3S/.ECMWF/.SEAS51_iri2/.appended",
        "iri_var": "prcp",
        "iri_unit_ops": _C3S_UNIT_OPS,
        "hindcast": (1993, 2016),
        "match": "exact",
    },
    "MeteoFrance": {
        "acmaddl_product": "c3s/meteofrance",
        "iri_type": "c3s",
        "iri_path": "SOURCES/.EU/.Copernicus/.CDS/.C3S/.Meteo_France/.System9/.hindcast",
        "iri_var": "prcp",
        "iri_unit_ops": _C3S_UNIT_OPS,
        "hindcast": (1993, 2016),
        "match": "exact",
    },
    "CMCC": {
        "acmaddl_product": "c3s/cmcc",
        "iri_type": "c3s",
        "iri_path": "SOURCES/.EU/.Copernicus/.CDS/.C3S/.CMCC/.SPSv3p5/.hindcast",
        "iri_var": "prcp",
        "iri_unit_ops": _C3S_UNIT_OPS,
        "hindcast": (1993, 2016),
        "match": "exact",
    },
    "CanSIPSv3": {
        "acmaddl_product": "c3s/eccc-cansipsv3",
        "iri_type": "nmme",
        # CanSIPS IC4 is under NMME on IRI, not C3S mirror
        "iri_url": f"{IRIDL_BASE}/SOURCES/.Models/.NMME/.CanSIPS-IC4/.HINDCAST/.MONTHLY",
        "iri_var": "prec",
        "hindcast": (1993, 2016),
        "match": "exact",
    },
    "JMA": {
        # Version-matched: CPS2 (acmaddl) vs CPS2 (IRI)
        "acmaddl_product": "c3s/jma-cps2",
        "iri_type": "c3s",
        "iri_path": "SOURCES/.EU/.Copernicus/.CDS/.C3S/.JMA/.CPS2/.hindcast",
        "iri_var": "prec",
        "iri_unit_ops": _C3S_UNIT_OPS,
        "hindcast": (1993, 2016),
        "match": "exact",
    },
    "JMA-CPS3": {
        # Cross-version: CPS3 (acmaddl) vs CPS2 (IRI) — expected r < 1.0
        "acmaddl_product": "c3s/jma",
        "iri_type": "c3s",
        "iri_path": "SOURCES/.EU/.Copernicus/.CDS/.C3S/.JMA/.CPS2/.hindcast",
        "iri_var": "prec",
        "iri_unit_ops": _C3S_UNIT_OPS,
        "hindcast": (1993, 2016),
        "match": "different",
    },
    "DWD": {
        # Version-matched: GCFS 2.1 (acmaddl) vs GCFS 2.1 (IRI)
        "acmaddl_product": "c3s/dwd-gcfs21",
        "iri_type": "c3s",
        "iri_path": "SOURCES/.EU/.Copernicus/.CDS/.C3S/.DWD/.GCFS2p1/.hindcast",
        "iri_var": "prcp",
        "iri_unit_ops": _C3S_UNIT_OPS,
        "hindcast": (1993, 2016),
        "match": "exact",
    },
    "DWD-GCFS22": {
        # Cross-version: GCFS 2.2 (acmaddl) vs GCFS 2.1 (IRI) — expected r < 1.0
        "acmaddl_product": "c3s/dwd",
        "iri_type": "c3s",
        "iri_path": "SOURCES/.EU/.Copernicus/.CDS/.C3S/.DWD/.GCFS2p1/.hindcast",
        "iri_var": "prcp",
        "iri_unit_ops": _C3S_UNIT_OPS,
        "hindcast": (1993, 2016),
        "match": "different",
    },

    # ---- ROS_ONLY (no IRI comparison available) ----
    "UKMO": {
        "acmaddl_product": "c3s/ukmo",
        "iri_type": None,
        "hindcast": (1993, 2016),
        "match": None,
    },
    "CanSIPS-IC3": {
        "acmaddl_product": "c3s/eccc-cansips",
        "iri_type": None,
        "hindcast": (1993, 2010),
        "match": None,
    },
    "CESM1": {
        "acmaddl_product": "nmme/cesm1-hindcast",
        "iri_type": None,
        "hindcast": (1991, 2016),
        "match": None,
    },
    "CanESM5": {
        "acmaddl_product": "nmme/canesm5-hindcast",
        "iri_type": None,
        "hindcast": (1991, 2016),
        "match": None,
    },
    "GEM-NEMO": {
        "acmaddl_product": "nmme/gemnemo-hindcast",
        "iri_type": None,
        "hindcast": (1991, 2016),
        "match": None,
    },
    "GEM5.2-NEMO": {
        "acmaddl_product": "nmme/gem52nemo-hindcast",
        "iri_type": None,
        "hindcast": (1991, 2016),
        "match": None,
    },

    # ---- Observational SST (monthly, no forecast structure) ----
    "ERSSTv5": {
        "acmaddl_product": "sst/ersst-v5",
        "iri_type": "obs_sst",
        "iri_url": f"{IRIDL_BASE}/SOURCES/.NOAA/.NCDC/.ERSST/.version5/.sst",
        "iri_var": "sst",
        "variable": "sst",
        "hindcast": (2010, 2012),
        "match": "exact",
    },
}


# ---------------------------------------------------------------------------
# IRI fetch functions
# ---------------------------------------------------------------------------

def _read_dlauth():
    """Read IRI DL auth key from ~/.pycpt_dlauth."""
    path = Path.home() / ".pycpt_dlauth"
    if not path.is_file():
        raise FileNotFoundError(
            "No ~/.pycpt_dlauth found. Needed for IRI C3S mirror access. "
            "See https://iridl.ldeo.columbia.edu/auth/signup"
        )
    with open(path, "rb") as f:
        return json.loads(f.read())["key"]


def fetch_iri_nmme(model_cfg, init_month, lead_months, region=None):
    """Fetch ensemble-mean seasonal forecast from IRI NMME via Ingrid.

    Uses IRI's server-side Ingrid operations (S filter, L average, M average,
    region subset) so we only download the small result — much faster than
    raw OPeNDAP which would pull all members client-side.

    Returns an xarray DataArray with dims (init_time, lat, lon) in seasonal mm,
    matching acmaddl's targeted-NMME contract.
    """
    import requests

    iri_var = model_cfg["iri_var"]
    y0, y1 = model_cfg["hindcast"]
    mon = THREELETTERS[init_month]

    # Build Ingrid URL with server-side processing
    parts = [f"{model_cfg['iri_url']}/.{iri_var}"]

    # S (init time) selection: only Feb inits in hindcast range
    parts.append(f"S/%280000%201%20{mon}%20{y0}-{y1}%29/VALUES")

    # Lead time range and averaging
    lo = min(lead_months) - 0.5
    hi = max(lead_months) - 0.5
    parts.append(f"L/{lo}/{hi}/RANGEEDGES")
    parts.append("%5BL%5D//keepgrids/average")

    # Ensemble mean
    parts.append("%5BM%5D/average")

    # Region
    if region:
        lat_s, lat_n, lon_w, lon_e = region
        parts.append(f"Y/{lat_s}/{lat_n}/RANGEEDGES")
        parts.append(f"X/{lon_w}/{lon_e}/RANGEEDGES")

    nc_url = "/".join(parts) + "/data.nc"
    print(f"  [iri] fetching NMME via Ingrid ({model_cfg['iri_url'].split('.')[-2]})")

    # Try with dlauth cookie if available, fall back to unauthenticated
    session = requests.Session()
    try:
        key = _read_dlauth()
        session.cookies.set("__dlauth_id", key, domain="iridl.ldeo.columbia.edu")
    except FileNotFoundError:
        pass  # NMME endpoints are often publicly accessible

    resp = session.get(nc_url, timeout=300)
    resp.raise_for_status()

    if resp.headers.get("content-type", "").startswith("text/html"):
        raise RuntimeError("IRI returned HTML instead of NetCDF — may need auth")

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(resp.content)
        tmp.flush()
        # decode_times=False: IRI uses "months since" with calendar=360
        # which xarray/cftime can't auto-decode
        ds = xr.open_dataset(tmp.name, decode_times=False).load()

    da = ds[iri_var]
    print(f"  [iri] got {dict(da.sizes)}")

    # Decode S from "months since 1960-01-01" to datetime64
    if "S" in da.dims:
        from acmaddl.normalize import decode_months_since
        import pandas as pd
        units = ds["S"].attrs.get("units", "")
        if "months since" in units:
            s_years, s_months = decode_months_since(units, da.S.values)
            dates = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in zip(s_years, s_months)])
            da = da.assign_coords(S=dates)
        da = da.rename({"S": "init_time"})

    # Rename spatial dims
    renames = {}
    if "Y" in da.dims:
        renames["Y"] = "lat"
    if "X" in da.dims:
        renames["X"] = "lon"
    if renames:
        da = da.rename(renames)

    # IRI's lead-averaged NMME precip is a daily rate. acmadDL converts a
    # targeted NMME fetch to seasonal mm, so apply the same exact calendar
    # duration to the independent reference (separately for every init year).
    import calendar
    day_counts = []
    for init in da["init_time"].values:
        init_year = int(init.astype("datetime64[Y]").astype(int)) + 1970
        total = 0
        for lead in lead_months:
            month_index = init_month + lead - 1
            year = init_year + (month_index - 1) // 12
            month = (month_index - 1) % 12 + 1
            total += calendar.monthrange(year, month)[1]
        day_counts.append(total)
    days = xr.DataArray(day_counts, dims="init_time", coords={"init_time": da.init_time})
    da = da * days
    da.attrs["units"] = "mm"

    return da


def fetch_iri_c3s(model_cfg, init_month, lead_months, region=None):
    """Fetch ensemble-mean seasonal forecast from IRI's C3S mirror.

    Uses the same Ingrid URL construction as the IRIDL adapter.
    Returns an xarray DataArray with dims (S, Y, X) in seasonal mm.
    """
    import requests

    key = _read_dlauth()
    iri_path = model_cfg["iri_path"]
    iri_var = model_cfg["iri_var"]
    y0, y1 = model_cfg["hindcast"]
    mon = THREELETTERS[init_month]

    # Build Ingrid URL with server-side processing
    parts = [f"{IRIDL_BASE}/{iri_path}/.{iri_var}"]

    # Year range
    parts.append(f"S/%280000%201%20{mon}%20{y0}-{y1}%29/VALUES")

    # Lead time range and averaging
    lo = min(lead_months) - 0.5
    hi = max(lead_months) - 0.5
    parts.append(f"L/{lo}/{hi}/RANGEEDGES")
    parts.append("%5BL%5D//keepgrids/average")

    # Ensemble mean
    parts.append("%5BM%5D/average")

    # Region
    if region:
        lat_s, lat_n, lon_w, lon_e = region
        parts.append(f"Y/{lat_s}/{lat_n}/RANGEEDGES")
        parts.append(f"X/{lon_w}/{lon_e}/RANGEEDGES")

    # Unit conversion
    if "iri_unit_ops" in model_cfg:
        parts.append(model_cfg["iri_unit_ops"])

    nc_url = "/".join(parts) + "/data.nc"
    print(f"  [iri] fetching C3S mirror ({iri_path.split('/')[-2]})")

    session = requests.Session()
    session.cookies.set("__dlauth_id", key, domain="iridl.ldeo.columbia.edu")
    resp = session.get(nc_url, timeout=120)
    resp.raise_for_status()

    if resp.headers.get("content-type", "").startswith("text/html"):
        raise RuntimeError("IRI returned HTML instead of NetCDF — auth failure?")

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(resp.content)
        tmp.flush()
        ds = xr.open_dataset(tmp.name, decode_times=False).load()

    da = ds[iri_var]

    # Decode S from "months since 1960-01-01" to datetime64
    if "S" in da.dims:
        from acmaddl.normalize import decode_months_since
        import pandas as pd
        units = ds["S"].attrs.get("units", "")
        if "months since" in units:
            s_years, s_months = decode_months_since(units, da.S.values)
            dates = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in zip(s_years, s_months)])
            da = da.assign_coords(S=dates)
        da = da.rename({"S": "init_time"})

    # Rename spatial dims
    renames = {}
    if "Y" in da.dims:
        renames["Y"] = "lat"
    if "X" in da.dims:
        renames["X"] = "lon"
    if renames:
        da = da.rename(renames)

    # The IRI mirror returns a lead-averaged daily rate; match acmaddl's
    # collapsed seasonal-total contract with the exact calendar duration.
    import calendar
    day_counts = []
    for init in da["init_time"].values:
        init_year = int(init.astype("datetime64[Y]").astype(int)) + 1970
        total = 0
        for lead in lead_months:
            month_index = init_month + lead - 1
            year = init_year + (month_index - 1) // 12
            month = (month_index - 1) % 12 + 1
            total += calendar.monthrange(year, month)[1]
        day_counts.append(total)
    days = xr.DataArray(day_counts, dims="init_time", coords={"init_time": da.init_time})
    da = da * days
    da.attrs["units"] = "mm"

    print(f"  [iri] got {dict(da.sizes)}")
    return da


def fetch_iri_obs_sst(model_cfg, region=None):
    """Fetch monthly observational SST from IRI via Ingrid.

    No forecast structure — just T/Y/X/zlev. Returns DataArray (time, lat, lon).
    """
    import pandas as pd
    import requests

    y0, y1 = model_cfg["hindcast"]
    parts = [model_cfg["iri_url"]]
    parts.append(f"T/%281%20Jan%20{y0}%29/%281%20Dec%20{y1}%29/RANGEEDGES")
    if region:
        lat_s, lat_n, lon_w, lon_e = region
        parts.append(f"Y/{lat_s}/{lat_n}/RANGEEDGES")
        parts.append(f"X/{lon_w}/{lon_e}/RANGEEDGES")

    nc_url = "/".join(parts) + "/data.nc"
    print(f"  [iri] fetching ERSSTv5 via Ingrid")

    session = requests.Session()
    try:
        session.cookies.set("__dlauth_id", _read_dlauth(),
                            domain="iridl.ldeo.columbia.edu")
    except FileNotFoundError:
        pass

    resp = session.get(nc_url, timeout=300)
    resp.raise_for_status()
    if resp.headers.get("content-type", "").startswith("text/html"):
        raise RuntimeError("IRI returned HTML — path or auth issue")

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(resp.content)
        tmp.flush()
        ds = xr.open_dataset(tmp.name, decode_times=False).load()

    da = ds[model_cfg["iri_var"]]
    if "T" in da.dims:
        import numpy as np
        units = ds["T"].attrs.get("units", "")
        if "months since" in units:
            # T values are month centers like 600.5 → month 600 from 1960-01.
            # Floor (not round) avoids banker's-rounding duplicates on *.5 values.
            ref_str = units.split("months since")[1].strip()
            ref = pd.Timestamp(ref_str)
            ref_total = ref.year * 12 + (ref.month - 1)
            t_vals = ds["T"].values
            total = ref_total + np.floor(t_vals).astype(int)
            years = total // 12
            months = total % 12 + 1
            dates = pd.to_datetime([f"{y}-{m:02d}-15" for y, m in zip(years, months)])
            da = da.assign_coords(T=dates)
        da = da.rename({"T": "time"})

    renames = {}
    if "Y" in da.dims:
        renames["Y"] = "lat"
    if "X" in da.dims:
        renames["X"] = "lon"
    if renames:
        da = da.rename(renames)

    if "zlev" in da.dims:
        da = da.squeeze("zlev", drop=True)

    print(f"  [iri] got {dict(da.sizes)}")
    return da


def fetch_acmaddl_obs_sst(model_cfg, region=None):
    """Fetch monthly observational SST from acmadDL. Returns DataArray (time, lat, lon)."""
    import acmaddl

    product = model_cfg["acmaddl_product"]
    variable = model_cfg.get("variable", "sst")
    y0, y1 = model_cfg["hindcast"]

    print(f"  [acmaddl] fetching {product} ({y0}-{y1})")
    ds = acmaddl.fetch(
        product, variable,
        hindcast=(y0, y1),
        region=region,
        verbose=False, progress=False, cache=False,
    )
    da = ds[variable]
    if "lev" in da.dims:
        da = da.squeeze("lev", drop=True)
    print(f"  [acmaddl] got {dict(da.sizes)}")
    return da


# ---------------------------------------------------------------------------
# acmadDL fetch
# ---------------------------------------------------------------------------

def fetch_acmaddl(model_cfg, init_month, target, region=None):
    """Fetch ensemble-mean seasonal forecast via acmaddl.

    Returns an xarray DataArray with dims (init_time, lat, lon) in seasonal mm.
    """
    import acmaddl

    product = model_cfg["acmaddl_product"]
    y0, y1 = model_cfg["hindcast"]

    print(f"  [acmaddl] fetching {product} ({y0}-{y1})")
    ds = acmaddl.fetch(
        product, "precip",
        init=f"2010-{init_month:02d}",  # year only sets month; hindcast overrides years
        target=target,
        hindcast=(y0, y1),
        year_index=True,
        region=region,
        verbose=False,
        progress=False,
        cache=False,  # always fetch fresh — validation must reflect current source data
    )

    da = ds["precip"]

    # Validation aligns against IRI's datetime init axis; year_index=True uses
    # integer years, so restore the requested initialization month as dates.
    if "year" in da.dims:
        import pandas as pd
        dates = pd.to_datetime([f"{int(y)}-{init_month:02d}-01" for y in da.year.values])
        da = da.assign_coords(year=dates).rename({"year": "init_time"})

    # Ensemble mean
    if "member" in da.dims:
        da = da.mean("member")
    if "M" in da.dims:
        da = da.mean("M")

    # Average lead times (CDS returns individual months; IRI averages server-side)
    if "lead_time" in da.dims:
        da = da.mean("lead_time")

    print(f"  [acmaddl] got {dict(da.sizes)}")
    return da


# ---------------------------------------------------------------------------
# Correlation metrics
# ---------------------------------------------------------------------------

def compute_correlations(ros_da, iri_da):
    """Compute temporal and spatial Pearson correlations.

    Args:
        ros_da: DataArray (init_time, lat, lon) from acmaddl
        iri_da: DataArray (init_time, lat, lon) from IRI

    Returns:
        (r_temporal, r_spatial) — mean correlations
    """
    # Align on common init_time, lat, lon
    ros, iri = xr.align(ros_da, iri_da, join="inner")

    if ros.sizes.get("init_time", 0) < 3:
        return np.nan, np.nan

    # Drop NaN grid points (coastlines, etc.)
    valid = ros.notnull() & iri.notnull()
    ros = ros.where(valid)
    iri = iri.where(valid)

    # r(ts): temporal correlation at each grid point, averaged over space
    r_ts = xr.corr(ros, iri, dim="init_time")
    r_ts_mean = float(r_ts.mean(skipna=True))

    # r(spat): spatial correlation at each time step, averaged over time
    r_spat = xr.corr(ros, iri, dim=["lat", "lon"])
    r_spat_mean = float(r_spat.mean(skipna=True))

    return r_ts_mean, r_spat_mean


def regrid_to_common(da1, da2):
    """Regrid two DataArrays to the coarser common grid via linear interp."""
    lat1, lat2 = da1.lat.values, da2.lat.values
    lon1, lon2 = da1.lon.values, da2.lon.values

    res1 = np.median(np.diff(lat1))
    res2 = np.median(np.diff(lat2))

    if abs(res1 - res2) < 0.01 and len(lat1) == len(lat2):
        return da1, da2  # already aligned

    # Use the coarser grid as target
    if res1 >= res2:
        target_lat, target_lon = lat1, lon1
        da2 = da2.interp(lat=target_lat, lon=target_lon, method="linear")
    else:
        target_lat, target_lon = lat2, lon2
        da1 = da1.interp(lat=target_lat, lon=target_lon, method="linear")

    return da1, da2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_model(name, model_cfg, init_month, target, lead_months, region):
    """Validate a single model. Returns (r_ts, r_spat, status, error)."""
    if model_cfg["iri_type"] is None:
        return None, None, "ROS_ONLY", None

    try:
        # Obs sources (no init/target/lead — just time-series comparison)
        if model_cfg["iri_type"] == "obs_sst":
            ros_da = fetch_acmaddl_obs_sst(model_cfg, region)
            iri_da = fetch_iri_obs_sst(model_cfg, region)
            # Rename time → init_time so existing compute_correlations works
            ros_da = ros_da.rename({"time": "init_time"})
            iri_da = iri_da.rename({"time": "init_time"})
            ros_da, iri_da = regrid_to_common(ros_da, iri_da)
            r_ts, r_spat = compute_correlations(ros_da, iri_da)
            match = model_cfg.get("match", "exact")
            if np.isnan(r_ts) or np.isnan(r_spat):
                status = "ERROR"
            elif r_ts >= 0.95 and r_spat >= 0.95:
                status = "PASS"
            else:
                status = "CHECK"
            return r_ts, r_spat, status, None

        # Fetch acmaddl
        ros_da = fetch_acmaddl(model_cfg, init_month, target, region)

        # Fetch IRI
        if model_cfg["iri_type"] == "nmme":
            iri_da = fetch_iri_nmme(model_cfg, init_month, lead_months, region)
        elif model_cfg["iri_type"] == "c3s":
            iri_da = fetch_iri_c3s(model_cfg, init_month, lead_months, region)
        else:
            return None, None, "SKIP", f"Unknown iri_type: {model_cfg['iri_type']}"

        # Regrid to common resolution
        ros_da, iri_da = regrid_to_common(ros_da, iri_da)

        # Compute correlations
        r_ts, r_spat = compute_correlations(ros_da, iri_da)

        match = model_cfg.get("match", "exact")
        if np.isnan(r_ts) or np.isnan(r_spat):
            status = "ERROR"
        elif r_ts >= 0.95 and r_spat >= 0.95:
            status = "PASS"
        else:
            status = "CHECK"

        return r_ts, r_spat, status, None

    except Exception as e:
        traceback.print_exc()
        return None, None, "ERROR", str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Validate acmaddl outputs against IRI Data Library"
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"Models to validate (default: all). Choices: {', '.join(MODELS)}",
    )
    parser.add_argument(
        "--skip-cds", action="store_true",
        help="Skip CDS-based models (slow API queue)",
    )
    parser.add_argument(
        "--init-month", type=int, default=2,
        help="Init month (default: 2 = February)",
    )
    parser.add_argument(
        "--target", default="MAM",
        help="Target season (default: MAM)",
    )
    parser.add_argument(
        "--region", nargs=4, type=float, default=[-20, 20, 20, 55],
        metavar=("LAT_S", "LAT_N", "LON_W", "LON_E"),
        help="Region [lat_s lat_n lon_w lon_e] (default: -20 20 20 55, East Africa)",
    )
    args = parser.parse_args()

    model_names = args.models or list(MODELS.keys())
    init_month = args.init_month
    target = args.target
    region = args.region

    # Compute lead months from init and target
    from acmaddl.fetch import parse_target, SEASON_MONTHS
    if target.upper() in SEASON_MONTHS:
        s_month, e_month = SEASON_MONTHS[target.upper()]
        if s_month <= e_month:
            target_months = list(range(s_month, e_month + 1))
        else:
            target_months = list(range(s_month, 13)) + list(range(1, e_month + 1))
    else:
        raise ValueError(f"Unknown target season: {target}")
    lead_months = [(m - init_month) % 12 + 1 for m in target_months]

    print(f"Validation setup: init={THREELETTERS[init_month]}, target={target}, "
          f"leads={lead_months}")
    print(f"Region: lat=[{region[0]}, {region[1]}], lon=[{region[2]}, {region[3]}]")
    print()

    # Run validation
    results = []
    for name in model_names:
        if name not in MODELS:
            print(f"Unknown model: {name}")
            continue
        cfg = MODELS[name]

        if args.skip_cds and cfg.get("iri_type") == "c3s":
            results.append((name, None, None, "SKIP_CDS", cfg.get("match")))
            continue

        print(f"{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        r_ts, r_spat, status, error = validate_model(
            name, cfg, init_month, target, lead_months, region,
        )
        results.append((name, r_ts, r_spat, status, cfg.get("match")))
        if error:
            print(f"  ERROR: {error}")
        elif r_ts is not None:
            print(f"  r(ts)={r_ts:.2f}, r(spat)={r_spat:.2f} -> {status}")
        else:
            print(f"  -> {status}")
        print()

    # Print summary table
    print()
    print(f"{'Model':<14} {'Match':<10} {'r(ts)':>6} {'r(spat)':>8} {'Status':<10}")
    print("-" * 52)
    for name, r_ts, r_spat, status, match in results:
        r_ts_str = f"{r_ts:.2f}" if r_ts is not None else "—"
        r_spat_str = f"{r_spat:.2f}" if r_spat is not None else "—"
        match_str = match or "—"
        print(f"{name:<14} {match_str:<10} {r_ts_str:>6} {r_spat_str:>8} {status:<10}")

    # Write JSON report for require-parity-verified.sh hook
    from datetime import datetime, timezone
    from acmaddl.validate import ValidationResult, write_report, read_report

    # Merge with any prior entries so partial runs don't wipe the report
    report_path = Path(__file__).parent.parent / "output" / "validation_report.json"
    prior = {}
    if report_path.exists():
        try:
            for e in read_report(str(report_path)):
                prior[e.get("acmaddl_key")] = e
        except Exception:
            prior = {}

    val_results = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for name, r_ts, r_spat, status, match in results:
        cfg = MODELS[name]
        if status in ("SKIP_CDS", "SKIP"):
            continue
        vr = ValidationResult(
            product=cfg["acmaddl_product"],
            variable=cfg.get("variable", "precip"),
            reference="iri" if cfg.get("iri_type") else "none",
            r_timeseries=float(r_ts) if r_ts is not None else float("nan"),
            r_spatial=float(r_spat) if r_spat is not None else float("nan"),
            status=status,
            init_month=init_month,
            target=target,
            region=list(region),
            hindcast=tuple(cfg["hindcast"]) if cfg.get("hindcast") else None,
            timestamp=now_iso,
        )
        prior[cfg["acmaddl_product"]] = vr.to_report_entry()
        val_results.append(vr)

    # Write merged report (preserves old entries not re-run this session)
    import json
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(list(prior.values()), f, indent=2, default=str)
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
