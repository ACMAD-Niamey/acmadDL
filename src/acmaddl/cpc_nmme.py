"""ACMAD/CPC pre-formatted NMME seasonal predictors — native, no IRIDL.

CPC publishes the NMME seasonal outlooks as ready-made CPT-format text on its public FTP
(ftp.cpc.ncep.noaa.gov/International/nmme/seasonal_nmme_{hindcast,forecast}_in_cpt_format),
one file per model x variable x IC-month x target-lead. These are exactly the files ACMAD's
operational pipeline ingests, so using them reproduces ACMAD's predictor inputs bit-for-bit
(verified max-abs-diff 0.0) with no IRI Data Library dependency.

Each file is a global 1deg seasonal field already aggregated to the target season: the
hindcast holds one field per training year, the forecast holds the target year. That is a
different data model from the raw NMME adapters (which fetch monthly output and aggregate),
so this is a small self-contained module rather than a fetch()/assemble() adapter. Use it
directly for the CCA predictor:

    hc, fc = acmaddl.cpc_nmme_predictor("cmc1", "sst", region=[-35,35,0,360],
                                        target="ASO", hindcast=(1991,2020), forecast_year=2026)
    # hc: (year, member, lat, lon); fc: (year=[2026], member, lat, lon) -- ready for deepscale

Models are ACMAD's roster names as CPC files them: cfsv2, cmc1, cmc2, nasa, ncar_ccsm4, gfdl,
nmme. IC month name defaults to the target's issuance (Jul IC -> ASO/SON/OND leads 8-10/9-11/
10-12). Results cache under the nuthatch root.
"""
from __future__ import annotations

import os
import pickle
import re
import tempfile
import urllib.request

import numpy as np
import xarray as xr

_HOST = "https://ftp.cpc.ncep.noaa.gov/International/nmme"
_HCST = f"{_HOST}/seasonal_nmme_hindcast_in_cpt_format"
_FCST = f"{_HOST}/seasonal_nmme_forecast_in_cpt_format"
# target season -> (IC month name, target-month lead label used in the CPC filenames)
_SEASON = {"ASO": ("Jul", "8-10"), "SON": ("Aug", "9-11"), "OND": ("Sep", "10-12")}


def _cache_dir():
    """Pickle-cache directory under ~/.nuthatch/acmaddl/cpc_nmme.

    Created on every call (not at import). Import-time ``makedirs`` is not enough:
    callers may replace ``~/.nuthatch`` after import (e.g. symlink onto a Shared
    Drive in Colab), which drops any dirs created at import and leaves
    ``open(..., "wb")`` failing with FileNotFoundError.
    """
    path = os.path.join(os.path.expanduser("~"), ".nuthatch", "acmaddl", "cpc_nmme")
    os.makedirs(path, exist_ok=True)
    return path


def _parse_cpt(text):
    """CPT v10 gridded text -> DataArray(time, lat, lon); missing (-999) -> NaN."""
    lines = text.splitlines()
    fields, times, i = [], [], 0
    while i < len(lines):
        if lines[i].startswith("cpt:field="):
            attrs = dict(re.findall(r"cpt:(\w+)=([^,]+)", lines[i]))
            nrow, ncol = int(attrs["nrow"]), int(attrs["ncol"])
            lons = np.array([v for v in re.split(r"\s+", lines[i + 1].strip()) if v][:ncol], float)
            data = np.full((nrow, ncol), np.nan); lats = np.empty(nrow)
            for r in range(nrow):
                parts = [v for v in re.split(r"\s+", lines[i + 2 + r].strip()) if v]
                lats[r] = float(parts[0]); row = np.array(parts[1:ncol + 1], float)
                data[r, :len(row)] = row
            data[data <= -998.0] = np.nan
            fields.append(xr.DataArray(data, dims=("lat", "lon"), coords={"lat": lats, "lon": lons}))
            times.append(attrs.get("T", str(len(times))).strip()); i += 2 + nrow
        else:
            i += 1
    da = xr.concat(fields, "time") if len(fields) > 1 else fields[0].expand_dims(time=[times[0]])
    return da.assign_coords(time=times).sortby("lat").sortby("lon")


def _download(url):
    return urllib.request.urlopen(url, timeout=90).read().decode("latin-1", "ignore")


def _fetch_global(model, var, target, hindcast, forecast_year):
    ic, lead = _SEASON[target.upper()]
    key = os.path.join(
        _cache_dir(),
        f"{model}_{var}_{ic}_{lead}_{hindcast[0]}-{hindcast[1]}_{forecast_year}.pkl",
    )
    if os.path.exists(key):
        with open(key, "rb") as f:
            d = pickle.load(f)
        return xr.DataArray.from_dict(d["hc"]), xr.DataArray.from_dict(d["fc"])
    hc = _parse_cpt(_download(f"{_HCST}/{model}_{var}_hcst_{ic}ic_{lead}_{hindcast[0]}-{hindcast[1]}.txt"))
    fc = _parse_cpt(_download(f"{_FCST}/{model}_{var}_fcst_{ic}ic_{lead}_{forecast_year}-{forecast_year}.txt"))
    hc = hc.rename({"time": "year"}).assign_coords(year=list(range(hindcast[0], hindcast[1] + 1)))
    fc = fc.rename({"time": "year"}).assign_coords(year=[forecast_year])
    with open(key, "wb") as f:
        pickle.dump({"hc": hc.to_dict(), "fc": fc.to_dict()}, f)
    return hc, fc


def _crop(da, region):
    lat_s, lat_n, lon_w, lon_e = region
    da = da.sortby("lat").sortby("lon").sel(lat=slice(lat_s, lat_n))
    if lon_w < 0:                                   # request -180..180; CPC source is 0..360
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")
    if lon_w <= lon_e:
        return da.sel(lon=slice(lon_w, lon_e))
    return da.sel(lon=(da.lon >= lon_w) | (da.lon <= lon_e))     # seam-crossing (e.g. TROP_PAC)


def cpc_nmme_predictor(model, variable, region, target="ASO", hindcast=(1991, 2020),
                       forecast_year=2026, regrid_to=None):
    """(hindcast, forecast) CCA-ready predictor from CPC's pre-formatted NMME CPT files.

    `variable` in {'sst','precip'}; `region` = [lat_s, lat_n, lon_w, lon_e]; `regrid_to` a
    DataArray (lat/lon) to interpolate onto (e.g. the predictand grid). Returns fields with a
    singleton `member` dim so deepscale's ensemble-mean is a no-op (CPC fields are already
    ensemble means)."""
    hc, fc = _fetch_global(model, variable, target, hindcast, forecast_year)
    hc, fc = _crop(hc, region), _crop(fc, region)
    if regrid_to is not None:
        hc = hc.interp(lat=regrid_to.lat, lon=regrid_to.lon)
        fc = fc.interp(lat=regrid_to.lat, lon=regrid_to.lon)
    return hc.expand_dims(member=[0]), fc.expand_dims(member=[0])
