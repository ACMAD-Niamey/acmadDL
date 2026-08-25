"""CPC raw big-endian GrADS binary monthly grids — native (no IRIDL).

CPC serves some legacy merged-analysis products (e.g. CAMS-OPI v0208) only as raw
GrADS "template" binaries on its public FTP/HTTP, one file per month, holding N
float32 fields on a fixed lon/lat grid with a sentinel UNDEF. This adapter reads
that format natively so those products need no IRI Data Library dependency.

Catalog config (see obs/cams-opi):
    adapter: cpc_binary
    source_url: <directory URL>
    file_pattern: "cams_opi_merged.{year}{month:02d}"   # one file per month
    binary: {nx, ny, x0, dx, y0, dy, n_fields, dtype: ">f4", undef: -999.0}
    variables:
        precip: {native_name: prcp, field_index: 3, units, target_units}

`field_index` selects which of the file's N sequential fields is the variable
(CAMS-OPI: 3 = `comb`, the blended CAMS+OPI merged precip = IRIDL's .mean/.prcp).
Older months may be gzip-compressed (`.gz`); both are tried.
"""
from __future__ import annotations

import gzip
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr

from .base import AdapterBase
from ..normalize import select_lon

_DEFAULT_LAG_MONTHS = 2


class CPCBinaryAdapter(AdapterBase):
    def fetch_data(self, product_config, variable, date_range=None, region=None):
        b = product_config["binary"]
        nx, ny, nfld = int(b["nx"]), int(b["ny"]), int(b["n_fields"])
        dtype = b.get("dtype", ">f4")
        undef = float(b.get("undef", -999.0))
        lat = b["y0"] + b["dy"] * np.arange(ny)
        lon = b["x0"] + b["dx"] * np.arange(nx)

        var_cfg = product_config["variables"][variable]
        fld = int(var_cfg["field_index"])
        native = var_cfg.get("native_name", variable)
        base = product_config["source_url"].rstrip("/")
        pattern = product_config["file_pattern"]
        verbose = product_config.get("_verbose", True)

        if date_range:
            y0, y1 = date_range
        else:
            import pandas as pd
            recent = pd.Timestamp.today() - pd.DateOffset(months=_DEFAULT_LAG_MONTHS)
            y0 = y1 = recent.year
        months = [(y, m) for y in range(y0, y1 + 1) for m in range(1, 13)]

        def one(ym):
            y, m = ym
            data = self._download(f"{base}/{pattern.format(year=y, month=m)}")
            if data is None:
                return None
            arr = np.frombuffer(data, dtype=dtype)
            if arr.size < nfld * ny * nx:
                return None
            f = arr.reshape(nfld, ny, nx)[fld].astype("float32").copy()
            f[f <= undef + 1e-3] = np.nan
            return np.datetime64(f"{y:04d}-{m:02d}"), f

        if verbose:
            print(f"[acmaddl:cpc_binary] fetching {len(months)} monthly grids from {base}")
        with ThreadPoolExecutor(max_workers=int(product_config.get("_max_workers", 8))) as ex:
            recs = [r for r in ex.map(one, months) if r is not None]
        recs.sort(key=lambda r: r[0])
        if not recs:
            raise RuntimeError(f"cpc_binary: no files found for {y0}-{y1} at {base}/{pattern}")
        times = np.array([t for t, _ in recs])
        cube = np.stack([a for _, a in recs])
        ds = xr.Dataset({native: (("time", "lat", "lon"), cube)},
                        coords={"time": times, "lat": lat, "lon": lon})
        if region:
            lat_s, lat_n, lon_w, lon_e = region
            ds = ds.sortby("lat").sel(lat=slice(lat_s - 1.0, lat_n + 1.0))
            ds = select_lon(ds, lon_w, lon_e, lon_name="lon")
        return ds.load()

    @staticmethod
    def _download(url):
        for u, gz in ((url, False), (url + ".gz", True)):
            try:
                raw = urllib.request.urlopen(u, timeout=60).read()
                if len(raw) < 100:        # FTP 404/partial bodies are tiny HTML/text
                    continue
                return gzip.decompress(raw) if gz else raw
            except Exception:
                continue
        return None

    def health_check(self, product_config, probe_remote=False):
        return {"healthy": True, "kind": "config",
                "message": "cpc_binary adapter config present.", "probe_remote": bool(probe_remote)}
