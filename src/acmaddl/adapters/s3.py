"""Adapter for reading forecast data archived on S3.

Used for hindcast data that has been bulk-downloaded from IRI and stored
on S3 as individual NetCDF files per init time. Each file contains all
members, lead times, and variables for a single initialization.

File layout: s3://{bucket}/{prefix}/{model}_{YYYY-MM}.nc
Dimensions: M (member), L (lead), Y (lat), X (lon)

Uses AWS CLI (no s3fs dependency).
"""

import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

from .base import AdapterBase



def _s3_exists(s3_path):
    result = subprocess.run(
        ["aws", "s3", "ls", s3_path],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and result.stdout.strip() != ""


def _s3_download(s3_path, local_path):
    subprocess.run(
        ["aws", "s3", "cp", s3_path, local_path, "--quiet"],
        check=True,
    )


class S3Adapter(AdapterBase):
    def health_check(self, product_config, probe_remote=False):
        bucket = product_config.get("s3_bucket")
        if not bucket:
            return {"healthy": False, "kind": "config",
                    "message": "Missing s3_bucket", "probe_remote": bool(probe_remote)}
        if not probe_remote:
            return {"healthy": True, "kind": "config",
                    "message": "S3 adapter config is valid.", "probe_remote": False}
        try:
            prefix = product_config.get("s3_prefix", "")
            path = f"s3://{bucket}/{prefix}".rstrip("/") + "/"
            exists = _s3_exists(path)
            return {"healthy": exists, "kind": "remote",
                    "message": f"S3 path {'exists' if exists else 'not found'}: {path}",
                    "probe_remote": True}
        except Exception as e:
            return {"healthy": False, "kind": "remote",
                    "message": f"S3 access failed: {e}", "probe_remote": True}

    def fetch_data(self, product_config, variable, date_range=None, region=None):
        verbose = product_config.get("_verbose", True)
        progress = product_config.get("_progress", True)
        bucket = product_config["s3_bucket"]
        prefix = product_config["s3_prefix"]
        file_template = product_config["file_template"]
        var_cfg = product_config["variables"][variable]
        native_var = var_cfg["native_name"]

        init_months = product_config.get("init_months", list(range(1, 13)))

        if date_range:
            years = list(range(date_range[0], date_range[1] + 1))
        else:
            raise ValueError("S3 adapter requires a date range (hindcast parameter)")

        # Build list of S3 paths
        paths = []
        for year in years:
            for month in init_months:
                fname = file_template.format(year=year, month=f"{month:02d}")
                paths.append(f"s3://{bucket}/{prefix}/{fname}")

        if verbose:
            print(f"[acmaddl:s3] checking {len(paths)} file(s)")

        existing = [p for p in paths if _s3_exists(p)]
        if not existing:
            raise RuntimeError(f"No S3 files found. Checked {len(paths)} paths, e.g. {paths[0]}")

        if verbose:
            print(f"[acmaddl:s3] downloading {len(existing)} file(s)")

        items = existing
        if progress:
            try:
                from tqdm.auto import tqdm
                items = tqdm(items, desc="acmadDL S3 download")
            except ImportError:
                pass

        datasets = []
        for s3_path in items:
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                try:
                    _s3_download(s3_path, tmp.name)
                    ds = xr.open_dataset(tmp.name, decode_times=False)

                    # Keep only the requested variable
                    drop_vars = [v for v in ds.data_vars if v != native_var]
                    if drop_vars:
                        ds = ds.drop_vars(drop_vars)

                    # Decode init_time (S) from "months since 1960-01-01"
                    s_coord = "S" if "S" in ds.coords else "init_time" if "init_time" in ds.coords else None
                    if s_coord is not None:
                        s_val = float(ds[s_coord].values)
                        ref_total = 1960 * 12
                        total = ref_total + int(round(s_val))
                        year, month = total // 12, total % 12 + 1
                        # ns precision at the source: a bare "YYYY-MM-DD" string
                        # resolves to day precision, and expand_dims below makes
                        # this a coordinate — non-ns there triggers xarray's
                        # non-nanosecond UserWarning.
                        init_dt = np.datetime64(f"{year}-{month:02d}-01", "ns")
                        ds = ds.drop_vars(s_coord)
                        ds = ds.expand_dims({"init_time": [init_dt]})

                    if region:
                        lat_dim = next((d for d in ("Y", "lat", "LAT", "latitude") if d in ds.dims), None)
                        lon_dim = next((d for d in ("X", "lon", "LON", "longitude") if d in ds.dims), None)
                        if lat_dim and lon_dim:
                            lat_s, lat_n, lon_w, lon_e = region
                            buf = 1.0
                            ds = ds.sortby(lat_dim)
                            ds = ds.sel({
                                lat_dim: slice(lat_s - buf, lat_n + buf),
                                lon_dim: slice(lon_w - buf, lon_e + buf),
                            })
                    ds = ds.load()
                    datasets.append(ds)
                finally:
                    os.unlink(tmp.name)

        if len(datasets) == 1:
            combined = datasets[0]
        else:
            combined = xr.concat(datasets, dim="init_time")

        # Filter and average target-season lead months (same convention as OPeNDAP:
        # NMME uses L = lead_month - 0.5, e.g. L=0.5 → month 1 after init)
        if "target_lead_months" in product_config and "L" in combined.dims:
            lead_months = product_config["target_lead_months"]
            target_L = [m - 0.5 for m in lead_months]
            avail_L = set(float(v) for v in combined.L.values)
            sel_L = [lt for lt in target_L if lt in avail_L]
            if sel_L:
                combined = combined.sel(L=sel_L).mean("L")

        return combined
