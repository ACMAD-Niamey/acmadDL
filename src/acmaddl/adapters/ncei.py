"""Adapter for NCEI North American Multi-Model Ensemble (NMME) archive.

Fetches daily seasonal forecast data from NCEI's HTTP archive. Each forecast
init/member/variable is a separate NetCDF file. The adapter scrapes the
directory listing, downloads matching files, and combines members into an
ensemble dimension.

No hindcast data is available from NCEI (2018+ only).
"""

import os
import re
import tempfile
import urllib.request
from datetime import datetime

import numpy as np
import xarray as xr

from .base import AdapterBase



def _scrape_filenames(url):
    """Scrape .nc filenames from an NCEI directory listing."""
    with urllib.request.urlopen(url, timeout=30) as resp:
        html = resp.read().decode()
    return re.findall(r'href="([^"]*\.nc)"', html)


def _subset_region(ds, region):
    if region is None:
        return ds
    lat_s, lat_n, lon_w, lon_e = region
    buf = 1.0
    lat_dim = next((d for d in ("lat", "LAT", "latitude") if d in ds.dims), None)
    lon_dim = next((d for d in ("lon", "LON", "longitude") if d in ds.dims), None)
    if lat_dim and lon_dim:
        ds = ds.sortby(lat_dim)
        ds = ds.sel({
            lat_dim: slice(lat_s - buf, lat_n + buf),
            lon_dim: slice(lon_w - buf, lon_e + buf),
        })
    return ds


class NCEIAdapter(AdapterBase):
    def health_check(self, product_config, probe_remote=False):
        url = product_config.get("source_url")
        if not url:
            return {"healthy": False, "kind": "config",
                    "message": "Missing source_url", "probe_remote": bool(probe_remote)}
        if not probe_remote:
            return {"healthy": True, "kind": "config",
                    "message": "NCEI adapter config is valid.", "probe_remote": False}
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10):
                pass
            return {"healthy": True, "kind": "remote",
                    "message": "NCEI source reachable.", "probe_remote": True}
        except Exception as e:
            return {"healthy": False, "kind": "remote",
                    "message": f"NCEI source unreachable: {e}", "probe_remote": True}

    def fetch_data(self, product_config, variable, date_range=None, region=None):
        verbose = product_config.get("_verbose", True)
        progress = product_config.get("_progress", True)
        base_url = product_config["source_url"]
        var_cfg = product_config["variables"][variable]
        native_var = var_cfg["native_name"]
        file_regex = product_config["file_regex"]
        member_list = product_config["member_list"]
        time_fix = product_config.get("time_fix")

        # Determine which years and init months to fetch
        if date_range:
            years = list(range(date_range[0], date_range[1] + 1))
        else:
            years = [datetime.now().year]

        init_months = product_config.get("init_months", list(range(1, 13)))

        # Collect URLs grouped by (year, month, member)
        targets = []
        for year in years:
            dir_url = f"{base_url}/{year}/"
            try:
                filenames = _scrape_filenames(dir_url)
            except Exception as e:
                if verbose:
                    print(f"[acmaddl:ncei] could not list {dir_url}: {e}")
                continue

            for month in init_months:
                init_str = f"{year}{month:02d}01"
                for member_id in member_list:
                    pattern = file_regex.format(
                        var=native_var, init=init_str, member=member_id,
                    )
                    matches = [f for f in filenames if re.match(pattern, f)]
                    if matches:
                        targets.append({
                            "url": f"{dir_url}{matches[0]}",
                            "member": member_id,
                            "init_year": year,
                            "init_month": month,
                        })

        if not targets:
            raise RuntimeError(
                f"No NCEI files found for {native_var}, years={years}, months={init_months}"
            )

        if verbose:
            print(f"[acmaddl:ncei] downloading {len(targets)} file(s)")

        items = targets
        if progress:
            try:
                from tqdm.auto import tqdm
                items = tqdm(items, desc="acmadDL NCEI download")
            except ImportError:
                pass

        member_datasets = []
        for t in items:
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                try:
                    urllib.request.urlretrieve(t["url"], tmp.name)
                    ds = xr.open_dataset(tmp.name)

                    # Fix files with unnamed dimensions (e.g., GEM-NEMO tasmax)
                    if "ncl0" in ds.dims:
                        import pandas as pd
                        rename_dims = {}
                        if "ncl0" in ds.dims:
                            rename_dims["ncl0"] = "time"
                        if "ncl1" in ds.dims:
                            rename_dims["ncl1"] = "lat"
                        if "ncl2" in ds.dims:
                            rename_dims["ncl2"] = "lon"
                        ds = ds.rename(rename_dims)
                        # Assign coordinates from init date and known grid
                        init_date = pd.Timestamp(f"{t['init_year']}-{t['init_month']:02d}-01")
                        # ns precision at the source: pd.date_range's inferred
                        # resolution isn't guaranteed to be ns (pandas picks the
                        # coarsest resolution that fits), and this becomes a
                        # coordinate via assign_coords below.
                        times = pd.date_range(
                            init_date, periods=ds.sizes["time"], freq="D"
                        ).values.astype("datetime64[ns]")
                        lats = np.linspace(-90, 90, ds.sizes["lat"])
                        lons = np.linspace(0, 359, ds.sizes["lon"])
                        ds = ds.assign_coords(time=times, lat=lats, lon=lons)

                    ds = _subset_region(ds, region)
                    ds = ds.load()

                    # Fix GEOS-5 time encoding bug (year hardcoded to 2022)
                    if time_fix == "geos5_year":
                        import pandas as pd
                        time_dim = "time" if "time" in ds.dims else "TIME"
                        times = pd.to_datetime(ds[time_dim].values)
                        year_offset = t["init_year"] - times[0].year
                        if year_offset != 0:
                            corrected = times + pd.DateOffset(years=year_offset)
                            # ns precision at the source: same rationale as above.
                            ds = ds.assign_coords(
                                {time_dim: corrected.values.astype("datetime64[ns]")}
                            )

                    # Add init_time coordinate. ns precision at the source: a bare
                    # "YYYY-MM-DD" string resolves to day precision, and
                    # expand_dims below makes this a coordinate.
                    init_time = np.datetime64(
                        f"{t['init_year']}-{t['init_month']:02d}-01", "ns"
                    )
                    ds = ds.expand_dims({"init_time": [init_time]})

                    # Add member coordinate
                    ds = ds.expand_dims({"member": [t["member"]]})

                    member_datasets.append(ds)
                except Exception as e:
                    if verbose:
                        print(f"[acmaddl:ncei] error fetching {t['url']}: {e}")
                finally:
                    os.unlink(tmp.name)

        if not member_datasets:
            raise RuntimeError("No data files retrieved")

        # Combine all members
        combined = xr.concat(member_datasets, dim="member")

        # Filter to target date range if specified (by month, to handle NOLEAP calendars)
        target_range = product_config.get("target_range")
        if target_range:
            time_dim = "time" if "time" in combined.dims else "TIME"
            s_month, e_month = target_range[0].month, target_range[1].month
            if s_month <= e_month:
                target_months = set(range(s_month, e_month + 1))
            else:
                target_months = set(range(s_month, 13)) | set(range(1, e_month + 1))
            time_months = combined[time_dim].dt.month
            combined = combined.sel({time_dim: time_months.isin(list(target_months))})

        return combined
