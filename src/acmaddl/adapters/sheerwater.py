"""Adapter for Sheerwater-backed datasets (public GCS Zarr stores).

Sheerwater (https://github.com/rhiza-research/sheerwater) exposes climate
datasets via plain Python functions that return xarray Datasets. This adapter
translates acmadDL's fetch arguments into Sheerwater's calling convention and
returns the raw dataset for acmadDL's normalization layer to process.

Caching note: Sheerwater is already Nuthatch-cached upstream. Do NOT apply
@nuthatch.cache() here — that would double-store the same data.
"""
import importlib
import logging
from datetime import date

import xarray as xr

from .base import AdapterBase


class _GcsfsBucketTypeFilter(logging.Filter):
    """Suppress gcsfs's 'Could not determine bucket type' warning.

    gcsfs's storage-control API call to detect bucket type (hierarchical vs
    zonal) fails for unauthenticated reads of public buckets (e.g.
    sheerwater-public-datalake). It falls back to UNKNOWN + standard
    GCSFileSystem, which works fine for anonymous reads — but the warning
    fires once per chunk read, flooding CI logs heavily enough to trigger
    GitHub Actions runner cancellation. The fallback is correct; the
    warning is benign; silencing it is the right move.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "Could not determine bucket type" not in record.getMessage()


# Installed at module import so any code path that ends up using gcsfs via
# sheerwater (including xarray's lazy chunk reads downstream of the adapter)
# benefits. Filters are deduplicated by identity, so re-imports are safe.
logging.getLogger("gcsfs").addFilter(_GcsfsBucketTypeFilter())


def _to_time_range(date_range: tuple[int, int] | None) -> tuple[str, str]:
    """Convert a (start_year, end_year) tuple to ISO date strings."""
    if date_range is None:
        from datetime import datetime
        today = datetime.today()
        start = date(today.year - 1, 1, 1)
        end = date(today.year, 12, 31)
    else:
        start_year, end_year = date_range
        start = date(start_year, 1, 1)
        end = date(end_year, 12, 31)
    return start.isoformat(), end.isoformat()


class SheerwaterAdapter(AdapterBase):
    def fetch_data(self, product_config: dict, variable: str,
                   date_range=None, region=None) -> xr.Dataset:
        source_fn_name = product_config["source"]
        # Support both top-level (e.g. "chirps_v2") and dotted submodule
        # paths (e.g. "chirps.chirps_raw_live") since not all sheerwater
        # functions are re-exported at the package root.
        if "." in source_fn_name:
            module_path, attr = source_fn_name.rsplit(".", 1)
            module = importlib.import_module(f"sheerwater.data.{module_path}")
            fn = getattr(module, attr)
        else:
            sheerwater_data = importlib.import_module("sheerwater.data")
            fn = getattr(sheerwater_data, source_fn_name)

        start_time, end_time = _to_time_range(date_range)
        source_kwargs = product_config.get("source_kwargs", {})

        # Sheerwater functions expect ``region`` to be a string identifier
        # (e.g. ``'global'``, country code, or a registered subdivision name).
        # acmadDL's calling convention passes a [lat_s, lat_n, lon_w, lon_e]
        # bbox list. When we get a bbox, fetch the global Sheerwater dataset
        # and crop client-side; pass strings through unchanged.
        if isinstance(region, (list, tuple)):
            bbox = list(region)
            sheerwater_region = "global"
        else:
            bbox = None
            sheerwater_region = region if region is not None else "global"

        ds = fn(
            start_time=start_time,
            end_time=end_time,
            region=sheerwater_region,
            **source_kwargs,
        )

        # Crop spatially *before* materializing. Sheerwater's region= can
        # only be a small set of named strings (e.g. 'global'), so for
        # country bboxes we fetch global lazily and crop client-side.
        # Lazy crop means only the bbox slab is loaded — for
        # obs/chirps-dekadal that's the difference between ~45 GB and a
        # few hundred MB. No-data sentinels from sources like
        # chirps_raw_live are converted to NaN downstream in normalize
        # via the variable's `fill_value` catalog directive.
        if bbox is not None:
            lat_s, lat_n, lon_w, lon_e = bbox
            lat_name = "lat" if "lat" in ds.dims else "latitude"
            lon_name = "lon" if "lon" in ds.dims else "longitude"
            # `.values` on a coord triggers a tiny coord-only load, not
            # the full data; safe even when the dataset is lazy.
            lat_vals = ds[lat_name].values
            if lat_vals[0] > lat_vals[-1]:
                lat_slice = slice(lat_n, lat_s)
            else:
                lat_slice = slice(lat_s, lat_n)
            ds = ds.sel({lat_name: lat_slice, lon_name: slice(lon_w, lon_e)})

        if hasattr(ds, "compute"):
            ds = ds.compute()

        return ds

    def health_check(self, product_config: dict, probe_remote: bool = False) -> dict:
        source = product_config.get("source")
        if not source:
            return {
                "healthy": False,
                "kind": "config",
                "message": "Missing 'source' field in product config.",
                "probe_remote": bool(probe_remote),
            }

        if not probe_remote:
            return {
                "healthy": True,
                "kind": "config",
                "message": f"Sheerwater adapter config valid (source={source}).",
                "probe_remote": False,
            }

        zarr_url = product_config.get("zarr_url")
        if not zarr_url:
            return {
                "healthy": False,
                "kind": "config",
                "message": "probe_remote=True requires 'zarr_url' in product config for the Sheerwater probe.",
                "probe_remote": True,
            }

        try:
            import xarray as xr
            ds = xr.open_dataset(zarr_url, engine="zarr", chunks={})
            hindcast_range = product_config.get("grid", {}).get("hindcast_range")
            if hindcast_range:
                time_coord = next((c for c in ("time", "T") if c in ds.coords), None)
                if time_coord:
                    import pandas as pd
                    t_min = pd.Timestamp(ds[time_coord].values.min())
                    assert t_min.year <= hindcast_range[0], (
                        f"Zarr store starts {t_min.year}, expected <= {hindcast_range[0]}"
                    )
            return {
                "healthy": True,
                "kind": "remote",
                "message": f"Sheerwater Zarr store reachable: {zarr_url}",
                "probe_remote": True,
            }
        except Exception as e:
            msg = str(e).lower()
            if any(kw in msg for kw in ("403", "permission", "credentials", "timeout", "connection")):
                kind = "transient"
            else:
                kind = "remote"
            return {
                "healthy": False,
                "kind": kind,
                "message": f"Sheerwater probe failed: {e}",
                "probe_remote": True,
            }
