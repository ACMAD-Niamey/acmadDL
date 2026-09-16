"""acmaddl MCP server: catalog, health, fetch, and zonal as agent tools.

Design rules (keep these when adding tools):

* Tools are thin. They call the same public verbs a Python user would
  (``acmaddl.fetch``, ``acmaddl.zonal``, ``acmaddl.catalog``, ...) and add
  nothing else. Behaviour lives in the library, not here.
* Arrays never cross the wire. Data-producing tools write NetCDF to
  ``ACMADDL_MCP_WORKDIR`` (default ``~/.acmaddl/mcp``) and return the path
  plus a compact JSON summary (dims, coords, variables, units).
* Library exceptions are re-raised as ``ToolError`` so the calling agent sees
  the message. The MCP SDK hides the text of any other exception.
* CDS / ECDS requests can queue for many minutes, longer than most MCP clients
  will wait on a single call. ``start_fetch`` + ``fetch_status`` run the same
  fetch on a background thread for that case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import acmaddl
from acmaddl import catalog

_HERE = Path(__file__).resolve().parent
# Source checkouts carry the Agent Skill at <repo>/skills/acmaddl; wheels do
# not, so resources degrade gracefully when the directory is absent.
_SKILL_DIR = _HERE.parents[2] / "skills" / "acmaddl"
_CATALOG_PATH = _HERE.parent / "catalog.yaml"

_INSTRUCTIONS = """\
acmaddl fetches seasonal / sub-seasonal climate data (NMME, C3S, ERA5, CHIRPS,
CHIRPS-GEFS, ERSST, TAMSAT, IMERG, S2S ...) and returns CF-aligned NetCDF with
canonical names: lat/lon (lat ascending, lon in [-180, 180]), time for
observations, init_time/lead_time/member for forecasts, variables precip/temp/sst.

Workflow: list_products -> describe_product -> (check_product) -> fetch.
fetch writes a NetCDF file and returns its path plus a summary; pass that path
to other tools or servers (e.g. africas2s-mcp). Use describe_dataset to
inspect any NetCDF file. For hindcast + forecast on a (year, member, lat, lon)
grid ready for downscaling, use fetch with year_index=true.

CDS/ECDS products can queue for a long time: prefer start_fetch + fetch_status
when a fetch might exceed your tool timeout. Results are cached locally, so a
repeated fetch of the same request is fast.

Read the acmaddl://skill resource for the full API and product conventions.
"""

mcp = MCPServer(
    name="acmaddl",
    title="acmadDL climate data",
    instructions=_INSTRUCTIONS,
    version=getattr(acmaddl, "__version__", ""),
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def workdir() -> Path:
    """Directory where data-producing tools write their NetCDF outputs."""
    root = os.environ.get("ACMADDL_MCP_WORKDIR") or (Path.home() / ".acmaddl" / "mcp")
    path = Path(root).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_safe(obj: Any) -> Any:
    """Recursively coerce numpy / datetime / Path values into JSON-native types."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.generic,)):
        return _json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _coord_summary(coord) -> dict:
    import numpy as np

    vals = coord.values
    out: dict[str, Any] = {"size": int(vals.size), "dtype": str(vals.dtype)}
    if vals.size == 0:
        return out
    flat = vals.ravel()
    if np.issubdtype(vals.dtype, np.datetime64):
        out["first"] = str(np.datetime_as_string(flat[0], unit="D"))
        out["last"] = str(np.datetime_as_string(flat[-1], unit="D"))
    elif np.issubdtype(vals.dtype, np.number):
        out["min"] = float(np.nanmin(flat))
        out["max"] = float(np.nanmax(flat))
        if vals.size > 1:
            step = np.diff(flat[: min(flat.size, 3)])
            if step.size and np.all(step == step[0]):
                out["step"] = float(step[0])
    else:
        out["first"] = str(flat[0])
        out["last"] = str(flat[-1])
    return out


def summarize(ds, *, path: str | Path | None = None, max_stats_elements: int = 20_000_000) -> dict:
    """Compact, JSON-safe description of a Dataset / DataArray.

    ``max_stats_elements`` caps the size at which per-variable min/max/NaN
    fraction are computed, so summarizing a large file stays cheap.
    """
    import numpy as np
    import xarray as xr

    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name=ds.name or "data")
    out: dict[str, Any] = {
        "dims": {k: int(v) for k, v in ds.sizes.items()},
        "coords": {k: _coord_summary(ds.coords[k]) for k in ds.coords},
        "variables": {},
        "attrs": _json_safe(dict(ds.attrs)),
    }
    if path is not None:
        out["path"] = str(path)
        try:
            out["size_bytes"] = os.path.getsize(path)
        except OSError:
            pass
    for name, da in ds.data_vars.items():
        entry: dict[str, Any] = {
            "dims": list(da.dims),
            "shape": [int(s) for s in da.shape],
            "dtype": str(da.dtype),
            "units": da.attrs.get("units"),
        }
        if da.size and da.size <= max_stats_elements and np.issubdtype(da.dtype, np.number):
            vals = da.values
            finite = np.isfinite(vals)
            entry["nan_fraction"] = float(1 - finite.mean())
            if finite.any():
                entry["min"] = float(vals[finite].min())
                entry["max"] = float(vals[finite].max())
        out["variables"][str(name)] = entry
    return out


def _open(path: str):
    import xarray as xr

    p = Path(path).expanduser()
    if not p.exists():
        raise ToolError(f"No such file: {p}")
    try:
        return xr.open_dataset(p)
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"Could not open {p} as NetCDF: {exc}") from exc


def _netcdf_safe(obj):
    """Cast object-dtype (Python str) coords/variables to fixed-width unicode.

    ``zonal`` indexes its new axis by shapefile attribute values, which pandas
    hands back as object arrays; netCDF4 refuses to write those.
    """
    import xarray as xr

    if isinstance(obj, xr.DataArray):
        return _netcdf_safe(obj.to_dataset(name=obj.name or "data"))
    for name in list(obj.coords) + list(obj.data_vars):
        if obj[name].dtype == object:
            obj[name] = obj[name].astype(str)
    return obj


def _slug(*parts: Any) -> str:
    text = "_".join(str(p) for p in parts if p not in (None, ""))
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in text).strip("-")


def _output_path(destination: str | None, *parts: Any, params: dict) -> str:
    """Resolve where a tool writes: explicit destination, else a stable name in workdir."""
    if destination:
        p = Path(destination).expanduser()
        if not str(destination).startswith("s3://"):
            p.parent.mkdir(parents=True, exist_ok=True)
            return str(p)
        return str(destination)
    digest = hashlib.sha1(json.dumps(_json_safe(params), sort_keys=True).encode()).hexdigest()[:8]
    return str(workdir() / f"{_slug(*parts)}_{digest}.nc")


def _region_arg(region):
    if region is None:
        return None
    if isinstance(region, str):
        return region
    if isinstance(region, (list, tuple)) and len(region) == 4:
        return [float(v) for v in region]
    raise ToolError("region must be [lat_s, lat_n, lon_w, lon_e] or a shapefile path.")


def _hindcast_arg(hindcast):
    if hindcast is None:
        return None
    if isinstance(hindcast, (list, tuple)) and len(hindcast) == 2:
        return (int(hindcast[0]), int(hindcast[1]))
    raise ToolError("hindcast must be [start_year, end_year].")


def _wrap(exc: Exception) -> ToolError:
    """Surface library exceptions to the agent with their real message."""
    if isinstance(exc, ToolError):
        return exc
    return ToolError(f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# catalog + health
# --------------------------------------------------------------------------

@mcp.tool()
def list_products(include_deprecated: bool = False) -> list[dict]:
    """List catalog products with adapter and the variables each one declares.

    Product ids look like "nmme/cfsv2", "obs/era5", "obs/chirps-v3-monthly". Use
    describe_product for the full catalog entry.
    """
    import warnings

    out = []
    for pid in catalog.list_products(include_deprecated=include_deprecated):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            entry = catalog.info(pid)
        out.append({
            "product": pid,
            "adapter": entry.get("adapter"),
            "variables": list((entry.get("variables") or {}).keys()),
            "deprecated": bool(entry.get("deprecated", False)),
            "notes": entry.get("notes"),
        })
    return out


@mcp.tool()
def describe_product(product: str) -> dict:
    """Full catalog entry for one product: adapter, source URL, variables with
    native names and units, grid, notes, deprecation status."""
    import warnings

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            entry = catalog.info(product)
    except KeyError as exc:
        raise ToolError(str(exc)) from exc
    result = _json_safe(entry)
    result["product"] = product
    notes = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    if notes:
        result["deprecation_warnings"] = notes
    return result


@mcp.tool()
def check_product(product: str, variable: str | None = None, probe_remote: bool = False) -> dict:
    """Health-check one product: credentials / config present, and (optionally)
    that the remote source answers. Pass variable to also confirm the product can
    serve it. probe_remote=true contacts the source and can be slow."""
    try:
        return _json_safe(acmaddl.check_product(product, probe_remote=probe_remote, variable=variable))
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


@mcp.tool()
def check_all_products(variable: str | None = None, probe_remote: bool = False) -> list[dict]:
    """Health-check every product. With variable set this doubles as a capability
    sweep: which products can serve precip / temp / sst."""
    try:
        return _json_safe(acmaddl.check_all_products(probe_remote=probe_remote, variable=variable))
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


# --------------------------------------------------------------------------
# fetch (sync + background job)
# --------------------------------------------------------------------------

def _fetch_kwargs(product, variable, init, target, region, hindcast, year_index,
                  months, seasonal, regrid_to, grid_res, cache, reforecast,
                  boundary, options) -> dict:
    kwargs: dict[str, Any] = dict(
        product=product, variable=variable, init=init, target=target,
        region=_region_arg(region), hindcast=_hindcast_arg(hindcast),
        year_index=year_index, cache=cache, reforecast=reforecast,
        boundary=boundary, verbose=False, progress=False,
    )
    if months:
        kwargs["months"] = [int(m) for m in months]
    if seasonal is not None:
        kwargs["seasonal"] = seasonal
    if regrid_to:
        kwargs["regrid_to"] = regrid_to
    if grid_res is not None:
        kwargs["grid_res"] = grid_res
    if options:
        kwargs.update(options)
    return kwargs


def _run_fetch(kwargs: dict, destination: str | None) -> dict:
    params = {k: v for k, v in kwargs.items() if k not in ("verbose", "progress")}
    init = kwargs.get("init")
    init_tag = init if isinstance(init, str) else (f"{init[0]}..{init[-1]}" if init else None)
    path = _output_path(destination, kwargs["product"], kwargs["variable"],
                        init_tag, kwargs.get("target"), params=params)
    try:
        ds = acmaddl.fetch(destination=path, **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc
    summary = summarize(ds, path=path)
    summary["request"] = _json_safe(params)
    return summary


_FETCH_DOC = """\
product: catalog id, e.g. "nmme/cfsv2", "c3s/ecmwf", "obs/era5", "obs/chirps-v3-monthly".
variable: canonical name — "precip", "temp", or "sst".
init: forecast issuance — "YYYY-MM" for seasonal products, "YYYY-MM-DD" for
  sub-seasonal / short-range ones, or a list of dates for issuance-keyed
  products (CHIRPS-GEFS). Omit for observations.
target: season, e.g. "MAM", "OND", "JJAS". Omit for raw observations.
region: [lat_s, lat_n, lon_w, lon_e] bbox, or a path to a .shp shapefile
  (result masked to the polygon; needs the geo extra).
hindcast: [start_year, end_year] to also pull the hindcast years.
year_index: true collapses to (year, member, lat, lon) seasonal totals, the
  shape africas2s downscaling expects.
months: restrict an observational fetch to calendar months, e.g. [6,7,8,9].
regrid_to / grid_res: regrid onto another product's grid or a fixed resolution.
options: any other acmaddl.fetch keyword (allow_partial, max_retries,
  request_interval, degenerate_attempts, region_buffer, ...).
destination: explicit output path (.nc or s3://...). Default: a stable
  name under ACMADDL_MCP_WORKDIR."""


@mcp.tool(description="Fetch one product/variable, normalize it, write NetCDF, and "
          "return the path plus a summary. Blocks until done; for slow CDS/ECDS "
          "requests use start_fetch instead.\n\n" + _FETCH_DOC)
def fetch(
    product: str,
    variable: str,
    init: str | list[str] | None = None,
    target: str | None = None,
    region: list[float] | str | None = None,
    hindcast: list[int] | None = None,
    year_index: bool = False,
    months: list[int] | None = None,
    seasonal: bool | None = None,
    regrid_to: str | None = None,
    grid_res: float | None = None,
    cache: bool = True,
    reforecast: bool = False,
    boundary: str = "center",
    options: dict[str, Any] | None = None,
    destination: str | None = None,
) -> dict:
    kwargs = _fetch_kwargs(product, variable, init, target, region, hindcast, year_index,
                           months, seasonal, regrid_to, grid_res, cache, reforecast,
                           boundary, options)
    return _run_fetch(kwargs, destination)


class _Jobs:
    """Minimal in-process job table for background fetches."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def submit(self, fn, *args) -> str:
        job_id = uuid.uuid4().hex[:12]
        record = {
            "job_id": job_id, "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None, "result": None, "error": None,
        }
        with self._lock:
            self._jobs[job_id] = record

        def _target():
            try:
                result = fn(*args)
                with self._lock:
                    record.update(status="done", result=result)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    record.update(status="error", error=str(exc),
                                  traceback=traceback.format_exc(limit=5))
            finally:
                with self._lock:
                    record["finished_at"] = datetime.now(timezone.utc).isoformat()

        threading.Thread(target=_target, name=f"acmaddl-mcp-{job_id}", daemon=True).start()
        return job_id

    def get(self, job_id: str) -> dict:
        with self._lock:
            record = self._jobs.get(job_id)
            return dict(record) if record else None

    def all(self) -> list[dict]:
        with self._lock:
            return [{k: v for k, v in r.items() if k != "result"} for r in self._jobs.values()]


_jobs = _Jobs()


@mcp.tool(description="Start a fetch on a background thread and return a job_id "
          "immediately. Poll fetch_status. Same parameters as fetch.\n\n" + _FETCH_DOC)
def start_fetch(
    product: str,
    variable: str,
    init: str | list[str] | None = None,
    target: str | None = None,
    region: list[float] | str | None = None,
    hindcast: list[int] | None = None,
    year_index: bool = False,
    months: list[int] | None = None,
    seasonal: bool | None = None,
    regrid_to: str | None = None,
    grid_res: float | None = None,
    cache: bool = True,
    reforecast: bool = False,
    boundary: str = "center",
    options: dict[str, Any] | None = None,
    destination: str | None = None,
) -> dict:
    kwargs = _fetch_kwargs(product, variable, init, target, region, hindcast, year_index,
                           months, seasonal, regrid_to, grid_res, cache, reforecast,
                           boundary, options)
    job_id = _jobs.submit(_run_fetch, kwargs, destination)
    return {"job_id": job_id, "status": "running"}


@mcp.tool()
def fetch_status(job_id: str) -> dict:
    """Status of a start_fetch job: running, done (with the fetch result), or
    error (with the message)."""
    record = _jobs.get(job_id)
    if record is None:
        raise ToolError(f"Unknown job_id {job_id!r}. Known: {[j['job_id'] for j in _jobs.all()]}")
    return record


@mcp.tool()
def list_jobs() -> list[dict]:
    """All start_fetch jobs in this server process, without their results."""
    return _jobs.all()


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------

@mcp.tool()
def describe_dataset(path: str) -> dict:
    """Summarize a NetCDF file: dims, coordinate ranges, variables, units, NaN
    fraction, global attributes. Use it on any path returned by another tool."""
    with _open(path) as ds:
        return summarize(ds, path=Path(path).expanduser())


@mcp.tool()
def zonal(
    path: str,
    geometries: str,
    by: str | None = None,
    label: str | None = None,
    stat: str = "mean",
    weights: str | None = "area",
    all_touched: bool = False,
    dim: str = "region",
    destination: str | None = None,
) -> dict:
    """Reduce a gridded NetCDF file to one value per polygon (district, basin,
    admin unit) and write the result as NetCDF.

    path: gridded NetCDF with lat/lon dims (e.g. a fetch output).
    geometries: shapefile / GeoJSON path; one output element per feature.
    by: unique column to index the new dim by (an admin code). label: display
    column carried as {dim}_label. stat: mean|sum|min|max|median|std|count.
    weights: "area" (default), "cos_lat", or null. all_touched: include every
    cell a polygon touches (needed for polygons smaller than a grid cell).
    Needs the geo extra (geopandas)."""
    with _open(path) as ds:
        try:
            out = acmaddl.zonal(ds.load(), geometries, by=by, label=label, stat=stat,
                                weights=weights, all_touched=all_touched, dim=dim)
        except Exception as exc:  # noqa: BLE001
            raise _wrap(exc) from exc
    params = dict(path=str(path), geometries=str(geometries), by=by, label=label,
                  stat=stat, weights=weights, all_touched=all_touched, dim=dim)
    out_path = _output_path(destination, Path(path).stem, "zonal", stat, params=params)
    try:
        out = _netcdf_safe(out)
        out.to_netcdf(out_path)
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc
    summary = summarize(out, path=out_path)
    summary["request"] = params
    return summary


# --------------------------------------------------------------------------
# resources: the Agent Skill and catalog, for harnesses that can't load skills
# --------------------------------------------------------------------------

def _read_text(path: Path, what: str) -> str:
    if not path.exists():
        return (f"{what} is not available in this install (looked for {path}). "
                "It ships with the source checkout: https://github.com/ACMAD-Niamey/acmadDL")
    return path.read_text(encoding="utf-8")


@mcp.resource("acmaddl://skill", mime_type="text/markdown",
              description="The acmaddl Agent Skill: API quick reference, conventions, gotchas.")
def skill_resource() -> str:
    return _read_text(_SKILL_DIR / "SKILL.md", "SKILL.md")


@mcp.resource("acmaddl://skill/references/{name}", mime_type="text/markdown",
              description="One skill reference doc: api, products, data-conventions, plotting, troubleshooting.")
def skill_reference(name: str) -> str:
    name = name.removesuffix(".md")
    return _read_text(_SKILL_DIR / "references" / f"{name}.md", f"reference {name!r}")


@mcp.resource("acmaddl://catalog", mime_type="application/yaml",
              description="The raw product catalog (catalog.yaml).")
def catalog_resource() -> str:
    return _read_text(_CATALOG_PATH, "catalog.yaml")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="acmaddl-mcp", description="Run the acmaddl MCP server.")
    parser.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.transport == "stdio":
        mcp.run("stdio")
    else:
        mcp.run(args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
