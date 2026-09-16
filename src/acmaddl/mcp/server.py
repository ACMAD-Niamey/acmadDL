"""acmaddl MCP server: catalog, health, fetch, and zonal as agent tools.

Design rules (keep these when adding tools):

* Tools are thin. They call the same public verbs a Python user would
  (``acmaddl.fetch``, ``acmaddl.zonal``, ``acmaddl.catalog``, ...) and add
  nothing else. Behaviour lives in the library, not here.
* Arrays never cross the wire. Data-producing tools write NetCDF to
  ``ACMADDL_MCP_WORKDIR`` (default ``~/.acmaddl/mcp``) and return the path
  plus a compact JSON summary (dims, coords, variables, units).
* Every parameter carries a schema-level description (``Annotated[...,
  Field(description=...)]``) and closed vocabularies are ``Literal`` enums
  derived from the catalog / library, so an agent sees valid values without
  a second call. Every tool documents what it returns and shows one example.
* Library exceptions are re-raised as ``ToolError`` so the calling agent sees
  the message. The MCP SDK hides the text of any other exception.
* CDS / ECDS requests can queue for many minutes, longer than most MCP clients
  will wait on a single call. ``start_fetch`` + ``fetch_status`` run the same
  fetch on a background thread for that case.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.metadata
import inspect
import json
import os
import threading
import traceback
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from mcp.server.caching import CacheHint
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

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
grid ready for downscaling, use fetch with year_index=true and a hindcast range.

CDS/ECDS products can queue for a long time: prefer start_fetch + fetch_status
when a fetch might exceed your tool timeout. Results are cached locally, so a
repeated fetch of the same request is fast.

Read the acmaddl://skill resource for the full API and product conventions.
"""

# MCP 2026-07-28 requires freshness hints (ttlMs / cacheScope) on every list
# and read result. Our tool list, resources, and skill text are static for the
# life of a server process, so clients (and prompt caches in front of them) may
# hold them for a day; nothing here is per-user, so intermediaries may share.
def _dist_version(dist: str) -> str:
    """Identify the server by the installed distribution version (serverInfo)."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return ""


_DAY_MS = 24 * 60 * 60 * 1000
_CACHE_HINTS = {
    "tools/list": CacheHint(ttl_ms=_DAY_MS, scope="public"),
    "prompts/list": CacheHint(ttl_ms=_DAY_MS, scope="public"),
    "resources/list": CacheHint(ttl_ms=_DAY_MS, scope="public"),
    "resources/templates/list": CacheHint(ttl_ms=_DAY_MS, scope="public"),
    "resources/read": CacheHint(ttl_ms=60 * 60 * 1000, scope="public"),
}

mcp = MCPServer(
    name="acmaddl",
    title="acmadDL climate data",
    instructions=_INSTRUCTIONS,
    version=_dist_version("acmadDL"),
    website_url="https://github.com/ACMAD-Niamey/acmadDL",
    cache_hints=_CACHE_HINTS,
)


# --------------------------------------------------------------------------
# vocabularies derived from the catalog, so the schema shows valid values
# --------------------------------------------------------------------------

def _all_products() -> list[str]:
    return catalog.list_products(include_deprecated=True)


def _catalog_variables() -> tuple[str, ...]:
    names: set[str] = set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for pid in _all_products():
            names |= set((catalog.info(pid).get("variables") or {}).keys())
    return tuple(sorted(names))


Variable = Literal[_catalog_variables()]
Stat = Literal["mean", "sum", "min", "max", "median", "std", "count"]
Weights = Literal["area", "cos_lat"]
Boundary = Literal["center", "cover"]


# --------------------------------------------------------------------------
# typed results (become the tools' output schemas)
# --------------------------------------------------------------------------

class CoordSummary(TypedDict, total=False):
    size: int
    dtype: str
    min: float
    max: float
    step: float
    first: str
    last: str


class VariableSummary(TypedDict, total=False):
    dims: list[str]
    shape: list[int]
    dtype: str
    units: str | None
    nan_fraction: float
    min: float
    max: float


class DatasetSummary(TypedDict):
    """What every data-producing tool returns: where the file is and what is in it."""
    dims: dict[str, int]
    coords: dict[str, CoordSummary]
    variables: dict[str, VariableSummary]
    attrs: dict[str, Any]
    path: NotRequired[str]
    size_bytes: NotRequired[int]
    request: NotRequired[dict[str, Any]]


class ProductListing(TypedDict):
    product: str
    adapter: str | None
    variables: list[str]
    deprecated: bool
    notes: str | None


class JobHandle(TypedDict):
    job_id: str
    status: Literal["running"]


class JobStatus(TypedDict):
    job_id: str
    status: Literal["running", "done", "error"]
    started_at: str
    finished_at: str | None
    result: DatasetSummary | None
    error: str | None
    traceback: NotRequired[str]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def tool(*, read_only: bool = False, idempotent: bool = True, open_world: bool = False,
         destructive: bool = False, **kwargs):
    """``mcp.tool`` with client-facing annotations and a dedented docstring.

    ``read_only``: no side effects. ``open_world``: talks to external services
    (data providers). ``idempotent``: repeating the call with the same
    arguments has no additional effect. ``destructive``: may delete or
    overwrite data the caller did not create.
    """
    annotations = ToolAnnotations(read_only_hint=read_only, destructive_hint=destructive,
                                  idempotent_hint=idempotent, open_world_hint=open_world)

    def decorate(fn):
        description = kwargs.pop("description", None) or inspect.cleandoc(fn.__doc__ or "")
        return mcp.tool(annotations=annotations, description=description, **kwargs)(fn)

    return decorate


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
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _coord_summary(coord) -> CoordSummary:
    import numpy as np

    vals = coord.values
    out: CoordSummary = {"size": int(vals.size), "dtype": str(vals.dtype)}
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


def summarize(ds, *, path: str | Path | None = None, max_stats_elements: int = 20_000_000) -> DatasetSummary:
    """Compact, JSON-safe description of a Dataset / DataArray.

    ``max_stats_elements`` caps the size at which per-variable min/max/NaN
    fraction are computed, so summarizing a large file stays cheap.
    """
    import numpy as np
    import xarray as xr

    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name=ds.name or "data")
    out: DatasetSummary = {
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
        units = da.attrs.get("units")
        entry: VariableSummary = {
            "dims": list(da.dims),
            "shape": [int(s) for s in da.shape],
            "dtype": str(da.dtype),
            "units": None if units is None else str(units),
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


def _require_product(product: str) -> None:
    """Fail early with suggestions instead of a bare KeyError from the catalog."""
    known = _all_products()
    if product in known:
        return
    close = difflib.get_close_matches(product, known, n=3, cutoff=0.5)
    hint = f" Did you mean {close}?" if close else ""
    raise ToolError(f"Product not found: {product!r}.{hint} Call list_products for the catalog.")


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

ProductArg = Annotated[str, Field(
    description='Catalog product id, e.g. "nmme/cfsv2", "c3s/ecmwf", "obs/era5", '
                '"obs/chirps-v3-monthly", "chc/chirps-gefs-daily". list_products shows them all.')]


@tool(read_only=True)
def list_products(
    include_deprecated: Annotated[bool, Field(
        description="Also list deprecated products and aliases (default: current products only).")] = False,
) -> list[ProductListing]:
    """List catalog products with their adapter and the variables each one declares.

    Start here to pick a product id for describe_product / check_product / fetch.
    Product ids are namespaced: nmme/* and c3s/* are seasonal forecast models,
    obs/* are observations and reanalyses, chc/* are CHC short-range products.

    Returns: a list of {product, adapter, variables, deprecated, notes}.
    Example: list_products() -> [{"product": "nmme/cfsv2", "adapter": "opendap",
    "variables": ["precip", "temp", "sst"], ...}, ...]
    """
    out: list[ProductListing] = []
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


@tool(read_only=True)
def describe_product(product: ProductArg) -> dict[str, Any]:
    """Full catalog entry for one product.

    Use it to learn what a product can serve before fetching: variables with
    native names and units, the adapter and source URL, grid, notes, and
    whether it is deprecated (and what replaces it).

    Returns: the catalog entry as JSON plus "product", and
    "deprecation_warnings" when the id is deprecated or an alias.
    Example: describe_product(product="nmme/cfsv2")
    """
    _require_product(product)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        entry = catalog.info(product)
    result = _json_safe(entry)
    result["product"] = product
    notes = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    if notes:
        result["deprecation_warnings"] = notes
    return result


@tool(read_only=True, open_world=True)
def check_product(
    product: ProductArg,
    variable: Annotated[Variable | None, Field(
        description="Also confirm the product can serve this variable.")] = None,
    probe_remote: Annotated[bool, Field(
        description="Contact the data source to confirm it answers. Slower; may need credentials.")] = False,
) -> dict[str, Any]:
    """Health-check one product: credentials and config present, and optionally
    that the remote source answers.

    Call it before a fetch you have not run before. "kind" tells you what to do:
    "capability" (product cannot serve that variable: pick another product),
    "config" (credentials or URL missing: fix the setup), "remote"/"transient"
    (source outage: retry later).

    Returns: {product, adapter, checked_at, healthy, kind, message, ...}.
    Example: check_product(product="c3s/ecmwf", variable="precip", probe_remote=true)
    """
    _require_product(product)
    try:
        return _json_safe(acmaddl.check_product(product, probe_remote=probe_remote, variable=variable))
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


@tool(read_only=True, open_world=True)
def check_all_products(
    variable: Annotated[Variable | None, Field(
        description="Restrict to products able to serve this variable (a capability sweep).")] = None,
    probe_remote: Annotated[bool, Field(
        description="Contact every data source. Slow; may need credentials.")] = False,
) -> list[dict[str, Any]]:
    """Health-check every product at once.

    With variable set this answers "which products can serve precip / temp / sst".
    Returns: one check_product result per product.
    Example: check_all_products(variable="sst")
    """
    try:
        return _json_safe(acmaddl.check_all_products(probe_remote=probe_remote, variable=variable))
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


# --------------------------------------------------------------------------
# fetch (sync + background job)
# --------------------------------------------------------------------------

VariableArg = Annotated[Variable, Field(
    description='Canonical variable name. Most products serve "precip" and "temp"; '
                'SST products serve "sst".')]
InitArg = Annotated[str | list[str] | None, Field(
    description='Forecast issuance: "YYYY-MM" for seasonal products, "YYYY-MM-DD" for '
                'sub-seasonal / short-range ones, or a list of dates for issuance-keyed '
                'products (CHIRPS-GEFS) to stack many issuances. Omit for observations.')]
TargetArg = Annotated[str | None, Field(
    description='Target season as consecutive month initials, e.g. "MAM", "OND", "JJAS". '
                'Omit for raw observations.')]
RegionArg = Annotated[list[float] | str | None, Field(
    description="[lat_s, lat_n, lon_w, lon_e] bounding box, or a path to a .shp shapefile "
                "(result masked to the polygon; needs the geo extra). Omit for the full domain.")]
HindcastArg = Annotated[list[int] | None, Field(
    description="[start_year, end_year] to also pull the hindcast years, e.g. [1993, 2016].")]
YearIndexArg = Annotated[bool, Field(
    description="Collapse to (year, member, lat, lon) seasonal totals, the shape africas2s "
                "downscaling expects. Requires target and hindcast.")]
MonthsArg = Annotated[list[int] | None, Field(
    description="Restrict an observational fetch to these calendar months, e.g. [6, 7, 8, 9]. "
                "Prunes the download for monthly-file products.")]
SeasonalArg = Annotated[bool | None, Field(
    description="Force seasonal (true) or monthly (false) handling; default lets the product decide.")]
RegridToArg = Annotated[str | None, Field(
    description="Regrid onto another product's grid, given as that product's id.")]
GridResArg = Annotated[float | None, Field(
    description="Regrid to a fixed resolution in degrees, e.g. 0.25.")]
CacheArg = Annotated[bool, Field(
    description="Use the local cache (default). false forces a fresh download.")]
ReforecastArg = Annotated[bool, Field(
    description="Fetch the reforecast (hindcast) suite for the issuance instead of the forecast. "
                "CDS s2s products only.")]
BoundaryArg = Annotated[Boundary, Field(
    description='Which grid cells count as inside the region: "center" (cell centre inside; '
                'unbiased for area means) or "cover" (every touched cell; best for display '
                'and small regions).')]
OptionsArg = Annotated[dict[str, Any] | None, Field(
    description="Any other acmaddl.fetch keyword: allow_partial, max_retries, retry_backoff, "
                "request_interval, degenerate_attempts, region_buffer, format.")]
DestinationArg = Annotated[str | None, Field(
    description="Explicit output path (.nc, or s3://...). Default: a stable name derived from "
                "the request under ACMADDL_MCP_WORKDIR, so repeating a request reuses the file.")]


def _fetch_kwargs(product, variable, init, target, region, hindcast, year_index,
                  months, seasonal, regrid_to, grid_res, cache, reforecast,
                  boundary, options) -> dict:
    _require_product(product)
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


def _run_fetch(kwargs: dict, destination: str | None) -> DatasetSummary:
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


@tool(open_world=True)
def fetch(
    product: ProductArg,
    variable: VariableArg,
    init: InitArg = None,
    target: TargetArg = None,
    region: RegionArg = None,
    hindcast: HindcastArg = None,
    year_index: YearIndexArg = False,
    months: MonthsArg = None,
    seasonal: SeasonalArg = None,
    regrid_to: RegridToArg = None,
    grid_res: GridResArg = None,
    cache: CacheArg = True,
    reforecast: ReforecastArg = False,
    boundary: BoundaryArg = "center",
    options: OptionsArg = None,
    destination: DestinationArg = None,
) -> DatasetSummary:
    """Fetch one product/variable, normalize it, write NetCDF, and return the
    path plus a summary of what is in the file.

    Blocks until the data is on disk. CDS/ECDS products (c3s/*, obs/era5) can
    queue for many minutes; if that may exceed your tool timeout use
    start_fetch + fetch_status instead. Repeating an identical request is fast
    because results are cached.

    Output shapes: forecasts (init_time, lead_time, member, lat, lon);
    observations (time, lat, lon); with year_index=true (year, member, lat,
    lon) seasonal totals ready for africas2s downscale.

    Returns: {path, dims, coords, variables, attrs, size_bytes, request}.
    Example: fetch(product="nmme/cfsv2", variable="precip", init="2025-02",
    target="MAM", region=[-12, 6, 28, 42], hindcast=[1993, 2016], year_index=true)
    """
    kwargs = _fetch_kwargs(product, variable, init, target, region, hindcast, year_index,
                           months, seasonal, regrid_to, grid_res, cache, reforecast,
                           boundary, options)
    return _run_fetch(kwargs, destination)


class _Jobs:
    """Job table for background fetches, persisted as JSON under the workdir.

    MCP 2026-07-28 removed protocol sessions: cross-call state must travel as
    server-minted handles in ordinary tool arguments. ``job_id`` is that
    handle. Records live on disk (``<workdir>/jobs/<job_id>.json``) rather
    than in process memory so any server process can answer ``fetch_status``
    for them, which is what a stateless HTTP deployment needs. The fetch
    itself still runs on a thread of the process that accepted it.
    """

    def __init__(self):
        self._lock = threading.Lock()

    @staticmethod
    def _dir() -> Path:
        d = workdir() / "jobs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write(self, record: dict) -> None:
        path = self._dir() / f"{record['job_id']}.json"
        tmp = path.with_suffix(".json.tmp")
        with self._lock:
            tmp.write_text(json.dumps(_json_safe(record)))
            os.replace(tmp, path)

    def submit(self, fn, *args) -> str:
        job_id = uuid.uuid4().hex[:12]
        record = {
            "job_id": job_id, "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None, "result": None, "error": None,
        }
        self._write(record)

        def _target():
            try:
                result = fn(*args)
                record.update(status="done", result=result)
            except Exception as exc:  # noqa: BLE001
                record.update(status="error", error=str(exc),
                              traceback=traceback.format_exc(limit=5))
            finally:
                record["finished_at"] = datetime.now(timezone.utc).isoformat()
                self._write(record)

        threading.Thread(target=_target, name=f"acmaddl-mcp-{job_id}", daemon=True).start()
        return job_id

    def get(self, job_id: str) -> dict | None:
        path = self._dir() / f"{job_id}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def all(self) -> list[dict]:
        records = []
        for path in sorted(self._dir().glob("*.json")):
            record = json.loads(path.read_text())
            records.append({k: v for k, v in record.items() if k != "result"})
        return records


_jobs = _Jobs()


@tool(open_world=True, idempotent=False)
def start_fetch(
    product: ProductArg,
    variable: VariableArg,
    init: InitArg = None,
    target: TargetArg = None,
    region: RegionArg = None,
    hindcast: HindcastArg = None,
    year_index: YearIndexArg = False,
    months: MonthsArg = None,
    seasonal: SeasonalArg = None,
    regrid_to: RegridToArg = None,
    grid_res: GridResArg = None,
    cache: CacheArg = True,
    reforecast: ReforecastArg = False,
    boundary: BoundaryArg = "center",
    options: OptionsArg = None,
    destination: DestinationArg = None,
) -> JobHandle:
    """Start the same fetch as `fetch` on a background thread and return at once.

    Use for CDS/ECDS products or large regions where a blocking fetch could
    time out. Poll fetch_status(job_id) until status is "done" (the result is
    the fetch summary) or "error". Jobs live only as long as this server process.

    Returns: {job_id, status: "running"}.
    Example: start_fetch(product="c3s/ecmwf", variable="precip", init="2025-02",
    target="MAM", region=[-12, 6, 28, 42], hindcast=[1993, 2016], year_index=true)
    """
    kwargs = _fetch_kwargs(product, variable, init, target, region, hindcast, year_index,
                           months, seasonal, regrid_to, grid_res, cache, reforecast,
                           boundary, options)
    job_id = _jobs.submit(_run_fetch, kwargs, destination)
    return {"job_id": job_id, "status": "running"}


@tool(read_only=True)
def fetch_status(
    job_id: Annotated[str, Field(description="The job_id returned by start_fetch.")],
) -> JobStatus:
    """Status of a start_fetch job. Handles persist under the workdir, so they
    remain valid across server restarts (a job interrupted by a restart stays
    "running"; start it again).

    Returns: {job_id, status: "running" | "done" | "error", started_at,
    finished_at, result (the fetch summary when done), error (message when failed)}.
    Example: fetch_status(job_id="3f9c2a1b7d4e")
    """
    record = _jobs.get(job_id)
    if record is None:
        raise ToolError(f"Unknown job_id {job_id!r}. Known: {[j['job_id'] for j in _jobs.all()]}")
    return record


@tool(read_only=True)
def list_jobs() -> list[dict[str, Any]]:
    """All start_fetch jobs recorded under the workdir, without their results.

    Returns: a list of {job_id, status, started_at, finished_at, error}.
    """
    return _jobs.all()


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------

@tool(read_only=True)
def describe_dataset(
    path: Annotated[str, Field(description="Path to a NetCDF file, e.g. one returned by fetch or zonal.")],
) -> DatasetSummary:
    """Summarize a NetCDF file without loading it into your context: dims,
    coordinate ranges, variables with units and NaN fraction, global attributes.

    Use it to check shapes before handing a file to another tool or server.
    Returns: {path, dims, coords, variables, attrs, size_bytes}.
    Example: describe_dataset(path="~/.acmaddl/mcp/nmme-cfsv2_precip_2025-02_MAM_1a2b3c4d.nc")
    """
    with _open(path) as ds:
        return summarize(ds, path=Path(path).expanduser())


@tool()
def zonal(
    path: Annotated[str, Field(description="Gridded NetCDF with lat/lon dims (e.g. a fetch output).")],
    geometries: Annotated[str, Field(
        description="Shapefile (.shp) or GeoJSON path with one feature per zone (district, basin, admin unit).")],
    by: Annotated[str | None, Field(
        description="Attribute column with a unique code to index the new axis by (e.g. \"shapeID\"). "
                    "Default: feature position.")] = None,
    label: Annotated[str | None, Field(
        description="Attribute column carried as a display name coordinate ({dim}_label). May repeat.")] = None,
    stat: Annotated[Stat, Field(description="The reduction over each zone's cells.")] = "mean",
    weights: Annotated[Weights | None, Field(
        description='Cell weighting for mean/sum: "area" (default, cos-latitude area), "cos_lat", or null.')] = "area",
    all_touched: Annotated[bool, Field(
        description="Include every cell a polygon touches, not only cells whose centre is inside. "
                    "Needed when zones are smaller than a grid cell.")] = False,
    dim: Annotated[str, Field(description='Name of the new zone axis (default "region").')] = "region",
    destination: DestinationArg = None,
) -> DatasetSummary:
    """Reduce a gridded NetCDF file to one value per polygon and write the
    result as NetCDF. lat/lon are replaced by a `dim` axis; all other dims
    (time, year, member, ...) are preserved. Needs the geo extra (geopandas).

    Returns: {path, dims, coords, variables, attrs, size_bytes, request}.
    Example: zonal(path="chirps.nc", geometries="ethiopia_woredas.shp",
    by="shapeID", label="shapeName", stat="mean", all_touched=true)
    """
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
    # HTTP+SSE is deprecated in MCP 2026-07-28; only stdio and Streamable HTTP are offered.
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--stateless", action="store_true",
                        help="Streamable HTTP without server-side session state (safe: job "
                             "handles are persisted under the workdir).")
    parser.add_argument("--json-response", action="store_true",
                        help="Streamable HTTP: reply with plain JSON instead of an event stream.")
    args = parser.parse_args(argv)
    if args.transport == "stdio":
        mcp.run("stdio")
    else:
        mcp.run("streamable-http", host=args.host, port=args.port,
                stateless_http=args.stateless, json_response=args.json_response)


if __name__ == "__main__":
    main()
