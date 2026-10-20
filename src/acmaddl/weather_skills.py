"""Run Rhiza weather-skills transforms, figures and agent tools on acmadDL data.

The adapter (``acmaddl.adapters.weather_skills``) turns their *fetchers* into
catalog products. This module covers the other skills: anything that takes
their standard-dataset Zarr through ``-i`` and writes a Zarr, writes a PNG, or
prints to stdout.

    from acmaddl import weather_skills as ws
    clipped = ws.run("clip-region", ds, bbox=[-2, 2, 36, 40])      # dataset, acmadDL shape
    png     = ws.run("plot", ds, variable="precip", output="map.png")
    bbox    = ws.run("resolve-region", "Kenya")                     # text

Their code is called unmodified, in-process, under the adapter's lock. All
reshaping is here: :func:`to_standard_dataset` (ours -> theirs, with
pint-parseable units and an honest provenance chain) and
:func:`from_standard_dataset` (theirs -> ours).
"""
from __future__ import annotations

import contextlib
import importlib.metadata
import io
import json
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import _paths
from .adapters import weather_skills as _adapter
from .adapters.weather_skills import (
    _SKILL_LOCK, _environ, _netcdf_safe_attrs, bbox_nwse, locate_skill, open_output,
    reshape_forecast, run_skill, skill_kind, stamp,
)
from .errors import WeatherSkillError

__all__ = ["to_standard_dataset", "from_standard_dataset", "run", "skills", "SkillInfo",
           "UNITS_OUT", "UNITS_IN"]

# acmadDL unit spelling -> pint/CF spelling their decorator quantifies.
UNITS_OUT = {
    "C": "degree_Celsius", "degC": "degree_Celsius", "K": "kelvin",
    "mm/day": "mm day-1", "mm/month": "mm month-1", "kg/m2": "kg m-2",
    "m3/m3": "1", "%": "percent",
}
# Their spelling -> ours.
UNITS_IN = {
    "degree_Celsius": "C", "degC": "C", "celsius": "C", "kelvin": "K",
    "mm day-1": "mm/day", "mm/day": "mm/day", "mm month-1": "mm/month",
    "kg m-2": "kg/m2", "percent": "%",
}
# Their classifier (weather_skills_core.units.STANDARD) requires a units attr
# on variables named like these; refuse here with a clear message rather than
# inside their decorator.
_UNITS_REQUIRED_HINTS = ("precip", "prcp", "rainfall", "rain", "tp", "pr",
                         "t2m", "2m_temperature", "temp", "tmax", "tmin", "tavg", "tas")
_RENAMES_IN = {"latitude": "lat", "longitude": "lon", "prediction_timedelta": "lead_time",
               "step": "lead_time", "number": "member"}
_CONVERTER_KWARGS = ("name", "source", "init", "lead", "init_month", "init_day")
# acmadDL spellings their decorator's pint pretty-print must map back to; checked
# by unit *equality* in pint, so "millimeter / day", "mm day-1" and "mm/day" all
# land on "mm/day".
_CANONICAL_UNITS = (("mm/day", "mm/day"), ("mm", "mm"), ("C", "degC"), ("K", "kelvin"),
                    ("kg/m2", "kg/m**2"), ("m/s", "m/s"), ("%", "percent"), ("1", "dimensionless"),
                    ("mm/month", "mm/month"), ("Pa", "Pa"), ("J/kg", "J/kg"), ("m3/m3", "m**3/m**3"))


def _canonical_units(text):
    """acmadDL spelling for a unit string, whatever spelling pint produced."""
    s = str(text).strip()
    if s in UNITS_IN:
        return UNITS_IN[s]
    try:
        import pint
        ureg = pint.UnitRegistry()
        parsed = ureg.parse_units(s)
    except Exception:
        return s
    for ours, theirs in _CANONICAL_UNITS:
        try:
            if parsed == ureg.parse_units(theirs):
                return ours
        except Exception:
            continue
    return s


def _acmaddl_version():
    try:
        return importlib.metadata.version("acmadDL")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _needs_units(varname):
    key = str(varname).lower()
    return any(key == h or key.startswith(h) or key.endswith(h) for h in _UNITS_REQUIRED_HINTS)


def _lead_to_timedelta(ds):
    lead = ds["lead_time"]
    if np.issubdtype(lead.dtype, np.timedelta64):
        return ds.assign_coords(lead_time=lead.values.astype("timedelta64[ns]"))
    units = str(lead.attrs.get("units", "")).lower()
    if units.startswith("hour"):
        vals = pd.to_timedelta(np.asarray(lead.values, dtype="float64"), unit="h")
    elif units.startswith("day"):
        vals = pd.to_timedelta(np.asarray(lead.values, dtype="float64"), unit="D")
    elif units.startswith("month"):
        raise ValueError(
            "lead_time is in months; no single timedelta represents a month on their "
            "prediction_timedelta axis. Select the target window and collapse the lead "
            "first (acmaddl year_index=True / africas2s), then convert.")
    else:
        raise ValueError(
            f"lead_time is numeric with units {units or 'missing'!r}; expected a timedelta "
            "axis or a numeric one with units 'hours' or 'days'.")
    return ds.assign_coords(lead_time=vals.values.astype("timedelta64[ns]"))


def _ensure_lead(ds, lead):
    if "lead_time" in ds.dims:
        return ds
    td = pd.Timedelta(lead) if lead is not None else pd.Timedelta(0)
    ds = ds.expand_dims(lead_time=[np.timedelta64(td.to_timedelta64(), "ns")])
    if lead is None:
        ds.attrs["acmaddl_lead_note"] = (
            "lead_time added as 0 days because the input had no lead axis and no lead= was given")
    return ds


def _existing_history(ds):
    raw = ds.attrs.get("weather_skills_history")
    if not raw:
        return []
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else list(raw)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def to_standard_dataset(obj, *, name=None, source=None, init=None, lead=None,
                        init_month=None, init_day=1, record=True):
    """acmadDL / africas2s xarray -> a weather-skills standard dataset.

    * Coordinates: ``lat``/``lon``/``time``/``init_time``/``member`` pass through
      (their vocabulary accepts them); ``lead_time`` must be a timedelta (hours/
      days are converted, months refused); a 2-D valid-time coordinate is dropped.
    * ``year`` (africas2s hindcasts) -> ``init_time`` at ``init_month``/``init_day``
      plus a single ``lead_time`` from ``lead=`` (e.g. ``"60 days"``).
    * A forecast with no time axis at all -> ``init_time`` from ``init=`` plus ``lead``.
    * Units become pint-parseable (``C`` -> ``degree_Celsius`` ...); variables their
      classifier requires units for are refused when they have none.
    * Dict attrs are JSON-encoded; encodings cleared.
    * Provenance: their ``weather_skills_history`` is appended (or started) with an
      ``acmaddl`` entry; ``weather_skills_source`` is ``source`` or ``acmaddl:<product>``.
      ``record=False`` writes the existing chain unchanged (used when a read-only tool
      such as ``provenance`` inspects the data: the inspection is not a step).
    """
    from weather_skills_core import provenance

    if isinstance(obj, xr.DataArray):
        da = obj.rename(name) if name else obj
        if da.name is None:
            raise ValueError("a bare DataArray needs a variable name: pass name='precip' (or similar)")
        ds = da.to_dataset()
    else:
        ds = obj.copy()

    # 1. coordinates
    if "time" in ds.coords and ds["time"].ndim >= 2:
        ds = ds.drop_vars("time")
    if "lead_time" in ds.dims:
        ds = _lead_to_timedelta(ds)
    # 2. africas2s hindcast on a year axis
    if "year" in ds.dims:
        if init_month is None:
            raise ValueError(
                "a 'year' axis has no place in their vocabulary: pass init_month= (and "
                "init_day=, default 1) so each year becomes an init_time, and lead= for the "
                "single lead_time (e.g. '60 days').")
        years = np.asarray(ds["year"].values).astype(int)
        inits = np.array([np.datetime64(datetime(int(y), int(init_month), int(init_day)), "ns")
                          for y in years])
        ds = ds.assign_coords(year=inits).rename({"year": "init_time"})
        ds = _ensure_lead(ds, lead)
    # 3. a single forecast / static field with no time axis
    elif "init_time" not in ds.dims and "time" not in ds.dims and "lead_time" not in ds.dims:
        if init is None:
            raise ValueError(
                "no time axis: pass init='YYYY-MM-DD' (the issuance) so the field becomes a "
                "forecast with an init_time (and lead= for its lead_time).")
        ds = ds.expand_dims(init_time=[np.datetime64(pd.Timestamp(init), "ns")])
        ds = _ensure_lead(ds, lead)
    elif "init_time" in ds.dims and "lead_time" not in ds.dims:
        ds = _ensure_lead(ds, lead)
    # 3b. Their forecast skills (step-to-time, deaccumulate) expect the layout their
    # fetchers write: a scalar `time` (the init) and a `step` lead axis. A single
    # issuance is written that way; several issuances keep an init_time dimension.
    if "init_time" in ds.dims and ds.sizes["init_time"] == 1 and "time" not in ds.coords:
        ds = ds.squeeze("init_time").rename({"init_time": "time"})
        if "lead_time" in ds.dims:
            ds = ds.rename({"lead_time": "step"})
    # 4. units
    for v in ds.data_vars:
        u = ds[v].attrs.get("units")
        if isinstance(u, str) and u.strip():
            ds[v].attrs["units"] = UNITS_OUT.get(u.strip(), u.strip())
        elif _needs_units(v):
            raise ValueError(
                f"variable {v!r} looks like a precipitation/temperature kind, which their "
                "skills require units for; set ds[...].attrs['units'] (e.g. 'mm/day', 'C').")
    # 5. attrs
    ds = _netcdf_safe_attrs(ds)
    # 6. provenance
    product = ds.attrs.get("acmaddl_product")
    src = source or (f"acmaddl:{product}" if product else ds.attrs.get("weather_skills_source") or "acmaddl")
    history = _existing_history(ds)
    if record:
        entry = provenance.build_entry(
            "acmaddl", _acmaddl_version(),
            {"source": src, "dims": {str(k): int(v) for k, v in ds.sizes.items()},
             "variables": [str(v) for v in ds.data_vars]},
            None)
        history = history + [entry]
    provenance.stamp_zarr(ds, history, source=src)
    return ds


def from_standard_dataset(ds, *, init=None):
    """A weather-skills standard dataset -> acmadDL's shape and spellings.

    Deliberately light (no catalog entry, so no ``normalize()``): their names
    to ours, their unit strings to ours, latitude ascending, helper coords
    dropped, and the adapter's forecast reshape when the forecast carries a
    scalar init ``time`` (their fetchers, ``step-to-time``) or ``init=`` is given.
    """
    ds = ds.copy()
    renames = {k: v for k, v in _RENAMES_IN.items()
               if (k in ds.dims or k in ds.coords) and v not in ds.dims and v not in ds.coords}
    if renames:
        ds = ds.rename(renames)
    if "step_bounds" in ds.variables:
        ds = ds.drop_vars("step_bounds")
    scalar_time = "time" in ds.coords and ds["time"].ndim == 0
    if "lead_time" in ds.dims and "init_time" not in ds.dims and (scalar_time or init is not None):
        init_ts = pd.Timestamp(init) if init is not None else pd.Timestamp(ds["time"].values)
        ds = ds.rename({"lead_time": "step"})
        ds = reshape_forecast(ds, init_ts.date()).rename({"step": "lead_time"})
        if init_ts != init_ts.normalize():                      # keep the hour if the init had one
            ds = ds.assign_coords(init_time=[np.datetime64(init_ts, "ns")])
            ds = ds.assign_coords(valid_time=ds["init_time"] + ds["lead_time"])
        ds = ds.rename({"valid_time": "time"})
    elif "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    elif "init_time" in ds.dims and "lead_time" in ds.dims and "time" not in ds.coords:
        # acmadDL's forecasts carry the valid time as a 2-D `time` coordinate.
        ds = ds.assign_coords(time=ds["init_time"] + ds["lead_time"])
    for v in ds.data_vars:
        u = ds[v].attrs.get("units")
        if isinstance(u, str):
            ds[v].attrs["units"] = _canonical_units(u)
    if "lat" in ds.dims:
        ds = ds.sortby("lat")
    return ds


# ── the runner ───────────────────────────────────────────────────────────────

def _input_action(parser):
    for a in parser._actions:
        if "-i" in a.option_strings or a.dest in ("input", "ds"):
            return a
    return None


def _positional_actions(parser):
    return [a for a in parser._actions if not a.option_strings and a.dest != "help"]


def _flag_value(action, key, value):
    """Render one keyword into argv tokens for ``action`` (None/False -> omit)."""
    opt = "--" + key.replace("_", "-")
    if value is None or value is False:
        return []
    if value is True:
        return [opt]
    if key == "bbox" and isinstance(value, (list, tuple)) and len(value) == 4:
        return [opt, bbox_nwse(value)]
    if isinstance(value, (date, datetime)):
        return [opt, value.isoformat()]
    if isinstance(value, (list, tuple)):
        if action.nargs in ("+", "*"):
            return [opt, *map(str, value)]
        out = []
        for v in value:
            out += [opt, str(v)]
        return out
    return [opt, str(value)]


def _render_argv(parser, paths, positionals, flags, skill):
    argv = []
    inp = _input_action(parser)
    if paths and inp is None:
        raise ValueError(f"{skill} takes no dataset input (it has no -i/--input flag)")
    if inp is not None and getattr(inp, "required", False) and not paths:
        raise ValueError(f"{skill} needs at least one dataset input (its -i/--input flag is required)")
    if inp is not None:
        if inp.nargs in ("+", "*"):
            if paths:
                argv += [inp.option_strings[0], *map(str, paths)]
        elif type(inp).__name__ == "_AppendAction":
            for p in paths:
                argv += [inp.option_strings[0], str(p)]
        else:
            if len(paths) != 1:
                raise ValueError(f"{skill} takes exactly one input; got {len(paths)}")
            argv += [inp.option_strings[0], str(paths[0])]
    pos = _positional_actions(parser)
    if len(positionals) > len(pos):
        raise ValueError(f"{skill} takes {len(pos)} positional argument(s); got {len(positionals)}")
    argv += positionals
    by_opt = {}
    for a in parser._actions:
        for o in a.option_strings:
            by_opt[o] = a
    for key, value in flags.items():
        opt = "--" + key.replace("_", "-")
        if opt not in by_opt:
            valid = sorted(o for o in by_opt if o.startswith("--") and o not in ("--help", "--output"))
            raise ValueError(f"{skill} has no flag {opt}; valid flags: {', '.join(valid)}")
        argv += _flag_value(by_opt[opt], key, value)
    return argv


def _run_stdout(fn, argv, skill, env):
    out, err = io.StringIO(), io.StringIO()
    with _SKILL_LOCK, _environ(env or {}), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            fn(list(argv))
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise WeatherSkillError(
                    f"{skill} {' '.join(argv)}: {err.getvalue().strip() or exc.code}") from None
    return out.getvalue()


def run(skill, *inputs, raw=False, provider=None, verbose=False, **flags):
    """Run one weather skill on acmadDL data and return what it produced.

    ``inputs``: xarray objects (converted with :func:`to_standard_dataset`), paths
    to existing Zarr stores (passed through), or strings for a skill's positional
    arguments (``resolve-region``'s query). ``flags``: the skill's own flags as
    keywords (``to_units="kelvin"`` -> ``--to-units kelvin``; ``True`` -> bare flag;
    lists repeat; ``bbox=[lat_s, lat_n, lon_w, lon_e]`` is reordered to N/W/S/E).
    Converter keywords (``name``, ``source``, ``init``, ``lead``, ``init_month``,
    ``init_day``) apply to every xarray input. ``output=`` for a figure is where
    the PNG lands.

    Returns a dataset in acmadDL's shape (``raw=True``: their standard dataset) for
    transforms, the PNG ``Path`` for figures, and the captured stdout text for
    agent tools. Fetchers are refused (use ``acmaddl.fetch``), as is
    ``submit-feedback``.
    """
    prov, _path = locate_skill(skill, provider)
    kind = skill_kind(prov, skill)
    if kind == "fetchers":
        raise ValueError(
            f"{skill} is a fetcher; fetch its data with acmaddl.fetch(...) and a weather-skills/* product")
    if skill == "submit-feedback":
        raise ValueError("submit-feedback posts to Rhiza's issue tracker and is not run from acmaddl")
    fn, version = _adapter.load_entrypoint(prov, skill, entrypoint=None)
    conv = {k: flags.pop(k) for k in _CONVERTER_KWARGS if k in flags}
    dest = flags.pop("output", None)
    if dest is not None and kind != "figure":
        raise ValueError(
            f"output= is only for figure skills (where the PNG lands); {skill} is a "
            f"{kind.rstrip('s')} skill and returns its result instead of writing a file")

    datasets, positionals = [], []
    for x in inputs:
        if isinstance(x, (xr.Dataset, xr.DataArray)):
            datasets.append(x)
        elif isinstance(x, (str, Path)) and Path(x).exists():
            datasets.append(Path(x))
        elif isinstance(x, str):
            positionals.append(x)
        else:
            raise TypeError(
                f"unsupported input {type(x).__name__}; pass xarray objects, Zarr paths or strings")

    env = {"MPLBACKEND": "Agg"} if kind == "figure" else {}
    with tempfile.TemporaryDirectory(prefix="acmaddl-weather-skills-run-") as tmp:
        tmp = Path(tmp)
        paths = []
        for k, d in enumerate(datasets):
            if isinstance(d, Path):
                paths.append(d)
                continue
            p = tmp / f"in{k}.zarr"
            # A read-only tool (provenance, inspect-zarr) is not a processing step.
            to_standard_dataset(d, record=(kind != "agent-tooling"), **conv).to_zarr(p, mode="w", consolidated=True)
            paths.append(p)
        argv = _render_argv(fn.parser, paths, positionals, flags, skill)
        has_output = any("-o" in a.option_strings for a in fn.parser._actions)
        if kind == "agent-tooling" or not has_output:
            return _run_stdout(fn, argv, skill, env)
        if kind == "figure":
            out = run_skill(fn, argv, tmp / "out.png", env=env, verbose=verbose, skill=skill)
            # acmadDL's scratch dir, not the system one: macOS reaps /var/folders/.../T.
            final = (Path(dest) if dest
                     else Path(_paths.get_tmpdir()) / f"acmaddl-{skill}-{uuid.uuid4().hex[:8]}.png")
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(out), str(final))
            return final
        out = run_skill(fn, argv, tmp / "out.zarr", env=env, verbose=verbose, skill=skill)
        ds = stamp(open_output(out), skill=skill, version=version, provider=prov)
    return ds if raw else from_standard_dataset(ds)


# ── inventory ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SkillInfo:
    name: str
    provider: str
    kind: str
    version: str
    flags: dict          # "--flag" -> help text (or "")


def skills(kind=None):
    """Every skill in the installed provider packages, by name.

    Read from the SKILL.md front matter and the script source (no module is
    imported), so it is cheap enough to call when deciding what to run.
    ``kind`` filters on the catalog group: ``transforms``, ``figure``,
    ``agent-tooling`` or ``fetchers``.
    """
    out = {}
    for prov in _adapter.PROVIDERS:
        try:
            files = importlib.metadata.files(prov) or []
        except importlib.metadata.PackageNotFoundError:
            continue
        by_skill = {}
        for f in files:
            m = re.match(r"skills/([^/]+)/(SKILL\.md|scripts/[^/]+\.py)$", str(f))
            if m:
                by_skill.setdefault(m.group(1), {})["md" if m.group(2) == "SKILL.md" else "py"] = f
        for name, parts in by_skill.items():
            if "md" not in parts or "py" not in parts:
                continue
            md = Path(str(parts["md"].locate())).read_text()
            src = Path(str(parts["py"].locate())).read_text()
            k = re.search(r"catalog-group:\s*(\S+)", md)
            v = re.search(r'_SKILL_VERSION\s*=\s*"([^"]+)"', src)
            flags = {}
            for mm in re.finditer(r"@weather_skill\.argument\((.*?)\)\n", src, re.S):
                body = mm.group(1)
                opts = re.findall(r'^\s*"(-[^"]+)"|\(\s*"(-[^"]+)"|,\s*"(--[^"]+)"', body, re.M)
                names = [o for tup in opts for o in tup if o]
                h = re.search(r'help=\s*(?:\(\s*)?"([^"]*)"', body)
                for o in names:
                    if o.startswith("--"):
                        flags[o] = h.group(1) if h else ""
            info = SkillInfo(name, prov, k.group(1) if k else "unknown", v.group(1) if v else "unknown", flags)
            if kind is None or info.kind == kind:
                out[name] = info
    return out
