"""Run Rhiza weather-skills fetchers in-process and hand their Zarr to acmadDL.

Design (docs/specs/2026-10-01-rhiza-weather-skills-adapter-design.md):

* Their code, unmodified. Each skill is a ``@weather_skill`` script bundled as a
  data file inside the provider wheel (``skills/<skill>/scripts/<file>.py``). We
  load it by path and call the decorated wrapper with an argv list, exactly as
  the CLI would. The wrapper writes a Zarr to ``-o``; we read it back.
* One adapter, many catalog entries. A ``rhiza/*`` product declares ``skill``,
  ``provider`` and an ``argv`` template; nothing per-skill lives here.
* Minimal reshaping: ``init_time`` from the requested init, ``valid_time``
  derived, helper coords dropped, provenance attrs kept. ``normalize()`` does
  the rest (renames, units, crop, lat order).
"""
from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import io
import json
import os
import warnings
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from .base import AdapterBase
from ..errors import RhizaNotInstalled, RhizaSkillError
from ..normalize import select_lon

INSTALL_HINT = "uv sync --group rhiza"

# (provider, skill, entrypoint) -> (wrapper, _SKILL_VERSION). Loading executes
# the script module (imports cfgrib, dynamical_catalog, ...), so do it once.
_ENTRYPOINTS: dict[tuple[str, str, str], tuple[object, str]] = {}


def locate_script(provider: str, skill: str) -> Path:
    """Path of ``skills/<skill>/scripts/*.py`` inside the installed ``provider``."""
    try:
        files = importlib.metadata.files(provider)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RhizaNotInstalled(provider) from exc
    if files is None:
        raise RhizaNotInstalled(provider)
    matches = [
        f for f in files
        if len(f.parts) == 4 and f.parts[0] == "skills" and f.parts[1] == skill
        and f.parts[2] == "scripts" and f.suffix == ".py"
    ]
    if len(matches) != 1:
        raise RhizaSkillError(
            f"{provider!r} ships {len(matches)} scripts for skill {skill!r}; "
            "expected exactly one under skills/<skill>/scripts/"
        )
    return Path(str(matches[0].locate()))


def load_entrypoint(provider: str, skill: str, entrypoint: str = "fetch"):
    """(wrapper, version) for a skill; the wrapper is its ``@weather_skill`` function."""
    key = (provider, skill, entrypoint)
    if key in _ENTRYPOINTS:
        return _ENTRYPOINTS[key]
    path = locate_script(provider, skill)
    name = "acmaddl_rhiza_" + skill.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, entrypoint, None)
    if fn is None or not hasattr(fn, "parser"):
        raise RhizaSkillError(
            f"{skill}: {entrypoint!r} in {path} is not a @weather_skill entrypoint"
        )
    version = str(getattr(module, "_SKILL_VERSION", "unknown"))
    _ENTRYPOINTS[key] = (fn, version)
    return _ENTRYPOINTS[key]


def provider_pin(provider: str) -> str:
    """The git commit the provider was installed from, or 'unknown'."""
    try:
        dist = importlib.metadata.distribution(provider)
        text = dist.read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
    if not text:
        return "unknown"
    try:
        return str(json.loads(text).get("vcs_info", {}).get("commit_id") or "unknown")
    except (ValueError, AttributeError):
        return "unknown"


def bbox_nwse(region) -> str:
    """acmadDL ``[lat_s, lat_n, lon_w, lon_e]`` -> weather-skills ``N/W/S/E``."""
    lat_s, lat_n, lon_w, lon_e = (float(v) for v in region)
    return f"{lat_n:g}/{lon_w:g}/{lat_s:g}/{lon_e:g}"


def render_argv(template, fields, *, requires_region=False, skill=""):
    """Fill a catalog ``argv`` template.

    ``{bbox}`` is special: when no region was requested the ``--bbox {bbox}``
    pair is dropped (most skills make it optional), unless the entry says
    ``requires_region: true``, in which case the caller must pass one.
    """
    if fields.get("bbox") is None and requires_region:
        raise ValueError(
            f"{skill or 'this skill'} needs a region: pass region=[lat_s, lat_n, lon_w, lon_e]"
        )
    out, i = [], 0
    while i < len(template):
        tok = str(template[i])
        nxt = str(template[i + 1]) if i + 1 < len(template) else None
        if nxt is not None and "{bbox}" in nxt and fields.get("bbox") is None:
            i += 2
            continue
        try:
            out.append(tok.format(**fields))
        except KeyError as exc:
            raise ValueError(
                f"{skill or 'skill'} argv template needs {exc.args[0]!r}, "
                f"which this request did not provide (fields: {sorted(fields)})"
            ) from None
        i += 1
    return out


@contextlib.contextmanager
def _environ(mapping):
    """Temporarily set environment variables; restore (or unset) afterwards."""
    saved = {k: os.environ.get(k) for k in mapping}
    try:
        os.environ.update({k: str(v) for k, v in mapping.items()})
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def run_skill(fn, argv, out_path, *, env=None, verbose=False, skill=""):
    """Call a ``@weather_skill`` wrapper as the CLI would, writing to ``out_path``.

    Their decorator prints the reason for a refusal to stderr and exits
    non-zero; that text becomes the ``RhizaSkillError`` message. Exit 0 with
    nothing written (the skill returned ``None``) is also an error here.
    """
    out_path = Path(out_path)
    full = [*argv, "-o", str(out_path)]
    err = io.StringIO()
    with _environ(env or {}), contextlib.redirect_stderr(err):
        try:
            fn(full)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
            if code != 0:
                text = err.getvalue().strip() or f"exit status {code}"
                raise RhizaSkillError(f"{skill or 'skill'} {' '.join(argv)}: {text}") from None
    if verbose and err.getvalue().strip():
        print(f"[acmaddl:rhiza] {skill}: {err.getvalue().strip()}")
    if not out_path.exists():
        raise RhizaSkillError(f"{skill or 'skill'} exited 0 but wrote no output at {out_path}")
    return out_path


def open_output(path):
    """Eagerly load a skill's Zarr so its temporary directory can go away."""
    with warnings.catch_warnings():
        # zarr 3 warns that consolidated metadata is not in the v3 spec; their
        # standard dataset contract writes it on purpose.
        warnings.simplefilter("ignore")
        with xr.open_zarr(path, consolidated=True) as ds:
            return ds.load()


def stamp(ds, *, skill, version, provider):
    """Record which skill produced this dataset next to their own provenance."""
    ds = ds.copy()
    ds.attrs["rhiza_skill"] = skill
    ds.attrs["rhiza_skill_version"] = version
    ds.attrs["rhiza_pin"] = provider_pin(provider)
    return ds


def reshape_forecast(ds, init):
    """Their forecast Zarr -> acmadDL's raw forecast shape, before ``normalize()``.

    * Drop any scalar ``time``/``valid_time``: for ecmwf-fetch and
      dynamical-fetch it is the init (which we know), for SubC it is the
      outlook's valid date (which must not be mistaken for the init).
    * Add ``init_time`` (length 1) from the requested issuance.
    * Derive ``valid_time = init_time + step`` as adapters/http.py does for
      CHIRPS-GEFS; ``normalize()`` maps it onto the canonical ``time`` name.
    * Drop ``step_bounds`` (cell-geometry helper; its ``nv`` dim goes with it).
    """
    for name in ("time", "valid_time"):
        if name in ds.coords and name not in ds.dims:
            ds = ds.drop_vars(name)
    if "step_bounds" in ds.variables:
        ds = ds.drop_vars("step_bounds")
    init_ns = np.datetime64(datetime(init.year, init.month, init.day), "ns")
    ds = ds.expand_dims(init_time=[init_ns])
    if "step" in ds.dims:
        ds = ds.assign_coords(valid_time=ds["init_time"] + ds["step"])
    return ds


def observation_windows(date_range, init_months, window_days, today):
    """[start, end] date windows for an observation request.

    acmadDL's observation API is year-based (``hindcast=(y0, y1)``) with an
    optional ``months=`` filter (``init_months``). Each year's selected months
    collapse into contiguous runs, clipped to yesterday, then split into
    chunks of at most ``window_days`` days. No ``hindcast`` at all means the
    trailing ``window_days`` (default 10) days ending yesterday.
    """
    yesterday = today - timedelta(days=1)
    if date_range is None:
        n = int(window_days or 10)
        spans = [(yesterday - timedelta(days=n - 1), yesterday)]
    else:
        y0, y1 = int(date_range[0]), int(date_range[1])
        months = sorted({int(m) for m in init_months}) if init_months else list(range(1, 13))
        spans = []
        for year in range(y0, y1 + 1):
            run = []
            for m in months + [None]:
                if m is None or (run and m != run[-1] + 1):
                    spans.append((date(year, run[0], 1), date(year, run[-1], monthrange(year, run[-1])[1])))
                    run = []
                if m is not None:
                    run.append(m)
        spans = [(s, min(e, yesterday)) for s, e in spans if s <= yesterday]
        if not spans:
            raise ValueError(
                f"observation window {date_range} / months={init_months} lies entirely in the future "
                f"(today is {today.isoformat()})"
            )
    if not window_days:
        return spans
    out = []
    for s, e in spans:
        cur = s
        while cur <= e:
            nxt = min(cur + timedelta(days=int(window_days) - 1), e)
            out.append((cur, nxt))
            cur = nxt + timedelta(days=1)
    return out


def crop_region(ds, region):
    """Crop a raw skill output to acmadDL's bbox before it is concatenated.

    Used for the global observation fetchers (no ``--bbox`` flag) so each
    chunk shrinks as soon as it loads. Latitude may be descending; longitude
    may be 0-360 — ``select_lon`` handles the convention and the seam.
    """
    lat_s, lat_n, lon_w, lon_e = (float(v) for v in region)
    lat_name = "lat" if "lat" in ds.dims else "latitude"
    lon_name = "lon" if "lon" in ds.dims else "longitude"
    lat_vals = ds[lat_name].values
    lat_slice = slice(lat_n, lat_s) if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1] else slice(lat_s, lat_n)
    ds = ds.sel({lat_name: lat_slice})
    return select_lon(ds, lon_w, lon_e, lon_name=lon_name)


def _read_rc(path):
    """``key: value`` lines of a cdsapirc-style file -> dict (missing file -> {})."""
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            k, sep, v = line.partition(":")
            if sep:
                out[k.strip()] = v.strip()
    return out


def ecds_environment(home=None):
    """Environment to give ecmwf-fetch so it finds ECMWF Data Store credentials.

    Their script requires ``ECMWF_DATASTORES_URL`` and ``ECMWF_DATASTORES_KEY``
    (or ``~/.ecmwfdatastoresrc``, which their client reads itself). acmadDL
    users usually hold the same token in ``~/.cdsapirc`` or ``CDSAPI_URL`` /
    ``CDSAPI_KEY`` for ``c3s/ecmwf-s2s``; map those across only when they point
    at the ECDS. A Copernicus CDS token is a different credential, so that
    case is refused with instructions rather than sent and rejected upstream.
    """
    home = Path(home) if home is not None else Path.home()
    if os.environ.get("ECMWF_DATASTORES_URL") and os.environ.get("ECMWF_DATASTORES_KEY"):
        return {}
    if (home / ".ecmwfdatastoresrc").exists():
        return {}
    url, key = os.environ.get("CDSAPI_URL"), os.environ.get("CDSAPI_KEY")
    if not (url and key):
        rc = _read_rc(home / ".cdsapirc")
        url, key = url or rc.get("url"), key or rc.get("key")
    if url and key and "ecds.ecmwf.int" in url:
        return {"ECMWF_DATASTORES_URL": url, "ECMWF_DATASTORES_KEY": key}
    raise RhizaSkillError(
        "rhiza/ecmwf-s2s needs ECMWF Data Store credentials: set ECMWF_DATASTORES_URL="
        "https://ecds.ecmwf.int/api and ECMWF_DATASTORES_KEY (or create ~/.ecmwfdatastoresrc), "
        "or point ~/.cdsapirc at https://ecds.ecmwf.int/api. A Copernicus CDS "
        "(cds.climate.copernicus.eu) key is a different token and will not work."
    )


class RhizaAdapter(AdapterBase):
    """Catalog entries: ``adapter: rhiza``; see module docstring and catalog.yaml header."""

    def fetch_data(self, product_config, variable, date_range=None, region=None):
        raise NotImplementedError  # Task 9
