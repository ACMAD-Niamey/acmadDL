"""Run Rhiza weather-skills fetchers in-process and hand their Zarr to acmadDL.

Design:

* Their code, unmodified. Each skill is a ``@weather_skill`` script bundled as a
  data file inside the provider wheel (``skills/<skill>/scripts/<file>.py``). We
  load it by path and call the decorated wrapper with an argv list, exactly as
  the CLI would. The wrapper writes a Zarr to ``-o``; we read it back.
* One adapter, many catalog entries. A ``weather-skills/*`` product declares ``skill``,
  ``provider`` and an ``argv`` template; nothing per-skill lives here.
* Minimal reshaping: ``init_time`` from the requested init, ``valid_time``
  derived, helper coords dropped, provenance attrs kept. ``normalize()`` does
  the rest (renames, units, crop, lat order).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import os
import re
import shutil
import tempfile
import threading
import warnings
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from .base import AdapterBase
from ..errors import WeatherSkillsNotInstalled, WeatherSkillError
from ..normalize import select_lon

INSTALL_HINT = "uv sync --group weather-skills"
PROVIDERS = ("weather-skills", "chc-skills")

# Their wrapper is called with sys.stdout/sys.stderr redirected and (for ECDS)
# os.environ edited — both process-wide. The MCP server runs fetches on
# threads, so every skill call, help probe and --probe-latest holds this lock.
_SKILL_LOCK = threading.Lock()

# (provider, skill, entrypoint) -> (wrapper, _SKILL_VERSION). Loading executes
# the script module (imports cfgrib, dynamical_catalog, ...), so do it once.
_ENTRYPOINTS: dict[tuple[str, str, str], tuple[object, str]] = {}


def locate_script(provider: str, skill: str) -> Path:
    """Path of ``skills/<skill>/scripts/*.py`` inside the installed ``provider``."""
    try:
        files = importlib.metadata.files(provider)
    except importlib.metadata.PackageNotFoundError as exc:
        raise WeatherSkillsNotInstalled(provider) from exc
    if files is None:
        raise WeatherSkillsNotInstalled(provider)
    matches = [
        f for f in files
        if len(f.parts) == 4 and f.parts[0] == "skills" and f.parts[1] == skill
        and f.parts[2] == "scripts" and f.suffix == ".py"
    ]
    if len(matches) != 1:
        raise WeatherSkillError(
            f"{provider!r} ships {len(matches)} scripts for skill {skill!r}; "
            "expected exactly one under skills/<skill>/scripts/"
        )
    return Path(str(matches[0].locate()))


def locate_skill(skill: str, provider: str | None = None):
    """(provider, script path) for a skill, searching the installed provider packages."""
    candidates = (provider,) if provider else PROVIDERS
    missing = []
    for prov in candidates:
        try:
            return prov, locate_script(prov, skill)
        except WeatherSkillsNotInstalled as exc:
            missing.append(str(exc))
        except WeatherSkillError:
            continue
    if missing and len(missing) == len(candidates):
        raise WeatherSkillsNotInstalled(candidates[0])
    raise WeatherSkillError(f"skill {skill!r} not found in {', '.join(candidates)}")


def skill_kind(provider: str, skill: str) -> str:
    """The SKILL.md ``catalog-group``: fetchers | transforms | figure | agent-tooling."""
    try:
        files = importlib.metadata.files(provider) or []
    except importlib.metadata.PackageNotFoundError as exc:
        raise WeatherSkillsNotInstalled(provider) from exc
    for f in files:
        if str(f) == f"skills/{skill}/SKILL.md":
            m = re.search(r"catalog-group:\s*(\S+)", Path(str(f.locate())).read_text())
            return m.group(1) if m else "unknown"
    raise WeatherSkillError(f"{provider!r} has no SKILL.md for {skill!r}")


def load_entrypoint(provider: str, skill: str, entrypoint: str | None = "fetch"):
    """(wrapper, version) for a skill; the wrapper is its ``@weather_skill`` function.

    ``entrypoint=None`` discovers it: the module-level function whose ``parser``
    attribute is an argparse parser (the check is explicit because Python
    3.13's ``pathlib.Path`` class also has a ``parser`` attribute).
    """
    key = (provider, skill, entrypoint)
    if key in _ENTRYPOINTS:
        return _ENTRYPOINTS[key]
    path = locate_script(provider, skill)
    name = "acmaddl_weather_skills_" + skill.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if entrypoint is None:
        found = [v for v in vars(module).values()
                 if inspect.isfunction(v)
                 and isinstance(getattr(v, "parser", None), argparse.ArgumentParser)]
        if len(found) != 1:
            raise WeatherSkillError(
                f"{skill}: expected one @weather_skill function in {path}, found {len(found)}")
        fn = found[0]
    else:
        fn = getattr(module, entrypoint, None)
        if fn is None or not hasattr(fn, "parser"):
            raise WeatherSkillError(
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
    non-zero; that text becomes the ``WeatherSkillError`` message. Exit 0 with
    nothing written (the skill returned ``None``) is also an error here.
    """
    out_path = Path(out_path)
    full = [*argv, "-o", str(out_path)]
    err = io.StringIO()
    with _SKILL_LOCK, _environ(env or {}), contextlib.redirect_stderr(err):
        try:
            fn(full)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
            if code != 0:
                text = err.getvalue().strip() or f"exit status {code}"
                raise WeatherSkillError(f"{skill or 'skill'} {' '.join(argv)}: {text}") from None
    if verbose:
        for line in err.getvalue().strip().splitlines():
            if line.strip():
                print(f"[acmaddl:weather-skills] {skill}: {line.strip()}")
    if not out_path.exists():
        raise WeatherSkillError(f"{skill or 'skill'} exited 0 but wrote no output at {out_path}")
    return out_path


def open_output(path):
    """Eagerly load a skill's Zarr so its temporary directory can go away."""
    with warnings.catch_warnings():
        # zarr 3 warns that consolidated metadata is not in the v3 spec; their
        # standard dataset contract writes it on purpose.
        warnings.simplefilter("ignore")
        with xr.open_zarr(path, consolidated=True) as ds:
            return ds.load()


_NETCDF_ATTR_TYPES = (str, bytes, int, float, list, tuple, np.ndarray, np.generic)


def _netcdf_safe_attrs(ds):
    """JSON-encode attrs NetCDF cannot store (dynamical.org stamps dict-valued
    ``statistics_approximate`` on coords). ``sanitize_for_netcdf`` rebuilds the
    dataset but keeps attrs as they are, so this has to happen here."""
    def fix(attrs):
        return {k: (v if isinstance(v, _NETCDF_ATTR_TYPES) else json.dumps(v, default=str))
                for k, v in attrs.items()}
    ds.attrs = fix(ds.attrs)
    for name in list(ds.variables):
        ds[name].attrs = fix(ds[name].attrs)
    return ds


def stamp(ds, *, skill, version, provider):
    """Record which skill produced this dataset next to their own provenance."""
    ds = _netcdf_safe_attrs(ds.copy())
    ds.attrs["weather_skills_name"] = skill
    ds.attrs["weather_skills_version"] = version
    ds.attrs["weather_skills_pin"] = provider_pin(provider)
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


def observation_windows(date_range, init_months, window_days, today, *, latest=None, allow_future=False):
    """[start, end] date windows for an observation request.

    acmadDL's observation API is year-based (``hindcast=(y0, y1)``) with an
    optional ``months=`` filter (``init_months``). Each year's selected months
    collapse into contiguous runs, clipped to the last published day, then
    split into chunks of at most ``window_days`` days. No ``hindcast`` at all
    means the trailing ``window_days`` (default 10) days ending there.

    The last published day is ``latest`` (the skill's own ``--probe-latest``)
    when known, else yesterday. Their fetchers refuse a window with no data in
    it, so asking past that day would fail inside the skill. ``allow_future``
    (CMIP6 scenarios) disables the clip altogether.
    """
    end_cap = today - timedelta(days=1)
    if latest is not None:
        end_cap = min(end_cap, latest)
    if date_range is None and init_months:
        date_range = (today.year, today.year)   # months= alone means those months of this year
    if date_range is None:
        n = int(window_days or 10)
        spans = [(end_cap - timedelta(days=n - 1), end_cap)]
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
        if not allow_future:
            spans = [(s, min(e, end_cap)) for s, e in spans if s <= end_cap]
        if not spans:
            what = "not published yet" if latest is not None else "entirely in the future"
            raise ValueError(
                f"observation window {date_range} / months={init_months} is {what} "
                f"(last available day {end_cap.isoformat()}, today {today.isoformat()})"
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


def earthdata_environment(home=None):
    """Refuse an Earthdata-backed skill before it can block on a login prompt.

    imerg-fetch calls ``earthaccess.login()`` with its default strategy, which
    falls through to an *interactive* prompt when it finds no credentials --
    inside acmaddl that is a hang, not an error. Policy: an Earthdata Login
    **token** only. ``EARTHDATA_TOKEN`` in the environment is used as is (their
    library reads it); otherwise ``~/.earthdatarc``, a file holding the bare
    token, is mapped onto that variable for the call. Username/password pairs
    (``EARTHDATA_USERNAME``/``EARTHDATA_PASSWORD``, ``~/.netrc``) are refused
    on purpose: a revocable, short-lived token is the safer thing to keep.
    """
    home = Path(home) if home is not None else Path.home()
    if os.environ.get("EARTHDATA_TOKEN"):
        return {}
    rc_path = home / ".earthdatarc"
    if rc_path.exists():
        lines = [ln.strip() for ln in rc_path.read_text().splitlines() if ln.strip()]
        if len(lines) == 1:
            token = re.sub(r"^token\s*:\s*", "", lines[0])
            if token and ":" not in token and " " not in token:
                return {"EARTHDATA_TOKEN": token}
        raise WeatherSkillError(
            f"{rc_path} must hold exactly one line: the Earthdata Login token itself "
            "(profile -> Generate Token at urs.earthdata.nasa.gov). Username/password "
            "entries are not accepted; acmaddl requires a token."
        )
    raise WeatherSkillError(
        "this product needs an Earthdata Login token: put the token in ~/.earthdatarc "
        "(one line, nothing else) or set EARTHDATA_TOKEN. Username/password and ~/.netrc "
        "are deliberately not accepted. (Without a token the skill would wait on an "
        "interactive login prompt.)"
    )


_ECDS_URL = "https://ecds.ecmwf.int/api"


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
    in the environment (it checks before the client would read its own
    ``~/.ecmwfdatastoresrc``). So both that file and an ECDS-pointing
    ``~/.cdsapirc`` / ``CDSAPI_URL`` / ``CDSAPI_KEY`` (what acmadDL users hold
    for ``c3s/ecmwf-s2s``) are mapped into the variables for the call. A Copernicus CDS token is a different credential, so that
    case is refused with instructions rather than sent and rejected upstream.
    """
    home = Path(home) if home is not None else Path.home()
    if os.environ.get("ECMWF_DATASTORES_URL") and os.environ.get("ECMWF_DATASTORES_KEY"):
        return {}
    # The client's own file (`url:` / `key:` lines). Their script checks the two
    # variables before the client would read it, so the file is mapped into them.
    rc_path = home / ".ecmwfdatastoresrc"
    if rc_path.exists():
        lines = [ln.strip() for ln in rc_path.read_text().splitlines() if ln.strip()]
        if len(lines) == 1 and ":" not in lines[0] and " " not in lines[0]:
            # acmadDL convenience, like ~/.earthdatarc: the bare key alone.
            return {"ECMWF_DATASTORES_URL": _ECDS_URL, "ECMWF_DATASTORES_KEY": lines[0]}
        rc = _read_rc(rc_path)
        if rc.get("key"):
            return {"ECMWF_DATASTORES_URL": rc.get("url") or _ECDS_URL, "ECMWF_DATASTORES_KEY": rc["key"]}
        raise WeatherSkillError(
            f"{rc_path} must hold either the bare ECDS key on one line, or 'url:' and 'key:' lines."
        )
    url, key = os.environ.get("CDSAPI_URL"), os.environ.get("CDSAPI_KEY")
    if not (url and key):
        rc = _read_rc(home / ".cdsapirc")
        url, key = url or rc.get("url"), key or rc.get("key")
    if url and key and "ecds.ecmwf.int" in url:
        return {"ECMWF_DATASTORES_URL": url, "ECMWF_DATASTORES_KEY": key}
    raise WeatherSkillError(
        "weather-skills/ecmwf-s2s needs ECMWF Data Store credentials: set ECMWF_DATASTORES_URL="
        "https://ecds.ecmwf.int/api and ECMWF_DATASTORES_KEY (or create ~/.ecmwfdatastoresrc), "
        "or point ~/.cdsapirc at https://ecds.ecmwf.int/api. A Copernicus CDS "
        "(cds.climate.copernicus.eu) key is a different token and will not work."
    )


def _credential_env(cfg):
    """Environment to run a skill (and its probe) with, per the entry's ``credentials`` knob."""
    creds = cfg.get("credentials")
    if not creds:
        return {}
    if creds == "ecds":
        return ecds_environment()
    if creds == "earthdata":
        return earthdata_environment()
    raise ValueError(
        f"{cfg.get('skill', '?')}: unknown credentials kind {creds!r} (expected 'ecds' or 'earthdata')")


def _today():
    """Patch point for tests."""
    return date.today()


class WeatherSkillsAdapter(AdapterBase):
    """Catalog entries: ``adapter: weather_skills``; see module docstring and catalog.yaml header."""

    def fetch_data(self, product_config, variable, date_range=None, region=None):
        cfg = product_config
        provider = cfg.get("provider", "weather-skills")
        skill = cfg["skill"]
        entrypoint = cfg.get("entrypoint", "fetch")
        template = [str(t) for t in cfg["argv"]]
        verbose = bool(cfg.get("_verbose"))
        native = cfg["variables"][variable]["native_name"]
        requires_region = bool(cfg.get("requires_region"))
        if cfg.get("_reforecast"):
            raise ValueError(
                f"{skill}: reforecasts are not available through weather-skills/* products (no reforecast "
                "stream). For ECMWF S2S hindcasts use c3s/ecmwf-s2s with reforecast=True."
            )
        if "forecast_type" in cfg:
            raise ValueError(
                f"{skill}: forecast_type is not supported on weather-skills/* products; member 0 is the "
                "control run and is always included."
            )
        fn, version = load_entrypoint(provider, skill, entrypoint)
        env = _credential_env(cfg)
        fields = {"variable": native, "bbox": bbox_nwse(region) if region else None}

        is_forecast = any("{init}" in t for t in template)
        is_obs = any("{start}" in t or "{end}" in t for t in template)
        if not (is_forecast or is_obs):
            raise ValueError(
                f"{skill}: argv template has neither {{init}} nor {{start}}/{{end}}; "
                "a weather-skills/* entry must be a forecast (init-keyed) or an observation (date-window) product"
            )

        with tempfile.TemporaryDirectory(prefix="acmaddl-weather-skills-") as tmp:
            tmp = Path(tmp)
            if is_forecast:
                init = cfg.get("_init_date")
                if not init:
                    raise ValueError(
                        f"{skill} is issuance-keyed: pass init='YYYY-MM-DD' (a full date; a month "
                        "alone selects nothing). Reforecasts are not available through weather-skills/* products."
                    )
                argv = render_argv(template, {**fields, "init": init}, requires_region=requires_region, skill=skill)
                if verbose:
                    print(f"[acmaddl:weather-skills] {skill} {' '.join(argv)}")
                out = run_skill(fn, argv, tmp / "out.zarr", env=env, verbose=verbose, skill=skill)
                ds = reshape_forecast(open_output(out), date.fromisoformat(init))
                shutil.rmtree(out, ignore_errors=True)
            else:
                latest = None
                if cfg.get("probe_latest") and not cfg.get("allow_future"):
                    text = self._probe_latest(fn, cfg, env=env)
                    try:
                        latest = date.fromisoformat(text)
                    except ValueError:
                        latest = None            # 'none': the skill has no realtime cap
                windows = observation_windows(date_range, cfg.get("init_months"), cfg.get("window_days"),
                                              _today(), latest=latest, allow_future=bool(cfg.get("allow_future")))
                # end_exclusive: the skill slices its time axis to `end 00:00`
                # (dynamical-fetch analyses), so ask for end+1 and trim back.
                end_exclusive = bool(cfg.get("end_exclusive"))
                parts = []
                for k, (start, end) in enumerate(windows):
                    asked_end = end + timedelta(days=1) if end_exclusive else end
                    argv = render_argv(template, {**fields, "start": start.isoformat(), "end": asked_end.isoformat()},
                                       requires_region=requires_region, skill=skill)
                    if verbose:
                        print(f"[acmaddl:weather-skills] {skill} {' '.join(argv)} ({k + 1}/{len(windows)})")
                    out = run_skill(fn, argv, tmp / f"out{k}.zarr", env=env, verbose=verbose, skill=skill)
                    part = open_output(out)
                    shutil.rmtree(out, ignore_errors=True)   # each global chunk leaves disk as soon as it is in memory
                    if end_exclusive and "time" in part.dims:
                        last = np.datetime64(datetime(end.year, end.month, end.day), "ns") + np.timedelta64(1, "D") - np.timedelta64(1, "ns")
                        part = part.sel(time=slice(None, last))
                    if region is not None:
                        part = crop_region(part, region)
                    parts.append(part)
                ds = parts[0] if len(parts) == 1 else xr.concat(parts, dim="time", combine_attrs="override")
        return stamp(ds, skill=skill, version=version, provider=provider)

    @staticmethod
    def _static_argv(template):
        """Flag/value pairs of a template that do not depend on the request (``--dataset X``)."""
        out, i = [], 0
        while i < len(template):
            tok, nxt = str(template[i]), (str(template[i + 1]) if i + 1 < len(template) else None)
            if tok.startswith("-") and nxt is not None and "{" not in nxt and tok not in ("-v", "--variable"):
                out += [tok, nxt]
                i += 2
            elif nxt is not None and "{" in nxt:
                i += 2
            else:
                i += 1
        return out

    def health_check(self, product_config, probe_remote=False):
        cfg = product_config
        provider, skill = cfg.get("provider", "weather-skills"), cfg.get("skill", "?")
        base = {"probe_remote": bool(probe_remote)}
        try:
            fn, version = load_entrypoint(provider, skill, cfg.get("entrypoint", "fetch"))
        except (WeatherSkillsNotInstalled, WeatherSkillError) as exc:
            return {**base, "healthy": False, "kind": "config", "message": str(exc)}
        err = io.StringIO()
        with _SKILL_LOCK, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            try:
                fn(["--help"])
            except SystemExit as exc:
                if exc.code not in (0, None):
                    return {**base, "healthy": False, "kind": "config",
                            "message": f"{skill} --help exited {exc.code}: {err.getvalue().strip()}"}
        ok = {**base, "healthy": True, "kind": "config",
              "message": f"{skill} v{version} ({provider}) loads and answers --help."}
        probe = cfg.get("probe_latest")
        if not probe_remote or not probe:
            return ok
        try:
            env = _credential_env(cfg)
        except WeatherSkillError as exc:          # no token / no ECDS key: a config problem, not a remote one
            return {**base, "healthy": False, "kind": "config", "message": str(exc)}
        try:
            latest = self._probe_latest(fn, cfg, env=env)
        except WeatherSkillError as exc:
            return {**base, "healthy": False, "kind": "remote", "message": str(exc)}
        return {**base, "healthy": True, "kind": "remote", "latest": latest,
                "message": f"{skill} v{version}: latest available {latest}."}

    def _probe_latest(self, fn, cfg, env=None):
        """Run the skill's own ``--probe-latest``; the last stdout line (a date, or 'none').

        ``env`` is the credential mapping the fetch itself runs with: some probes
        (imerg-fetch, smap-fetch) authenticate too.
        """
        probe = cfg.get("probe_latest")
        argv = self._static_argv(cfg.get("argv", [])) + (
            list(probe) if isinstance(probe, (list, tuple)) else ["--probe-latest"])
        out, err = io.StringIO(), io.StringIO()
        with _SKILL_LOCK, _environ(env or {}), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                fn(argv)
            except SystemExit as exc:
                if exc.code not in (0, None):
                    raise WeatherSkillError(
                        f"{cfg.get('skill', '?')} {' '.join(argv)}: {err.getvalue().strip() or exc.code}"
                    ) from None
        lines = [ln.strip() for ln in out.getvalue().splitlines() if ln.strip()]
        return lines[-1] if lines else "none"
