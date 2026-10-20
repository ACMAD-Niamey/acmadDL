"""Capture small real outputs of the Rhiza skills as test fixtures.

    uv run python scripts/capture_weather_skills_fixtures.py            # credential-free skills
    uv run python scripts/capture_weather_skills_fixtures.py --real     # also ECDS / Earthdata skills, whichever credentials exist

Each fixture is the skill's raw Zarr output, subset to a few members / steps /
cells and saved as NetCDF under tests/fixtures/weather_skills/. Re-run after moving a
provider pin in pyproject.toml. The three credentialed shapes (ecmwf-fetch,
imerg-fetch, smap-fetch) are captured for real only with --real and the matching
credentials (an ECDS token; an Earthdata token in ~/.earthdatarc or
EARTHDATA_TOKEN); otherwise they are synthesised from their documented schemas so
the unit suite is complete offline.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from acmaddl.adapters import weather_skills

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "weather_skills"
BBOX = "1/36/-1/38"            # N/W/S/E, 2x2 degrees over Kenya
BBOX_S2S = "3/36/-3/40"        # ECMWF S2S is 1.5 deg: a 2x2 box holds one latitude row, which the skill squeezes away
YESTERDAY = date.today() - timedelta(days=1)


def _run(provider, skill, argv, env=None):
    fn, version = weather_skills.load_entrypoint(provider, skill)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.zarr"
        err = io.StringIO()
        with weather_skills._environ(env or {}), contextlib.redirect_stderr(err):
            try:
                fn([*argv, "-o", str(out)])
            except SystemExit as exc:
                if exc.code not in (0, None):
                    raise SystemExit(f"{skill} failed:\n{err.getvalue()}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = xr.open_zarr(out, consolidated=True).load()
    return ds


def _subset(ds):
    """<=3 members, <=4 steps/times, and the 3x3 cells nearest 0N 37E (so a Kenya bbox hits them)."""
    sel = {}
    for dim, n in (("number", 3), ("step", 4), ("time", 4)):
        if dim in ds.dims:
            sel[dim] = slice(0, n)
    for dim, centre in (("latitude", 0.0), ("longitude", 37.0)):
        if dim in ds.dims:
            i = int(np.argmin(np.abs(ds[dim].values - centre)))
            lo = max(0, min(i - 1, ds.sizes[dim] - 3))
            sel[dim] = slice(lo, lo + 3)
    ds = ds.isel(**sel)
    for v in ds.data_vars:          # keep files tiny and encoding-free
        ds[v] = ds[v].astype("float32")
        ds[v].encoding = {}
    for c in ds.coords:
        ds[c].encoding = {}
    return _netcdf_safe_attrs(ds)


def _netcdf_safe_attrs(ds):
    """NetCDF attrs must be scalars/strings/arrays; dynamical.org stamps dicts
    (``statistics_approximate``). JSON-encode anything else so the fixture
    keeps the information without failing to_netcdf."""
    def fix(attrs):
        return {k: (v if isinstance(v, (str, int, float, bytes, list, tuple, np.ndarray, np.generic))
                    else json.dumps(v, default=str)) for k, v in attrs.items()}
    ds.attrs = fix(ds.attrs)
    for name in list(ds.variables):
        ds[name].attrs = fix(ds[name].attrs)
    return ds


def _probe(provider, skill, argv):
    fn, _ = weather_skills.load_entrypoint(provider, skill)
    if "--probe-latest" not in argv:
        argv = [*argv, "--probe-latest"]
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        try:
            fn(list(argv))
        except SystemExit:
            pass
    return out.getvalue().strip().splitlines()[-1].strip()


def _history(skill, version, args):
    return json.dumps([{"skill": skill, "version": version, "args": args, "input": None}])


def synthetic_forecast(source, skill, var, units, nsteps=4, step_hours=24, members=3, init="2026-09-25"):
    """ecmwf-fetch shape: (number, step, latitude, longitude) + scalar time."""
    step = pd.to_timedelta(np.arange(nsteps) * step_hours, unit="h")
    lat = np.array([1.5, 0.0, -1.5])
    lon = np.array([36.0, 37.5, 39.0])
    data = np.random.default_rng(0).random((members, nsteps, 3, 3)).astype("float32")
    return xr.Dataset(
        {var: (("number", "step", "latitude", "longitude"), data, {"units": units})},
        coords={"number": np.arange(members, dtype="int16"), "step": step,
                "latitude": lat, "longitude": lon, "time": np.datetime64(init, "ns")},
        attrs={"Conventions": "CF-1.13", "weather_skills_source": source,
               "weather_skills_history": _history(skill, "synthetic", {"date": init, "bbox": BBOX}),
               "acmaddl_fixture": "synthetic — re-capture with scripts/capture_weather_skills_fixtures.py --real"},
    )


def synthetic_obs(source, skill, var, units, start="2026-09-20", days=4):
    time = pd.date_range(start, periods=days, freq="D")
    lat = np.array([1.5, 0.0, -1.5])
    lon = np.array([36.0, 37.5, 39.0])
    data = np.random.default_rng(1).random((days, 3, 3)).astype("float32")
    return xr.Dataset(
        {var: (("time", "latitude", "longitude"), data, {"units": units})},
        coords={"time": time, "latitude": lat, "longitude": lon},
        attrs={"Conventions": "CF-1.13", "weather_skills_source": source,
               "weather_skills_history": _history(skill, "synthetic", {"start": start}),
               "acmaddl_fixture": "synthetic — re-capture with scripts/capture_weather_skills_fixtures.py --real"},
    )


def main(real: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    ws, chc = "weather-skills", "chc-skills"
    jobs = {}

    ifs = "ecmwf-ifs-ens-forecast-15-day-0-25-degree"
    latest = _probe(ws, "dynamical-fetch", ["--dataset", ifs])
    jobs["ensemble_forecast"] = (ws, "dynamical-fetch",
        ["--dataset", ifs, "--date", latest, "--bbox", BBOX, "-v", "precipitation_surface"])
    gfs = "noaa-gfs-forecast"
    jobs["single_forecast"] = (ws, "dynamical-fetch",
        ["--dataset", gfs, "--date", _probe(ws, "dynamical-fetch", ["--dataset", gfs]), "--bbox", BBOX, "-v", "temperature_2m"])
    ana = "noaa-gefs-analysis"
    end = _probe(ws, "dynamical-fetch", ["--dataset", ana])
    start = (date.fromisoformat(end) - timedelta(days=1)).isoformat()
    jobs["analysis"] = (ws, "dynamical-fetch",
        ["--dataset", ana, "--start-time", start, "--end-time", end, "--bbox", BBOX, "-v", "precipitation_surface"])
    cend = _probe(ws, "chirps-fetch", [])
    cstart = (date.fromisoformat(cend) - timedelta(days=1)).isoformat()
    jobs["daily_obs"] = (ws, "chirps-fetch", ["--start-time", cstart, "--end-time", cend])
    sinit = _probe(chc, "subc-mme-fetch", ["--outlook", "7d", "--probe-latest", "pr"])
    jobs["subc_envelope"] = (chc, "subc-mme-fetch",
        ["--date", sinit, "--outlook", "7d", "--bbox", BBOX, "-v", "pr"])

    for name, (provider, skill, argv) in jobs.items():
        print(f"capturing {name} <- {skill} {' '.join(argv)}", flush=True)
        ds = _subset(_run(provider, skill, argv))
        ds.to_netcdf(OUT / f"{name}.nc")

    # Credentialed shapes: captured for real when the credentials are present
    # (checked the same way the adapter checks them), synthesised otherwise, so
    # --real on a machine with only one kind of credential still does what it can.
    from acmaddl.errors import WeatherSkillError
    def have(check):
        try:
            return dict(check())
        except WeatherSkillError as exc:
            print(f"  skipping real capture: {str(exc)[:90]}...")
            return None
    week_ago = (YESTERDAY - timedelta(days=7)).isoformat()
    six_ago = (YESTERDAY - timedelta(days=6)).isoformat()
    ecds = have(weather_skills.ecds_environment) if real else None
    earthdata = have(weather_skills.earthdata_environment) if real else None

    if ecds is not None:
        s2s_init = (date.today() - timedelta(days=4)).isoformat()
        print("capturing s2s_forecast <- ecmwf-fetch", flush=True)
        _subset(_run(ws, "ecmwf-fetch", ["--date", s2s_init, "--bbox", BBOX_S2S, "-v", "tp"], env=ecds)).to_netcdf(OUT / "s2s_forecast.nc")
    else:
        print("synthesising s2s_forecast (no ECDS credentials)")
        synthetic_forecast("ecmwf-s2s", "ecmwf-fetch", "tp", "mm day-1").to_netcdf(OUT / "s2s_forecast.nc")

    if earthdata is not None:
        for name, skill, argv in (
            ("imerg_daily", "imerg-fetch", ["--start-time", week_ago, "--end-time", six_ago]),
            ("smap_daily", "smap-fetch", ["--start-time", week_ago, "--end-time", six_ago, "--bbox", BBOX]),
        ):
            print(f"capturing {name} <- {skill} {' '.join(argv)}", flush=True)
            _subset(_run(ws, skill, argv, env=earthdata)).to_netcdf(OUT / f"{name}.nc")
    else:
        print("synthesising imerg_daily, smap_daily (no Earthdata token)")
        synthetic_obs("imerg", "imerg-fetch", "precip", "mm/day").to_netcdf(OUT / "imerg_daily.nc")
        synthetic_obs("smap", "smap-fetch", "soil_moisture", "m3 m-3").to_netcdf(OUT / "smap_daily.nc")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", action="store_true", help="also capture the credentialed skills")
    main(p.parse_args().real)
