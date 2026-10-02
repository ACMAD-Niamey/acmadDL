"""Unit tests for the Rhiza weather-skills adapter. No network.

The provider packages (dependency group `rhiza`) are optional in a dev env;
tests that need the real scripts skip without them. Everything else runs
against fake wrappers and the fixtures in tests/fixtures/rhiza/.
"""
import importlib.metadata
from pathlib import Path

import pytest

from acmaddl.adapters import _ADAPTERS, get_adapter
from acmaddl.adapters import rhiza
from acmaddl.errors import RhizaNotInstalled, RhizaSkillError


def _group_installed() -> bool:
    try:
        importlib.metadata.distribution("weather-skills")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


needs_group = pytest.mark.skipif(not _group_installed(), reason="rhiza group not installed")


def test_adapter_registered():
    assert "rhiza" in _ADAPTERS
    assert type(get_adapter("rhiza")).__name__ == "RhizaAdapter"


def test_locate_script_missing_distribution_names_install_hint(monkeypatch):
    def boom(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, "files", boom)
    with pytest.raises(RhizaNotInstalled) as e:
        rhiza.locate_script("weather-skills", "chirps-fetch")
    assert rhiza.INSTALL_HINT in str(e.value)


def test_locate_script_requires_exactly_one_script(monkeypatch):
    from importlib.metadata import PackagePath
    fake = [PackagePath("skills/x/scripts/a.py"), PackagePath("skills/x/scripts/b.py")]
    monkeypatch.setattr(importlib.metadata, "files", lambda name: fake)
    with pytest.raises(RhizaSkillError, match="2 scripts"):
        rhiza.locate_script("weather-skills", "x")


@needs_group
def test_locate_and_load_real_entrypoint():
    path = rhiza.locate_script("weather-skills", "chirps-fetch")
    assert path.name == "fetch.py" and path.exists()
    fn, version = rhiza.load_entrypoint("weather-skills", "chirps-fetch")
    assert hasattr(fn, "parser")           # a @weather_skill wrapper
    assert version and version != "unknown"
    assert rhiza.load_entrypoint("weather-skills", "chirps-fetch")[0] is fn   # memoised


@needs_group
def test_provider_pin_is_a_commit():
    pin = rhiza.provider_pin("weather-skills")
    assert len(pin) == 40 and all(c in "0123456789abcdef" for c in pin)


# ── argv rendering ──────────────────────────────────────────────────────────

def test_bbox_nwse_reorders_acmaddl_bbox():
    # acmadDL: [lat_s, lat_n, lon_w, lon_e]; weather-skills: N/W/S/E
    assert rhiza.bbox_nwse([-5.0, 5.5, 33.5, 42.0]) == "5.5/33.5/-5/42"


def test_render_argv_fills_fields_in_order():
    tpl = ["--dataset", "noaa-gefs-forecast-35-day", "--date", "{init}", "--bbox", "{bbox}", "-v", "{variable}"]
    out = rhiza.render_argv(tpl, {"init": "2026-09-28", "bbox": "5/34/-5/42", "variable": "precipitation_surface"})
    assert out == ["--dataset", "noaa-gefs-forecast-35-day", "--date", "2026-09-28",
                   "--bbox", "5/34/-5/42", "-v", "precipitation_surface"]


def test_render_argv_drops_bbox_pair_when_no_region():
    tpl = ["--date", "{init}", "--bbox", "{bbox}", "-v", "{variable}"]
    out = rhiza.render_argv(tpl, {"init": "2026-09-28", "bbox": None, "variable": "tp"})
    assert out == ["--date", "2026-09-28", "-v", "tp"]


def test_render_argv_requires_region_when_declared():
    tpl = ["--date", "{init}", "--bbox", "{bbox}"]
    with pytest.raises(ValueError, match="region"):
        rhiza.render_argv(tpl, {"init": "2026-09-28", "bbox": None}, requires_region=True, skill="ecmwf-fetch")


def test_render_argv_missing_field_is_a_clear_error():
    with pytest.raises(ValueError, match="start"):
        rhiza.render_argv(["--start-time", "{start}"], {"init": "2026-09-28"}, skill="chirps-fetch")


# ── fixtures: raw skill outputs, one per output shape ───────────────────────

import json
import numpy as np
import xarray as xr

FIXTURES = Path(__file__).parent / "fixtures" / "rhiza"
FORECAST_FIXTURES = ["ensemble_forecast", "single_forecast", "s2s_forecast", "subc_envelope"]
OBS_FIXTURES = ["analysis", "daily_obs", "imerg_daily", "smap_daily"]


def fixture(name: str) -> xr.Dataset:
    ds = xr.open_dataset(FIXTURES / f"{name}.nc").load()
    ds.encoding = {}
    for v in ds.variables:
        ds[v].encoding = {}
    return ds


@pytest.mark.parametrize("name", FORECAST_FIXTURES)
def test_forecast_fixtures_have_their_raw_schema(name):
    ds = fixture(name)
    assert {"step", "latitude", "longitude"} <= set(ds.dims)
    assert np.issubdtype(ds["step"].dtype, np.timedelta64)
    assert "time" in ds.coords and "time" not in ds.dims       # scalar init (or SubC valid date)
    json.loads(ds.attrs["weather_skills_history"])
    assert ds.sizes["latitude"] <= 3 and ds.sizes["longitude"] <= 3


@pytest.mark.parametrize("name", OBS_FIXTURES)
def test_obs_fixtures_have_their_raw_schema(name):
    ds = fixture(name)
    assert {"time", "latitude", "longitude"} <= set(ds.dims)
    assert ds.sizes["time"] <= 4
    json.loads(ds.attrs["weather_skills_history"])


# ── running a skill, reading its output, stamping ───────────────────────────

import os
import sys


def fake_wrapper(fixture_name: str, *, exit_code=None, stderr_text="", write=True):
    """A stand-in for a @weather_skill wrapper: copies a fixture to the -o path."""
    def fn(argv):
        fn.calls.append(list(argv))
        if stderr_text:
            print(stderr_text, file=sys.stderr)
        if exit_code is not None:
            raise SystemExit(exit_code)
        if write:
            out = Path(argv[argv.index("-o") + 1])
            fixture(fixture_name).to_zarr(out, mode="w", consolidated=True)
    fn.calls = []
    fn.parser = object()
    return fn


def test_run_skill_writes_and_open_output_loads(tmp_path):
    fn = fake_wrapper("ensemble_forecast")
    out = rhiza.run_skill(fn, ["--date", "2026-09-28"], tmp_path / "o.zarr", skill="dynamical-fetch")
    assert fn.calls == [["--date", "2026-09-28", "-o", str(tmp_path / "o.zarr")]]
    ds = rhiza.open_output(out)
    assert "step" in ds.dims and ds["precipitation_surface"].values.any()


def test_run_skill_nonzero_exit_surfaces_their_message(tmp_path):
    fn = fake_wrapper("ensemble_forecast", exit_code=2, stderr_text="Error: init 2026-09-30 is under the 2-day embargo")
    with pytest.raises(RhizaSkillError, match="2-day embargo"):
        rhiza.run_skill(fn, [], tmp_path / "o.zarr", skill="ecmwf-fetch")


def test_run_skill_exit_zero_without_output_is_an_error(tmp_path):
    fn = fake_wrapper("ensemble_forecast", exit_code=0, write=False)
    with pytest.raises(RhizaSkillError, match="wrote no output"):
        rhiza.run_skill(fn, [], tmp_path / "o.zarr", skill="x")


def test_run_skill_other_exceptions_propagate(tmp_path):
    def fn(argv):
        raise RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        rhiza.run_skill(fn, [], tmp_path / "o.zarr")


def test_run_skill_env_is_scoped(tmp_path, monkeypatch):
    monkeypatch.delenv("ACMADDL_T", raising=False)
    seen = {}
    def fn(argv):
        seen["v"] = os.environ.get("ACMADDL_T")
        fixture("daily_obs").to_zarr(Path(argv[argv.index("-o") + 1]), mode="w", consolidated=True)
    rhiza.run_skill(fn, [], tmp_path / "o.zarr", env={"ACMADDL_T": "1"})
    assert seen["v"] == "1" and "ACMADDL_T" not in os.environ


def test_stamp_adds_ours_and_keeps_theirs():
    ds = rhiza.stamp(fixture("daily_obs"), skill="chirps-fetch", version="0.0.2", provider="weather-skills")
    assert ds.attrs["rhiza_skill"] == "chirps-fetch"
    assert ds.attrs["rhiza_skill_version"] == "0.0.2"
    assert len(ds.attrs["rhiza_pin"]) >= 7
    assert "weather_skills_history" in ds.attrs and "weather_skills_source" in ds.attrs


# ── forecast reshape + normalize round trip ─────────────────────────────────

from datetime import date
from acmaddl.normalize import normalize

FC_CONFIG = {
    "adapter": "rhiza",
    "variables": {"precip": {"native_name": "precipitation_surface", "units": "mm/day", "target_units": "mm/day"}},
    "grid": {"lat_res": 0.25, "lon_res": 0.25},
}


def test_reshape_forecast_builds_init_and_valid_time():
    ds = rhiza.reshape_forecast(fixture("ensemble_forecast"), date(2026, 9, 28))
    assert ds.sizes["init_time"] == 1
    assert "time" not in ds.coords and "step_bounds" not in ds.coords and "nv" not in ds.dims
    assert set(ds["valid_time"].dims) == {"init_time", "step"}
    assert ds["valid_time"].values[0, 0] == np.datetime64("2026-09-28T00:00:00", "ns")
    assert ds["valid_time"].values[0, 1] == np.datetime64("2026-09-28", "ns") + ds["step"].values[1]


def test_reshape_forecast_subc_valid_date_is_not_the_init():
    raw = fixture("subc_envelope")                 # scalar time = valid date, not init
    ds = rhiza.reshape_forecast(raw, date(2026, 9, 21))
    assert ds["init_time"].values[0] == np.datetime64("2026-09-21", "ns")
    assert ds["valid_time"].values[0, 0] == np.datetime64("2026-09-21", "ns") + ds["step"].values[0]


@pytest.mark.parametrize("name,var", [("ensemble_forecast", "precipitation_surface"), ("s2s_forecast", "tp")])
def test_forecast_round_trip_through_normalize(name, var):
    cfg = {**FC_CONFIG, "variables": {"precip": {"native_name": var, "units": "mm/day", "target_units": "mm/day"}}}
    raw = rhiza.reshape_forecast(fixture(name), date(2026, 9, 28))
    out = normalize(raw, cfg, "precip", region=[-1.0, 1.0, 36.0, 38.0])
    assert set(out["precip"].dims) == {"init_time", "lead_time", "member", "lat", "lon"}
    assert np.issubdtype(out["lead_time"].dtype, np.timedelta64)
    assert 0 in out["member"].values                        # control kept
    assert np.all(np.diff(out["lat"].values) > 0)           # ascending
    assert out["precip"].attrs["units"] == "mm/day"
    assert set(out["time"].dims) == {"init_time", "lead_time"}   # valid_time -> time
    assert out["time"].values[0, 0] == out["init_time"].values[0]


def test_single_forecast_has_no_member_dim():
    cfg = {**FC_CONFIG, "variables": {"temp": {"native_name": "temperature_2m", "units": "C", "target_units": "C"}}}
    out = normalize(rhiza.reshape_forecast(fixture("single_forecast"), date(2026, 9, 28)), cfg, "temp")
    assert "member" not in out.dims and {"init_time", "lead_time", "lat", "lon"} <= set(out["temp"].dims)


# ── observation windows + pre-concat crop ───────────────────────────────────

def test_windows_default_is_trailing_days_ending_yesterday():
    today = date(2026, 10, 1)
    assert rhiza.observation_windows(None, None, 10, today) == [(date(2026, 9, 21), date(2026, 9, 30))]
    assert rhiza.observation_windows(None, None, None, today) == [(date(2026, 9, 21), date(2026, 9, 30))]


def test_windows_year_and_months_become_contiguous_runs():
    w = rhiza.observation_windows((2020, 2020), [3, 4, 5, 10], None, date(2026, 10, 1))
    assert w == [(date(2020, 3, 1), date(2020, 5, 31)), (date(2020, 10, 1), date(2020, 10, 31))]


def test_windows_are_chunked_and_clipped_to_yesterday():
    w = rhiza.observation_windows((2026, 2026), [9], 10, date(2026, 9, 25))
    assert w == [(date(2026, 9, 1), date(2026, 9, 10)), (date(2026, 9, 11), date(2026, 9, 20)),
                 (date(2026, 9, 21), date(2026, 9, 24))]


def test_windows_entirely_in_the_future_raise():
    with pytest.raises(ValueError, match="future"):
        rhiza.observation_windows((2031, 2031), None, 10, date(2026, 10, 1))


def test_crop_region_handles_descending_latitude_and_native_names():
    raw = fixture("daily_obs")                       # latitude descending, global CHIRPS names
    out = rhiza.crop_region(raw, [-0.5, 0.5, 36.5, 37.5])
    assert out.sizes["latitude"] >= 1 and out.sizes["longitude"] >= 1
    assert float(out["latitude"].min()) >= -0.6 and float(out["latitude"].max()) <= 0.6


def test_crop_region_seam_crossing_box_uses_select_lon():
    lon = np.arange(0.0, 360.0, 60.0)               # 0..300, a 0-360 source
    ds = xr.Dataset({"x": (("latitude", "longitude"), np.ones((2, 6)))},
                    coords={"latitude": [1.0, 0.0], "longitude": lon})
    out = rhiza.crop_region(ds, [0.0, 1.0, -70.0, 70.0])
    assert sorted(out["longitude"].values.tolist()) == [0.0, 60.0, 300.0]
