"""Unit tests for the Rhiza weather-skills adapter. No network.

The provider packages (dependency group `weather-skills`) are optional in a dev env;
tests that need the real scripts skip without them. Everything else runs
against fake wrappers and the fixtures in tests/fixtures/weather_skills/.
"""
import importlib.metadata
from pathlib import Path

import pytest

from acmaddl.adapters import _ADAPTERS, get_adapter
from acmaddl.adapters import weather_skills
from acmaddl.errors import WeatherSkillsNotInstalled, WeatherSkillError


def _group_installed() -> bool:
    try:
        importlib.metadata.distribution("weather-skills")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


needs_group = pytest.mark.skipif(not _group_installed(), reason="weather-skills group not installed")


def test_adapter_registered():
    assert "weather_skills" in _ADAPTERS
    assert type(get_adapter("weather_skills")).__name__ == "WeatherSkillsAdapter"


def test_locate_script_missing_distribution_names_install_hint(monkeypatch):
    def boom(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, "files", boom)
    with pytest.raises(WeatherSkillsNotInstalled) as e:
        weather_skills.locate_script("weather-skills", "chirps-fetch")
    assert weather_skills.INSTALL_HINT in str(e.value)


def test_locate_script_requires_exactly_one_script(monkeypatch):
    from importlib.metadata import PackagePath
    fake = [PackagePath("skills/x/scripts/a.py"), PackagePath("skills/x/scripts/b.py")]
    monkeypatch.setattr(importlib.metadata, "files", lambda name: fake)
    with pytest.raises(WeatherSkillError, match="2 scripts"):
        weather_skills.locate_script("weather-skills", "x")


@needs_group
def test_locate_and_load_real_entrypoint():
    path = weather_skills.locate_script("weather-skills", "chirps-fetch")
    assert path.name == "fetch.py" and path.exists()
    fn, version = weather_skills.load_entrypoint("weather-skills", "chirps-fetch")
    assert hasattr(fn, "parser")           # a @weather_skill wrapper
    assert version and version != "unknown"
    assert weather_skills.load_entrypoint("weather-skills", "chirps-fetch")[0] is fn   # memoised


@needs_group
def test_provider_pin_is_a_commit():
    pin = weather_skills.provider_pin("weather-skills")
    assert len(pin) == 40 and all(c in "0123456789abcdef" for c in pin)


# ── argv rendering ──────────────────────────────────────────────────────────

def test_bbox_nwse_reorders_acmaddl_bbox():
    # acmadDL: [lat_s, lat_n, lon_w, lon_e]; weather-skills: N/W/S/E
    assert weather_skills.bbox_nwse([-5.0, 5.5, 33.5, 42.0]) == "5.5/33.5/-5/42"


def test_render_argv_fills_fields_in_order():
    tpl = ["--dataset", "noaa-gefs-forecast-35-day", "--date", "{init}", "--bbox", "{bbox}", "-v", "{variable}"]
    out = weather_skills.render_argv(tpl, {"init": "2026-09-28", "bbox": "5/34/-5/42", "variable": "precipitation_surface"})
    assert out == ["--dataset", "noaa-gefs-forecast-35-day", "--date", "2026-09-28",
                   "--bbox", "5/34/-5/42", "-v", "precipitation_surface"]


def test_render_argv_drops_bbox_pair_when_no_region():
    tpl = ["--date", "{init}", "--bbox", "{bbox}", "-v", "{variable}"]
    out = weather_skills.render_argv(tpl, {"init": "2026-09-28", "bbox": None, "variable": "tp"})
    assert out == ["--date", "2026-09-28", "-v", "tp"]


def test_render_argv_requires_region_when_declared():
    tpl = ["--date", "{init}", "--bbox", "{bbox}"]
    with pytest.raises(ValueError, match="region"):
        weather_skills.render_argv(tpl, {"init": "2026-09-28", "bbox": None}, requires_region=True, skill="ecmwf-fetch")


def test_render_argv_missing_field_is_a_clear_error():
    with pytest.raises(ValueError, match="start"):
        weather_skills.render_argv(["--start-time", "{start}"], {"init": "2026-09-28"}, skill="chirps-fetch")


# ── fixtures: raw skill outputs, one per output shape ───────────────────────

import json
import numpy as np
import xarray as xr

FIXTURES = Path(__file__).parent / "fixtures" / "weather_skills"
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
    out = weather_skills.run_skill(fn, ["--date", "2026-09-28"], tmp_path / "o.zarr", skill="dynamical-fetch")
    assert fn.calls == [["--date", "2026-09-28", "-o", str(tmp_path / "o.zarr")]]
    ds = weather_skills.open_output(out)
    assert "step" in ds.dims and ds["precipitation_surface"].values.any()


def test_run_skill_nonzero_exit_surfaces_their_message(tmp_path):
    fn = fake_wrapper("ensemble_forecast", exit_code=2, stderr_text="Error: init 2026-09-30 is under the 2-day embargo")
    with pytest.raises(WeatherSkillError, match="2-day embargo"):
        weather_skills.run_skill(fn, [], tmp_path / "o.zarr", skill="ecmwf-fetch")


def test_run_skill_exit_zero_without_output_is_an_error(tmp_path):
    fn = fake_wrapper("ensemble_forecast", exit_code=0, write=False)
    with pytest.raises(WeatherSkillError, match="wrote no output"):
        weather_skills.run_skill(fn, [], tmp_path / "o.zarr", skill="x")


def test_run_skill_other_exceptions_propagate(tmp_path):
    def fn(argv):
        raise RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        weather_skills.run_skill(fn, [], tmp_path / "o.zarr")


def test_run_skill_env_is_scoped(tmp_path, monkeypatch):
    monkeypatch.delenv("ACMADDL_T", raising=False)
    seen = {}
    def fn(argv):
        seen["v"] = os.environ.get("ACMADDL_T")
        fixture("daily_obs").to_zarr(Path(argv[argv.index("-o") + 1]), mode="w", consolidated=True)
    weather_skills.run_skill(fn, [], tmp_path / "o.zarr", env={"ACMADDL_T": "1"})
    assert seen["v"] == "1" and "ACMADDL_T" not in os.environ


def test_stamp_adds_ours_and_keeps_theirs():
    ds = weather_skills.stamp(fixture("daily_obs"), skill="chirps-fetch", version="0.0.2", provider="weather-skills")
    assert ds.attrs["weather_skills_name"] == "chirps-fetch"
    assert ds.attrs["weather_skills_version"] == "0.0.2"
    assert len(ds.attrs["weather_skills_pin"]) >= 7
    assert "weather_skills_history" in ds.attrs and "weather_skills_source" in ds.attrs


# ── forecast reshape + normalize round trip ─────────────────────────────────

from datetime import date
from acmaddl.normalize import normalize

FC_CONFIG = {
    "adapter": "weather_skills",
    "variables": {"precip": {"native_name": "precipitation_surface", "units": "mm/day", "target_units": "mm/day"}},
    "grid": {"lat_res": 0.25, "lon_res": 0.25},
}


def test_reshape_forecast_builds_init_and_valid_time():
    ds = weather_skills.reshape_forecast(fixture("ensemble_forecast"), date(2026, 9, 28))
    assert ds.sizes["init_time"] == 1
    assert "time" not in ds.coords and "step_bounds" not in ds.coords and "nv" not in ds.dims
    assert set(ds["valid_time"].dims) == {"init_time", "step"}
    assert ds["valid_time"].values[0, 0] == np.datetime64("2026-09-28T00:00:00", "ns")
    assert ds["valid_time"].values[0, 1] == np.datetime64("2026-09-28", "ns") + ds["step"].values[1]


def test_reshape_forecast_subc_valid_date_is_not_the_init():
    raw = fixture("subc_envelope")                 # scalar time = valid date, not init
    ds = weather_skills.reshape_forecast(raw, date(2026, 9, 21))
    assert ds["init_time"].values[0] == np.datetime64("2026-09-21", "ns")
    assert ds["valid_time"].values[0, 0] == np.datetime64("2026-09-21", "ns") + ds["step"].values[0]


@pytest.mark.parametrize("name,var", [("ensemble_forecast", "precipitation_surface"), ("s2s_forecast", "tp")])
def test_forecast_round_trip_through_normalize(name, var):
    cfg = {**FC_CONFIG, "variables": {"precip": {"native_name": var, "units": "mm/day", "target_units": "mm/day"}}}
    raw = weather_skills.reshape_forecast(fixture(name), date(2026, 9, 28))
    out = normalize(raw, cfg, "precip", region=[-1.0, 1.0, 36.0, 38.0])
    assert set(out["precip"].dims) == {"init_time", "lead_time", "member", "lat", "lon"}
    assert np.issubdtype(out["lead_time"].dtype, np.timedelta64)
    assert 0 in out["member"].values                        # control kept
    assert np.all(np.diff(out["lat"].values) > 0)           # ascending
    assert out["precip"].attrs["units"] == "mm/day"
    assert set(out["time"].dims) == {"init_time", "lead_time"}   # valid_time -> time
    # First valid time = init + first lead. ECMWF S2S starts at 24 h (no zero-hour
    # accumulation); the dynamical products start at lead 0.
    assert out["time"].values[0, 0] == out["init_time"].values[0] + out["lead_time"].values[0]


def test_single_forecast_has_no_member_dim():
    cfg = {**FC_CONFIG, "variables": {"temp": {"native_name": "temperature_2m", "units": "C", "target_units": "C"}}}
    out = normalize(weather_skills.reshape_forecast(fixture("single_forecast"), date(2026, 9, 28)), cfg, "temp")
    assert "member" not in out.dims and {"init_time", "lead_time", "lat", "lon"} <= set(out["temp"].dims)


# ── observation windows + pre-concat crop ───────────────────────────────────

def test_windows_default_is_trailing_days_ending_yesterday():
    today = date(2026, 10, 1)
    assert weather_skills.observation_windows(None, None, 10, today) == [(date(2026, 9, 21), date(2026, 9, 30))]
    assert weather_skills.observation_windows(None, None, None, today) == [(date(2026, 9, 21), date(2026, 9, 30))]


def test_windows_year_and_months_become_contiguous_runs():
    w = weather_skills.observation_windows((2020, 2020), [3, 4, 5, 10], None, date(2026, 10, 1))
    assert w == [(date(2020, 3, 1), date(2020, 5, 31)), (date(2020, 10, 1), date(2020, 10, 31))]


def test_windows_are_chunked_and_clipped_to_yesterday():
    w = weather_skills.observation_windows((2026, 2026), [9], 10, date(2026, 9, 25))
    assert w == [(date(2026, 9, 1), date(2026, 9, 10)), (date(2026, 9, 11), date(2026, 9, 20)),
                 (date(2026, 9, 21), date(2026, 9, 24))]


def test_windows_entirely_in_the_future_raise():
    with pytest.raises(ValueError, match="future"):
        weather_skills.observation_windows((2031, 2031), None, 10, date(2026, 10, 1))


def test_crop_region_handles_descending_latitude_and_native_names():
    raw = fixture("daily_obs")                       # latitude descending, global CHIRPS names
    out = weather_skills.crop_region(raw, [-0.5, 0.5, 36.5, 37.5])
    assert out.sizes["latitude"] >= 1 and out.sizes["longitude"] >= 1
    assert float(out["latitude"].min()) >= -0.6 and float(out["latitude"].max()) <= 0.6


def test_crop_region_seam_crossing_box_uses_select_lon():
    lon = np.arange(0.0, 360.0, 60.0)               # 0..300, a 0-360 source
    ds = xr.Dataset({"x": (("latitude", "longitude"), np.ones((2, 6)))},
                    coords={"latitude": [1.0, 0.0], "longitude": lon})
    out = weather_skills.crop_region(ds, [0.0, 1.0, -70.0, 70.0])
    assert sorted(out["longitude"].values.tolist()) == [0.0, 60.0, 300.0]


# ── ECDS credential mapping ─────────────────────────────────────────────────

def _clear_cred_env(monkeypatch):
    for k in ("ECMWF_DATASTORES_URL", "ECMWF_DATASTORES_KEY", "CDSAPI_URL", "CDSAPI_KEY"):
        monkeypatch.delenv(k, raising=False)


def test_ecds_env_noop_when_their_variables_exist(monkeypatch, tmp_path):
    _clear_cred_env(monkeypatch)
    monkeypatch.setenv("ECMWF_DATASTORES_URL", "https://ecds.ecmwf.int/api")
    monkeypatch.setenv("ECMWF_DATASTORES_KEY", "k")
    assert weather_skills.ecds_environment(home=tmp_path) == {}


def test_ecds_env_maps_their_rc_file_into_the_variables(monkeypatch, tmp_path):
    """ecmwf-fetch's require_env insists on the variables before the client would
    read ~/.ecmwfdatastoresrc, so the standard file has to be mapped, not trusted."""
    _clear_cred_env(monkeypatch)
    (tmp_path / ".ecmwfdatastoresrc").write_text("url: https://ecds.ecmwf.int/api\nkey: k-123\n")
    assert weather_skills.ecds_environment(home=tmp_path) == {
        "ECMWF_DATASTORES_URL": "https://ecds.ecmwf.int/api", "ECMWF_DATASTORES_KEY": "k-123"}


def test_ecds_env_rc_file_without_key_is_a_clear_error(monkeypatch, tmp_path):
    _clear_cred_env(monkeypatch)
    (tmp_path / ".ecmwfdatastoresrc").write_text("url: https://ecds.ecmwf.int/api\n")
    with pytest.raises(WeatherSkillError, match="ecmwfdatastoresrc"):
        weather_skills.ecds_environment(home=tmp_path)


def test_ecds_env_maps_an_ecds_cdsapirc(monkeypatch, tmp_path):
    _clear_cred_env(monkeypatch)
    (tmp_path / ".cdsapirc").write_text("url: https://ecds.ecmwf.int/api\nkey: abc-123\n")
    assert weather_skills.ecds_environment(home=tmp_path) == {
        "ECMWF_DATASTORES_URL": "https://ecds.ecmwf.int/api", "ECMWF_DATASTORES_KEY": "abc-123"}


def test_ecds_env_refuses_a_copernicus_cdsapirc(monkeypatch, tmp_path):
    _clear_cred_env(monkeypatch)
    (tmp_path / ".cdsapirc").write_text("url: https://cds.climate.copernicus.eu/api\nkey: abc\n")
    with pytest.raises(WeatherSkillError, match="ecds.ecmwf.int"):
        weather_skills.ecds_environment(home=tmp_path)


def test_ecds_env_prefers_cdsapi_env_vars(monkeypatch, tmp_path):
    _clear_cred_env(monkeypatch)
    monkeypatch.setenv("CDSAPI_URL", "https://ecds.ecmwf.int/api")
    monkeypatch.setenv("CDSAPI_KEY", "zzz")
    assert weather_skills.ecds_environment(home=tmp_path)["ECMWF_DATASTORES_KEY"] == "zzz"


# ── fetch_data end to end through acmaddl.fetch ─────────────────────────────

import acmaddl

WEATHER_SKILLS_FC = {
    "adapter": "weather_skills", "provider": "weather-skills", "skill": "dynamical-fetch",
    "argv": ["--dataset", "ecmwf-ifs-ens-forecast-15-day-0-25-degree", "--date", "{init}", "--bbox", "{bbox}", "-v", "{variable}"],
    "variables": {"precip": {"native_name": "precipitation_surface", "units": "mm/day", "target_units": "mm/day"}},
    "grid": {"lat_res": 0.25, "lon_res": 0.25, "forecast_members": 51},
}
WEATHER_SKILLS_OBS = {
    "adapter": "weather_skills", "provider": "weather-skills", "skill": "chirps-fetch",
    "argv": ["--start-time", "{start}", "--end-time", "{end}"], "window_days": 2,
    "variables": {"precip": {"native_name": "precip", "units": "mm/day", "target_units": "mm/day"}},
    "grid": {"lat_res": 0.05, "lon_res": 0.05, "temporal": "daily"},
}


@pytest.fixture
def fake_catalog(monkeypatch):
    """Register two throwaway weather-skills/* products and route skill loading to fakes."""
    from acmaddl import catalog
    entries = {"weather-skills/_test-fc": WEATHER_SKILLS_FC, "weather-skills/_test-obs": WEATHER_SKILLS_OBS}
    real_info = catalog.info
    monkeypatch.setattr(catalog, "info", lambda p: dict(entries[p]) | {"deprecated": False} if p in entries else real_info(p))
    monkeypatch.setattr(catalog, "get", catalog.info)
    wrappers = {"dynamical-fetch": fake_wrapper("ensemble_forecast"), "chirps-fetch": fake_wrapper("daily_obs")}
    monkeypatch.setattr(weather_skills, "load_entrypoint", lambda provider, skill, entrypoint="fetch": (wrappers[skill], "9.9.9"))
    return wrappers


def test_fetch_forecast_end_to_end(fake_catalog):
    ds = acmaddl.fetch("weather-skills/_test-fc", "precip", init="2026-09-28", region=[-1, 1, 36, 38], cache=False)
    argv = fake_catalog["dynamical-fetch"].calls[0]
    assert argv[:8] == ["--dataset", "ecmwf-ifs-ens-forecast-15-day-0-25-degree", "--date", "2026-09-28",
                        "--bbox", "1/36/-1/38", "-v", "precipitation_surface"]
    assert set(ds["precip"].dims) == {"init_time", "lead_time", "member", "lat", "lon"}
    assert ds["init_time"].values[0] == np.datetime64("2026-09-28", "ns")
    assert ds.attrs["weather_skills_name"] == "dynamical-fetch" and ds.attrs["weather_skills_version"] == "9.9.9"
    assert "weather_skills_history" in ds.attrs


def test_fetch_forecast_month_only_init_is_rejected(fake_catalog):
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        acmaddl.fetch("weather-skills/_test-fc", "precip", init="2026-09", region=[-1, 1, 36, 38], cache=False)


def test_fetch_forecast_without_init_is_rejected(fake_catalog):
    with pytest.raises(ValueError, match="init="):
        acmaddl.fetch("weather-skills/_test-fc", "precip", region=[-1, 1, 36, 38], cache=False)


def test_fetch_obs_chunks_and_concatenates(fake_catalog, monkeypatch):
    monkeypatch.setattr(weather_skills, "_today", lambda: date(2026, 10, 1))
    ds = acmaddl.fetch("weather-skills/_test-obs", "precip", hindcast=(2026, 2026), months=[9], region=[-1, 1, 36, 38], cache=False)
    calls = fake_catalog["chirps-fetch"].calls
    assert calls[0][:4] == ["--start-time", "2026-09-01", "--end-time", "2026-09-02"]
    assert calls[-1][:4] == ["--start-time", "2026-09-29", "--end-time", "2026-09-30"]
    assert len(calls) == 15
    assert set(ds["precip"].dims) == {"time", "lat", "lon"}
    assert ds.sizes["time"] == 15 * fixture("daily_obs").sizes["time"]   # each chunk contributes the fixture's days


def test_fetch_data_template_without_dates_is_rejected():
    cfg = {**WEATHER_SKILLS_OBS, "argv": ["--bbox", "{bbox}"]}
    with pytest.raises(ValueError, match="neither"):
        weather_skills.WeatherSkillsAdapter().fetch_data(cfg, "precip", date_range=(2026, 2026), region=[-1, 1, 36, 38])


def test_fetch_output_round_trips_to_netcdf(fake_catalog, tmp_path):
    """dynamical.org stamps dict-valued coord attrs; the adapter must leave the
    result writable (sanitize_for_netcdf does not JSON-encode attrs)."""
    wrapper = fake_catalog["dynamical-fetch"]
    def fn(argv):
        wrapper.calls.append(list(argv))
        ds = fixture("ensemble_forecast")
        ds["latitude"].attrs["statistics_approximate"] = {"min": -90.0, "max": 90.0}
        ds.attrs["nested"] = {"a": [1, 2]}
        ds.to_zarr(Path(argv[argv.index("-o") + 1]), mode="w", consolidated=True)
    fn.parser = object()
    fake_catalog["dynamical-fetch"] = fn
    out = acmaddl.fetch("weather-skills/_test-fc", "precip", init="2026-09-28", region=[-1, 1, 36, 38], cache=False,
                        destination=str(tmp_path / "fc.nc"), format="netcdf")
    assert (tmp_path / "fc.nc").exists()
    assert isinstance(out["lat"].attrs["statistics_approximate"], str)


# ── health check ────────────────────────────────────────────────────────────

def test_health_config_reports_missing_group(monkeypatch):
    def missing(*a, **k):
        raise WeatherSkillsNotInstalled("weather-skills")
    monkeypatch.setattr(weather_skills, "load_entrypoint", missing)
    r = weather_skills.WeatherSkillsAdapter().health_check(WEATHER_SKILLS_FC)
    assert r["healthy"] is False and r["kind"] == "config" and "uv sync --group weather-skills" in r["message"]


def test_health_config_ok_runs_help(monkeypatch):
    fn = fake_wrapper("ensemble_forecast")
    def help_fn(argv):
        fn.calls.append(list(argv))
        raise SystemExit(0)
    help_fn.parser = object()
    monkeypatch.setattr(weather_skills, "load_entrypoint", lambda *a, **k: (help_fn, "0.0.2"))
    r = weather_skills.WeatherSkillsAdapter().health_check(WEATHER_SKILLS_FC)
    assert r["healthy"] is True and r["kind"] == "config" and "0.0.2" in r["message"]
    assert fn.calls == [["--help"]]


def test_health_remote_probe_parses_latest(monkeypatch):
    def probe_fn(argv):
        if argv == ["--help"]:
            raise SystemExit(0)
        assert argv == ["--dataset", "ecmwf-ifs-ens-forecast-15-day-0-25-degree", "--probe-latest"]
        print("2026-09-30")
        raise SystemExit(0)
    probe_fn.parser = object()
    monkeypatch.setattr(weather_skills, "load_entrypoint", lambda *a, **k: (probe_fn, "0.0.2"))
    r = weather_skills.WeatherSkillsAdapter().health_check({**WEATHER_SKILLS_FC, "probe_latest": True}, probe_remote=True)
    assert r["healthy"] is True and r["kind"] == "remote" and r["latest"] == "2026-09-30"


def test_health_remote_probe_list_form_and_failure(monkeypatch):
    def probe_fn(argv):
        if argv == ["--help"]:
            raise SystemExit(0)
        assert argv == ["--outlook", "7d", "--probe-latest", "ts"]
        print("archive unreachable", file=sys.stderr)
        raise SystemExit(1)
    probe_fn.parser = object()
    monkeypatch.setattr(weather_skills, "load_entrypoint", lambda *a, **k: (probe_fn, "0.1.0"))
    cfg = {**WEATHER_SKILLS_FC, "argv": ["--date", "{init}", "--outlook", "7d", "-v", "{variable}"], "probe_latest": ["--probe-latest", "ts"]}
    r = weather_skills.WeatherSkillsAdapter().health_check(cfg, probe_remote=True)
    assert r["healthy"] is False and r["kind"] == "remote" and "archive unreachable" in r["message"]


# ── catalog contract ────────────────────────────────────────────────────────

WEATHER_SKILLS_PRODUCTS = [
    "weather-skills/ecmwf-s2s", "weather-skills/ifs-ens-15d", "weather-skills/ifs-ens-46d", "weather-skills/ifs-ens-46d-6h",
    "weather-skills/aifs-ens", "weather-skills/aifs-single", "weather-skills/gefs-35d", "weather-skills/gfs",
    "weather-skills/icon-eu-5d",
    "weather-skills/subc-mme-7d", "weather-skills/subc-mme-15d", "weather-skills/subc-mme-30d",
    "weather-skills/chirps-daily", "weather-skills/imerg-daily", "weather-skills/imerg-daily-final",
    "weather-skills/imerg-early-30min", "weather-skills/imerg-late-30min",
    "weather-skills/gefs-analysis", "weather-skills/gfs-analysis", "weather-skills/mrms-hourly",
    "weather-skills/era5", "weather-skills/oisst-daily", "weather-skills/smap-daily", "weather-skills/cmip6",
]


def test_catalog_has_every_rhiza_product():
    from acmaddl import catalog
    listed = [p for p in catalog.list_products() if p.startswith("weather-skills/")]
    assert sorted(listed) == sorted(WEATHER_SKILLS_PRODUCTS)


@pytest.mark.parametrize("product", WEATHER_SKILLS_PRODUCTS)
def test_rhiza_entry_contract(product):
    from acmaddl import catalog
    cfg = catalog.info(product)
    assert cfg["adapter"] == "weather_skills"
    assert cfg.get("provider", "weather-skills") in ("weather-skills", "chc-skills")
    assert cfg["skill"] and isinstance(cfg["argv"], list) and cfg["argv"]
    joined = " ".join(map(str, cfg["argv"]))
    assert ("{init}" in joined) != ("{start}" in joined), "forecast XOR observation"
    assert "{variable}" in joined or cfg["skill"] in ("oisst-fetch", "chirps-fetch", "imerg-fetch", "smap-fetch")
    for v in cfg["variables"].values():
        assert {"native_name", "units", "target_units"} <= set(v)
    assert "notes" in cfg and "grid" in cfg
    if cfg.get("window_days"):
        assert "{start}" in joined
    if cfg.get("credentials"):
        assert cfg["credentials"] in ("ecds", "earthdata")


@needs_group
@pytest.mark.parametrize("product", WEATHER_SKILLS_PRODUCTS)
def test_rhiza_entry_skill_exists_in_pinned_package(product):
    from acmaddl import catalog
    cfg = catalog.info(product)
    fn, version = weather_skills.load_entrypoint(cfg.get("provider", "weather-skills"), cfg["skill"], cfg.get("entrypoint", "fetch"))
    assert hasattr(fn, "parser") and version


# ── windows clipped to the product's published latest; future allowed for projections ──

def test_windows_clip_to_latest_and_drop_windows_past_it():
    today = date(2026, 10, 2)
    w = weather_skills.observation_windows((2026, 2026), [9], 10, today, latest=date(2026, 9, 25))
    assert w == [(date(2026, 9, 1), date(2026, 9, 10)), (date(2026, 9, 11), date(2026, 9, 20)),
                 (date(2026, 9, 21), date(2026, 9, 25))]
    with pytest.raises(ValueError, match="not published"):
        weather_skills.observation_windows((2026, 2026), [10], 10, today, latest=date(2026, 9, 25))


def test_windows_trailing_default_ends_at_latest():
    today = date(2026, 10, 2)
    assert weather_skills.observation_windows(None, None, 3, today, latest=date(2026, 9, 25)) == [
        (date(2026, 9, 23), date(2026, 9, 25))]


def test_windows_allow_future_skips_clipping():
    w = weather_skills.observation_windows((2030, 2030), [1], None, date(2026, 10, 2), allow_future=True)
    assert w == [(date(2030, 1, 1), date(2030, 1, 31))]


def test_fetch_obs_probes_latest_when_entry_supports_it(fake_catalog, monkeypatch):
    monkeypatch.setattr(weather_skills, "_today", lambda: date(2026, 10, 2))
    from acmaddl import catalog
    base = fake_catalog["chirps-fetch"]
    def fn(argv):
        if "--probe-latest" in argv:
            print("2026-09-25")
            raise SystemExit(0)
        return base(argv)
    fn.parser = object()
    monkeypatch.setattr(weather_skills, "load_entrypoint", lambda *a, **k: (fn, "9.9.9"))
    cfg = {**WEATHER_SKILLS_OBS, "probe_latest": True}
    weather_skills.WeatherSkillsAdapter().fetch_data(cfg | {"init_months": [9]}, "precip", date_range=(2026, 2026), region=[-1, 1, 36, 38])
    assert base.calls[-1][:4] == ["--start-time", "2026-09-25", "--end-time", "2026-09-25"]


# ── final-review fixes ──────────────────────────────────────────────────────

def test_run_skill_and_probe_hold_the_skill_lock(tmp_path):
    """stdout/stderr redirects and os.environ edits are process-wide; the MCP server
    runs fetches on threads, so every skill call must hold one lock."""
    seen = {}
    def fn(argv):
        seen["locked"] = weather_skills._SKILL_LOCK.locked()
        if "--probe-latest" in argv:
            print("2026-09-25")
            raise SystemExit(0)
        fixture("daily_obs").to_zarr(Path(argv[argv.index("-o") + 1]), mode="w", consolidated=True)
    fn.parser = object()
    weather_skills.run_skill(fn, [], tmp_path / "o.zarr")
    assert seen["locked"] is True
    seen.clear()
    assert weather_skills.WeatherSkillsAdapter()._probe_latest(fn, {"probe_latest": True, "argv": []}) == "2026-09-25"
    assert seen["locked"] is True
    assert not weather_skills._SKILL_LOCK.locked()


def test_fetch_obs_deletes_each_chunk_before_the_next(fake_catalog, monkeypatch):
    monkeypatch.setattr(weather_skills, "_today", lambda: date(2026, 10, 1))
    base = fake_catalog["chirps-fetch"]
    outs = []
    def fn(argv):
        out = Path(argv[argv.index("-o") + 1])
        assert all(not p.exists() for p in outs), "previous chunk still on disk"
        outs.append(out)
        return base(argv)
    fn.parser = object()
    monkeypatch.setattr(weather_skills, "load_entrypoint", lambda *a, **k: (fn, "9.9.9"))
    acmaddl.fetch("weather-skills/_test-obs", "precip", hindcast=(2026, 2026), months=[9], region=[-1, 1, 36, 38], cache=False)
    assert len(outs) == 15


def test_end_exclusive_entries_get_the_last_day_in_full(fake_catalog, monkeypatch):
    """dynamical-fetch's analysis branch slices time to `end 00:00`; an end_exclusive
    entry asks for end+1 and trims back, so the last day keeps its sub-daily steps."""
    import pandas as pd
    monkeypatch.setattr(weather_skills, "_today", lambda: date(2026, 10, 2))
    def fn(argv):
        s = pd.Timestamp(argv[argv.index("--start-time") + 1])
        e = pd.Timestamp(argv[argv.index("--end-time") + 1])
        times = pd.date_range(s, e, freq="3h")          # inclusive of e 00:00, like their slice
        ds = xr.Dataset({"precip": (("time", "latitude", "longitude"), np.ones((len(times), 3, 3), "float32"), {"units": "mm/day"})},
                        coords={"time": times, "latitude": [1.0, 0.0, -1.0], "longitude": [36.0, 37.0, 38.0]},
                        attrs={"weather_skills_history": "[]", "weather_skills_source": "t"})
        ds.to_zarr(Path(argv[argv.index("-o") + 1]), mode="w", consolidated=True)
    fn.parser = object()
    monkeypatch.setattr(weather_skills, "load_entrypoint", lambda *a, **k: (fn, "9.9.9"))
    cfg = {**WEATHER_SKILLS_OBS, "window_days": None, "end_exclusive": True}
    ds = weather_skills.WeatherSkillsAdapter().fetch_data(cfg | {"init_months": [9]}, "precip", date_range=(2026, 2026), region=[-1, 1, 36, 38])
    assert str(ds["time"].values.max())[:16] == "2026-09-30T21:00"
    assert ds.sizes["time"] == 30 * 8


def test_cache_key_folds_today_for_windows_reaching_the_present(fake_catalog, monkeypatch):
    """The trailing window and a clipped current month must not be cached under a
    day-independent key, or the first fetch would be served forever."""
    fetch_mod = sys.modules["acmaddl.fetch"]        # acmaddl.fetch the attribute is the function
    seen = []
    def fake_cached(product, variable, config, date_range, region, init_months=None, init_date=None, target_months=None, **kw):
        seen.append(init_date)
        return weather_skills.WeatherSkillsAdapter().fetch_data(config, variable, date_range=date_range, region=region)
    monkeypatch.setattr(fetch_mod, "_fetch_raw_cached", fake_cached)
    today = date.today()
    acmaddl.fetch("weather-skills/_test-obs", "precip", region=[-1, 1, 36, 38])                                   # trailing
    acmaddl.fetch("weather-skills/_test-obs", "precip", hindcast=(today.year, today.year), months=[today.month], region=[-1, 1, 36, 38])
    acmaddl.fetch("weather-skills/_test-obs", "precip", hindcast=(2020, 2020), months=[3], region=[-1, 1, 36, 38])  # historical
    assert seen == [today.isoformat(), today.isoformat(), None]


def test_windows_months_without_hindcast_mean_the_current_year():
    today = date(2026, 10, 2)
    assert weather_skills.observation_windows(None, [9], None, today) == [(date(2026, 9, 1), date(2026, 9, 30))]


def test_reforecast_and_forecast_type_are_refused(fake_catalog):
    with pytest.raises(ValueError, match="reforecast"):
        acmaddl.fetch("weather-skills/_test-fc", "precip", init="2026-09-28", region=[-1, 1, 36, 38], cache=False, reforecast=True)
    with pytest.raises(ValueError, match="forecast_type"):
        acmaddl.fetch("weather-skills/_test-fc", "precip", init="2026-09-28", region=[-1, 1, 36, 38], cache=False, forecast_type="control_forecast")


# ── Earthdata credentials: token only ─────────────────────────────────────────
#
# imerg-fetch's earthaccess.login() falls through to an interactive prompt and
# blocks when nothing is configured; the adapter checks first. Policy: a token
# (EARTHDATA_TOKEN, or the bare token in ~/.earthdatarc). Username/password
# pairs and .netrc are refused on purpose.

def _clear_earthdata(monkeypatch):
    for k in ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
        monkeypatch.delenv(k, raising=False)


def test_earthdata_env_ok_when_token_variable_is_set(monkeypatch, tmp_path):
    _clear_earthdata(monkeypatch)
    monkeypatch.setenv("EARTHDATA_TOKEN", "eyJ.token")
    assert weather_skills.earthdata_environment(home=tmp_path) == {}


def test_earthdatarc_bare_token_is_mapped_into_the_environment(monkeypatch, tmp_path):
    _clear_earthdata(monkeypatch)
    (tmp_path / ".earthdatarc").write_text("eyJ.abc\n")
    assert weather_skills.earthdata_environment(home=tmp_path) == {"EARTHDATA_TOKEN": "eyJ.abc"}


def test_earthdatarc_tolerates_a_token_prefix(monkeypatch, tmp_path):
    _clear_earthdata(monkeypatch)
    (tmp_path / ".earthdatarc").write_text("token: eyJ.abc\n")
    assert weather_skills.earthdata_environment(home=tmp_path) == {"EARTHDATA_TOKEN": "eyJ.abc"}


def test_earthdatarc_loses_to_the_environment_variable(monkeypatch, tmp_path):
    _clear_earthdata(monkeypatch)
    monkeypatch.setenv("EARTHDATA_TOKEN", "from-env")
    (tmp_path / ".earthdatarc").write_text("from-file\n")
    assert weather_skills.earthdata_environment(home=tmp_path) == {}      # their library reads the env itself


def test_earthdatarc_empty_or_multiline_is_a_clear_error(monkeypatch, tmp_path):
    _clear_earthdata(monkeypatch)
    (tmp_path / ".earthdatarc").write_text("\n")
    with pytest.raises(WeatherSkillError, match="earthdatarc"):
        weather_skills.earthdata_environment(home=tmp_path)
    (tmp_path / ".earthdatarc").write_text("username: u\npassword: p\n")
    with pytest.raises(WeatherSkillError, match="token"):
        weather_skills.earthdata_environment(home=tmp_path)


def test_earthdata_username_password_and_netrc_are_refused(monkeypatch, tmp_path):
    _clear_earthdata(monkeypatch)
    monkeypatch.setenv("EARTHDATA_USERNAME", "u")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "p")
    (tmp_path / ".netrc").write_text("machine urs.earthdata.nasa.gov login u password p\n")
    with pytest.raises(WeatherSkillError, match="token"):
        weather_skills.earthdata_environment(home=tmp_path)


def test_fetch_with_earthdata_credentials_knob_fails_fast(fake_catalog, monkeypatch, tmp_path):
    _clear_earthdata(monkeypatch)
    monkeypatch.setattr(weather_skills.Path, "home", classmethod(lambda cls: tmp_path))
    base = fake_catalog["chirps-fetch"]
    with pytest.raises(WeatherSkillError, match="Earthdata"):
        weather_skills.WeatherSkillsAdapter().fetch_data({**WEATHER_SKILLS_OBS, "credentials": "earthdata"}, "precip",
                                        date_range=(2020, 2020), region=[-1, 1, 36, 38])
    assert base.calls == []          # the skill never ran


def test_fetch_passes_earthdatarc_token_to_the_skill(fake_catalog, monkeypatch, tmp_path):
    _clear_earthdata(monkeypatch)
    monkeypatch.setattr(weather_skills.Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / ".earthdatarc").write_text("eyJ.abc\n")
    seen = {}
    base = fake_catalog["chirps-fetch"]
    def fn(argv):
        seen["token"] = os.environ.get("EARTHDATA_TOKEN")
        return base(argv)
    fn.parser = object()
    monkeypatch.setattr(weather_skills, "load_entrypoint", lambda *a, **k: (fn, "9.9.9"))
    weather_skills.WeatherSkillsAdapter().fetch_data({**WEATHER_SKILLS_OBS, "credentials": "earthdata"}, "precip",
                                    date_range=(2020, 2020), region=[-1, 1, 36, 38])
    assert seen["token"] == "eyJ.abc" and "EARTHDATA_TOKEN" not in os.environ


def test_probe_latest_runs_with_the_mapped_credentials(fake_catalog, monkeypatch, tmp_path):
    """Found live: the --probe-latest call before an observation fetch ran without
    the ~/.earthdatarc token, so IMERG's probe fell into its login prompt."""
    _clear_earthdata(monkeypatch)
    monkeypatch.setattr(weather_skills.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(weather_skills, "_today", lambda: date(2026, 10, 2))
    (tmp_path / ".earthdatarc").write_text("eyJ.abc\n")
    seen = {}
    base = fake_catalog["chirps-fetch"]
    def fn(argv):
        if argv == ["--help"]:
            raise SystemExit(0)
        if "--probe-latest" in argv:
            seen["probe_token"] = os.environ.get("EARTHDATA_TOKEN")
            print("2026-09-25")
            raise SystemExit(0)
        return base(argv)
    fn.parser = object()
    monkeypatch.setattr(weather_skills, "load_entrypoint", lambda *a, **k: (fn, "9.9.9"))
    cfg = {**WEATHER_SKILLS_OBS, "credentials": "earthdata", "probe_latest": True}
    weather_skills.WeatherSkillsAdapter().fetch_data(cfg, "precip", date_range=None, region=[-1, 1, 36, 38])
    assert seen["probe_token"] == "eyJ.abc"
    r = weather_skills.WeatherSkillsAdapter().health_check(cfg, probe_remote=True)
    assert r["healthy"] and r["latest"] == "2026-09-25" and seen["probe_token"] == "eyJ.abc"


def test_health_check_reports_missing_credentials(monkeypatch, tmp_path):
    _clear_earthdata(monkeypatch)
    monkeypatch.setattr(weather_skills.Path, "home", classmethod(lambda cls: tmp_path))
    def fn(argv):
        raise SystemExit(0)          # answers --help; the probe must never be reached
    fn.parser = object()
    monkeypatch.setattr(weather_skills, "load_entrypoint", lambda *a, **k: (fn, "9.9.9"))
    r = weather_skills.WeatherSkillsAdapter().health_check({**WEATHER_SKILLS_OBS, "credentials": "earthdata", "probe_latest": True}, probe_remote=True)
    assert r["healthy"] is False and r["kind"] == "config" and "token" in r["message"]


def test_ecds_env_accepts_a_bare_key_in_the_rc_file(monkeypatch, tmp_path):
    """Like ~/.earthdatarc: a single bare line is the key, URL defaults to the ECDS API."""
    _clear_cred_env(monkeypatch)
    (tmp_path / ".ecmwfdatastoresrc").write_text("abcd-1234-efgh\n")
    assert weather_skills.ecds_environment(home=tmp_path) == {
        "ECMWF_DATASTORES_URL": "https://ecds.ecmwf.int/api", "ECMWF_DATASTORES_KEY": "abcd-1234-efgh"}


def test_fetch_stamps_the_product_id(fake_catalog):
    ds = acmaddl.fetch("weather-skills/_test-fc", "precip", init="2026-09-28", region=[-1, 1, 36, 38], cache=False)
    assert ds.attrs["acmaddl_product"] == "weather-skills/_test-fc"


# ── skill discovery for the runner ──────────────────────────────────────────

@needs_group
def test_load_entrypoint_discovers_the_decorated_function():
    fn, version = weather_skills.load_entrypoint("weather-skills", "clip-region", entrypoint=None)
    assert fn.__name__ == "clip_region" and version


@needs_group
def test_locate_skill_searches_both_providers():
    assert weather_skills.locate_skill("clip-region")[0] == "weather-skills"
    assert weather_skills.locate_skill("subc-mme-fetch")[0] == "chc-skills"
    with pytest.raises(WeatherSkillError, match="not found"):
        weather_skills.locate_skill("no-such-skill")


@needs_group
def test_skill_kind_reads_the_catalog_group():
    assert weather_skills.skill_kind("weather-skills", "clip-region") == "transforms"
    assert weather_skills.skill_kind("weather-skills", "plot") == "figure"
    assert weather_skills.skill_kind("weather-skills", "resolve-region") == "agent-tooling"
    assert weather_skills.skill_kind("weather-skills", "chirps-fetch") == "fetchers"
