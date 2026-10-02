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


# ── ECDS credential mapping ─────────────────────────────────────────────────

def _clear_cred_env(monkeypatch):
    for k in ("ECMWF_DATASTORES_URL", "ECMWF_DATASTORES_KEY", "CDSAPI_URL", "CDSAPI_KEY"):
        monkeypatch.delenv(k, raising=False)


def test_ecds_env_noop_when_their_variables_exist(monkeypatch, tmp_path):
    _clear_cred_env(monkeypatch)
    monkeypatch.setenv("ECMWF_DATASTORES_URL", "https://ecds.ecmwf.int/api")
    monkeypatch.setenv("ECMWF_DATASTORES_KEY", "k")
    assert rhiza.ecds_environment(home=tmp_path) == {}


def test_ecds_env_noop_when_their_rc_file_exists(monkeypatch, tmp_path):
    _clear_cred_env(monkeypatch)
    (tmp_path / ".ecmwfdatastoresrc").write_text("url: https://ecds.ecmwf.int/api\nkey: k\n")
    assert rhiza.ecds_environment(home=tmp_path) == {}


def test_ecds_env_maps_an_ecds_cdsapirc(monkeypatch, tmp_path):
    _clear_cred_env(monkeypatch)
    (tmp_path / ".cdsapirc").write_text("url: https://ecds.ecmwf.int/api\nkey: abc-123\n")
    assert rhiza.ecds_environment(home=tmp_path) == {
        "ECMWF_DATASTORES_URL": "https://ecds.ecmwf.int/api", "ECMWF_DATASTORES_KEY": "abc-123"}


def test_ecds_env_refuses_a_copernicus_cdsapirc(monkeypatch, tmp_path):
    _clear_cred_env(monkeypatch)
    (tmp_path / ".cdsapirc").write_text("url: https://cds.climate.copernicus.eu/api\nkey: abc\n")
    with pytest.raises(RhizaSkillError, match="ecds.ecmwf.int"):
        rhiza.ecds_environment(home=tmp_path)


def test_ecds_env_prefers_cdsapi_env_vars(monkeypatch, tmp_path):
    _clear_cred_env(monkeypatch)
    monkeypatch.setenv("CDSAPI_URL", "https://ecds.ecmwf.int/api")
    monkeypatch.setenv("CDSAPI_KEY", "zzz")
    assert rhiza.ecds_environment(home=tmp_path)["ECMWF_DATASTORES_KEY"] == "zzz"


# ── fetch_data end to end through acmaddl.fetch ─────────────────────────────

import acmaddl

RHIZA_FC = {
    "adapter": "rhiza", "provider": "weather-skills", "skill": "dynamical-fetch",
    "argv": ["--dataset", "ecmwf-ifs-ens-forecast-15-day-0-25-degree", "--date", "{init}", "--bbox", "{bbox}", "-v", "{variable}"],
    "variables": {"precip": {"native_name": "precipitation_surface", "units": "mm/day", "target_units": "mm/day"}},
    "grid": {"lat_res": 0.25, "lon_res": 0.25, "forecast_members": 51},
}
RHIZA_OBS = {
    "adapter": "rhiza", "provider": "weather-skills", "skill": "chirps-fetch",
    "argv": ["--start-time", "{start}", "--end-time", "{end}"], "window_days": 2,
    "variables": {"precip": {"native_name": "precip", "units": "mm/day", "target_units": "mm/day"}},
    "grid": {"lat_res": 0.05, "lon_res": 0.05, "temporal": "daily"},
}


@pytest.fixture
def fake_catalog(monkeypatch):
    """Register two throwaway rhiza/* products and route skill loading to fakes."""
    from acmaddl import catalog
    entries = {"rhiza/_test-fc": RHIZA_FC, "rhiza/_test-obs": RHIZA_OBS}
    real_info = catalog.info
    monkeypatch.setattr(catalog, "info", lambda p: dict(entries[p]) | {"deprecated": False} if p in entries else real_info(p))
    monkeypatch.setattr(catalog, "get", catalog.info)
    wrappers = {"dynamical-fetch": fake_wrapper("ensemble_forecast"), "chirps-fetch": fake_wrapper("daily_obs")}
    monkeypatch.setattr(rhiza, "load_entrypoint", lambda provider, skill, entrypoint="fetch": (wrappers[skill], "9.9.9"))
    return wrappers


def test_fetch_forecast_end_to_end(fake_catalog):
    ds = acmaddl.fetch("rhiza/_test-fc", "precip", init="2026-09-28", region=[-1, 1, 36, 38], cache=False)
    argv = fake_catalog["dynamical-fetch"].calls[0]
    assert argv[:8] == ["--dataset", "ecmwf-ifs-ens-forecast-15-day-0-25-degree", "--date", "2026-09-28",
                        "--bbox", "1/36/-1/38", "-v", "precipitation_surface"]
    assert set(ds["precip"].dims) == {"init_time", "lead_time", "member", "lat", "lon"}
    assert ds["init_time"].values[0] == np.datetime64("2026-09-28", "ns")
    assert ds.attrs["rhiza_skill"] == "dynamical-fetch" and ds.attrs["rhiza_skill_version"] == "9.9.9"
    assert "weather_skills_history" in ds.attrs


def test_fetch_forecast_month_only_init_is_rejected(fake_catalog):
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        acmaddl.fetch("rhiza/_test-fc", "precip", init="2026-09", region=[-1, 1, 36, 38], cache=False)


def test_fetch_forecast_without_init_is_rejected(fake_catalog):
    with pytest.raises(ValueError, match="init="):
        acmaddl.fetch("rhiza/_test-fc", "precip", region=[-1, 1, 36, 38], cache=False)


def test_fetch_obs_chunks_and_concatenates(fake_catalog, monkeypatch):
    monkeypatch.setattr(rhiza, "_today", lambda: date(2026, 10, 1))
    ds = acmaddl.fetch("rhiza/_test-obs", "precip", hindcast=(2026, 2026), months=[9], region=[-1, 1, 36, 38], cache=False)
    calls = fake_catalog["chirps-fetch"].calls
    assert calls[0][:4] == ["--start-time", "2026-09-01", "--end-time", "2026-09-02"]
    assert calls[-1][:4] == ["--start-time", "2026-09-29", "--end-time", "2026-09-30"]
    assert len(calls) == 15
    assert set(ds["precip"].dims) == {"time", "lat", "lon"}
    assert ds.sizes["time"] == 15 * fixture("daily_obs").sizes["time"]   # each chunk contributes the fixture's days


def test_fetch_data_template_without_dates_is_rejected():
    cfg = {**RHIZA_OBS, "argv": ["--bbox", "{bbox}"]}
    with pytest.raises(ValueError, match="neither"):
        rhiza.RhizaAdapter().fetch_data(cfg, "precip", date_range=(2026, 2026), region=[-1, 1, 36, 38])


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
    out = acmaddl.fetch("rhiza/_test-fc", "precip", init="2026-09-28", region=[-1, 1, 36, 38], cache=False,
                        destination=str(tmp_path / "fc.nc"), format="netcdf")
    assert (tmp_path / "fc.nc").exists()
    assert isinstance(out["lat"].attrs["statistics_approximate"], str)


# ── health check ────────────────────────────────────────────────────────────

def test_health_config_reports_missing_group(monkeypatch):
    def missing(*a, **k):
        raise RhizaNotInstalled("weather-skills")
    monkeypatch.setattr(rhiza, "load_entrypoint", missing)
    r = rhiza.RhizaAdapter().health_check(RHIZA_FC)
    assert r["healthy"] is False and r["kind"] == "config" and "uv sync --group rhiza" in r["message"]


def test_health_config_ok_runs_help(monkeypatch):
    fn = fake_wrapper("ensemble_forecast")
    def help_fn(argv):
        fn.calls.append(list(argv))
        raise SystemExit(0)
    help_fn.parser = object()
    monkeypatch.setattr(rhiza, "load_entrypoint", lambda *a, **k: (help_fn, "0.0.2"))
    r = rhiza.RhizaAdapter().health_check(RHIZA_FC)
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
    monkeypatch.setattr(rhiza, "load_entrypoint", lambda *a, **k: (probe_fn, "0.0.2"))
    r = rhiza.RhizaAdapter().health_check({**RHIZA_FC, "probe_latest": True}, probe_remote=True)
    assert r["healthy"] is True and r["kind"] == "remote" and r["latest"] == "2026-09-30"


def test_health_remote_probe_list_form_and_failure(monkeypatch):
    def probe_fn(argv):
        if argv == ["--help"]:
            raise SystemExit(0)
        assert argv == ["--outlook", "7d", "--probe-latest", "ts"]
        print("archive unreachable", file=sys.stderr)
        raise SystemExit(1)
    probe_fn.parser = object()
    monkeypatch.setattr(rhiza, "load_entrypoint", lambda *a, **k: (probe_fn, "0.1.0"))
    cfg = {**RHIZA_FC, "argv": ["--date", "{init}", "--outlook", "7d", "-v", "{variable}"], "probe_latest": ["--probe-latest", "ts"]}
    r = rhiza.RhizaAdapter().health_check(cfg, probe_remote=True)
    assert r["healthy"] is False and r["kind"] == "remote" and "archive unreachable" in r["message"]


# ── catalog contract ────────────────────────────────────────────────────────

RHIZA_PRODUCTS = [
    "rhiza/ecmwf-s2s", "rhiza/ifs-ens-15d", "rhiza/ifs-ens-46d", "rhiza/ifs-ens-46d-6h",
    "rhiza/aifs-ens", "rhiza/aifs-single", "rhiza/gefs-35d", "rhiza/gfs",
    "rhiza/icon-eu-5d", "rhiza/hrdps", "rhiza/hrrr-48h",
    "rhiza/subc-mme-7d", "rhiza/subc-mme-15d", "rhiza/subc-mme-30d",
    "rhiza/chirps-daily", "rhiza/imerg-daily", "rhiza/imerg-daily-final",
    "rhiza/imerg-early-30min", "rhiza/imerg-late-30min",
    "rhiza/gefs-analysis", "rhiza/gfs-analysis", "rhiza/hrrr-analysis", "rhiza/mrms-hourly",
    "rhiza/era5", "rhiza/oisst-daily", "rhiza/smap-daily", "rhiza/cmip6",
]


def test_catalog_has_every_rhiza_product():
    from acmaddl import catalog
    listed = [p for p in catalog.list_products() if p.startswith("rhiza/")]
    assert sorted(listed) == sorted(RHIZA_PRODUCTS)


@pytest.mark.parametrize("product", RHIZA_PRODUCTS)
def test_rhiza_entry_contract(product):
    from acmaddl import catalog
    cfg = catalog.info(product)
    assert cfg["adapter"] == "rhiza"
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
        assert cfg["credentials"] == "ecds"


@needs_group
@pytest.mark.parametrize("product", RHIZA_PRODUCTS)
def test_rhiza_entry_skill_exists_in_pinned_package(product):
    from acmaddl import catalog
    cfg = catalog.info(product)
    fn, version = rhiza.load_entrypoint(cfg.get("provider", "weather-skills"), cfg["skill"], cfg.get("entrypoint", "fetch"))
    assert hasattr(fn, "parser") and version


# ── windows clipped to the product's published latest; future allowed for projections ──

def test_windows_clip_to_latest_and_drop_windows_past_it():
    today = date(2026, 10, 2)
    w = rhiza.observation_windows((2026, 2026), [9], 10, today, latest=date(2026, 9, 25))
    assert w == [(date(2026, 9, 1), date(2026, 9, 10)), (date(2026, 9, 11), date(2026, 9, 20)),
                 (date(2026, 9, 21), date(2026, 9, 25))]
    with pytest.raises(ValueError, match="not published"):
        rhiza.observation_windows((2026, 2026), [10], 10, today, latest=date(2026, 9, 25))


def test_windows_trailing_default_ends_at_latest():
    today = date(2026, 10, 2)
    assert rhiza.observation_windows(None, None, 3, today, latest=date(2026, 9, 25)) == [
        (date(2026, 9, 23), date(2026, 9, 25))]


def test_windows_allow_future_skips_clipping():
    w = rhiza.observation_windows((2030, 2030), [1], None, date(2026, 10, 2), allow_future=True)
    assert w == [(date(2030, 1, 1), date(2030, 1, 31))]


def test_fetch_obs_probes_latest_when_entry_supports_it(fake_catalog, monkeypatch):
    monkeypatch.setattr(rhiza, "_today", lambda: date(2026, 10, 2))
    from acmaddl import catalog
    base = fake_catalog["chirps-fetch"]
    def fn(argv):
        if "--probe-latest" in argv:
            print("2026-09-25")
            raise SystemExit(0)
        return base(argv)
    fn.parser = object()
    monkeypatch.setattr(rhiza, "load_entrypoint", lambda *a, **k: (fn, "9.9.9"))
    cfg = {**RHIZA_OBS, "probe_latest": True}
    rhiza.RhizaAdapter().fetch_data(cfg | {"init_months": [9]}, "precip", date_range=(2026, 2026), region=[-1, 1, 36, 38])
    assert base.calls[-1][:4] == ["--start-time", "2026-09-25", "--end-time", "2026-09-25"]
