"""acmaddl.weather_skills: run Rhiza's transforms/figures/tools on acmadDL data.

Transforms are pure functions, so they run for real on the fixtures with no
network. Figure tests need matplotlib (Agg). Everything skips without the
`weather-skills` dependency group.
"""
import importlib.metadata
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from acmaddl.errors import WeatherSkillError

try:
    importlib.metadata.distribution("weather-skills")
    HAVE_GROUP = True
except importlib.metadata.PackageNotFoundError:
    HAVE_GROUP = False
needs_group = pytest.mark.skipif(not HAVE_GROUP, reason="weather-skills group not installed")

from acmaddl import weather_skills as ws   # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "weather_skills"


def fixture(name):
    ds = xr.open_dataset(FIXTURES / f"{name}.nc").load()
    ds.encoding = {}
    for v in ds.variables:
        ds[v].encoding = {}
    return ds


def _obs(units="mm/day", name="precip", **attrs):
    time = pd.date_range("2026-09-01", periods=4, freq="D")
    da = xr.DataArray(np.ones((4, 3, 3), "float32"), dims=("time", "lat", "lon"),
                      coords={"time": time, "lat": [-1.0, 0.0, 1.0], "lon": [36.0, 37.0, 38.0]},
                      attrs={"units": units} if units else {})
    return da.rename(name).to_dataset().assign_attrs(**attrs)


# ── to_standard_dataset ──────────────────────────────────────────────────────

def test_units_are_rewritten_to_pint_strings():
    out = ws.to_standard_dataset(_obs(units="C", name="temp"))
    assert out["temp"].attrs["units"] == "degree_Celsius"
    assert ws.to_standard_dataset(_obs(units="mm/day"))["precip"].attrs["units"] == "mm day-1"
    assert ws.to_standard_dataset(_obs(units="K", name="sst"))["sst"].attrs["units"] == "kelvin"
    assert ws.to_standard_dataset(_obs(units="m3/m3", name="soil_moisture"))["soil_moisture"].attrs["units"] == "1"
    assert ws.to_standard_dataset(_obs(units="Pa", name="msl"))["msl"].attrs["units"] == "Pa"   # unknown: untouched


def test_units_required_variable_without_units_is_refused():
    with pytest.raises(ValueError, match="precip"):
        ws.to_standard_dataset(_obs(units=None))
    ws.to_standard_dataset(_obs(units=None, name="soil_moisture"))     # not a units-required kind


def test_bare_dataarray_needs_a_name():
    da = _obs()["precip"].rename(None)
    with pytest.raises(ValueError, match="name="):
        ws.to_standard_dataset(da)
    assert "rain" in ws.to_standard_dataset(da, name="rain").data_vars


def test_numeric_hour_leads_become_timedeltas_and_months_are_refused():
    fc = xr.Dataset({"precip": (("init_time", "lead_time", "lat", "lon"), np.ones((1, 3, 2, 2), "float32"), {"units": "mm/day"})},
                    coords={"init_time": pd.to_datetime(["2026-09-01"]), "lead_time": ("lead_time", [24.0, 48.0, 72.0], {"units": "hours"}),
                            "lat": [0.0, 1.0], "lon": [36.0, 37.0]})
    out = ws.to_standard_dataset(fc)                            # one init -> their fetcher layout
    assert np.issubdtype(out["step"].dtype, np.timedelta64)
    assert out["step"].values[1] == np.timedelta64(48, "h")
    assert out["time"].ndim == 0 and "init_time" not in out.dims
    fc["lead_time"].attrs["units"] = "months"
    with pytest.raises(ValueError, match="month"):
        ws.to_standard_dataset(fc)


def test_two_dimensional_valid_time_is_dropped():
    fc = xr.Dataset({"precip": (("init_time", "lead_time", "lat", "lon"), np.ones((1, 2, 2, 2), "float32"), {"units": "mm/day"})},
                    coords={"init_time": pd.to_datetime(["2026-09-01"]), "lead_time": pd.to_timedelta([1, 2], unit="D"),
                            "lat": [0.0, 1.0], "lon": [36.0, 37.0]})
    fc = fc.assign_coords(time=fc["init_time"] + fc["lead_time"])
    out = ws.to_standard_dataset(fc)
    assert out["time"].ndim == 0                                 # the 2-D valid time went; scalar init stays


def test_year_hindcast_becomes_init_time_with_a_lead():
    hc = xr.Dataset({"precip": (("year", "member", "lat", "lon"), np.ones((3, 2, 2, 2), "float32"), {"units": "mm"})},
                    coords={"year": [2001, 2002, 2003], "member": [0, 1], "lat": [0.0, 1.0], "lon": [36.0, 37.0]})
    with pytest.raises(ValueError, match="init_month"):
        ws.to_standard_dataset(hc)
    out = ws.to_standard_dataset(hc, init_month=2, lead="60 days")
    assert set(out.dims) == {"init_time", "lead_time", "member", "lat", "lon"}
    assert out["init_time"].values[0] == np.datetime64("2001-02-01", "ns")
    assert out.sizes["lead_time"] == 1 and out["lead_time"].values[0] == np.timedelta64(60, "D")


def test_single_forecast_needs_an_init():
    fc = xr.Dataset({"precip": (("member", "lat", "lon"), np.ones((2, 2, 2), "float32"), {"units": "mm/day"})},
                    coords={"member": [0, 1], "lat": [0.0, 1.0], "lon": [36.0, 37.0]})
    with pytest.raises(ValueError, match="init="):
        ws.to_standard_dataset(fc)
    out = ws.to_standard_dataset(fc, init="2026-09-28")
    assert out["time"].values == np.datetime64("2026-09-28", "ns") and out.sizes["step"] == 1
    assert "acmaddl_lead_note" in out.attrs                     # default lead recorded


def test_several_issuances_keep_an_init_time_dimension():
    fc = xr.Dataset({"precip": (("init_time", "lead_time", "lat", "lon"), np.ones((2, 2, 2, 2), "float32"), {"units": "mm/day"})},
                    coords={"init_time": pd.to_datetime(["2026-09-01", "2026-09-02"]), "lead_time": pd.to_timedelta([1, 2], unit="D"),
                            "lat": [0.0, 1.0], "lon": [36.0, 37.0]})
    out = ws.to_standard_dataset(fc)
    assert out.sizes["init_time"] == 2 and "lead_time" in out.dims


def test_provenance_is_started_or_appended():
    out = ws.to_standard_dataset(_obs(acmaddl_product="obs/chirps-v3-daily"))
    hist = json.loads(out.attrs["weather_skills_history"])
    assert len(hist) == 1 and hist[0]["skill"] == "acmaddl" and hist[0]["args"]["source"] == "acmaddl:obs/chirps-v3-daily"
    assert out.attrs["weather_skills_source"] == "acmaddl:obs/chirps-v3-daily"
    again = ws.to_standard_dataset(out)                          # already carries a chain
    assert len(json.loads(again.attrs["weather_skills_history"])) == 2


def test_dict_attrs_are_json_encoded_and_encodings_cleared():
    ds = _obs()
    ds["lat"].attrs["statistics_approximate"] = {"min": -90.0, "max": 90.0}
    ds["precip"].encoding["dtype"] = "float32"
    out = ws.to_standard_dataset(ds)
    assert isinstance(out["lat"].attrs["statistics_approximate"], str) and out["precip"].encoding == {}


# ── from_standard_dataset ────────────────────────────────────────────────────

def fixture_as_acmaddl(name):
    return ws.from_standard_dataset(fixture(name))


def test_from_standard_renames_and_sorts_latitude():
    out = fixture_as_acmaddl("daily_obs")
    assert set(out["precip"].dims) == {"time", "lat", "lon"}
    assert np.all(np.diff(out["lat"].values) > 0)
    assert out["precip"].attrs["units"] == "mm/day"


def test_from_standard_reshapes_a_scalar_time_forecast():
    out = fixture_as_acmaddl("ensemble_forecast")
    assert set(out["precipitation_surface"].dims) == {"init_time", "lead_time", "member", "lat", "lon"}
    assert "step_bounds" not in out.coords and "nv" not in out.dims
    assert set(out["time"].dims) == {"init_time", "lead_time"}
    assert out["precipitation_surface"].attrs["units"] == "mm/day"


def test_from_standard_s2s_units_and_init():
    out = ws.from_standard_dataset(fixture("s2s_forecast"), init="2026-10-01")
    assert out["init_time"].values[0] == np.datetime64("2026-10-01", "ns")
    assert out["tp"].attrs["units"] == "mm/day"


@pytest.mark.parametrize("name", ["daily_obs", "analysis", "imerg_daily", "smap_daily"])
def test_round_trip_obs_is_lossless(name):
    ours = fixture_as_acmaddl(name)
    back = ws.from_standard_dataset(ws.to_standard_dataset(ours))
    var = list(ours.data_vars)[0]
    xr.testing.assert_allclose(back[var], ours[var])
    assert back[var].attrs.get("units") == ours[var].attrs.get("units")   # smap carries none
    assert list(back.dims) == list(ours.dims)


def test_round_trip_forecast_is_lossless():
    ours = fixture_as_acmaddl("ensemble_forecast")
    back = ws.from_standard_dataset(ws.to_standard_dataset(ours))
    xr.testing.assert_allclose(back["precipitation_surface"], ours["precipitation_surface"])
    assert np.array_equal(back["lead_time"].values, ours["lead_time"].values)


# ── run(): transforms ────────────────────────────────────────────────────────

@needs_group
def test_run_clip_region_matches_acmaddl_crop():
    ours = fixture_as_acmaddl("daily_obs")                     # 3x3 cells around 0N 37E
    out = ws.run("clip-region", ours, bbox=[-0.05, 0.05, 36.95, 37.05])
    assert set(out["precip"].dims) == {"time", "lat", "lon"}
    assert out.sizes["lat"] >= 1 and out.sizes["lon"] >= 1 and out.sizes["lat"] < ours.sizes["lat"]
    assert out["precip"].attrs["units"] == "mm/day"
    hist = json.loads(out.attrs["weather_skills_history"])
    assert [h["skill"] for h in hist[-2:]] == ["acmaddl", "clip-region"]
    assert out.attrs["weather_skills_name"] == "clip-region"


@needs_group
def test_run_raw_returns_their_shape():
    out = ws.run("clip-region", fixture_as_acmaddl("daily_obs"), bbox=[-1, 1, 36, 38], raw=True)
    assert out["precip"].attrs["units"] in ("mm day-1", "millimeter / day")   # their decorator's pint spelling
    assert "latitude" in out.dims or "lat" in out.dims


@needs_group
def test_run_unit_convert_to_standard():
    # Their --to-units path fails on their own layout too ("has no units attr" after
    # pint quantification); --to-standard, the documented form, works.
    ours = fixture_as_acmaddl("single_forecast")                # temperature_2m in C
    out = ws.run("unit-convert", ours, variable="temperature_2m", to_standard=True)
    assert out["temperature_2m"].attrs["units"] == "C"
    assert float(out["temperature_2m"].max()) < 100


@needs_group
def test_run_summarize_dim_and_select():
    ours = fixture_as_acmaddl("ensemble_forecast")
    mean = ws.run("summarize-dim", ours, dim="member", method="mean")
    assert "member" not in mean.dims
    one = ws.run("select", ours, dim="member", index=0)
    assert one.sizes.get("member", 1) == 1


@needs_group
def test_run_rename_and_difference_and_concat():
    a = fixture_as_acmaddl("daily_obs")
    b = a.copy()
    b["precip"] = b["precip"] * 2
    renamed = ws.run("rename", a, variable="precip", to_name="rain")
    assert "rain" in renamed.data_vars
    diff = ws.run("difference", a, b)
    assert np.allclose(np.abs(diff["precip"].values), np.abs(a["precip"].values), equal_nan=True)
    cat = ws.run("concat", a, b, dim="time")
    assert cat.sizes["time"] == 2 * a.sizes["time"]


@needs_group
def test_run_aggregate_temporal_daily_from_analysis():
    ours = fixture_as_acmaddl("analysis")                       # 3-hourly GEFS analysis, 4 steps
    out = ws.run("aggregate-temporal", ours, period="daily", method="mean")
    assert out.sizes["time"] >= 1 and set(out["precipitation_surface"].dims) == {"time", "lat", "lon"}


@needs_group
def test_run_step_to_time_and_deaccumulate_on_a_forecast():
    ours = fixture_as_acmaddl("ensemble_forecast")
    valid = ws.run("step-to-time", ours, raw=True)
    assert "time" in valid.dims or "time" in valid.coords
    try:
        ws.run("deaccumulate", ours, variable="precipitation_surface")
    except WeatherSkillError as exc:                             # already a rate: their refusal is fine
        assert "rate" in str(exc).lower() or "accum" in str(exc).lower()


# `downscale` and `coarsen` are not exercised offline: both fail in this environment on
# their own Zarr layout too ("Mixing chunked array types ... pint.Quantity ... dask"),
# an xarray-regrid vs pint-xarray incompatibility at the pinned versions. See
# troubleshooting.md; they run unchanged once that is resolved upstream.


@needs_group
def test_run_refuses_unknown_flags_and_wrong_input_counts():
    ours = fixture_as_acmaddl("daily_obs")
    with pytest.raises(ValueError, match="--bbox"):
        ws.run("clip-region", ours, bounding_box="x")
    with pytest.raises(ValueError, match="one input"):
        ws.run("clip-region", ours, ours, bbox=[-1, 1, 36, 38])
    with pytest.raises(ValueError, match="acmaddl.fetch"):
        ws.run("chirps-fetch", start_time="2026-01-01", end_time="2026-01-02")
    with pytest.raises(ValueError, match="submit-feedback"):
        ws.run("submit-feedback", title="x", body="y")


@needs_group
def test_run_surfaces_their_refusal():
    with pytest.raises(WeatherSkillError, match="clip-region"):
        ws.run("clip-region", fixture_as_acmaddl("daily_obs"), bbox=[80, 85, 170, 175])   # empty selection


# ── run(): figures and agent tools; skills() ─────────────────────────────────

@needs_group
def test_run_plot_writes_a_png_where_asked(tmp_path):
    pytest.importorskip("matplotlib")
    ours = fixture_as_acmaddl("daily_obs")
    png = ws.run("plot", ours, variable="precip", output=tmp_path / "map.png")
    assert png == tmp_path / "map.png" and png.exists() and png.stat().st_size > 1000


@needs_group
def test_run_plot_without_output_survives_the_temp_dir():
    pytest.importorskip("matplotlib")
    png = ws.run("plot-timeseries", fixture_as_acmaddl("daily_obs"), variable="precip", reduce=["lat", "lon"])
    try:
        assert png.exists() and png.suffix == ".png" and "plot-timeseries" in png.name
    finally:
        png.unlink(missing_ok=True)


@needs_group
def test_run_agent_tools_return_text():
    bbox = ws.run("resolve-region", "Kenya").strip()
    parts = bbox.split("/")
    assert len(parts) == 4 and float(parts[0]) > float(parts[2])      # N/W/S/E
    text = ws.run("resolve-time", "last-2w", emit="json")
    assert "start" in text.lower()
    info = ws.run("inspect-zarr", fixture_as_acmaddl("daily_obs"), format="json")
    assert "precip" in info
    prov = ws.run("provenance", fixture_as_acmaddl("daily_obs"), format="json")
    chain = json.loads(prov)
    assert [e["skill"] for e in chain] == ["chirps-fetch"]        # inspecting is not a step


@needs_group
def test_skills_lists_every_non_fetcher():
    allk = ws.skills()
    assert {"clip-region", "plot", "resolve-region", "subc-mme-fetch", "iod-mode-index"} <= set(allk)
    assert allk["clip-region"].kind == "transforms" and "--bbox" in allk["clip-region"].flags
    assert allk["subc-mme-fetch"].provider == "chc-skills"
    non_fetchers = ws.skills(kind="transforms") | ws.skills(kind="figure") | ws.skills(kind="agent-tooling")
    assert len(non_fetchers) == 28          # 15 transforms + 8 figures + 5 tools


# ── final-review fixes ──────────────────────────────────────────────────────

@needs_group
def test_output_keyword_is_refused_for_non_figure_skills(tmp_path):
    with pytest.raises(ValueError, match="figure"):
        ws.run("rename", fixture_as_acmaddl("daily_obs"), variable="precip", to_name="rain", output=tmp_path / "x.zarr")
    with pytest.raises(ValueError, match="figure"):
        ws.run("inspect-zarr", fixture_as_acmaddl("daily_obs"), output=tmp_path / "x.txt")


@needs_group
def test_figure_default_location_honours_acmaddl_tmpdir(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("ACMADDL_TMP_DIR", str(tmp_path / "acmaddl-tmp"))
    png = ws.run("plot", fixture_as_acmaddl("daily_obs"), variable="precip")
    assert png.exists() and (tmp_path / "acmaddl-tmp") in png.parents


@needs_group
def test_zero_inputs_to_a_required_input_flag_is_refused_before_argparse():
    with pytest.raises(ValueError, match="dataset input"):
        ws.run("plot-timeseries", variable="precip")
    with pytest.raises(ValueError, match="dataset input"):
        ws.run("difference")
