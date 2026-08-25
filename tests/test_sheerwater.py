import numpy as np
import pandas as pd
import pytest
import xarray as xr
from unittest.mock import MagicMock, patch


def _make_raw_ds(variable="precip"):
    times = pd.date_range("2010-01-01", "2010-03-01", freq="MS")
    lat = np.arange(-2.0, 3.0, 1.0)
    lon = np.arange(36.0, 41.0, 1.0)
    data = np.random.rand(len(times), len(lat), len(lon)).astype(np.float32)
    return xr.Dataset(
        {variable: (["time", "lat", "lon"], data)},
        coords={"time": times, "lat": lat, "lon": lon},
    )


def test_sheerwater_adapter_fetch_calls_correct_function():
    from acmaddl.adapters.sheerwater import SheerwaterAdapter
    raw = _make_raw_ds("precip")
    mock_fn = MagicMock(return_value=raw)
    entry = {
        "source": "chirps_v3",
        "variables": {
            "precip": {"native_name": "precip", "units": "mm/day", "target_units": "mm/day"}
        },
        "grid": {"hindcast_range": [2010, 2010]},
    }
    with patch("sheerwater.data.chirps_v3", mock_fn, create=True):
        adapter = SheerwaterAdapter()
        result = adapter.fetch_data(entry, "precip", date_range=(2010, 2010), region=[-2, 2, 36, 40])
    mock_fn.assert_called_once()
    call_kwargs = mock_fn.call_args.kwargs
    assert "start_time" in call_kwargs
    assert "end_time" in call_kwargs
    assert isinstance(result, xr.Dataset)


def test_sheerwater_adapter_passes_region():
    from acmaddl.adapters.sheerwater import SheerwaterAdapter
    raw = _make_raw_ds("precip")
    mock_fn = MagicMock(return_value=raw)
    entry = {
        "source": "chirps_v3",
        "variables": {
            "precip": {"native_name": "precip", "units": "mm/day", "target_units": "mm/day"}
        },
        "grid": {"hindcast_range": [2010, 2010]},
    }
    with patch("sheerwater.data.chirps_v3", mock_fn, create=True):
        adapter = SheerwaterAdapter()
        result = adapter.fetch_data(entry, "precip", date_range=(2010, 2010), region=[-2, 2, 36, 40])
    call_kwargs = mock_fn.call_args.kwargs
    # Sheerwater takes a named string region; the adapter translates a
    # acmaddl-style bbox list to a global fetch and crops the result.
    assert call_kwargs["region"] == "global"
    # Result was cropped client-side to the bbox.
    assert float(result["lat"].min()) >= -2.0
    assert float(result["lat"].max()) <= 2.0
    assert float(result["lon"].min()) >= 36.0
    assert float(result["lon"].max()) <= 40.0


def test_sheerwater_adapter_passes_string_region_unchanged():
    """A string region (e.g. 'africa') flows through to sheerwater unchanged."""
    from acmaddl.adapters.sheerwater import SheerwaterAdapter
    raw = _make_raw_ds("precip")
    mock_fn = MagicMock(return_value=raw)
    entry = {
        "source": "chirps_v3",
        "variables": {
            "precip": {"native_name": "precip", "units": "mm/day", "target_units": "mm/day"}
        },
        "grid": {"hindcast_range": [2010, 2010]},
    }
    with patch("sheerwater.data.chirps_v3", mock_fn, create=True):
        adapter = SheerwaterAdapter()
        adapter.fetch_data(entry, "precip", date_range=(2010, 2010), region="africa")
    call_kwargs = mock_fn.call_args.kwargs
    assert call_kwargs["region"] == "africa"


def test_sheerwater_adapter_passes_source_kwargs():
    from acmaddl.adapters.sheerwater import SheerwaterAdapter
    raw = _make_raw_ds("precip")
    mock_fn = MagicMock(return_value=raw)
    entry = {
        "source": "chirps_v3",
        "source_kwargs": {"agg_days": 1},
        "variables": {
            "precip": {"native_name": "precip", "units": "mm/day", "target_units": "mm/day"}
        },
        "grid": {"hindcast_range": [2010, 2010]},
    }
    with patch("sheerwater.data.chirps_v3", mock_fn, create=True):
        adapter = SheerwaterAdapter()
        adapter.fetch_data(entry, "precip", date_range=(2010, 2010), region=None)
    call_kwargs = mock_fn.call_args.kwargs
    assert call_kwargs.get("agg_days") == 1


def test_sheerwater_adapter_returns_xarray_dataset():
    from acmaddl.adapters.sheerwater import SheerwaterAdapter
    raw = _make_raw_ds("precip")
    entry = {
        "source": "chirps_v3",
        "variables": {
            "precip": {"native_name": "precip", "units": "mm/day", "target_units": "mm/day"}
        },
        "grid": {"hindcast_range": [2010, 2010]},
    }
    with patch("sheerwater.data.chirps_v3", MagicMock(return_value=raw), create=True):
        adapter = SheerwaterAdapter()
        result = adapter.fetch_data(entry, "precip", date_range=(2010, 2010), region=None)
    assert isinstance(result, xr.Dataset)


def test_get_adapter_sheerwater():
    from acmaddl.adapters import get_adapter
    from acmaddl.adapters.sheerwater import SheerwaterAdapter
    adapter = get_adapter("sheerwater")
    assert isinstance(adapter, SheerwaterAdapter)


def test_sheerwater_health_check_config_only():
    from acmaddl.adapters.sheerwater import SheerwaterAdapter
    adapter = SheerwaterAdapter()
    entry = {
        "source": "chirps_v3",
        "variables": {"precip": {"native_name": "precip", "units": "mm/day", "target_units": "mm/day"}},
        "grid": {"hindcast_range": [1981, 2024]},
    }
    result = adapter.health_check(entry, probe_remote=False)
    assert result["healthy"] is True
    assert result["kind"] == "config"
    assert result["probe_remote"] is False


def test_sheerwater_health_check_missing_source():
    from acmaddl.adapters.sheerwater import SheerwaterAdapter
    adapter = SheerwaterAdapter()
    result = adapter.health_check({}, probe_remote=False)
    assert result["healthy"] is False
    assert "source" in result["message"]


def test_sheerwater_health_check_remote_requires_zarr_url():
    from acmaddl.adapters.sheerwater import SheerwaterAdapter
    adapter = SheerwaterAdapter()
    entry = {"source": "chirps_v3", "variables": {}, "grid": {}}
    result = adapter.health_check(entry, probe_remote=True)
    assert result["healthy"] is False
    assert "zarr_url" in result["message"]


def test_sheerwater_health_check_remote_success():
    from acmaddl.adapters.sheerwater import SheerwaterAdapter
    adapter = SheerwaterAdapter()
    entry = {
        "source": "chirps_v3",
        "zarr_url": "gs://fake-bucket/chirps",
        "variables": {},
        "grid": {"hindcast_range": [1981, 2024]},
    }
    mock_ds = xr.Dataset(
        {"precip": (["time", "lat", "lon"], np.zeros((3, 2, 2)))},
        coords={
            "time": pd.date_range("1981-01-01", periods=3, freq="MS"),
            "lat": [0.0, 1.0],
            "lon": [30.0, 31.0],
        },
    )
    with patch("xarray.open_dataset", return_value=mock_ds):
        result = adapter.health_check(entry, probe_remote=True)
    assert result["healthy"] is True
    assert result["kind"] == "remote"
