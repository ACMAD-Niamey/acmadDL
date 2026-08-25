"""_resolve_native_name: tolerance to source renames and stale caches."""
import numpy as np
import pytest
import xarray as xr

from acmaddl.normalize import _resolve_native_name


def _ds(**vars_):
    coords = {"year": [2000, 2001], "lat": [0.0, 1.0], "lon": [30.0, 31.0]}
    return xr.Dataset(
        {name: (("year", "lat", "lon"), np.ones((2, 2, 2))) for name in vars_},
        coords=coords)


def test_native_name_direct_hit():
    cfg = {"native_name": "pr"}
    assert _resolve_native_name(_ds(pr=1), cfg, "precip") == "pr"


def test_falls_back_through_declared_aliases():
    # cached data still carries the pre-rename name declared as path_name
    cfg = {"native_name": "pr", "path_name": "prcp"}
    assert _resolve_native_name(_ds(prcp=1), cfg, "precip") == "prcp"
    # dict-form (per-stream) path_name is flattened
    cfg = {"native_name": "pr", "path_name": {"hindcast": "prcp", "forecast": "pr"}}
    assert _resolve_native_name(_ds(prcp=1), cfg, "precip") == "prcp"


def test_lone_variable_heuristic_warns():
    cfg = {"native_name": "pr"}
    with pytest.warns(RuntimeWarning, match="only data variable"):
        assert _resolve_native_name(_ds(rainfall_total=1), cfg, "precip") == "rainfall_total"


def test_ambiguous_dataset_raises_with_cache_hint():
    cfg = {"native_name": "pr"}
    with pytest.raises(ValueError, match="cache"):
        _resolve_native_name(_ds(a=1, b=2), cfg, "precip")
