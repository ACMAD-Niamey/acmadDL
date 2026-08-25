"""cpc_nmme pickle cache must survive ~/.nuthatch remounts after import."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import xarray as xr


def test_cache_dir_recreated_after_nuthatch_tree_replaced(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    import acmaddl.cpc_nmme as m

    first = Path(m._cache_dir())
    assert first.is_dir()
    assert first == tmp_path / ".nuthatch" / "acmaddl" / "cpc_nmme"

    # Colab / Shared Drive pattern: replace ~/.nuthatch after import.
    shutil.rmtree(tmp_path / ".nuthatch")
    (tmp_path / ".nuthatch").mkdir()

    second = Path(m._cache_dir())
    assert second.is_dir()
    assert second == first


def test_fetch_global_writes_after_cache_dir_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    import acmaddl.cpc_nmme as m

    Path(m._cache_dir())
    shutil.rmtree(tmp_path / ".nuthatch")

    da = xr.DataArray(
        np.zeros((1, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={"time": ["t0"], "lat": [-1.0, 1.0], "lon": [10.0, 11.0]},
    )
    monkeypatch.setattr(m, "_download", lambda url: "unused")
    monkeypatch.setattr(m, "_parse_cpt", lambda text: da)

    hc, fc = m._fetch_global("cfsv2", "sst", "ASO", (1991, 1991), 2026)
    assert hc.sizes["year"] == 1
    assert fc.sizes["year"] == 1
    cached = list((tmp_path / ".nuthatch" / "acmaddl" / "cpc_nmme").glob("*.pkl"))
    assert len(cached) == 1
