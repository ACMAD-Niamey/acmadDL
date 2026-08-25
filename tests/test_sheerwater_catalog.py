"""§3.2 Sheerwater-backed catalog entries (acmaddl #7) + ERA5 enrichment.

CHIRPS now uses an explicit obs/chirps-{version}-{cadence}[-rhiza] naming
(see tests/test_chirps_catalog.py): native UCSB products on the http adapter,
Rhiza/Sheerwater mirrors ending in -rhiza. obs/imerg and obs/ghcn are
Sheerwater-backed precip entries. obs/era5 (cds) gains a
precip variable so ERA5 stays broadly usable without Sheerwater (the installed
sheerwater build exposes no era5 data function).
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from unittest.mock import MagicMock, patch

from acmaddl import catalog


def test_obs_chirps_v3_daily_rhiza_routes_to_sheerwater():
    e = catalog.info("obs/chirps-v3-daily-rhiza")
    assert e["adapter"] == "sheerwater"
    assert e["source"] == "chirps_v3"
    assert "precip" in e["variables"]


def test_obs_chirps_v3_monthly_is_native_and_not_deprecated():
    # The old obs/chirps-direct UCSB COG path is now obs/chirps-v3-monthly,
    # the native v3 monthly product, no longer deprecated (its old `successor`
    # pointed at the truncated mirror, which was backwards).
    e = catalog.info("obs/chirps-v3-monthly")
    assert e["adapter"] == "http"
    assert not e.get("deprecated")
    assert "successor" not in e
    assert e["format"] == "cog"


@pytest.mark.parametrize("prod,src", [
    ("obs/imerg", "imerg_final"),
    ("obs/ghcn", "ghcn_avg"),
])
def test_new_sheerwater_obs_entries(prod, src):
    e = catalog.info(prod)
    assert e["adapter"] == "sheerwater"
    assert e["source"] == src
    assert "precip" in e["variables"]


def test_obs_era5_enriched_with_precip():
    e = catalog.info("obs/era5")
    assert e["adapter"] == "cds"             # stays on CDS (no sheerwater era5 fn)
    assert "precip" in e["variables"]
    assert "temp" in e["variables"]          # original variable preserved


@pytest.mark.parametrize("prod", ["obs/chirps-v3-daily-rhiza", "obs/imerg", "obs/ghcn"])
def test_new_entries_config_health_ok(prod):
    from acmaddl.adapters import get_adapter
    e = catalog.info(prod)
    adapter = get_adapter(e["adapter"])
    result = adapter.health_check(e, probe_remote=False)
    assert result["healthy"] is True
    assert result["kind"] == "config"


def _raw_monthly(variable="precip"):
    times = pd.date_range("2010-01-01", "2010-12-01", freq="MS")
    lat = np.arange(-3.0, 4.0, 1.0)
    lon = np.arange(29.0, 36.0, 1.0)
    data = np.random.rand(len(times), len(lat), len(lon)).astype(np.float32)
    return xr.Dataset({variable: (["time", "lat", "lon"], data)},
                      coords={"time": times, "lat": lat, "lon": lon})


def test_fetch_obs_chirps_v3_daily_rhiza_through_sheerwater():
    """End-to-end (mocked sheerwater fn): catalog -> adapter -> normalize."""
    import acmaddl
    with patch("sheerwater.data.chirps_v3",
               MagicMock(return_value=_raw_monthly("precip")), create=True):
        ds = acmaddl.fetch("obs/chirps-v3-daily-rhiza", "precip",
                           hindcast=(2010, 2010), region=[-2, 2, 30, 34],
                           cache=False, verbose=False)
    assert ds is not None
    assert ("lat" in ds.coords) or ("latitude" in ds.coords)
    da = ds["precip"] if "precip" in getattr(ds, "data_vars", {}) else ds[list(ds.data_vars)[0]]
    assert float(np.isfinite(da).mean()) > 0.0
