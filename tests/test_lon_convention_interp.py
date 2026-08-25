"""_match_lon_convention: interpolating a 0..360 source onto a seam-crossing target
grid (negative longitudes) must not NaN the western band. Added on `acmad`."""
import numpy as np
import xarray as xr
from acmaddl.fetch import _match_lon_convention


def test_0_360_source_to_negative_target_grid():
    lon = np.arange(1.25, 360, 2.5)            # 0..360 source (CMAP-like)
    lat = np.array([-2.0, 0.0, 2.0])
    da = xr.DataArray(np.ones((len(lat), len(lon))), dims=("lat", "lon"),
                      coords={"lat": lat, "lon": lon})
    target = np.arange(-25, 55.5, 0.5)         # Africa grid crossing the seam
    rolled = _match_lon_convention(da, target)
    assert float(rolled.lon.min()) < 0         # source shifted into -180..180
    interp = rolled.interp(lon=target, lat=lat)
    west = interp.sel(lon=slice(-25, -0.5))
    assert float(np.isfinite(west).mean()) > 0.8   # western band now covered


def test_noop_when_conventions_already_match():
    lon = np.arange(-179.5, 180, 1.0)
    da = xr.DataArray(np.ones((1, len(lon))), dims=("lat", "lon"),
                      coords={"lat": [0.0], "lon": lon})
    out = _match_lon_convention(da, np.arange(-25, 55.5, 0.5))
    assert np.array_equal(out.lon.values, da.lon.values)   # unchanged
