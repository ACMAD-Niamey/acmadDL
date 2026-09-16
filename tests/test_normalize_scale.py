"""Per-variable `scale`: a catalog-declared multiplier applied by normalize().

Added for ERA5-Land potential evaporation, which CDS serves in metres with a
NEGATIVE sign (ERA5's flux convention: negative = upward/evaporation). The
existing m -> mm/day conversion handles the magnitude; `scale: -1.0` handles
the sign, so the catalog can express the whole transform declaratively instead
of the fetch caller remembering to flip it.

`scale` runs AFTER sentinel masking, so a `fill_value` sentinel is still
compared against the value the source actually wrote.
"""
import numpy as np
import pandas as pd
import xarray as xr

from acmaddl.normalize import normalize


def _config(**var_extra):
    cfg = {"units": "m", "target_units": "mm/day"}
    cfg.update(var_extra)
    return {"variables": {"pev": cfg}}


def _ds(value):
    times = pd.date_range("2020-07-01", periods=1, freq="MS")
    lat = np.arange(5.0, 8.0, 1.0)
    lon = np.arange(-2.0, 1.0, 1.0)
    data = np.full((len(times), len(lat), len(lon)), value, dtype="float32")
    return xr.Dataset(
        {"pev": (["time", "latitude", "longitude"], data)},
        coords={"time": times.values.astype("datetime64[ns]"),
                "latitude": lat, "longitude": lon},
    )


def test_scale_flips_the_sign_after_the_unit_conversion():
    """-0.009 m -> x1000 -> -9.0 mm/day -> x-1 -> +9.0 mm/day."""
    out = normalize(_ds(-0.009), _config(scale=-1.0), "pev")
    np.testing.assert_allclose(float(out["pev"].mean()), 9.0, atol=1e-4)
    assert out["pev"].attrs["units"] == "mm/day"


def test_no_scale_declared_leaves_the_converted_values_untouched():
    out = normalize(_ds(-0.009), _config(), "pev")
    np.testing.assert_allclose(float(out["pev"].mean()), -9.0, atol=1e-4)


def test_scale_runs_after_sentinel_masking():
    """A sentinel is compared against the source value, not the scaled one.

    Masking after scaling would leave -999 -> +999 unmasked (or, with the
    conversion in between, never match at all), silently promoting a no-data
    sentinel to a plausible evaporation rate.
    """
    ds = _ds(-999.0)
    out = normalize(ds, _config(units="mm/day", scale=-1.0, fill_value=-999.0), "pev")
    assert bool(np.isnan(out["pev"]).all()), (
        f"sentinel survived masking: {np.unique(out['pev'].values)}")
