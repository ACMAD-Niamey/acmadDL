"""Cross-adapter contract tests for collapsed seasonal precipitation."""

from datetime import datetime

import numpy as np
import xarray as xr

from acmaddl.normalize import normalize


def _forecast(values, init_times, lead_times):
    data = np.asarray(values, dtype=float)[:, :, None, None, None]
    return xr.Dataset(
        {"pr": (("S", "L", "M", "Y", "X"), data)},
        coords={
            "S": np.asarray(init_times, dtype="datetime64[ns]"),
            "L": lead_times,
            "M": [0],
            "Y": [0.0],
            "X": [30.0],
        },
    )


def _config(**extra):
    return {
        "variables": {"precip": {
            "native_name": "pr", "units": "mm/day", "target_units": "mm/day",
        }},
        "target_range": (datetime(2011, 2, 1), datetime(2011, 4, 30)),
        **extra,
    }


def test_monthly_rates_are_calendar_weighted_to_seasonal_mm():
    raw = _forecast(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        ["2011-01-01", "2012-01-01"],
        [1, 2, 3],
    )
    out = normalize(raw, _config(leadtime_month=[2, 3, 4]), "precip", year_index=True)
    got = out["precip"].isel(member=0, lat=0, lon=0)
    np.testing.assert_allclose(got.values, [89.0, 90.0])
    assert got.attrs["units"] == "mm"


def test_daily_rates_are_summed_to_seasonal_mm():
    raw = _forecast([[2.0, 3.0, 4.0]], ["2011-01-01"], [24, 48, 72])
    out = normalize(raw, _config(leadtime_hour=[24, 48, 72]), "precip", year_index=True)
    got = out["precip"].isel(member=0, lat=0, lon=0)
    np.testing.assert_allclose(got.values, [9.0])
    assert got.attrs["units"] == "mm"


def test_lead_resolved_fetch_keeps_per_step_rate_units():
    raw = _forecast([[1.0, 1.0, 1.0]], ["2011-01-01"], [1, 2, 3])
    out = normalize(raw, _config(leadtime_month=[2, 3, 4]), "precip", year_index=False)
    assert "lead_time" in out.dims
    assert out["precip"].attrs["units"] == "mm/day"


def test_target_season_after_init_rolls_into_next_year():
    raw = _forecast([[1.0, 1.0, 1.0]], ["2011-09-01"], [7, 8, 9])
    out = normalize(raw, _config(leadtime_month=[7, 8, 9]), "precip", year_index=True)
    got = out["precip"].isel(member=0, lat=0, lon=0)
    np.testing.assert_allclose(got.values, [90.0])  # FMA 2012 includes leap day
