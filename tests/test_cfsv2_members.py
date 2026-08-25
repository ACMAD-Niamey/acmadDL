"""Generic member reduction plus CFSv2's native-member contract."""
import numpy as np
import xarray as xr
from datetime import datetime

from acmaddl.normalize import normalize
from acmaddl import catalog


def _member_ds(n_members, n_time=1):
    """A minimal (S, L, M, Y, X) NMME-style raw dataset with n_members members."""
    # months since 1960-01-01 -> Feb 2010, Feb 2011, ...
    s_vals = 601.0 + 12 * np.arange(n_time)
    l_vals = np.array([0.5])
    m_vals = np.arange(n_members)
    lat = np.arange(-2, 3, 1.0)
    lon = np.arange(36, 41, 1.0)
    shape = (len(s_vals), len(l_vals), len(m_vals), len(lat), len(lon))
    # member index broadcast into the data so a mean over a known slice is
    # checkable: value at member i is i (float), so mean(0..23) = 11.5
    data = np.broadcast_to(
        m_vals.astype(np.float32)[None, None, :, None, None], shape
    ).copy()
    ds = xr.Dataset(
        {"prec": (["S", "L", "M", "Y", "X"], data)},
        coords={"S": s_vals, "L": l_vals, "M": m_vals, "Y": lat, "X": lon},
    )
    ds["S"].attrs["units"] = "months since 1960-01-01"
    return ds


_CFSV2_LIKE_CONFIG = {
    "variables": {
        "precip": {
            "native_name": "prec",
            "units": "mm/day",
            "target_units": "mm/day",
        }
    },
    "member_reduce": {"first": 24, "op": "mean"},
}

_NO_REDUCE_CONFIG = {
    "variables": {
        "precip": {
            "native_name": "prec",
            "units": "mm/day",
            "target_units": "mm/day",
        }
    },
}


def test_member_reduce_averages_first_24_members():
    """With member_reduce: {first: 24, op: mean}, 28 members collapse to the
    mean of the first 24 (native order), matching
    isel(member=slice(0, 24)).mean("member")."""
    raw = _member_ds(28)
    clean = normalize(raw, _CFSV2_LIKE_CONFIG, "precip")
    da = clean["precip"]

    assert "member" in da.dims
    assert da.sizes["member"] == 1
    # member values are 0..27 broadcast; mean of first 24 (0..23) = 11.5
    expected = np.mean(np.arange(24))
    np.testing.assert_allclose(da.isel(member=0).values, expected)


def test_member_reduce_matches_manual_isel_mean():
    """Directly matches the reference's isel(member=slice(0,24)).mean('member')."""
    raw = _member_ds(28)
    clean = normalize(raw, _CFSV2_LIKE_CONFIG, "precip")
    got = clean["precip"]

    baseline_raw = _member_ds(28)
    baseline = normalize(baseline_raw, _NO_REDUCE_CONFIG, "precip")
    want = (
        baseline["precip"]
        .isel(member=slice(0, 24))
        .mean("member")
        .expand_dims(member=[0])
    )
    xr.testing.assert_allclose(
        got.transpose(*want.dims), want, rtol=0, atol=1e-6
    )


def test_member_reduce_applies_under_year_index():
    """The reduce must run whether or not year_index reshaping is requested,
    since assemble()/prepare_inputs() both fetch with year_index=True."""
    raw = _member_ds(28)
    clean = normalize(raw, _CFSV2_LIKE_CONFIG, "precip", year_index=True)
    da = clean["precip"]

    assert "member" in da.dims
    assert da.sizes["member"] == 1
    expected = np.mean(np.arange(24))
    np.testing.assert_allclose(da.isel(member=0).values, expected)


def test_member_reduce_absent_is_a_no_op():
    """Backward-compat: a product WITHOUT member_reduce keeps all its members
    (proves the knob is opt-in and default behaviour is unchanged)."""
    raw = _member_ds(28)
    clean = normalize(raw, _NO_REDUCE_CONFIG, "precip")
    da = clean["precip"]

    assert "member" in da.dims
    assert da.sizes["member"] == 28
    np.testing.assert_allclose(sorted(da["member"].values), np.arange(28))


def test_cfsv2_catalog_preserves_all_usable_members():
    """CFSv2 keeps its 24 populated members instead of averaging them."""
    config = catalog.get("nmme/cfsv2")
    assert "member_reduce" not in config
    clean = normalize(_member_ds(28), config, "precip")
    assert clean.sizes["member"] == 28
    assert clean["precip"].attrs["units"] == "mm/day"


def test_cfsv2_drops_only_structurally_empty_member_slots():
    config = catalog.get("nmme/cfsv2")
    raw = _member_ds(28)
    raw["prec"].loc[{"M": [24, 25, 26, 27]}] = np.nan
    clean = normalize(raw, config, "precip")
    assert clean.sizes["member"] == 24
    np.testing.assert_array_equal(clean.member.values, np.arange(24))


def test_cfsv2_target_converts_daily_rate_to_season_total():
    """A targeted CFSv2 rate is multiplied by the exact target day count."""
    config = dict(catalog.get("nmme/cfsv2"))
    config["target_range"] = (datetime(2010, 10, 1), datetime(2010, 12, 31))
    raw = _member_ds(28)
    raw["prec"][:] = 2.0

    clean = normalize(raw, config, "precip")
    assert clean["precip"].attrs["units"] == "mm"
    np.testing.assert_allclose(clean["precip"].values, 184.0)


def test_cfsv2_season_total_respects_each_hindcast_years_calendar():
    """FMA has 90 days in leap-year 2012 and 89 in 2010/2011."""
    config = dict(catalog.get("nmme/cfsv2"))
    config["target_range"] = (datetime(2010, 2, 1), datetime(2010, 4, 30))
    raw = _member_ds(1, n_time=3)
    raw["prec"][:] = 1.0

    clean = normalize(raw, config, "precip")
    got = clean["precip"].isel(lead_time=0, member=0, lat=0, lon=0)
    np.testing.assert_allclose(got.values, [89.0, 89.0, 90.0])


def test_cfsv2_collapsed_rate_rolls_future_target_into_next_year():
    config = dict(catalog.get("nmme/cfsv2"))
    config["target_range"] = (datetime(2011, 2, 1), datetime(2011, 4, 30))
    raw = _member_ds(1).assign_coords(S=[620.0]).mean("L", keep_attrs=True)  # Sep 2011 init
    raw["S"].attrs["units"] = "months since 1960-01-01"
    raw["prec"][:] = 1.0

    clean = normalize(raw, config, "precip")
    got = clean["precip"].isel(member=0, lat=0, lon=0)
    np.testing.assert_allclose(got.values, [90.0])  # FMA 2012 includes leap day
