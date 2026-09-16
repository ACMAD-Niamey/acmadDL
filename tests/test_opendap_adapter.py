"""IRI OPeNDAP adapter — stream routing.

The IRI NMME CFSv2 hindcast and forecast live at sibling endpoints that differ
only by a mid-path token (`.HINDCAST` vs `.FORECAST`). A `split_streams: true`
entry expresses both as one product whose `source_url` carries a `{stream}`
placeholder; the adapter picks the token from the requested years (past the
hindcast range -> forecast), mirroring the CCSR adapter's split-stream routing.

These tests capture the opened URL (no network).
"""
import numpy as np
import pytest
import xarray as xr


def _capture_url(monkeypatch):
    """Patch the adapter's xr.open_dataset to record the opened URL and return a
    trivial dataset, so URL routing can be tested without network."""
    from acmaddl.adapters import opendap as opendap_mod
    captured = {}

    def fake_open(url, **kwargs):
        captured["url"] = url
        return xr.Dataset({"prec": (("y", "x"), np.zeros((2, 2)))})

    monkeypatch.setattr(opendap_mod.xr, "open_dataset", fake_open)
    return captured


_SPLIT_CFG = {
    "adapter": "opendap",
    "split_streams": True,
    "source_url": "https://iri/SOURCES/.NMME/.NCEP-CFSv2/.{stream}/.PENTAD_SAMPLES/.MONTHLY",
    "variables": {"precip": {"native_name": "prec", "units": "mm/day", "target_units": "mm/day"}},
    "grid": {"hindcast_range": [1982, 2010]},
    "_verbose": False,
}


def test_split_streams_routes_hindcast_year_to_hindcast(monkeypatch):
    from acmaddl.adapters.opendap import OPeNDAPAdapter
    captured = _capture_url(monkeypatch)
    OPeNDAPAdapter().fetch_data(_SPLIT_CFG, "precip", date_range=(2010, 2010), region=None)
    assert captured["url"].endswith("/.NCEP-CFSv2/.HINDCAST/.PENTAD_SAMPLES/.MONTHLY/.prec/dods")
    assert "{stream}" not in captured["url"]


def test_split_streams_routes_forecast_year_to_forecast(monkeypatch):
    from acmaddl.adapters.opendap import OPeNDAPAdapter
    captured = _capture_url(monkeypatch)
    OPeNDAPAdapter().fetch_data(_SPLIT_CFG, "precip", date_range=(2024, 2024), region=None)
    assert captured["url"].endswith("/.NCEP-CFSv2/.FORECAST/.PENTAD_SAMPLES/.MONTHLY/.prec/dods")
    assert "{stream}" not in captured["url"]


def test_no_split_streams_uses_source_url_verbatim(monkeypatch):
    """Without split_streams, the variable hangs directly off source_url (the
    existing non-CFSv2 IRI behaviour is unchanged)."""
    from acmaddl.adapters.opendap import OPeNDAPAdapter
    captured = _capture_url(monkeypatch)
    cfg = {
        "adapter": "opendap",
        "source_url": "https://iri/SOURCES/.NMME/.NCEP-CFSv2/.HINDCAST/.PENTAD_SAMPLES/.MONTHLY",
        "variables": {"precip": {"native_name": "prec", "units": "mm/day", "target_units": "mm/day"}},
        "grid": {"hindcast_range": [1982, 2010]},
        "_verbose": False,
    }
    OPeNDAPAdapter().fetch_data(cfg, "precip", date_range=(2010, 2010), region=None)
    assert captured["url"].endswith("/.HINDCAST/.PENTAD_SAMPLES/.MONTHLY/.prec/dods")


def _ds_for_years(years, months_init=2):
    # S = months since 1960-01-01 for (Feb of each year)
    s = np.array([(y - 1960) * 12 + (months_init - 1) for y in years], dtype=float)
    lat = np.arange(-2, 3, 1.0); lon = np.arange(36, 41, 1.0)
    data = np.zeros((len(s), len(lat), len(lon)), dtype="float32")
    ds = xr.Dataset({"prec": (["S", "Y", "X"], data)},
                    coords={"S": s, "Y": lat, "X": lon})
    ds["S"].attrs["units"] = "months since 1960-01-01"
    return ds


def _ds_for_years_with_leads(years, months_init=1, n_members=28):
    """Real-shape IRI PENTAD_SAMPLES layout: dims (S, L, M, Y, X), L = the
    half-integer lead convention (0.5..9.5), M = ensemble member."""
    s = np.array([(y - 1960) * 12 + (months_init - 1) for y in years], dtype=float)
    L = np.arange(0.5, 10.5, 1.0)
    M = np.arange(n_members)
    lat = np.arange(-2, 3, 1.0); lon = np.arange(36, 41, 1.0)
    data = np.zeros((len(s), len(L), len(M), len(lat), len(lon)), dtype="float32")
    ds = xr.Dataset({"prec": (["S", "L", "M", "Y", "X"], data)},
                    coords={"S": s, "L": L, "M": M, "Y": lat, "X": lon})
    ds["S"].attrs["units"] = "months since 1960-01-01"
    return ds


def test_append_streams_concats_both_streams(monkeypatch):
    from acmaddl.adapters import opendap as opendap_mod
    from acmaddl.adapters.opendap import OPeNDAPAdapter
    calls = []

    def fake_open(url, **kw):
        calls.append(url)
        # hindcast URL -> 2009,2010 ; forecast URL -> 2011,2012
        if "HINDCAST" in url:
            return _ds_for_years([2009, 2010])
        return _ds_for_years([2011, 2012])

    monkeypatch.setattr(opendap_mod.xr, "open_dataset", fake_open)
    cfg = dict(_SPLIT_CFG, append_streams=True)
    out = OPeNDAPAdapter().fetch_data(cfg, "precip", date_range=(2009, 2012), region=None)
    assert any("HINDCAST" in u for u in calls) and any("FORECAST" in u for u in calls)
    # decoded later, but S length should be 4 (2009,2010,2011,2012)
    assert out.sizes["S"] == 4


def test_no_append_flag_single_stream(monkeypatch):
    from acmaddl.adapters import opendap as opendap_mod
    from acmaddl.adapters.opendap import OPeNDAPAdapter
    calls = []
    monkeypatch.setattr(opendap_mod.xr, "open_dataset",
                        lambda url, **kw: calls.append(url) or _ds_for_years([2009, 2010]))
    # spanning range but NO append_streams -> one open, hindcast only
    OPeNDAPAdapter().fetch_data(_SPLIT_CFG, "precip", date_range=(2009, 2012), region=None)
    assert len(calls) == 1 and "HINDCAST" in calls[0]


_OVERLAP_CFG = dict(_SPLIT_CFG, append_streams=True,
                     grid={"hindcast_range": [1982, 2011], "forecast_range": [2011, None]})


def test_append_streams_boundary_overlap_year_comes_from_both_streams(monkeypatch):
    """cfsv2-style boundary overlap (hindcast_range/forecast_range share 2011):
    the split year must be requested from BOTH streams, and each stream's own
    (real, server-side) S-filter contributes only the inits it actually has --
    here simulated as HINDCAST having a Feb-2011 init and FORECAST having a
    separate Aug-2011 init. Both must survive the union with no double count."""
    from acmaddl.adapters import opendap as opendap_mod
    from acmaddl.adapters.opendap import OPeNDAPAdapter
    calls = []

    def fake_open(url, **kw):
        calls.append(url)
        if "HINDCAST" in url:
            # hindcast stream: 2009, 2010, and its 2011 Jan-Mar tail (simulated Feb)
            return _ds_for_years([2009, 2010, 2011], months_init=2)
        # forecast stream: 2011 Apr-Dec (simulated Aug) through 2012
        return _ds_for_years([2011, 2012], months_init=8)

    monkeypatch.setattr(opendap_mod.xr, "open_dataset", fake_open)
    out = OPeNDAPAdapter().fetch_data(_OVERLAP_CFG, "precip", date_range=(2009, 2012), region=None)
    assert any("HINDCAST" in u for u in calls) and any("FORECAST" in u for u in calls)
    # both streams' 2011 inits (Feb from hindcast, Aug from forecast) survive:
    # 2009, 2010, 2011(Feb), 2011(Aug), 2012 = 5 distinct S values, none dropped.
    assert out.sizes["S"] == 5


def test_append_streams_dedups_shared_init_on_concat(monkeypatch):
    """Defensive: if both streams ever returned the SAME init (S value) for the
    overlap year -- not expected in practice, but must never double-count -- the
    concat must dedup rather than silently doubling that timestep's weight in
    downstream ensemble/mean math."""
    from acmaddl.adapters import opendap as opendap_mod
    from acmaddl.adapters.opendap import OPeNDAPAdapter

    def fake_open(url, **kw):
        # both streams return the identical 2011 (Feb) init plus their own year
        if "HINDCAST" in url:
            return _ds_for_years([2010, 2011], months_init=2)
        return _ds_for_years([2011, 2012], months_init=2)  # duplicate S vs hindcast's 2011

    monkeypatch.setattr(opendap_mod.xr, "open_dataset", fake_open)
    out = OPeNDAPAdapter().fetch_data(_OVERLAP_CFG, "precip", date_range=(2010, 2012), region=None)
    # 2010, 2011 (deduped, kept once), 2012 = 3, not 4
    assert out.sizes["S"] == 3
    assert len(set(out["S"].values.tolist())) == 3


def test_resolve_streams_tolerates_sparse_forecast_segment(monkeypatch):
    """A stream segment whose S-filter yields fewer inits than its nominal year
    span (e.g. FORECAST asked for 2011 but has no Jan-2011 init for a Jan-init
    request) must be tolerated -- contribute what it has, not error."""
    from acmaddl.adapters import opendap as opendap_mod
    from acmaddl.adapters.opendap import OPeNDAPAdapter

    def fake_open(url, **kw):
        if "HINDCAST" in url:
            return _ds_for_years([2010, 2011], months_init=1)  # has Jan-2011
        return _ds_for_years([2012], months_init=1)  # no Jan-2011 on forecast at all

    monkeypatch.setattr(opendap_mod.xr, "open_dataset", fake_open)
    out = OPeNDAPAdapter().fetch_data(_OVERLAP_CFG, "precip", date_range=(2010, 2012), region=None)
    # forecast segment contributed nothing for 2011 but didn't error; hindcast's
    # 2011 Jan init + 2010 + forecast's 2012 all present = 3
    assert out.sizes["S"] == 3


def test_empty_forecast_segment_with_target_lead_months_drops_L_and_concats(monkeypatch):
    """Regression test for a real bug found against the live IRI endpoint: with
    `target_lead_months` set (the normal case for any `acmaddl.fetch(..., target=)`
    call), a FORECAST segment with ZERO matching inits (real case: a Jan-2011
    request -- FORECAST has no Jan-2011 init) must still have its L dim reduced
    away just like the non-empty HINDCAST segment, so the two segments' `prec`
    arrays have the SAME number of dimensions and `xr.concat` succeeds. Before
    the fix, the empty segment could keep L at full size (or the reduction could
    corrupt the M dim), producing a 4-dims-vs-5-dims (or dim-size) concat error."""
    from acmaddl.adapters import opendap as opendap_mod
    from acmaddl.adapters.opendap import OPeNDAPAdapter

    def fake_open(url, **kw):
        if "HINDCAST" in url:
            return _ds_for_years_with_leads([2010, 2011], months_init=1)  # has Jan-2011
        # forecast stream: its only init in range is a Feb (not Jan) 2012 init,
        # so filtering to init_months=[1] over (2011,2012) matches ZERO inits --
        # the true empty-segment case (not just "fewer than nominal").
        return _ds_for_years_with_leads([2012], months_init=2)

    monkeypatch.setattr(opendap_mod.xr, "open_dataset", fake_open)
    cfg = dict(_OVERLAP_CFG, init_months=[1], target_lead_months=[3, 4, 5])  # Jan-init, MAM leads
    out = OPeNDAPAdapter().fetch_data(cfg, "precip", date_range=(2010, 2012), region=None)
    assert "L" not in out.dims, f"L should be reduced away, got dims {out.dims}"
    assert out["prec"].ndim == 4, out["prec"].dims
    # forecast segment matched ZERO inits (no Jan init in range) and contributed
    # nothing; hindcast's 2010 + 2011(Jan) = 2 total, not an error.
    assert out.sizes["S"] == 2


def test_empty_segment_drops_L_even_when_no_lead_matches_at_all(monkeypatch):
    """Tighter regression guard: even if the empty (S=0) segment's L coordinate
    fails to match ANY of the target leads (observed live against IRI -- an
    xarray/netCDF4 lazy-backend quirk at a 0-length outer dim that a numpy-backed
    fixture doesn't reproduce, so `sel_L` computes empty), the S=0 segment must
    STILL collapse its L dim so its shape matches the populated sibling segment.
    Relying on `sel_L` being non-empty is what caused the original live failure."""
    from acmaddl.adapters import opendap as opendap_mod
    from acmaddl.adapters.opendap import OPeNDAPAdapter

    def fake_open(url, **kw):
        if "HINDCAST" in url:
            return _ds_for_years_with_leads([2010, 2011], months_init=1)
        # forecast: zero Jan inits in range (empty segment), AND give it L
        # values that don't overlap the target leads at all, to force sel_L==[].
        ds = _ds_for_years_with_leads([2012], months_init=2)
        return ds.assign_coords(L=ds.L.values + 0.001)  # perturbed, no exact/near match

    monkeypatch.setattr(opendap_mod.xr, "open_dataset", fake_open)
    cfg = dict(_OVERLAP_CFG, init_months=[1], target_lead_months=[3, 4, 5])
    out = OPeNDAPAdapter().fetch_data(cfg, "precip", date_range=(2010, 2012), region=None)
    assert "L" not in out.dims, f"L should be reduced away, got dims {out.dims}"
    assert out["prec"].ndim == 4, out["prec"].dims
    assert out.sizes["S"] == 2


# ── Seam-crossing regions must become SEPARATE contiguous requests ──────────
# NOAA PSL answers a lazily seam-concatenated selection with a zero-filled
# array. Probe-verified 2026-09-02 against obs/oisst-v2-highres: an 80x90 deg
# Africa box truncated at every chunk size tried (230k values upward), while the
# same 115,200 cells/step loaded fine — up to 2.76M values — as one contiguous
# request, and loading each side of the seam separately then joining returned
# real data. So the defect is the shape of the request, not its size.
#
# The observable signature is a NON-MONOTONIC lon axis reaching the loader:
# [330..360, 0..60]. Each request must instead carry an ascending lon axis.

from acmaddl.adapters import opendap as opendap_mod  # noqa: E402


@pytest.fixture
def obs_source(monkeypatch):
    """A lazy-looking obs dataset on a 0..360 grid, plus the loads it receives."""
    import pandas as pd
    times = pd.date_range("2015-01-01", "2016-12-01", freq="MS")
    lat = np.arange(-40.0, 40.5, 1.0)
    lon = np.arange(0.0, 360.0, 1.0)
    data = np.random.default_rng(0).random(
        (len(times), len(lat), len(lon))).astype("float32")
    ds = xr.Dataset(
        {"sst": (["time", "lat", "lon"], data)},
        coords={"time": times.values.astype("datetime64[ns]"), "lat": lat, "lon": lon},
    )
    monkeypatch.setattr(opendap_mod.xr, "open_dataset", lambda *a, **k: ds)

    seen = []
    real = opendap_mod._load_obs_chunks

    def spy(sub, variable, y0, y1, *args, **kwargs):
        seen.append(sub)
        return real(sub, variable, y0, y1, *args, **kwargs)
    monkeypatch.setattr(opendap_mod, "_load_obs_chunks", spy)
    return ds, seen


OBS_CONFIG = {
    "adapter": "opendap",
    "source_url": "https://psl.example.invalid/sst.mon.mean.nc",
    "url_template": "{base}",
    "decode_times": True,
    "max_request_years": 2,
    "variables": {"sst": {"native_name": "sst", "units": "degC", "target_units": "C"}},
    "_verbose": False,
}


def _ascending(values):
    return bool(np.all(np.diff(values) > 0))


def test_a_seam_crossing_region_is_loaded_as_two_contiguous_requests(obs_source):
    _, seen = obs_source
    opendap_mod.OPeNDAPAdapter().fetch_data(
        OBS_CONFIG, "sst", date_range=(2015, 2016), region=[-40, 40, -30, 60])
    assert len(seen) == 2, f"expected one request per seam segment, got {len(seen)}"
    for sub in seen:
        assert _ascending(sub.lon.values), (
            f"request carries a non-monotonic lon axis "
            f"({sub.lon.values[:3]} .. {sub.lon.values[-3:]}) — this is the "
            f"malformed DAP request PSL zero-fills")


def test_the_two_segments_are_joined_into_the_full_requested_box(obs_source):
    source, _ = obs_source
    out = opendap_mod.OPeNDAPAdapter().fetch_data(
        OBS_CONFIG, "sst", date_range=(2015, 2016), region=[-40, 40, -30, 60])
    lons = out.lon.values
    assert (lons >= 330).any() and (lons <= 60).any()
    # Same cells select_lon would have chosen, and the real values, not zeros.
    expected = np.concatenate([np.arange(330.0, 360.0), np.arange(0.0, 61.0)])
    np.testing.assert_array_equal(np.sort(lons), np.sort(expected))
    assert float(out["sst"].std()) > 0


def test_a_contiguous_region_is_still_a_single_request(obs_source):
    _, seen = obs_source
    opendap_mod.OPeNDAPAdapter().fetch_data(
        OBS_CONFIG, "sst", date_range=(2015, 2016), region=[-40, 40, 30, 60])
    assert len(seen) == 1
    assert _ascending(seen[0].lon.values)


# ── Response-size budget on top of max_request_years ───────────────────────
# `max_request_years` cannot bound a DAP response on its own, because the size
# of one time step depends on how many cells the caller asked for. Measured
# against NOAA PSL 2026-09-02 (contiguous requests, global 0.25 deg OISST,
# 1,036,800 cells/step): 6.2M values (24.9 MB) OK, 8.3M (33.2 MB) truncated —
# i.e. a ~32 MiB server cap. A global 2-year request at max_request_years=2 is
# 24.9M values and always truncated, so the year cap needs a size companion.

from acmaddl.adapters.opendap import _load_obs_chunks, _steps_per_chunk  # noqa: E402


def _monthly(n_months, cells_lat=2, cells_lon=2, start="1991-01-01"):
    import pandas as pd
    times = pd.date_range(start, periods=n_months, freq="MS")
    lat = np.arange(float(cells_lat))
    lon = np.arange(float(cells_lon))
    data = np.arange(n_months * cells_lat * cells_lon, dtype="float32").reshape(
        n_months, cells_lat, cells_lon)
    return xr.Dataset(
        {"sst": (["time", "lat", "lon"], data)},
        coords={"time": times.values.astype("datetime64[ns]"), "lat": lat, "lon": lon},
    )


def test_a_small_request_is_bounded_by_the_year_cap():
    """4 cells/step: the size budget is nowhere near binding, so 5 years = 60."""
    window = _monthly(120)
    assert _steps_per_chunk(window, "sst", max_years=5, max_values=4_000_000) == 60


def test_a_wide_request_is_bounded_by_the_size_budget():
    """1,036,800 cells/step (global 0.25 deg) against a 4M budget -> 3 steps,
    not the 24 that max_request_years=2 would ask for."""
    window = _monthly(24, cells_lat=720, cells_lon=1440)
    assert _steps_per_chunk(window, "sst", max_years=2, max_values=4_000_000) == 3


def test_a_single_step_over_budget_still_yields_one_step():
    """Never zero: one oversized step must still be attempted (and the
    degenerate guard will report it) rather than looping forever."""
    window = _monthly(6, cells_lat=3000, cells_lon=3000)
    assert _steps_per_chunk(window, "sst", max_years=5, max_values=1000) == 1


def test_the_budget_never_exceeds_the_steps_available():
    window = _monthly(7)
    assert _steps_per_chunk(window, "sst", max_years=5, max_values=4_000_000) == 7


def test_a_size_split_reassembles_the_record_exactly():
    """The whole point: more requests, identical data, in order."""
    ds = _monthly(36)
    got = _load_obs_chunks(ds, "sst", 1991, 1993, 5, verbose=False, label="test",
                           max_values=8)   # 4 cells/step -> 2 steps per request
    assert got.sizes["time"] == 36
    np.testing.assert_array_equal(got.sst.values, ds.sst.values)
    np.testing.assert_array_equal(got.time.values, ds.time.values)


def test_a_size_split_actually_issues_more_requests(monkeypatch):
    calls = []
    monkeypatch.setattr(opendap_mod, "_reject_degenerate",
                        lambda *a, **k: calls.append(a[2]))
    ds = _monthly(36)
    _load_obs_chunks(ds, "sst", 1991, 1993, 5, verbose=False, label="test",
                     max_values=8)
    assert len(calls) == 18, f"expected 18 two-step requests, got {len(calls)}"
