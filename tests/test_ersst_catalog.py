"""ERSSTv5 via the OPeNDAP adapter, and the two adapter generalizations it needed.

The adapter was IRIDL-shaped in two places: it built an Ingrid variable-path URL,
and it opened every dataset with `decode_times=False` because NMME's "months
since" axis is not CF-decodable. A THREDDS-served observational file needs
neither. Both are now catalog-declared rather than assumed.
"""
import numpy as np
import pytest
import xarray as xr

import acmaddl
from acmaddl.adapters.opendap import (
    _build_url,
    _load_obs_chunks,
    _reject_degenerate,
    _sort_ascending,
)
from acmaddl.catalog import _catalog

# An SST product is NaN over land, so the sample region must be open ocean.
# The western Indian Ocean, which is also the box the WIO/IOD indices use.
REGION = [-5, 5, 55, 65]


# --- URL templating --------------------------------------------------------


def test_url_template_defaults_to_the_iridl_ingrid_form():
    url = _build_url({}, "sst", "https://iridl.ldeo.columbia.edu/SOURCES/.X")
    assert url == "https://iridl.ldeo.columbia.edu/SOURCES/.X/.sst/dods"


def test_url_template_can_be_overridden_to_a_plain_dataset_url():
    cfg = {"url_template": "{base}"}
    assert _build_url(cfg, "sst", "https://psl.noaa.gov/x/sst.mnmean.nc") == (
        "https://psl.noaa.gov/x/sst.mnmean.nc"
    )


def test_url_template_can_interpolate_the_native_variable_name():
    cfg = {"url_template": "{base}/{native_name}.nc"}
    assert _build_url(cfg, "sst", "https://h/d") == "https://h/d/sst.nc"


def test_every_opendap_entry_builds_a_url_without_leftover_placeholders():
    for name, entry in _catalog.items():
        if not isinstance(entry, dict) or entry.get("adapter") != "opendap":
            continue
        base = entry["source_url"]
        if entry.get("split_streams"):
            base = base.format(stream="HINDCAST")
        for variable, cfg in entry["variables"].items():
            url = _build_url(entry, cfg["native_name"], base)
            assert "{" not in url, f"{name}/{variable} left a placeholder: {url}"


# --- descending axes -------------------------------------------------------


def _descending():
    return xr.Dataset(
        {"sst": (("lat", "lon"), np.arange(6.0).reshape(3, 2))},
        coords={"lat": [88.0, 0.0, -88.0], "lon": [0.0, 180.0]},
    )


def test_sort_ascending_flips_a_descending_axis():
    got = _sort_ascending(_descending(), "lat", "lon")
    np.testing.assert_array_equal(got.lat.values, [-88.0, 0.0, 88.0])


def test_sort_ascending_leaves_an_ascending_axis_untouched():
    ds = _descending().sortby("lat")
    got = _sort_ascending(ds, "lat", "lon")
    np.testing.assert_array_equal(got.lat.values, ds.lat.values)
    np.testing.assert_array_equal(got.sst.values, ds.sst.values)


def test_a_descending_axis_would_otherwise_slice_to_nothing():
    """The bug this guards: an ascending slice against a descending coord returns
    an empty selection, with no error."""
    ds = _descending()
    assert ds.sel(lat=slice(-10, 10)).sizes["lat"] == 0
    assert _sort_ascending(ds, "lat").sel(lat=slice(-10, 10)).sizes["lat"] == 1


def test_sort_ascending_ignores_missing_and_scalar_coords():
    ds = _descending().isel(lat=0)
    _sort_ascending(ds, "lat", "nope")  # must not raise


# --- truncated DAP responses -----------------------------------------------


def _series(values):
    n = len(values)
    return xr.Dataset(
        {"sst": (("time", "lat"), np.repeat(np.asarray(values, float)[:, None], 2, 1))},
        coords={"time": np.array([np.datetime64(f"{1991 + i // 12}-{i % 12 + 1:02d}-01")
                                  for i in range(n)]), "lat": [0.0, 2.0]},
    )


def test_a_constant_field_is_rejected_as_a_truncated_response():
    """netCDF4 hands back a zero-filled array for an over-large DAP request,
    prints ERR to stderr, and does not raise. Twenty years of 0 degC SST with no
    land mask would otherwise flow straight into a skill metric."""
    with pytest.raises(RuntimeError, match="constant 'sst' field"):
        _reject_degenerate(_series([0.0] * 24), "sst", "test")


def test_an_all_nan_field_is_rejected():
    with pytest.raises(RuntimeError, match="no finite values"):
        _reject_degenerate(_series([np.nan] * 4), "sst", "test")


def test_real_varying_data_passes_the_guard():
    _reject_degenerate(_series(np.linspace(20, 30, 24)), "sst", "test")


def test_a_single_value_response_is_not_treated_as_degenerate():
    _reject_degenerate(_series([27.0]).isel(time=0, lat=0), "sst", "test")


def test_obs_chunks_request_the_years_in_blocks_and_reassemble_them():
    ds = _series(np.linspace(20, 30, 36))  # 1991-1993, monthly
    got = _load_obs_chunks(ds, "sst", 1991, 1993, 1, verbose=False, label="test")
    assert got.sizes["time"] == 36
    np.testing.assert_allclose(got.sst.values, ds.sst.values)


def test_obs_chunks_reject_a_truncated_block():
    """A good first year must not excuse a zeroed second one: the guard runs per
    chunk, and names the chunk that failed."""
    ds = _series(list(np.linspace(20, 30, 12)) + [0.0] * 12)
    with pytest.raises(RuntimeError, match=r"\(1992-1992\)"):
        _load_obs_chunks(ds, "sst", 1991, 1992, 1, verbose=False, label="test")


def test_the_guard_runs_before_the_result_can_be_cached():
    """`_reject_degenerate` is called inside the adapter, so a truncated response
    raises out of `_fetch_raw` and nuthatch never stores it. A corrupt fetch that
    reached the cache would keep being served long after the server recovered."""
    import inspect

    from acmaddl.adapters import opendap

    source = inspect.getsource(opendap._load_obs_chunks)
    assert "_reject_degenerate" in source
    assert "load()" in source  # values are materialized, so the check sees them


def test_obs_chunks_raise_when_the_window_is_empty():
    with pytest.raises(RuntimeError, match="no data in"):
        _load_obs_chunks(_series([25.0] * 12), "sst", 2050, 2051, 5,
                         verbose=False, label="test")


def test_ersst_declares_a_request_chunk_small_enough_for_the_psl_server():
    """Probed 2026-07-09: 120 monthly steps of the global 2-degree grid returns
    real data; 240 returns zeros. Ten years is the observed edge, so the
    catalogued chunk must be strictly under it."""
    entry = _catalog["obs/ersst-v5"]
    assert entry.get("max_request_years", 5) < 10


# --- the catalog entry -----------------------------------------------------


def test_ersst_entry_declares_what_the_opendap_adapter_needs():
    entry = _catalog["obs/ersst-v5"]
    assert entry["adapter"] == "opendap"
    assert entry["url_template"] == "{base}"
    assert entry["decode_times"] is True
    assert entry["variables"]["sst"]["native_name"] == "sst"
    # ERSST is already in Celsius, so no conversion fires; the target label is
    # acmaddl's canonical "C", matching every other temperature product.
    assert entry["variables"]["sst"]["units"] == "degC"
    assert entry["variables"]["sst"]["target_units"] == "C"


def test_ersst_is_not_declared_as_a_forecast_product():
    grid = _catalog["obs/ersst-v5"]["grid"]
    assert "forecast_range" not in grid
    assert grid["temporal"] == "monthly"


# --- live -----------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.network
def test_fetch_ersst_v5_monthly_sst():
    """Guards the PSL endpoint, the time decoding and the latitude ordering."""
    ds = acmaddl.fetch(product="obs/ersst-v5", variable="sst",
                       hindcast=(1991, 1992), region=REGION, verbose=True)
    assert "sst" in ds and ds.sizes["time"] == 24
    assert ds["sst"].attrs["units"] == "C"
    assert float(ds.lat[0]) < float(ds.lat[-1]), "latitude not sorted ascending"
    # Equatorial Indian Ocean SST: warm, and certainly not a Kelvin value.
    assert 20.0 < float(ds["sst"].mean()) < 32.0


@pytest.mark.integration
@pytest.mark.network
def test_ersst_supports_the_named_ocean_indices():
    """The reason to catalogue ERSST: it is the reference SST for ONI/RONI/IOD."""
    deepscale = pytest.importorskip("deepscale")
    ds = acmaddl.fetch(product="obs/ersst-v5", variable="sst",
                       hindcast=(1991, 2020), verbose=False, progress=False)
    sst = ds["sst"].groupby("time.year").mean("time")

    roni = deepscale.Index.named("roni").reduce(sst)
    dmi = deepscale.Index.named("dmi").reduce(sst)
    wio = deepscale.Index.named("wio").reduce(sst)

    assert roni.sizes["year"] == 30
    # RONI is an anomaly difference in degrees, so it is centred near zero and
    # bounded by a couple of degrees; 1997 was a very strong El Nino.
    assert abs(float(roni.mean())) < 0.5 and float(abs(roni).max()) < 4.0
    assert float(roni.sel(year=1997)) > float(roni.sel(year=1996))
    assert abs(float(dmi.mean())) < 0.5
    # WIO is absolute SST, the quantity the >29 C threshold is about.
    assert 26.0 < float(wio.mean()) < 30.0
