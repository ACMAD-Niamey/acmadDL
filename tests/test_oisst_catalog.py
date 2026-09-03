"""NOAA OISST v2.1 high-resolution SST via NOAA PSL OPeNDAP.

The 0.25° sibling of obs/ersst-v5 (2°). PSL also publishes the whole record as
a single ~2.2 GB file over plain HTTP; this entry deliberately uses the THREDDS
OPeNDAP endpoint instead so a request is subset server-side, and inherits the
`max_request_years` chunking that guards against PSL's silent DAP truncation.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from acmaddl import catalog
from acmaddl.adapters import get_adapter
from acmaddl.normalize import normalize

PRODUCT = "obs/oisst-v2-highres"


def test_uses_the_psl_opendap_endpoint_not_the_bulk_file():
    e = catalog.info(PRODUCT)
    assert e["adapter"] == "opendap"
    assert e["source_url"].startswith("https://psl.noaa.gov/thredds/dodsC/")
    assert "noaa.oisst.v2.highres" in e["source_url"]
    assert "downloads.psl.noaa.gov" not in e["source_url"]


def test_whole_file_endpoint_has_no_ingrid_variable_path():
    assert catalog.info(PRODUCT)["url_template"] == "{base}"


def test_ordinary_cf_times_are_decoded():
    assert catalog.info(PRODUCT)["decode_times"] is True


def test_requests_are_chunked_against_psl_truncation():
    """PSL zero-fills long DAP responses instead of erroring; obs/ersst-v5 and
    obs/cmap chunk for the same reason. At 0.25° a year of global data is ~64x
    the cells ERSST returns, so the chunk is correspondingly smaller."""
    e = catalog.info(PRODUCT)
    assert isinstance(e["max_request_years"], int)
    assert e["max_request_years"] <= catalog.info("obs/ersst-v5")["max_request_years"]


def test_quarter_degree_monthly_sst_only():
    e = catalog.info(PRODUCT)
    assert set(e["variables"]) == {"sst"}
    assert e["grid"]["lat_res"] == 0.25 and e["grid"]["lon_res"] == 0.25
    assert e["grid"]["temporal"] == "monthly"


def test_finer_than_its_ersst_sibling():
    """Its reason to exist next to obs/ersst-v5."""
    assert (catalog.info(PRODUCT)["grid"]["lat_res"]
            < catalog.info("obs/ersst-v5")["grid"]["lat_res"])


def test_declares_source_degc_to_canonical_celsius():
    v = catalog.info(PRODUCT)["variables"]["sst"]
    assert v["native_name"] == "sst"
    assert v["units"] == "degC" and v["target_units"] == "C"


def test_record_starts_in_1981():
    assert catalog.info(PRODUCT)["grid"]["hindcast_range"][0] == 1981


def test_config_health_ok():
    e = catalog.info(PRODUCT)
    result = get_adapter(e["adapter"]).health_check(e, probe_remote=False)
    assert result["healthy"] is True and result["kind"] == "config"


def _synthetic_oisst():
    times = pd.date_range("2020-01-01", "2020-12-01", freq="MS")
    lat = np.arange(-10.0, 10.0, 0.25)
    lon = np.arange(340.0, 360.0, 0.25)
    data = np.full((len(times), len(lat), len(lon)), 27.5, dtype="float32")
    return xr.Dataset(
        {"sst": (["time", "lat", "lon"], data)},
        coords={"time": times.values.astype("datetime64[ns]"), "lat": lat, "lon": lon},
    )


def test_normalize_keeps_celsius_without_a_kelvin_conversion():
    """degC -> C is a relabel, not a conversion; a stray K->C would land at -245."""
    out = normalize(_synthetic_oisst(), catalog.info(PRODUCT), "sst")
    assert out["sst"].attrs["units"] == "C"
    np.testing.assert_allclose(float(out["sst"].mean()), 27.5, atol=1e-4)
