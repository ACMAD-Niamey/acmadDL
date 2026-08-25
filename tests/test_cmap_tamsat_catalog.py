"""ACMAD predictands from native (non-IRIDL) stores: obs/cmap (NOAA PSL OpenDAP)
and obs/tamsat (JASMIN HTTP). Added on the `acmad` branch.

The ACMAD reproduction is IRIDL-independent: the CPC-merged-precip predictand is
CMAP, served natively as NetCDF by NOAA PSL, in place of the IRIDL-only CAMS-OPI.
"""
from acmaddl import catalog


def test_cmap_registered_as_psl_opendap():
    c = catalog.get("obs/cmap")
    assert c["adapter"] == "opendap"
    assert "psl.noaa.gov" in c["source_url"]         # NOAA PSL, not IRIDL
    assert c["url_template"] == "{base}"             # whole-file THREDDS
    assert c["variables"]["precip"]["native_name"] == "precip"
    assert c["variables"]["precip"]["target_units"] == "mm/day"


def test_tamsat_registered_as_http():
    c = catalog.get("obs/tamsat")
    assert c["adapter"] == "http"
    assert c["source_url"].startswith("https://gws-access.jasmin.ac.uk")
    assert "{year}" in c["file_pattern"] and "{month" in c["file_pattern"]
    assert c["variables"]["precip"]["native_name"] == "rfe"


def test_no_iridl_dependency_for_acmad_obs_predictands():
    """The ACMAD obs predictands must not depend on the IRI Data Library."""
    for product in ("obs/cmap", "obs/tamsat"):
        assert catalog.get(product)["adapter"] != "iridl"
    assert "obs/camsopi" not in catalog.list_products()
