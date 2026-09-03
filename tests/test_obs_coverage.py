"""Live audit of the newly added observational entries against their sources.

Four questions per product, none of which can be answered offline:

* **Do the units and values look right?** — `validate.check_structure` against
  the real catalog config: units present and as declared, lat/lon dims, not
  all-NaN, values physically plausible, fetched years inside declared coverage.
* **Does the FIRST year the catalog claims actually have data?** — catches
  over-claiming (a declared start earlier than the source really goes).
* **Does the year BEFORE it have none?** — catches under-claiming (years the
  source has that the catalog hides). The two together pin the boundary exactly,
  which is what caught `obs/gpcc-first-guess` really starting in 2013 and not
  2004: its 2004-2012 directories hold only legacy GrADS binaries.
* **Does the LAST year the catalog claims actually have data?** — a failure here
  means the declared end year has outrun what the source publishes.

Run:  pytest -m network tests/test_obs_coverage.py -v

Adding a product is one row in PRODUCTS. Deliberately scoped to the entries
added in this batch — pointing it at the older obs entries (`obs/cmap`,
`obs/tamsat`, `obs/ersst-v5`, `obs/cams-opi`) is a one-line change each, but
their declared ranges have not been probe-verified, so that is its own job.
"""
import pandas as pd
import pytest

import acmaddl
from acmaddl import catalog
from acmaddl.validate import check_structure

SAHEL = [8, 18, -8, 8]          # crosses the prime meridian on purpose
ATLANTIC = [-5, 15, -30, -10]   # SST is ocean-only; a land bbox is legitimately all-NaN

# product, variable, region, sample month (None = the whole year), declared units
PRODUCTS = [
    ("obs/gpcc-monitoring-v2020", "precip", SAHEL, 8, "mm/month"),
    ("obs/gpcc-first-guess", "precip", SAHEL, 8, "mm/month"),
    ("obs/gpcp-v2-3", "precip", SAHEL, 8, "mm/day"),
    ("obs/oisst-v2-highres", "sst", ATLANTIC, None, "C"),
]
IDS = [p[0] for p in PRODUCTS]

# A missing year surfaces differently per adapter: the http adapter refuses a
# partial pull (RuntimeError) after a 404, a missing NCEI year directory raises
# HTTPError, an unmatched wildcard raises FileNotFoundError, and OPeNDAP reports
# "has no data in <range>" (RuntimeError). HTTPError and FileNotFoundError are
# both OSError. Narrow enough that a broken test raises something else.
MISSING_DATA_ERRORS = (RuntimeError, OSError)


def _fetch(product, variable, region, month, years):
    kw = dict(region=region, hindcast=years, verbose=False, progress=False)
    if month is not None:
        kw["months"] = [month]
    return acmaddl.fetch(product, variable, **kw)


def _years(ds):
    return pd.DatetimeIndex(ds.time.values).year


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.parametrize("product,variable,region,month,units", PRODUCTS, ids=IDS)
def test_units_and_values_are_plausible(product, variable, region, month, units):
    config = catalog.info(product)
    lo, hi = config["grid"]["hindcast_range"]
    mid = (lo + hi) // 2                      # a year safely inside the record
    ds = _fetch(product, variable, region, month, (mid, mid))

    assert ds[variable].attrs["units"] == units, (
        f"{product} returned {ds[variable].attrs['units']!r}, catalog says {units!r}")

    checks = check_structure(ds, product, variable, config)
    failed = {name: c["detail"] for name, c in checks.items() if not c["passed"]}
    assert not failed, f"{product} failed structural checks: {failed}"


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.parametrize("product,variable,region,month,units", PRODUCTS, ids=IDS)
def test_the_declared_first_year_has_data(product, variable, region, month, units):
    lo = catalog.info(product)["grid"]["hindcast_range"][0]
    ds = _fetch(product, variable, region, month, (lo, lo))
    assert ds[variable].size > 0
    assert int(_years(ds).min()) == lo, (
        f"{product} declares coverage from {lo} but returned {int(_years(ds).min())}")


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.parametrize("product,variable,region,month,units", PRODUCTS, ids=IDS)
def test_the_year_before_the_declared_start_has_no_data(
        product, variable, region, month, units):
    """If this fails, the source has years the catalog is hiding — widen it."""
    before = catalog.info(product)["grid"]["hindcast_range"][0] - 1
    with pytest.raises(MISSING_DATA_ERRORS):
        _fetch(product, variable, region, month, (before, before))


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.parametrize("product,variable,region,month,units", PRODUCTS, ids=IDS)
def test_the_declared_last_year_has_data(product, variable, region, month, units):
    """January, not December: these products lag the present by 1-3 months, so
    the declared end year is normally still in progress."""
    hi = catalog.info(product)["grid"]["hindcast_range"][1]
    ds = _fetch(product, variable, region, 1 if month is not None else None, (hi, hi))
    assert ds[variable].size > 0
    assert int(_years(ds).max()) == hi


# ── ERA5-Land pev ───────────────────────────────────────────────────────────
# Units/plausibility only. The coverage boundary belongs to obs/era5-land-monthly,
# which predates this variable and whose 1950 start was not probe-verified here;
# and each boundary probe would be a full CDS round trip.

@pytest.mark.integration
@pytest.mark.network
@pytest.mark.cds
def test_era5_land_pev_units_and_values_are_plausible():
    product, config = "obs/era5-land-monthly", catalog.info("obs/era5-land-monthly")
    ds = _fetch(product, "pev", SAHEL, 7, (2020, 2020))
    assert ds["pev"].attrs["units"] == "mm/day"
    checks = check_structure(ds, product, "pev", config)
    failed = {name: c["detail"] for name, c in checks.items() if not c["passed"]}
    assert not failed, f"{product} pev failed structural checks: {failed}"
