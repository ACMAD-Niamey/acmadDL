"""Tests for acmaddl.validate — structural checks, comparison, and report I/O.

Synthetic tests (no network):  pytest tests/test_validate.py
Integration tests (network):   pytest tests/test_validate.py -m integration
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from acmaddl.validate import (
    ValidationResult,
    check_structure,
    compare,
    compute_correlations,
    read_report,
    regrid_to_common,
    validate_product,
    write_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def good_gcm_ds():
    """A well-formed GCM hindcast dataset after normalization."""
    init_times = pd.to_datetime([f"{y}-02-01" for y in range(2000, 2010)])
    members = np.arange(5)
    lat = np.arange(-10, 11, 1.0)
    lon = np.arange(30, 51, 1.0)
    shape = (len(init_times), len(members), len(lat), len(lon))
    data = np.random.rand(*shape).astype(np.float32) * 10
    ds = xr.Dataset(
        {"precip": (["init_time", "member", "lat", "lon"], data)},
        coords={
            "init_time": init_times,
            "member": members,
            "lat": lat,
            "lon": lon,
        },
    )
    ds["precip"].attrs["units"] = "mm/day"
    return ds


@pytest.fixture
def all_nan_ds():
    """Dataset where the variable is entirely NaN."""
    lat = np.arange(-5, 6, 1.0)
    lon = np.arange(30, 41, 1.0)
    data = np.full((len(lat), len(lon)), np.nan, dtype=np.float32)
    ds = xr.Dataset(
        {"precip": (["lat", "lon"], data)},
        coords={"lat": lat, "lon": lon},
    )
    ds["precip"].attrs["units"] = "mm/day"
    return ds


@pytest.fixture
def missing_var_ds():
    """Dataset missing the expected variable."""
    lat = np.arange(-5, 6, 1.0)
    lon = np.arange(30, 41, 1.0)
    data = np.ones((len(lat), len(lon)), dtype=np.float32)
    return xr.Dataset(
        {"wrong_name": (["lat", "lon"], data)},
        coords={"lat": lat, "lon": lon},
    )


@pytest.fixture
def no_units_ds():
    """Dataset where the variable has no units attribute."""
    lat = np.arange(-5, 6, 1.0)
    lon = np.arange(30, 41, 1.0)
    data = np.ones((len(lat), len(lon)), dtype=np.float32) * 5.0
    return xr.Dataset(
        {"precip": (["lat", "lon"], data)},
        coords={"lat": lat, "lon": lon},
    )


@pytest.fixture
def implausible_ds():
    """Dataset with physically implausible precip values (negative)."""
    lat = np.arange(-5, 6, 1.0)
    lon = np.arange(30, 41, 1.0)
    data = np.full((len(lat), len(lon)), -999.0, dtype=np.float32)
    ds = xr.Dataset(
        {"precip": (["lat", "lon"], data)},
        coords={"lat": lat, "lon": lon},
    )
    ds["precip"].attrs["units"] = "mm/day"
    return ds


@pytest.fixture
def correlated_pair():
    """Two DataArrays that are perfectly correlated (r=1.0)."""
    init_times = pd.to_datetime([f"{y}-02-01" for y in range(2000, 2015)])
    lat = np.arange(-5, 6, 1.0)
    lon = np.arange(30, 41, 1.0)
    shape = (len(init_times), len(lat), len(lon))
    data = np.random.rand(*shape).astype(np.float32) * 10
    coords = {"init_time": init_times, "lat": lat, "lon": lon}
    da1 = xr.DataArray(data, dims=["init_time", "lat", "lon"], coords=coords)
    da2 = da1.copy(deep=True)
    return da1, da2


@pytest.fixture
def uncorrelated_pair():
    """Two DataArrays with independent random data."""
    np.random.seed(42)
    init_times = pd.to_datetime([f"{y}-02-01" for y in range(2000, 2015)])
    lat = np.arange(-5, 6, 1.0)
    lon = np.arange(30, 41, 1.0)
    shape = (len(init_times), len(lat), len(lon))
    coords = {"init_time": init_times, "lat": lat, "lon": lon}
    da1 = xr.DataArray(
        np.random.rand(*shape).astype(np.float32),
        dims=["init_time", "lat", "lon"], coords=coords,
    )
    da2 = xr.DataArray(
        np.random.rand(*shape).astype(np.float32),
        dims=["init_time", "lat", "lon"], coords=coords,
    )
    return da1, da2


# ---------------------------------------------------------------------------
# check_structure tests
# ---------------------------------------------------------------------------

class TestCheckStructure:
    def test_good_dataset_passes_all(self, good_gcm_ds):
        checks = check_structure(good_gcm_ds, "test/product", "precip")
        assert checks["variable_present"]["passed"]
        assert checks["has_units"]["passed"]
        assert checks["spatial_dims"]["passed"]
        assert checks["not_all_nan"]["passed"]
        assert checks["value_range"]["passed"]

    def test_missing_variable_fails(self, missing_var_ds):
        checks = check_structure(missing_var_ds, "test/product", "precip")
        assert not checks["variable_present"]["passed"]
        assert len(checks) == 1

    def test_all_nan_fails(self, all_nan_ds):
        checks = check_structure(all_nan_ds, "test/product", "precip")
        assert not checks["not_all_nan"]["passed"]

    def test_no_units_fails(self, no_units_ds):
        checks = check_structure(no_units_ds, "test/product", "precip")
        assert not checks["has_units"]["passed"]

    def test_implausible_values_fail(self, implausible_ds):
        checks = check_structure(implausible_ds, "test/product", "precip")
        assert not checks["value_range"]["passed"]

    def test_seasonal_mm_uses_accumulation_range(self, good_gcm_ds):
        ds = good_gcm_ds.copy(deep=True)
        ds["precip"][:] = 1000.0
        ds["precip"].attrs["units"] = "mm"
        checks = check_structure(ds, "test/product", "precip")
        assert checks["value_range"]["passed"]

    def test_member_count_check(self, good_gcm_ds):
        # actual member count matches one of the two declared ensemble sizes
        config = {"grid": {"forecast_members": 25, "hindcast_members": 5}}
        checks = check_structure(good_gcm_ds, "test/product", "precip", config)
        assert checks["member_count"]["passed"]

    def test_member_count_mismatch(self, good_gcm_ds):
        # actual (5) matches neither declared size
        config = {"grid": {"forecast_members": 10, "hindcast_members": 12}}
        checks = check_structure(good_gcm_ds, "test/product", "precip", config)
        assert not checks["member_count"]["passed"]


# ---------------------------------------------------------------------------
# Correlation tests
# ---------------------------------------------------------------------------

class TestCorrelations:
    def test_perfect_correlation(self, correlated_pair):
        da1, da2 = correlated_pair
        r_ts, r_spat = compute_correlations(da1, da2)
        assert r_ts == pytest.approx(1.0, abs=1e-6)
        assert r_spat == pytest.approx(1.0, abs=1e-6)

    def test_uncorrelated_low_r(self, uncorrelated_pair):
        da1, da2 = uncorrelated_pair
        r_ts, r_spat = compute_correlations(da1, da2)
        assert abs(r_ts) < 0.5
        assert abs(r_spat) < 0.5

    def test_insufficient_time_returns_nan(self):
        lat = np.arange(-2, 3, 1.0)
        lon = np.arange(30, 35, 1.0)
        init_times = pd.to_datetime(["2010-02-01", "2011-02-01"])
        shape = (2, len(lat), len(lon))
        coords = {"init_time": init_times, "lat": lat, "lon": lon}
        da1 = xr.DataArray(np.ones(shape), dims=["init_time", "lat", "lon"], coords=coords)
        da2 = da1.copy(deep=True)
        r_ts, r_spat = compute_correlations(da1, da2)
        assert np.isnan(r_ts)


class TestCompare:
    def test_identical_passes(self, correlated_pair):
        da1, da2 = correlated_pair
        r_ts, r_spat, status = compare(da1, da2, threshold=0.95)
        assert status == "PASS"
        assert r_ts >= 0.95

    def test_uncorrelated_does_not_pass(self, uncorrelated_pair):
        da1, da2 = uncorrelated_pair
        r_ts, r_spat, status = compare(da1, da2, threshold=0.95)
        assert status in ("CHECK", "ERROR")


class TestRegrid:
    def test_same_grid_noop(self, correlated_pair):
        da1, da2 = correlated_pair
        r1, r2 = regrid_to_common(da1, da2)
        assert r1.sizes == da1.sizes
        assert r2.sizes == da2.sizes

    def test_different_grids_regridded(self):
        lat1 = np.arange(-10, 11, 1.0)
        lon1 = np.arange(30, 51, 1.0)
        lat2 = np.arange(-10, 11, 2.0)
        lon2 = np.arange(30, 51, 2.0)
        init_times = pd.to_datetime([f"{y}-02-01" for y in range(2000, 2005)])
        da1 = xr.DataArray(
            np.random.rand(len(init_times), len(lat1), len(lon1)),
            dims=["init_time", "lat", "lon"],
            coords={"init_time": init_times, "lat": lat1, "lon": lon1},
        )
        da2 = xr.DataArray(
            np.random.rand(len(init_times), len(lat2), len(lon2)),
            dims=["init_time", "lat", "lon"],
            coords={"init_time": init_times, "lat": lat2, "lon": lon2},
        )
        r1, r2 = regrid_to_common(da1, da2)
        assert r1.sizes["lat"] == r2.sizes["lat"]
        assert r1.sizes["lon"] == r2.sizes["lon"]


# ---------------------------------------------------------------------------
# ValidationResult tests
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_passed_statuses(self):
        assert ValidationResult(product="x", variable="p", reference="iri", status="PASS").passed()
        assert ValidationResult(product="x", variable="p", reference="none", status="ROS_ONLY").passed()
        assert not ValidationResult(product="x", variable="p", reference="iri", status="CHECK").passed()
        assert not ValidationResult(product="x", variable="p", reference="iri", status="ERROR").passed()

    def test_report_entry_format(self):
        r = ValidationResult(
            product="c3s/ecmwf-monthly", variable="precip", reference="iri",
            r_timeseries=0.9876, r_spatial=0.9912, status="PASS",
            init_month=2, target="MAM", region=[-20, 20, 20, 55],
            hindcast=(1993, 2016), threshold=0.95,
            timestamp="2026-04-16T00:00:00+00:00",
        )
        entry = r.to_report_entry()
        assert entry["acmaddl_key"] == "c3s/ecmwf-monthly"
        assert entry["r_timeseries"] == 0.9876
        assert entry["status"] == "PASS"

    def test_nan_r_becomes_none_in_report(self):
        r = ValidationResult(product="x", variable="p", reference="none", status="ROS_ONLY")
        entry = r.to_report_entry()
        assert entry["r_timeseries"] is None
        assert entry["r_spatial"] is None


# ---------------------------------------------------------------------------
# Report I/O tests
# ---------------------------------------------------------------------------

class TestReportIO:
    def test_write_and_read_roundtrip(self):
        results = [
            ValidationResult(
                product="c3s/ecmwf-monthly", variable="precip", reference="iri",
                r_timeseries=1.0, r_spatial=1.0, status="PASS",
                timestamp="2026-04-16T00:00:00+00:00",
            ),
            ValidationResult(
                product="c3s/ukmo", variable="precip", reference="none",
                status="ROS_ONLY",
                timestamp="2026-04-16T00:00:00+00:00",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.json")
            written = write_report(results, path)
            assert Path(written).exists()

            data = read_report(path)
            assert len(data) == 2
            assert data[0]["acmaddl_key"] == "c3s/ecmwf-monthly"
            assert data[0]["r_timeseries"] == 1.0
            assert data[0]["status"] == "PASS"
            assert data[1]["status"] == "ROS_ONLY"

    def test_hook_compatible_format(self):
        """The JSON matches what require-parity-verified.sh parses."""
        results = [
            ValidationResult(
                product="nmme/cfsv2", variable="precip", reference="iri",
                r_timeseries=0.9999, r_spatial=0.9999, status="PASS",
                timestamp="2026-04-16T00:00:00+00:00",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.json")
            write_report(results, path)
            with open(path) as f:
                data = json.load(f)

            assert isinstance(data, list)
            entry = data[0]
            assert "acmaddl_key" in entry
            assert "r_timeseries" in entry
            assert "status" in entry
            assert isinstance(entry["r_timeseries"], float)
            assert entry["r_timeseries"] >= 0.95


# ---------------------------------------------------------------------------
# validate_product (synthetic, no network)
# ---------------------------------------------------------------------------

class TestValidateProductSynthetic:
    def test_self_validation_with_mock(self, monkeypatch, good_gcm_ds):
        """validate_product with reference='self' runs structural checks only."""
        def mock_fetch(*args, **kwargs):
            return good_gcm_ds
        monkeypatch.setattr("acmaddl.validate._fetch_acmaddl_da",
                            lambda *a, **kw: good_gcm_ds["precip"].mean("member", keep_attrs=True))

        result = validate_product(
            "nmme/cfsv2", variable="precip", reference="self",
            hindcast=(2000, 2009), verbose=False,
        )
        assert result.status == "PASS"
        assert result.structural_checks["variable_present"]["passed"]
        assert np.isnan(result.r_timeseries)

    def test_comparison_with_reference_da(self, monkeypatch, correlated_pair):
        """validate_product with reference_da runs comparison."""
        da1, da2 = correlated_pair
        monkeypatch.setattr("acmaddl.validate._fetch_acmaddl_da", lambda *a, **kw: da1)

        result = validate_product(
            "nmme/cfsv2", variable="precip", reference="iri",
            reference_da=da2, hindcast=(2000, 2014), verbose=False,
        )
        assert result.status == "PASS"
        assert result.r_timeseries == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Integration tests (require network + credentials)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.network
def test_validate_cfsv2_self():
    """Structural validation of CFSv2 via live fetch."""
    result = validate_product(
        "nmme/cfsv2", variable="precip", reference="self",
        init_month=2, target="MAM", region=[-2, 2, 36, 40],
        hindcast=(2008, 2010), verbose=True,
    )
    assert result.status == "PASS"
    assert result.structural_checks["variable_present"]["passed"]
    assert result.structural_checks["value_range"]["passed"]


# ── Monthly-total precip and observational coverage ─────────────────────────
# Added with the GPCC entries. Two gaps showed up:
#   * the precip value range is a mm/day rate range, so a legitimate mm/month
#     total from a monsoon region reads as implausible
#   * the hindcast_range check only looked at `init_time` and hard-coded
#     passed=True, so an observational product's declared coverage was never
#     actually checked against the data that came back

import numpy as np
import pandas as pd
import xarray as xr

from acmaddl.validate import check_structure


def _obs(years, value=100.0, units="mm/month", variable="precip"):
    times = pd.to_datetime([f"{y}-08-01" for y in years])
    lat = np.arange(8.0, 12.0, 1.0)
    lon = np.arange(-2.0, 2.0, 1.0)
    data = np.full((len(times), len(lat), len(lon)), value, dtype="float32")
    ds = xr.Dataset(
        {variable: (["time", "lat", "lon"], data)},
        coords={"time": times.values.astype("datetime64[ns]"), "lat": lat, "lon": lon},
    )
    ds[variable].attrs["units"] = units
    return ds


def _cfg(hindcast_range):
    return {"grid": {"hindcast_range": list(hindcast_range)},
            "variables": {"precip": {}}}


def test_a_monsoon_monthly_total_is_plausible():
    """1200 mm/month is a real monsoon month, not a broken unit conversion.

    Against the mm/day rate ceiling it reads as implausible, which would make
    the check cry wolf on every obs/gpcc-* or obs/tamsat fetch over South Asia.
    """
    checks = check_structure(_obs([2020], value=1200.0), "obs/gpcc-monitoring-v2020",
                             "precip", _cfg((1982, 2026)))
    assert checks["value_range"]["passed"], checks["value_range"]["detail"]


def test_an_absurd_monthly_total_is_still_rejected():
    """The ceiling must still catch a genuine unit or scale error."""
    checks = check_structure(_obs([2020], value=50_000.0), "obs/gpcc-monitoring-v2020",
                             "precip", _cfg((1982, 2026)))
    assert not checks["value_range"]["passed"]


def test_a_daily_rate_keeps_the_stricter_rate_ceiling():
    """Regression: mm/day products must not inherit the monthly headroom."""
    checks = check_structure(_obs([2020], value=1200.0, units="mm/day"),
                             "obs/gpcp-v2-3", "precip", _cfg((1979, 2026)))
    assert not checks["value_range"]["passed"]


def test_observed_years_inside_the_declared_coverage_pass():
    checks = check_structure(_obs([2000, 2010, 2020]), "obs/gpcc-monitoring-v2020",
                             "precip", _cfg((1982, 2026)))
    assert checks["coverage_range"]["passed"], checks["coverage_range"]["detail"]


def test_data_earlier_than_the_declared_start_is_flagged():
    """The catalog claims 1982-; data from 1975 means the declared start is wrong."""
    checks = check_structure(_obs([1975, 2020]), "obs/gpcc-monitoring-v2020",
                             "precip", _cfg((1982, 2026)))
    assert not checks["coverage_range"]["passed"]
    assert "1975" in checks["coverage_range"]["detail"]


def test_data_later_than_the_declared_end_is_flagged():
    """Catches a declared end year that has gone stale."""
    checks = check_structure(_obs([2020, 2026]), "obs/gpcc-monitoring-v2020",
                             "precip", _cfg((1982, 2025)))
    assert not checks["coverage_range"]["passed"]


def test_a_forecasts_hindcast_range_stays_informational():
    """`hindcast_range` is NOT a cap for forecasts — real-time inits run past it
    by design (see the catalog.yaml header), so it must never be a failure."""
    times = pd.to_datetime(["2024-02-01"])
    ds = xr.Dataset(
        {"precip": (["init_time", "lat", "lon"], np.full((1, 2, 2), 3.0, dtype="float32"))},
        coords={"init_time": times.values.astype("datetime64[ns]"),
                "lat": [8.0, 9.0], "lon": [1.0, 2.0]},
    )
    ds["precip"].attrs["units"] = "mm/day"
    checks = check_structure(ds, "nmme/cfsv2", "precip", _cfg((1982, 2011)))
    assert checks["hindcast_range"]["passed"]
    assert "coverage_range" not in checks


# ── pev, and a slack that doesn't swallow a sign flip ───────────────────────
# The plausibility check allowed ±50 outside every variable's range. That slack
# suits temp/sst (wide absolute magnitudes) but makes the check useless for a
# NON-NEGATIVE rate: -9 mm/day of potential evaporation — exactly the ERA5
# sign-convention regression the pev entry guards against — sat comfortably
# inside `0 - 50`.

def test_a_positive_pev_rate_is_plausible():
    ds = _obs([2020], value=9.0, units="mm/day", variable="pev")
    checks = check_structure(ds, "obs/era5-land-monthly", "pev",
                             {"grid": {"hindcast_range": [1950, 2026]},
                              "variables": {"pev": {}}})
    assert checks["value_range"]["passed"], checks["value_range"]["detail"]


def test_a_negative_pev_rate_is_flagged():
    """The ERA5 sign convention leaking through would look exactly like this."""
    ds = _obs([2020], value=-9.0, units="mm/day", variable="pev")
    checks = check_structure(ds, "obs/era5-land-monthly", "pev",
                             {"grid": {"hindcast_range": [1950, 2026]},
                              "variables": {"pev": {}}})
    assert not checks["value_range"]["passed"], checks["value_range"]["detail"]


def test_negative_precipitation_is_flagged():
    """Negative rainfall is never physical, whatever the slack."""
    checks = check_structure(_obs([2020], value=-20.0, units="mm/day"),
                             "obs/gpcp-v2-3", "precip", _cfg((1979, 2026)))
    assert not checks["value_range"]["passed"]


def test_a_slightly_negative_interpolation_artifact_is_tolerated():
    """Regridding/interpolation can undershoot zero by a hair; don't cry wolf."""
    checks = check_structure(_obs([2020], value=-0.4, units="mm/day"),
                             "obs/gpcp-v2-3", "precip", _cfg((1979, 2026)))
    assert checks["value_range"]["passed"], checks["value_range"]["detail"]


def test_extreme_but_real_cold_still_passes():
    """Regression: temp keeps its wide slack (Vostok has hit -89.2 C)."""
    checks = check_structure(_obs([2020], value=-89.2, units="C", variable="temp"),
                             "obs/era5", "temp",
                             {"grid": {"hindcast_range": [1940, 2026]},
                              "variables": {"temp": {}}})
    assert checks["value_range"]["passed"], checks["value_range"]["detail"]
