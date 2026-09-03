"""Validation module — structural checks + comparison logic for acmadDL products.

Public API:
    validate_product()   — validate one catalog product (structural + optional comparison)
    check_structure()    — fetch-only sanity checks (dims, units, NaN, value ranges)
    compare()            — compute correlations between acmadDL output and a reference
    write_report()       — write JSON in the format require-parity-verified.sh expects
    read_report()        — load a previously written report

IRI-specific fetch logic and the REFERENCE_MAP (which IRI endpoint pairs with which
acmadDL product) live in scripts/validate_against_iri.py, outside the production
source tree. This module is provider-agnostic — it accepts a reference DataArray
from any source.
"""

import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    product: str
    variable: str
    reference: str                       # "iri", "self", "none", or custom label
    r_timeseries: float = float("nan")
    r_spatial: float = float("nan")
    status: str = "PENDING"              # PASS, CHECK, FAIL, ROS_ONLY, ERROR
    init_month: int = 2
    target: str = "MAM"
    region: list = field(default_factory=lambda: [-20, 20, 20, 55])
    hindcast: Optional[tuple] = None
    threshold: float = 0.95
    acmaddl_shape: dict = field(default_factory=dict)
    reference_shape: dict = field(default_factory=dict)
    structural_checks: dict = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = ""

    def passed(self) -> bool:
        return self.status in ("PASS", "ROS_ONLY")

    def to_report_entry(self) -> dict:
        """Format for the JSON report that require-parity-verified.sh reads."""
        return {
            "acmaddl_key": self.product,
            "r_timeseries": None if np.isnan(self.r_timeseries) else round(self.r_timeseries, 4),
            "r_spatial": None if np.isnan(self.r_spatial) else round(self.r_spatial, 4),
            "status": self.status,
            "reference": self.reference,
            "variable": self.variable,
            "init_month": self.init_month,
            "target": self.target,
            "region": self.region,
            "hindcast": list(self.hindcast) if self.hindcast else None,
            "threshold": self.threshold,
            "structural_checks": self.structural_checks,
            "error": self.error,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Structural (fetch-only) checks
# ---------------------------------------------------------------------------

_VALUE_RANGES = {
    "precip": (0.0, 500.0),      # default rate range (mm/day)
    "temp": (-90.0, 70.0),       # Celsius
    "sst": (-3.0, 45.0),         # Celsius (ERSST valid_min/max)
    "pev": (0.0, 30.0),          # potential evaporation, mm/day
}

# How far outside its declared range a value may stray before the check fails.
# Wide for temperatures, whose absolute magnitudes are large and whose K->C
# boundary is rough. Tight for a NON-NEGATIVE rate: a hair below zero is an
# interpolation/regrid artifact, but a substantially negative one is a sign or
# unit bug — ERA5 serving potential evaporation as a negative upward flux is the
# case in point, and a 50-wide slack would wave a flipped pev field straight
# through.
_RANGE_SLACK = {"precip": 1.0, "pev": 1.0}
_DEFAULT_RANGE_SLACK = 50.0


def check_structure(ds, product, variable, product_config=None):
    """Run structural sanity checks on a fetched dataset.

    Args:
        ds: xarray Dataset as returned by acmaddl.fetch()
        product: catalog key (for diagnostics)
        variable: expected variable name
        product_config: catalog config dict (optional, enables member/range checks)

    Returns:
        dict of {check_name: {"passed": bool, "detail": str}}

    Checks: ``variable_present``, ``has_units``, ``spatial_dims``,
    ``not_all_nan``, ``value_range``, ``member_count``, and one coverage check —
    ``hindcast_range`` (forecasts, informational) or ``coverage_range``
    (observations, a real pass/fail against the catalog's declared years).
    """
    checks = {}

    # Variable present
    if variable in ds:
        checks["variable_present"] = {"passed": True, "detail": f"{variable} found"}
        da = ds[variable]
    else:
        checks["variable_present"] = {
            "passed": False,
            "detail": f"{variable} not in {list(ds.data_vars)}",
        }
        return checks

    # Units
    units = da.attrs.get("units", "")
    if units:
        checks["has_units"] = {"passed": True, "detail": units}
    else:
        checks["has_units"] = {"passed": False, "detail": "no units attribute"}

    # Required spatial dims
    has_lat = "lat" in ds.dims
    has_lon = "lon" in ds.dims
    checks["spatial_dims"] = {
        "passed": has_lat and has_lon,
        "detail": f"dims={list(ds.dims)}",
    }

    # No all-NaN variable
    all_nan = bool(da.isnull().all())
    checks["not_all_nan"] = {
        "passed": not all_nan,
        "detail": "all NaN" if all_nan else "has valid data",
    }

    # Physically plausible values
    if variable in _VALUE_RANGES:
        lo, hi = _VALUE_RANGES[variable]
        if variable == "precip" and da.attrs.get("units") == "mm":
            hi = 5000.0  # accumulated target-season / multi-day precipitation
        elif variable == "precip" and da.attrs.get("units") == "mm/month":
            # Monthly TOTALS (obs/gpcc-*, obs/tamsat), not a daily rate. Gridded
            # monthly maxima reach ~2000-2500 mm in Meghalaya/Chocó, so 3000
            # leaves headroom while still catching a x30 rate/total mix-up.
            hi = 3000.0
        slack = _RANGE_SLACK.get(variable, _DEFAULT_RANGE_SLACK)
        vmin = float(da.min(skipna=True))
        vmax = float(da.max(skipna=True))
        plausible = vmin >= lo - slack and vmax <= hi + slack
        checks["value_range"] = {
            "passed": plausible,
            "detail": f"min={vmin:.2f}, max={vmax:.2f} "
                      f"(expected ~{lo}..{hi}, slack {slack})",
        }

    # Member count (if catalog config provided). A model's real-time forecast and
    # reforecast ensembles differ, so a fetched dataset's member count should match
    # one of the two declared sizes (forecast_members or hindcast_members).
    if product_config and "grid" in product_config:
        grid = product_config["grid"]
        expected = [n for n in (grid.get("forecast_members"),
                                grid.get("hindcast_members")) if n]
        if expected and "member" in ds.dims:
            actual = ds.sizes["member"]
            checks["member_count"] = {
                "passed": actual in expected,
                "detail": f"actual={actual}, expected one of {expected}",
            }

        expected_range = product_config["grid"].get("hindcast_range")
        if expected_range and "init_time" in ds.dims:
            # Forecasts: `hindcast_range` is the model's reforecast window, NOT a
            # data cap — real-time inits run past it by design (see the
            # catalog.yaml header). Report it, never fail on it.
            import pandas as pd
            times = pd.DatetimeIndex(ds.init_time.values)
            actual_years = (int(times.min().year), int(times.max().year))
            checks["hindcast_range"] = {
                "passed": True,
                "detail": f"actual={actual_years}, catalog={expected_range} "
                          f"(informational: not a cap for forecasts)",
            }
        elif expected_range and "time" in ds.dims:
            # Observations/reanalysis: for these `hindcast_range` DOES denote data
            # coverage, so data outside it means the catalog is wrong — either a
            # declared start that is too late, or an end year that has gone stale.
            import pandas as pd
            times = pd.DatetimeIndex(ds.time.values)
            actual_years = (int(times.min().year), int(times.max().year))
            lo, hi_year = expected_range
            inside = actual_years[0] >= lo and actual_years[1] <= hi_year
            checks["coverage_range"] = {
                "passed": inside,
                "detail": f"actual={actual_years}, catalog={expected_range}",
            }

    return checks


# ---------------------------------------------------------------------------
# Comparison helpers (provider-agnostic)
# ---------------------------------------------------------------------------

def regrid_to_common(da1, da2):
    """Regrid two DataArrays to the coarser common grid via linear interp."""
    lat1, lat2 = da1.lat.values, da2.lat.values

    res1 = np.median(np.diff(lat1))
    res2 = np.median(np.diff(lat2))

    if abs(res1 - res2) < 0.01 and len(lat1) == len(lat2):
        return da1, da2

    if res1 >= res2:
        target_lat, target_lon = lat1, da1.lon.values
        da2 = da2.interp(lat=target_lat, lon=target_lon, method="linear")
    else:
        target_lat, target_lon = lat2, da2.lon.values
        da1 = da1.interp(lat=target_lat, lon=target_lon, method="linear")

    return da1, da2


def compute_correlations(ros_da, ref_da):
    """Compute temporal and spatial Pearson r between two DataArrays.

    Both must have (init_time, lat, lon) dims.

    Returns:
        (r_timeseries, r_spatial) — area-mean temporal r, time-mean spatial r.
    """
    ros, ref = xr.align(ros_da, ref_da, join="inner")

    if ros.sizes.get("init_time", 0) < 3:
        return float("nan"), float("nan")

    valid = ros.notnull() & ref.notnull()
    ros = ros.where(valid)
    ref = ref.where(valid)

    r_ts = xr.corr(ros, ref, dim="init_time")
    r_ts_mean = float(r_ts.mean(skipna=True))

    r_spat = xr.corr(ros, ref, dim=["lat", "lon"])
    r_spat_mean = float(r_spat.mean(skipna=True))

    return r_ts_mean, r_spat_mean


def compare(acmaddl_da, reference_da, threshold=0.95):
    """Compare a acmadDL DataArray against a reference.

    Regrids to a common grid and computes correlations.

    Returns:
        (r_timeseries, r_spatial, status) where status is "PASS", "CHECK", or "ERROR".
    """
    ros, ref = regrid_to_common(acmaddl_da, reference_da)
    r_ts, r_spat = compute_correlations(ros, ref)

    if np.isnan(r_ts) or np.isnan(r_spat):
        return r_ts, r_spat, "ERROR"
    elif r_ts >= threshold and r_spat >= threshold:
        return r_ts, r_spat, "PASS"
    else:
        return r_ts, r_spat, "CHECK"


# ---------------------------------------------------------------------------
# acmadDL fetch helper
# ---------------------------------------------------------------------------

def _fetch_acmaddl_da(product, variable, init_month, target, hindcast, region):
    """Fetch ensemble-mean seasonal data via acmaddl.fetch(), return DataArray."""
    from .fetch import fetch as _do_fetch

    ds = _do_fetch(
        product, variable,
        init=f"2010-{init_month:02d}",
        target=target,
        hindcast=hindcast,
        region=region,
        verbose=False,
        progress=False,
        cache=False,
    )

    da = ds[variable]

    # keep_attrs=True so the `units` attribute set by normalize() survives the
    # ensemble + lead-time reductions; check_structure["has_units"] reads it.
    if "member" in da.dims:
        da = da.mean("member", keep_attrs=True)
    if "M" in da.dims:
        da = da.mean("M", keep_attrs=True)
    if "lead_time" in da.dims:
        da = da.mean("lead_time", keep_attrs=True)

    return da


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------

def validate_product(
    product: str,
    variable: str = "precip",
    reference: str = "self",
    reference_da: xr.DataArray = None,
    init_month: int = 2,
    target: str = "MAM",
    region: list = None,
    hindcast: tuple = None,
    threshold: float = 0.95,
    verbose: bool = True,
) -> ValidationResult:
    """Validate a single acmadDL catalog product.

    Args:
        product: acmadDL catalog key, e.g. "c3s/ecmwf-monthly"
        variable: variable to validate, e.g. "precip" or "temp"
        reference: label for the reference source ("iri", "self", or custom)
        reference_da: pre-fetched reference DataArray for comparison. If None
            and reference!="self", only structural checks run.
        init_month: initialization month (1-12)
        target: target season, e.g. "MAM", "OND"
        region: [lat_s, lat_n, lon_w, lon_e] or None for default
        hindcast: (start_year, end_year) or None to use catalog default
        threshold: correlation threshold for PASS (default 0.95)
        verbose: print progress

    Returns:
        ValidationResult with status, correlations, and structural checks.
    """
    if region is None:
        region = [-20, 20, 20, 55]

    result = ValidationResult(
        product=product,
        variable=variable,
        reference=reference,
        init_month=init_month,
        target=target,
        region=region,
        threshold=threshold,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Resolve hindcast range
    if hindcast is None:
        try:
            from . import catalog
            cfg = catalog.get(product)
            hr = cfg.get("grid", {}).get("hindcast_range")
            if hr:
                hindcast = tuple(hr)
        except Exception:
            pass
    if hindcast is None:
        hindcast = (2020, 2024)
    result.hindcast = hindcast

    # Fetch via acmadDL
    try:
        if verbose:
            print(f"[validate] fetching {product} via acmaddl ({hindcast[0]}-{hindcast[1]})")
        ros_da = _fetch_acmaddl_da(product, variable, init_month, target, hindcast, region)
        result.acmaddl_shape = dict(ros_da.sizes)
    except Exception as e:
        result.status = "ERROR"
        result.error = f"acmadDL fetch failed: {e}"
        if verbose:
            traceback.print_exc()
        return result

    # Structural checks (always run)
    try:
        from . import catalog
        product_config = catalog.get(product)
    except Exception:
        product_config = None

    ds_for_check = ros_da.to_dataset(name=variable)
    result.structural_checks = check_structure(ds_for_check, product, variable, product_config)

    # If self-only validation, stop here
    if reference_da is None and reference == "self":
        all_passed = all(c["passed"] for c in result.structural_checks.values())
        result.status = "PASS" if all_passed else "CHECK"
        return result

    # If no reference data provided, mark as ROS_ONLY
    if reference_da is None:
        result.status = "ROS_ONLY"
        result.reference = "none"
        return result

    # Compare against reference
    try:
        result.reference_shape = dict(reference_da.sizes)
        r_ts, r_spat, status = compare(ros_da, reference_da, threshold)
        result.r_timeseries = r_ts
        result.r_spatial = r_spat
        result.status = status
    except Exception as e:
        result.status = "ERROR"
        result.error = f"Comparison failed: {e}"
        if verbose:
            traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Report I/O
# ---------------------------------------------------------------------------

DEFAULT_REPORT_PATH = "output/validation_report.json"


def write_report(results: list, path: str = None):
    """Write validation results as JSON in the format require-parity-verified.sh expects.

    The hook reads entries with keys: acmaddl_key, r_timeseries, status.

    Args:
        results: list of ValidationResult
        path: output path. Defaults to lab/acmaddl/output/validation_report.json

    Returns:
        str: the path written to
    """
    if path is None:
        path = str(Path(__file__).parent.parent.parent / DEFAULT_REPORT_PATH)

    entries = [r.to_report_entry() for r in results]

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(entries, f, indent=2, default=str)

    return str(out)


def read_report(path: str = None) -> list:
    """Read a previously written validation report."""
    if path is None:
        path = str(Path(__file__).parent.parent.parent / DEFAULT_REPORT_PATH)
    with open(path) as f:
        return json.load(f)
