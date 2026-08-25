"""Unit tests for OPeNDAP + CCSR retry behavior (Task R6).

Verifies that transient OSError from xr.open_dataset is retried by the
opendap and ccsr adapters via the shared _with_retry helper, and that a
clearly permanent HTTP 4xx is not retried.

No network required — all xr.open_dataset calls are monkeypatched.
"""
import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_opendap_ds():
    """Minimal dataset shaped like an IRI OPeNDAP response (no S coord)."""
    return xr.Dataset({"prec": (("Y", "X"), np.zeros((3, 3)))})


def _fake_ccsr_ds(init_month=2, init_years=(1991,)):
    """Minimal dataset shaped like a CCSR OPeNDAP response."""
    from tests.test_ccsr_adapter import _synthetic_ccsr  # reuse existing fixture
    return _synthetic_ccsr(init_month=init_month, init_years=init_years, native="prcp")


# Minimal product config — small retry budget so the test runs fast
_OPENDAP_CFG = {
    "adapter": "opendap",
    "source_url": "https://example/SOURCES/.MODEL",
    "variables": {"precip": {"native_name": "prec", "units": "mm/day",
                              "target_units": "mm/day"}},
    "_verbose": False,
    "_max_retries": 2,
    "_retry_backoff": 0.0,
}


# ---------------------------------------------------------------------------
# OPeNDAP retry tests
# ---------------------------------------------------------------------------

def test_opendap_retries_on_transient_oserror(monkeypatch):
    """First xr.open_dataset raises OSError (NetCDF I/O failure); adapter
    retries and succeeds on the second attempt."""
    from acmaddl.adapters import opendap as opendap_mod
    import acmaddl.adapters._robust as robust_mod
    from acmaddl.adapters.opendap import OPeNDAPAdapter

    calls = {"n": 0}

    def flaky_open(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("NetCDF: I/O failure")
        return _fake_opendap_ds()

    monkeypatch.setattr(opendap_mod.xr, "open_dataset", flaky_open)
    # Suppress actual sleeps so the test is instant.
    monkeypatch.setattr(robust_mod.time, "sleep", lambda _s: None)

    result = OPeNDAPAdapter().fetch_data(_OPENDAP_CFG, "precip",
                                          date_range=None, region=None)
    assert calls["n"] == 2          # 1 failure + 1 success
    assert "prec" in result.data_vars


def test_opendap_does_not_retry_http_4xx(monkeypatch):
    """HTTP 4xx in the exception message is permanent — adapter must not retry."""
    from acmaddl.adapters import opendap as opendap_mod
    import acmaddl.adapters._robust as robust_mod
    from acmaddl.adapters.opendap import OPeNDAPAdapter

    calls = {"n": 0}

    def four_oh_four(url, **kwargs):
        calls["n"] += 1
        raise OSError("HTTP response code: 404 Not Found")

    monkeypatch.setattr(opendap_mod.xr, "open_dataset", four_oh_four)
    monkeypatch.setattr(robust_mod.time, "sleep", lambda _s: None)

    with pytest.raises(OSError, match="404"):
        OPeNDAPAdapter().fetch_data(_OPENDAP_CFG, "precip",
                                     date_range=None, region=None)
    assert calls["n"] == 1          # no retries for a 4xx permanent error


def test_opendap_exhausts_retries_then_raises(monkeypatch):
    """After max_retries is spent, the OSError propagates to the caller."""
    from acmaddl.adapters import opendap as opendap_mod
    import acmaddl.adapters._robust as robust_mod
    from acmaddl.adapters.opendap import OPeNDAPAdapter

    calls = {"n": 0}

    def always_fail(url, **kwargs):
        calls["n"] += 1
        raise OSError("NetCDF: I/O failure")

    monkeypatch.setattr(opendap_mod.xr, "open_dataset", always_fail)
    monkeypatch.setattr(robust_mod.time, "sleep", lambda _s: None)

    cfg = dict(_OPENDAP_CFG, _max_retries=2)
    with pytest.raises(OSError, match="I/O failure"):
        OPeNDAPAdapter().fetch_data(cfg, "precip", date_range=None, region=None)
    # 1 initial attempt + 2 retries = 3 total
    assert calls["n"] == 3


# ---------------------------------------------------------------------------
# CCSR retry test (mirrors opendap, uses the main fetch_data path)
# ---------------------------------------------------------------------------

def test_ccsr_retries_on_transient_oserror(monkeypatch):
    """CCSR adapter retries a transient OSError on xr.open_dataset."""
    from acmaddl.adapters import ccsr as ccsr_mod
    import acmaddl.adapters._robust as robust_mod
    from acmaddl.adapters.ccsr import CCSRAdapter
    from datetime import datetime

    calls = {"n": 0}

    def flaky_open(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("NetCDF: I/O failure")
        # Return a minimal CCSR-shaped dataset
        S = np.array([0], dtype="int64")  # 1970-01-01 in hours
        L = np.arange(6)
        M = np.array([1, 2, 3])
        Y = np.array([-1.0, 0.0, 1.0])
        X = np.array([30.0, 35.0, 40.0])
        tgt = np.zeros((1, 6), dtype="int64")
        data = np.zeros((1, 3, 6, 3, 3), dtype="float32")
        ds = xr.Dataset(
            {"prcp": (["S", "M", "L", "Y", "X"], data),
             "target": (["S", "L"], tgt)},
            coords={"S": S, "M": M, "L": L, "Y": Y, "X": X},
        )
        ds["S"].attrs["units"] = "hours since 1960-01-01"
        ds["target"].attrs["units"] = "hours since 1960-01-01"
        return ds

    monkeypatch.setattr(ccsr_mod.xr, "open_dataset", flaky_open)
    monkeypatch.setattr(robust_mod.time, "sleep", lambda _s: None)

    cfg = {
        "adapter": "ccsr",
        "source_url": "https://example/NMME/NOAA-GFDL/SPEAR",
        "variables": {"precip": {"native_name": "prcp", "units": "mm/day",
                                  "target_units": "mm/day"}},
        "init_months": [1],   # January, matching S=0 (1960-01-01)
        "target_range": (datetime(1960, 2, 1), datetime(1960, 4, 30)),
        "_verbose": False,
        "_max_retries": 2,
        "_retry_backoff": 0.0,
    }
    result = CCSRAdapter().fetch_data(cfg, "precip", date_range=None, region=None)
    assert calls["n"] == 2          # 1 failure + 1 success
    assert "prcp" in result.data_vars
