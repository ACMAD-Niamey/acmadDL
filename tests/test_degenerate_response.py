"""The fetch-level degenerate-response guard (all adapters), lifted from AGU's safe_fetch.

A truncated DAP packet / zero-filled server error yields a bitwise-constant field. The guard
rejects it before it can be memoized, for every adapter — not just the OPeNDAP obs path where the
check previously lived.
"""
import numpy as np
import pytest
import xarray as xr

from acmaddl.adapters._robust import reject_if_degenerate, DegenerateResponseError


def _ds(values, var="precip"):
    a = np.asarray(values, dtype=float)
    return xr.Dataset({var: (("time", "lat", "lon"), a.reshape(a.shape[0], 1, -1))},
                      coords={"time": np.arange(a.shape[0])})


# ---- detector -----------------------------------------------------------------------

def test_rejects_all_zero_field():
    with pytest.raises(DegenerateResponseError, match="constant 'precip' field"):
        reject_if_degenerate(_ds([[0, 0, 0]] * 6), "precip", "test")


def test_rejects_constant_nonzero_field():
    with pytest.raises(DegenerateResponseError, match="constant"):
        reject_if_degenerate(_ds([[5.0, 5.0]] * 6), "precip", "test")


def test_accepts_real_varying_field():
    rng = np.random.default_rng(0)
    reject_if_degenerate(_ds(rng.normal(size=(6, 4)) + 10), "precip", "test")   # no raise


def test_single_value_is_tolerated():
    reject_if_degenerate(_ds([[7.0]]), "precip", "test")                        # size < 2 -> skip


def test_all_nan_toggle():
    ds = _ds([[np.nan, np.nan]] * 4)
    with pytest.raises(DegenerateResponseError, match="no finite values"):
        reject_if_degenerate(ds, "precip", "test", reject_all_nan=True)
    reject_if_degenerate(ds, "precip", "test", reject_all_nan=False)            # land-mask ok


def test_constant_over_finite_cells_with_some_nan():
    # all-zero over the finite (land) cells, NaN elsewhere -> still a truncation signature
    with pytest.raises(DegenerateResponseError):
        reject_if_degenerate(_ds([[0.0, np.nan], [0.0, np.nan]]), "precip", "test",
                             reject_all_nan=False)


def test_degenerate_error_is_a_runtimeerror():
    # so existing `except RuntimeError` handlers (e.g. the OPeNDAP obs-chunk path) still catch it
    assert issubclass(DegenerateResponseError, RuntimeError)


# ---- integration with fetch() via a stub adapter ------------------------------------

def test_fetch_raw_rejects_before_caching(monkeypatch):
    import importlib
    F = importlib.import_module("acmaddl.fetch")   # the module, not the shadowing fetch() function

    class _ConstAdapter:
        def fetch_data(self, config, variable, *, date_range, region):
            return _ds([[0.0, 0.0, 0.0]] * 5, var=variable)

    monkeypatch.setattr(F, "get_adapter", lambda name: _ConstAdapter())
    # opt-in: the guard only fires with reject_degenerate=True (fetch's degenerate_attempts>1)
    with pytest.raises(DegenerateResponseError):
        F._fetch_raw("nmme/x", "precip", {"adapter": "stub"}, (1993, 2016), [-5, 5, 34, 42],
                     reject_degenerate=True)
    # default (opt-out) returns the constant field unchanged — mechanics that use constant stubs
    # and callers that don't opt in are unaffected
    out = F._fetch_raw("nmme/x", "precip", {"adapter": "stub"}, (1993, 2016), [-5, 5, 34, 42])
    assert float(out["precip"].max()) == 0.0


def test_fetch_raw_accepts_good_data(monkeypatch):
    import importlib
    F = importlib.import_module("acmaddl.fetch")   # the module, not the shadowing fetch() function
    rng = np.random.default_rng(1)

    class _GoodAdapter:
        def fetch_data(self, config, variable, *, date_range, region):
            return _ds(rng.normal(size=(5, 4)) + 20, var=variable)

    monkeypatch.setattr(F, "get_adapter", lambda name: _GoodAdapter())
    out = F._fetch_raw("nmme/x", "precip", {"adapter": "stub"}, (1993, 2016), [-5, 5, 34, 42])
    assert "precip" in out
