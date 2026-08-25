"""Adapter registry wiring.

Regression: the IcechunkAdapter existed as a file (adapters/icechunk.py) but was
never imported into adapters/__init__.py, so it was missing from _ADAPTERS and
unreachable via get_adapter(). Every adapter that ships should be registered and
constructible (heavy optional deps must be imported lazily, not at module load).
"""
from acmaddl.adapters import _ADAPTERS, get_adapter


def test_icechunk_adapter_registered():
    assert "icechunk" in _ADAPTERS
    assert type(get_adapter("icechunk")).__name__ == "IcechunkAdapter"


def test_every_registered_adapter_is_constructible():
    assert _ADAPTERS, "no adapters registered"
    for name in _ADAPTERS:
        assert get_adapter(name) is not None


def test_unknown_adapter_raises():
    import pytest
    with pytest.raises(KeyError):
        get_adapter("does-not-exist")
