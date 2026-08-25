"""Deprecation aliases in the catalog.

Old product ids (e.g. nmme/spear-hindcast) are kept as `alias_of` stubs that
resolve to the canonical product (nmme/spear) so existing callers keep working,
while emitting a deprecation warning and staying out of the live product list.
"""
import warnings

import pytest

from acmaddl import catalog


def test_alias_resolves_to_target_config():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = catalog.info("nmme/spear-hindcast")
    assert any(issubclass(x.category, DeprecationWarning) for x in w), "expected a deprecation warning"
    target = catalog.info("nmme/spear")
    assert cfg["adapter"] == target["adapter"] == "ccsr"
    assert cfg["source_url"] == target["source_url"]
    assert cfg["deprecated"] is True
    assert cfg["aliased_from"] == "nmme/spear-hindcast"


@pytest.mark.parametrize("old,new", [
    ("nmme/geoss2s-hindcast", "nmme/geoss2s"),
    ("nmme/geoss2s-forecast", "nmme/geoss2s"),
    ("nmme/cansipsic4-hindcast", "nmme/cansipsic4"),
    ("nmme/spearb-hindcast", "nmme/spearb"),
    ("nmme/cfsv2-forecast", "nmme/cfsv2"),
    ("nmme/ccsm4-hindcast", "nmme/ccsm4"),
    ("nmme/ccsm4-iri", "nmme/ccsm4"),
    ("nmme/cesm1-hindcast", "nmme/cesm1"),
])
def test_known_aliases_point_at_canonical_product(old, new):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert catalog.info(old)["source_url"] == catalog.info(new)["source_url"]


def test_aliases_excluded_from_live_product_list():
    live = catalog.list_products(include_deprecated=False)
    assert "nmme/spear" in live and "nmme/geoss2s" in live
    for alias in ("nmme/spear-hindcast", "nmme/geoss2s-hindcast",
                  "nmme/cansipsic4-hindcast", "nmme/spearb-hindcast",
                  "nmme/cfsv2-forecast", "nmme/ccsm4-hindcast",
                  "nmme/cesm1-hindcast"):
        assert alias not in live, f"{alias} should be excluded as a deprecated alias"


def test_unknown_alias_target_raises():
    # Sanity: a well-formed alias must point at a real product.
    from acmaddl.catalog import _catalog
    assert all(v["alias_of"] in _catalog
               for v in _catalog.values() if "alias_of" in v)
