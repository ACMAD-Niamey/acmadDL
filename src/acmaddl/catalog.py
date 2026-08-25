import yaml
from datetime import date
from pathlib import Path

from .errors import VariableNotSupported

_CATALOG_PATH = Path(__file__).parent / "catalog.yaml"
with open(_CATALOG_PATH) as f:
    _catalog = yaml.safe_load(f)


def _enrich(entry: dict) -> dict:
    result = dict(entry)
    deprecated_after = result.get("deprecated_after")
    if deprecated_after:
        result["deprecated"] = date.fromisoformat(deprecated_after) <= date.today()
    else:
        result["deprecated"] = False
    return result


def list_products(include_deprecated: bool = True) -> list[str]:
    if include_deprecated:
        return list(_catalog.keys())
    return [k for k, v in _catalog.items()
            if "alias_of" not in v and not _enrich(v).get("deprecated", False)]


def info(product_name: str) -> dict:
    if product_name not in _catalog:
        raise KeyError(f"Product not found: {product_name}")
    entry = _catalog[product_name]
    # Deprecation alias: an old product id that resolves to its successor's
    # config (so existing callers keep working) with a deprecation warning.
    alias = entry.get("alias_of")
    if alias:
        if alias not in _catalog:
            raise KeyError(
                f"Product {product_name!r} aliases unknown product {alias!r}"
            )
        import warnings
        warnings.warn(
            f"Product {product_name!r} is deprecated; use {alias!r} instead.",
            DeprecationWarning, stacklevel=2,
        )
        resolved = _enrich(_catalog[alias])
        resolved["deprecated"] = True
        resolved["aliased_from"] = product_name
        return resolved

    resolved = _enrich(entry)
    # Date-based deprecation: a product whose `deprecated_after` has passed.
    # Warn on every resolution (info/get is on the fetch path, fetch.py), so
    # callers get a clear notice at the point of use, not just in health checks.
    if resolved.get("deprecated"):
        import warnings
        bits = []
        if resolved.get("deprecated_after"):
            bits.append(f"scheduled sunset {resolved['deprecated_after']}")
        if resolved.get("successor"):
            bits.append(f"use {resolved['successor']} instead")
        tail = f" ({'; '.join(bits)})" if bits else ""
        msg = f"Product {product_name!r} is deprecated{tail}."
        if resolved.get("deprecation_note"):
            msg += " " + resolved["deprecation_note"]
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return resolved


get = info


def variables(product_name: str) -> list[str]:
    """The variable names a product declares, in catalog order."""
    return list(info(product_name).get("variables") or {})


def require_variable(product_name: str, variable: str, config: dict | None = None):
    """Raise :class:`VariableNotSupported` unless the product declares ``variable``.

    The catalog knows statically which variables a product carries, so a
    capability mismatch can be reported before any network I/O — and typed, so
    callers can tell it apart from a transient availability gap. Pass an
    already-resolved ``config`` to reuse it (and avoid re-emitting a
    deprecation warning for aliased products).
    """
    cfg = info(product_name) if config is None else config
    available = list(cfg.get("variables") or {})
    if variable not in available:
        raise VariableNotSupported(product_name, variable, available)
