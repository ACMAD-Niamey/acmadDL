from datetime import datetime, timezone

from . import catalog
from .adapters import get_adapter
from .errors import VariableNotSupported


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def check_product(product, probe_remote=False, variable=None):
    """Return health status for one catalog product.

    Pass ``variable`` to also check that the product can serve it. A variable
    the catalog doesn't declare is reported as ``kind="capability"`` — a
    permanent property of the product, distinct from a ``config`` problem or a
    ``remote``/``transient`` outage, which is the distinction a caller needs to
    decide between "deselect this product" and "retry later". It is answerable
    from the catalog alone, so it never costs a remote probe.
    """
    # catalog.get() resolves the config and, for deprecated/alias products,
    # emits the DeprecationWarning (single source of truth — see catalog.info).
    config = catalog.get(product)

    if variable is not None:
        try:
            catalog.require_variable(product, variable, config=config)
        except VariableNotSupported as e:
            return {
                "product": product,
                "adapter": config.get("adapter", "unknown"),
                "checked_at": _utc_now(),
                "healthy": False,
                "kind": "capability",
                "message": str(e),
                "variable": e.variable,
                "available": e.available,
                "probe_remote": bool(probe_remote),
            }

    if config.get("pending_url"):
        return {
            "product": product,
            "adapter": config["adapter"],
            "checked_at": _utc_now(),
            "healthy": False,
            "kind": "config",
            "message": config.get("pending_url_note", "Source URL not yet confirmed. See issue tracker."),
            "probe_remote": bool(probe_remote),
        }

    adapter_name = config["adapter"]
    adapter = get_adapter(adapter_name)
    result = adapter.health_check(config, probe_remote=probe_remote)
    return {
        "product": product,
        "adapter": adapter_name,
        "checked_at": _utc_now(),
        **result,
    }


def check_all_products(probe_remote=False, variable=None):
    """Return health status for all catalog products.

    ``variable`` is threaded through to :func:`check_product`, so this doubles
    as a capability sweep: "which products can serve precip?"
    """
    statuses = []
    for product in catalog.list_products():
        try:
            statuses.append(check_product(product, probe_remote=probe_remote,
                                          variable=variable))
        except Exception as e:
            config = catalog.get(product)
            statuses.append(
                {
                    "product": product,
                    "adapter": config.get("adapter", "unknown"),
                    "checked_at": _utc_now(),
                    "healthy": False,
                    "kind": "runtime",
                    "message": f"Health check failed: {e}",
                    "probe_remote": bool(probe_remote),
                }
            )
    return statuses
