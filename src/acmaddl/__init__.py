"""acmadDL — federated data integration for seasonal climate forecasting.

Provides a unified fetch() API across CDS, OPeNDAP, NCEI, S3, HTTP, and
Sheerwater data sources. All outputs are CF-aligned xarray Datasets.

Quick start:
    import acmaddl
    ds = acmaddl.fetch("c3s/ecmwf", variable="precip", init="2025-02",
                       target="MAM", region=[-12, 6, 28, 42], hindcast=(2010, 2015))
"""

import os as _os
from pathlib import Path as _Path

# Pin nuthatch's cache to a local directory at import time.
#
# nuthatch discovers its config by walking *up* the filesystem from the calling
# module and adopting the first pyproject.toml / nuthatch.toml it finds. When
# acmaddl is pip-installed, that search leaves site-packages/acmaddl/ and can pick
# up a nuthatch.toml shipped by a co-installed package — notably `sheerwater`, which
# points the cache root at a private GCS bucket (gs://sheerwater-datalake/...).
# Anyone without those credentials then hits a 401 / interactive prompt on their
# first fetch(). We defend against that on two fronts:
#   1. Ship src/acmaddl/nuthatch.toml (see there) so the upward search finds *our*
#      local config first and never reaches the ambient one.
#   2. Pin the root/local cache filesystem here via the standard NUTHATCH_* env
#      vars, since an installed package's own config is demoted to a "mirror" and
#      cannot set the actual root.
# The file:// prefix is required: without it fsspec's split_protocol() mis-parses
# the '://' that adapter source URLs embed in cache-key paths. setdefault() means
# ACMADDL_CACHE_DIR, the NUTHATCH_* env vars, and ~/.nuthatch.toml all still win.
from ._paths import env_with_legacy as _env_with_legacy

_cache_dir = _Path(
    _env_with_legacy("ACMADDL_CACHE_DIR") or _Path.home() / ".nuthatch" / "caches"
).expanduser()
_cache_uri = f"file://{_cache_dir}"
_os.environ.setdefault("NUTHATCH_ROOT_FILESYSTEM", _cache_uri)
_os.environ.setdefault("NUTHATCH_LOCAL_FILESYSTEM", _cache_uri)

from . import catalog
from .errors import VariableNotSupported
from .fetch import fetch, parse_target, parse_init, season_to_months
from .cpc_nmme import cpc_nmme_predictor  # noqa: F401
from .assemble import assemble, obs_predictor
from .zonal import zonal
from .health import check_product, check_all_products

__all__ = [
    "catalog",
    "fetch",
    "VariableNotSupported",
    "zonal",
    "parse_target",
    "season_to_months",
    "parse_init",
    "assemble",
    "obs_predictor",
    "check_product",
    "check_all_products",
]
