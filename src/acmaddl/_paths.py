"""Filesystem locations used by acmaddl adapters."""
import os
import warnings

# Pre-rename spellings of the env vars below. The package was renamed
# rosetta -> acmaddl in 0.2.0; the old names are still honored so an exported
# ROSETTA_CACHE_DIR does not silently relocate an existing cache.
_LEGACY_ENV = {
    "ACMADDL_CACHE_DIR": "ROSETTA_CACHE_DIR",
    "ACMADDL_TMP_DIR": "ROSETTA_TMP_DIR",
}


def env_with_legacy(name: str) -> str | None:
    """Read ``name``, falling back to its deprecated ``ROSETTA_*`` spelling.

    Returns ``None`` if neither is set. Reading the legacy name emits a
    ``DeprecationWarning`` — it still works, but it is on its way out.
    """
    value = os.environ.get(name)
    if value:
        return value
    legacy = _LEGACY_ENV[name]
    value = os.environ.get(legacy)
    if value:
        warnings.warn(
            f"{legacy} is deprecated and will be removed in a future release; "
            f"use {name} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return value or None


def get_tmpdir() -> str:
    """Return the directory adapters should use for scratch downloads.

    Defaults to ``~/.nuthatch/acmaddl/_tmp/`` (co-located with the nuthatch
    cache so scratch and cache live in the same tree). Override via the
    ``ACMADDL_TMP_DIR`` environment variable (or the deprecated
    ``ROSETTA_TMP_DIR``).

    Adapters MUST use this rather than the system tempdir: macOS reaps
    ``/var/folders/.../T/`` on its own schedule, which broke cache entries
    that pickled lazy datasets referencing those temp files (issue #24).
    """
    path = env_with_legacy("ACMADDL_TMP_DIR") or os.path.expanduser(
        "~/.nuthatch/acmaddl/_tmp"
    )
    os.makedirs(path, exist_ok=True)
    return path
