"""Run Rhiza weather-skills fetchers in-process and hand their Zarr to acmadDL.

Design (docs/specs/2026-10-01-rhiza-weather-skills-adapter-design.md):

* Their code, unmodified. Each skill is a ``@weather_skill`` script bundled as a
  data file inside the provider wheel (``skills/<skill>/scripts/<file>.py``). We
  load it by path and call the decorated wrapper with an argv list, exactly as
  the CLI would. The wrapper writes a Zarr to ``-o``; we read it back.
* One adapter, many catalog entries. A ``rhiza/*`` product declares ``skill``,
  ``provider`` and an ``argv`` template; nothing per-skill lives here.
* Minimal reshaping: ``init_time`` from the requested init, ``valid_time``
  derived, helper coords dropped, provenance attrs kept. ``normalize()`` does
  the rest (renames, units, crop, lat order).
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path

from .base import AdapterBase
from ..errors import RhizaNotInstalled, RhizaSkillError

INSTALL_HINT = "uv sync --group rhiza"

# (provider, skill, entrypoint) -> (wrapper, _SKILL_VERSION). Loading executes
# the script module (imports cfgrib, dynamical_catalog, ...), so do it once.
_ENTRYPOINTS: dict[tuple[str, str, str], tuple[object, str]] = {}


def locate_script(provider: str, skill: str) -> Path:
    """Path of ``skills/<skill>/scripts/*.py`` inside the installed ``provider``."""
    try:
        files = importlib.metadata.files(provider)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RhizaNotInstalled(provider) from exc
    if files is None:
        raise RhizaNotInstalled(provider)
    matches = [
        f for f in files
        if len(f.parts) == 4 and f.parts[0] == "skills" and f.parts[1] == skill
        and f.parts[2] == "scripts" and f.suffix == ".py"
    ]
    if len(matches) != 1:
        raise RhizaSkillError(
            f"{provider!r} ships {len(matches)} scripts for skill {skill!r}; "
            "expected exactly one under skills/<skill>/scripts/"
        )
    return Path(str(matches[0].locate()))


def load_entrypoint(provider: str, skill: str, entrypoint: str = "fetch"):
    """(wrapper, version) for a skill; the wrapper is its ``@weather_skill`` function."""
    key = (provider, skill, entrypoint)
    if key in _ENTRYPOINTS:
        return _ENTRYPOINTS[key]
    path = locate_script(provider, skill)
    name = "acmaddl_rhiza_" + skill.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, entrypoint, None)
    if fn is None or not hasattr(fn, "parser"):
        raise RhizaSkillError(
            f"{skill}: {entrypoint!r} in {path} is not a @weather_skill entrypoint"
        )
    version = str(getattr(module, "_SKILL_VERSION", "unknown"))
    _ENTRYPOINTS[key] = (fn, version)
    return _ENTRYPOINTS[key]


def provider_pin(provider: str) -> str:
    """The git commit the provider was installed from, or 'unknown'."""
    try:
        dist = importlib.metadata.distribution(provider)
        text = dist.read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
    if not text:
        return "unknown"
    try:
        return str(json.loads(text).get("vcs_info", {}).get("commit_id") or "unknown")
    except (ValueError, AttributeError):
        return "unknown"


class RhizaAdapter(AdapterBase):
    """Catalog entries: ``adapter: rhiza``; see module docstring and catalog.yaml header."""

    def fetch_data(self, product_config, variable, date_range=None, region=None):
        raise NotImplementedError  # Task 9
