"""Unit tests for the Rhiza weather-skills adapter. No network.

The provider packages (dependency group `rhiza`) are optional in a dev env;
tests that need the real scripts skip without them. Everything else runs
against fake wrappers and the fixtures in tests/fixtures/rhiza/.
"""
import importlib.metadata
from pathlib import Path

import pytest

from acmaddl.adapters import _ADAPTERS, get_adapter
from acmaddl.adapters import rhiza
from acmaddl.errors import RhizaNotInstalled, RhizaSkillError


def _group_installed() -> bool:
    try:
        importlib.metadata.distribution("weather-skills")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


needs_group = pytest.mark.skipif(not _group_installed(), reason="rhiza group not installed")


def test_adapter_registered():
    assert "rhiza" in _ADAPTERS
    assert type(get_adapter("rhiza")).__name__ == "RhizaAdapter"


def test_locate_script_missing_distribution_names_install_hint(monkeypatch):
    def boom(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, "files", boom)
    with pytest.raises(RhizaNotInstalled) as e:
        rhiza.locate_script("weather-skills", "chirps-fetch")
    assert rhiza.INSTALL_HINT in str(e.value)


def test_locate_script_requires_exactly_one_script(monkeypatch):
    from importlib.metadata import PackagePath
    fake = [PackagePath("skills/x/scripts/a.py"), PackagePath("skills/x/scripts/b.py")]
    monkeypatch.setattr(importlib.metadata, "files", lambda name: fake)
    with pytest.raises(RhizaSkillError, match="2 scripts"):
        rhiza.locate_script("weather-skills", "x")


@needs_group
def test_locate_and_load_real_entrypoint():
    path = rhiza.locate_script("weather-skills", "chirps-fetch")
    assert path.name == "fetch.py" and path.exists()
    fn, version = rhiza.load_entrypoint("weather-skills", "chirps-fetch")
    assert hasattr(fn, "parser")           # a @weather_skill wrapper
    assert version and version != "unknown"
    assert rhiza.load_entrypoint("weather-skills", "chirps-fetch")[0] is fn   # memoised


@needs_group
def test_provider_pin_is_a_commit():
    pin = rhiza.provider_pin("weather-skills")
    assert len(pin) == 40 and all(c in "0123456789abcdef" for c in pin)
