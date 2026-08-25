"""Compatibility guarantees for the rosetta -> acmaddl rename (0.2.0).

Two promises were made to existing users when the package was renamed
(distribution `accord-rosetta` -> `acmadDL`, import name `rosetta` ->
`acmaddl`):

1. `import rosetta` keeps working, warns, and resolves to the *same* module
   objects as `acmaddl` — not a parallel copy.
2. The `ROSETTA_*` environment variables are still honored, so an exported
   `ROSETTA_CACHE_DIR` does not silently relocate someone's cache.

The shim installs a `sys.meta_path` finder and rebinds `sys.modules["rosetta"]`,
so the import tests run in subprocesses to keep that out of the test session.
"""
import subprocess
import sys
import textwrap

import pytest

from acmaddl._paths import env_with_legacy


def run_py(source: str) -> str:
    """Run `source` in a clean subprocess and return its stdout."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"
    return result.stdout.strip()


class TestImportShim:
    def test_import_rosetta_emits_a_deprecation_warning(self):
        """The old name still works, but says so. Silent aliasing would leave
        users on a deprecated path with no signal to migrate."""
        out = run_py("""
            import warnings
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                import rosetta
            hits = [w for w in caught if issubclass(w.category, DeprecationWarning)]
            print(len(hits), "acmaddl" in str(hits[0].message))
        """)
        assert out == "1 True"

    def test_rosetta_is_the_acmaddl_module_itself(self):
        out = run_py("""
            import warnings; warnings.simplefilter("ignore")
            import rosetta, acmaddl
            print(rosetta is acmaddl, rosetta.fetch is acmaddl.fetch)
        """)
        assert out == "True True"

    def test_deep_submodule_imports_resolve_to_the_same_objects(self):
        """`from rosetta.adapters.base import AdapterBase` must hand back the
        object `acmaddl.adapters.base` holds. A shim that re-executed the
        modules under the old names would produce look-alike duplicates."""
        out = run_py("""
            import warnings; warnings.simplefilter("ignore")
            import sys
            from rosetta.adapters.base import AdapterBase
            import acmaddl.adapters.base
            print(
                AdapterBase is acmaddl.adapters.base.AdapterBase,
                sys.modules["rosetta.adapters"] is acmaddl.adapters,
            )
        """)
        assert out == "True True"

    def test_exceptions_raised_through_the_old_name_are_catchable_by_the_new(self):
        """The failure mode duplicate modules cause: `except
        acmaddl.errors.VariableNotSupported` silently not catching an error
        that came in via `rosetta`."""
        out = run_py("""
            import warnings; warnings.simplefilter("ignore")
            from rosetta.errors import VariableNotSupported as Old
            from acmaddl.errors import VariableNotSupported as New
            try:
                raise Old("nmme/cfsv2", "precip", ["temp"])
            except New:
                print("caught")
        """)
        assert out == "caught"

    def test_aliasing_does_not_rewrite_the_real_modules_identity(self):
        """The import machinery stamps __spec__/__loader__ onto whatever a
        loader returns; the shim puts the originals back. Otherwise
        `acmaddl.adapters.base.__spec__.name` would read "rosetta.adapters.base"."""
        out = run_py("""
            import warnings; warnings.simplefilter("ignore")
            import rosetta.adapters.base
            import acmaddl.adapters.base as real
            print(real.__name__, real.__spec__.name)
        """)
        assert out == "acmaddl.adapters.base acmaddl.adapters.base"


class TestLegacyEnvVars:
    def test_new_name_is_used_when_set(self, monkeypatch):
        monkeypatch.setenv("ACMADDL_TMP_DIR", "/tmp/new")
        monkeypatch.delenv("ROSETTA_TMP_DIR", raising=False)
        assert env_with_legacy("ACMADDL_TMP_DIR") == "/tmp/new"

    def test_new_name_wins_over_the_legacy_one(self, monkeypatch):
        monkeypatch.setenv("ACMADDL_TMP_DIR", "/tmp/new")
        monkeypatch.setenv("ROSETTA_TMP_DIR", "/tmp/old")
        assert env_with_legacy("ACMADDL_TMP_DIR") == "/tmp/new"

    def test_legacy_name_is_honored_with_a_warning(self, monkeypatch):
        """Honored, so nobody's exported ROSETTA_CACHE_DIR silently starts
        pointing somewhere else after the upgrade."""
        monkeypatch.delenv("ACMADDL_TMP_DIR", raising=False)
        monkeypatch.setenv("ROSETTA_TMP_DIR", "/tmp/old")
        with pytest.warns(DeprecationWarning, match="ROSETTA_TMP_DIR is deprecated"):
            assert env_with_legacy("ACMADDL_TMP_DIR") == "/tmp/old"

    def test_returns_none_when_neither_is_set(self, monkeypatch):
        monkeypatch.delenv("ACMADDL_TMP_DIR", raising=False)
        monkeypatch.delenv("ROSETTA_TMP_DIR", raising=False)
        assert env_with_legacy("ACMADDL_TMP_DIR") is None

    def test_no_warning_when_only_the_new_name_is_set(self, monkeypatch, recwarn):
        monkeypatch.setenv("ACMADDL_TMP_DIR", "/tmp/new")
        monkeypatch.delenv("ROSETTA_TMP_DIR", raising=False)
        env_with_legacy("ACMADDL_TMP_DIR")
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]
