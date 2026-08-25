"""Deprecated alias for :mod:`acmaddl`.

The project was renamed in 0.2.0: import name ``rosetta`` -> ``acmaddl``,
distribution ``accord-rosetta`` -> ``acmadDL``. This package keeps existing
code working — both ``import rosetta`` and deep imports like
``from rosetta.adapters.base import AdapterBase`` — while warning that the old
spelling is on its way out.

It is a redirect, not a copy. Every ``rosetta.X`` resolves to the *same* module
object as ``acmaddl.X``, so identity holds across the two spellings and an
exception raised through ``rosetta`` is caught by ``except
acmaddl.errors.VariableNotSupported``. A second, parallel copy of the package
would silently break that.
"""
import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import acmaddl

_OLD = "rosetta"
_NEW = "acmaddl"


class _AliasLoader(importlib.abc.Loader):
    """Resolve ``rosetta.X`` to the already-imported ``acmaddl.X`` object."""

    def create_module(self, spec):
        module = importlib.import_module(_NEW + spec.name[len(_OLD):])
        # The import machinery calls _init_module_attrs() on whatever we return,
        # which would stamp this module's __spec__/__loader__ with the aliased
        # name. Stash the real ones so exec_module() can put them back.
        self._real_attrs = (module.__spec__, getattr(module, "__loader__", None))
        return module

    def exec_module(self, module):
        # Already executed under its real name; only undo the attribute stamping.
        spec, loader = self._real_attrs
        module.__spec__ = spec
        module.__loader__ = loader


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(_OLD + "."):
            return importlib.util.spec_from_loader(fullname, _AliasLoader())
        return None


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

warnings.warn(
    "The 'rosetta' package has been renamed to 'acmaddl' (distribution "
    "'accord-rosetta' -> 'acmadDL'). 'import rosetta' still works but is "
    "deprecated and will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Bind the top-level name to the real module object, so `rosetta.fetch is
# acmaddl.fetch` and no attribute forwarding is needed.
sys.modules[__name__] = acmaddl
