"""Typed exceptions for acmaddl's public API.

Callers need to route failures by *kind*, not by string-matching whatever
exception happened to escape the fetch path. The distinction that matters
downstream is:

  * capability gap  — this product will never serve this variable (permanent,
    a configuration error on the caller's side);
  * availability gap — this init/target isn't published yet (transient, retry
    later or pick another season).

Both used to surface as a bare ``KeyError``, which is indistinguishable
without inspecting the message.
"""


class VariableNotSupported(ValueError):
    """A product's catalog entry does not declare the requested variable.

    A ``ValueError`` because the caller asked for something the catalog can
    answer statically — not an I/O or availability failure. Carries the
    structured payload (``product``, ``variable``, ``available``) so callers
    can react without parsing the message.
    """

    def __init__(self, product: str, variable: str, available):
        self.product = product
        self.variable = variable
        self.available = list(available)
        super().__init__(
            f"{product} does not provide {variable!r}; it serves "
            f"{', '.join(self.available) or 'no variables'}"
        )


class RhizaNotInstalled(ImportError):
    """The Rhiza weather-skills provider packages are not in this environment.

    They live in the uv dependency group ``rhiza`` (git-pinned, so they cannot
    be a PyPI extra). Raised before any network I/O.
    """

    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(
            f"Rhiza weather-skills package {provider!r} is not installed. "
            f"Install the group: uv sync --group rhiza"
        )


class RhizaSkillError(RuntimeError):
    """A Rhiza skill script refused the request or failed while running.

    Their ``@weather_skill`` decorator prints the reason to stderr and exits
    non-zero; the adapter captures that text and surfaces it here verbatim, so
    an embargoed ECMWF init or a missing credential reads the same whether the
    skill ran from a shell or from acmaddl.
    """
