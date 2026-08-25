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
