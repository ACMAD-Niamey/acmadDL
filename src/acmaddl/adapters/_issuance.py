"""Enumerating the files of an issuance-keyed forecast archive.

Observational archives are keyed by the time an observation describes. Forecast
archives are keyed by *two* times: when the forecast was issued, and what it is
about. CHC's CHIRPS-GEFS puts the first in the directory path and the second in
the filename::

    .../CHIRPS-GEFS/v3/daily/global/2026/07/05/c3g_2026.07.20.tif
                                    ^^^^^^^^^^        ^^^^^^^^^^
                                    issued 5 July     valid 20 July, lead 15

Other archives put the lead number in the filename, or the init date in both
places, or serve one accumulated file per issuance with no lead at all. Rather
than a new adapter per layout, a catalog entry declares an ``issuance`` block
and gets `strftime` templating over three names — ``init``, ``valid`` and
``lead``::

    issuance:
      path_pattern: "{init:%Y}/{init:%m}/{init:%d}"
      file_pattern: "c3g_{valid:%Y}.{valid:%m}.{valid:%d}.tif"
      leads: [0, 15]
      lead_units: days

This module turns that declaration plus a list of issuance dates into the file
list, and the file list back into the ``(init_time, lead_time)`` coordinates the
rest of acmadDL expects. It knows nothing about HTTP, so an S3 or OPeNDAP
archive with the same shape can reuse it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

_LEAD_UNITS = {
    "days": timedelta(days=1),
    "hours": timedelta(hours=1),
}


@dataclass(frozen=True)
class IssuanceFile:
    """One remote file, and the coordinates it will occupy."""

    url: str
    init: datetime
    lead: int


def issuance_config(product_config: dict) -> dict | None:
    """The validated ``issuance`` block, or None for a plain time-series product."""
    block = product_config.get("issuance")
    if block is None:
        return None
    if "file_pattern" not in block:
        raise ValueError("issuance block requires 'file_pattern'")
    leads = block.get("leads", [0, 0])
    if len(leads) != 2 or leads[0] > leads[1]:
        raise ValueError(
            f"issuance 'leads' must be an inclusive [min, max] pair, got {leads!r}"
        )
    units = block.get("lead_units", "days")
    if units not in _LEAD_UNITS:
        raise ValueError(
            f"issuance 'lead_units' must be one of {sorted(_LEAD_UNITS)}, got {units!r}"
        )
    return {
        "path_pattern": block.get("path_pattern", ""),
        "file_pattern": block["file_pattern"],
        "leads": list(range(int(leads[0]), int(leads[1]) + 1)),
        "lead_units": units,
        "lead_step": _LEAD_UNITS[units],
    }


def parse_init_dates(init_dates) -> list[datetime]:
    """Normalize whatever the caller passed as issuance dates to datetimes."""
    parsed = []
    for value in init_dates:
        if isinstance(value, datetime):
            parsed.append(value)
        else:
            text = str(value)
            if len(text) != 10:
                raise ValueError(
                    f"issuance dates must be full YYYY-MM-DD dates, got {value!r}. "
                    "An issuance-keyed forecast has a day, not just a month."
                )
            parsed.append(datetime.strptime(text, "%Y-%m-%d"))
    return sorted(set(parsed))


def enumerate_files(base_url: str, issuance: dict, init_dates) -> list[IssuanceFile]:
    """Every (issuance, lead) file implied by the config, in a stable order."""
    inits = parse_init_dates(init_dates)
    if not inits:
        raise ValueError("no issuance dates given")

    step = issuance["lead_step"]
    files = []
    for init in inits:
        for lead in issuance["leads"]:
            valid = init + lead * step
            fields = {"init": init, "valid": valid, "lead": lead}
            parts = [base_url.rstrip("/")]
            path = issuance["path_pattern"].format(**fields)
            if path:
                parts.append(path.strip("/"))
            parts.append(issuance["file_pattern"].format(**fields))
            files.append(IssuanceFile("/".join(parts), init, lead))
    return files


def lead_timedelta(leads, lead_units: str) -> np.ndarray:
    """Lead numbers -> numpy timedeltas, so ``lead_time`` carries its own units.

    An integer ``lead_time`` is ambiguous the moment two products disagree on
    whether it counts days or hours; a timedelta cannot be misread.
    """
    unit = "D" if lead_units == "days" else "h"
    return np.array([np.timedelta64(int(lead), unit) for lead in leads]).astype(
        "timedelta64[ns]"
    )
