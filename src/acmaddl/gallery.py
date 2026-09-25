"""Dataset reference sheet: every catalogued product on one page.

``show_datasets()`` renders the full acmadDL catalog — the visual counterpart
of ``acmaddl.catalog.list_products()``: sections by source family, one row per
product with its id, source, variables (units), grid resolution and cadence,
and a **temporal-coverage bar** on a shared year axis, so the native hindcast
window of every product can be read at a glance (the per-model year windows
are the single most common surprise when assembling a multi-model roster).

Aliases and date-deprecated products are excluded; the sheet always reflects
the live ``catalog.yaml``. Rendering needs matplotlib (the ``demo`` extra).

From a shell::

    acmaddl datasets                 # opens a window
    acmaddl datasets datasets.png    # writes a file
    python -m acmaddl.gallery [out.png]
"""
from __future__ import annotations

from pathlib import Path

import yaml

__all__ = ["show_datasets"]

_CATALOG_PATH = Path(__file__).parent / "catalog.yaml"

_SECTIONS = (
    ("obs", "Observations"),
    ("nmme", "NMME seasonal forecasts"),
    ("c3s", "C3S / Copernicus seasonal & sub-seasonal"),
    ("chc", "CHC forecasts"),
)
_SECTION_COLOR = {"obs": "#2f7ed8", "nmme": "#31a354", "c3s": "#e6842a", "chc": "#8e6bb8"}

_ADAPTER_LABEL = {
    "ccsr": "CCSR (Columbia)",
    "cds": "Copernicus CDS",
    "opendap": "IRI DL / OPeNDAP",
    "iridl": "IRI Data Library",
    "sheerwater": "Sheerwater",
    "cpc_binary": "NOAA CPC",
}

_YEAR_MIN, _YEAR_MAX = 1978, 2028


def _source(entry):
    a = entry.get("adapter", "?")
    if a == "http":
        url = entry.get("source_url", "")
        host = url.split("//")[-1].split("/")[0]
        return host or "http"
    return _ADAPTER_LABEL.get(a, a)


def _meta_line(entry):
    g = entry.get("grid") or {}
    parts = [_source(entry)]
    vs = entry.get("variables") or {}
    if vs:
        parts.append(", ".join(
            f"{n} ({d.get('target_units') or d.get('units') or '?'})"
            for n, d in vs.items()))
    if g.get("lat_res") is not None:
        parts.append(f"{g['lat_res']:g}\N{DEGREE SIGN}")
    cadence = g.get("temporal") or "seasonal (init/lead)"
    parts.append(cadence)
    return "  ·  ".join(parts)


def _collect():
    """[(family, name, entry)] for every non-alias, non-deprecated product,
    in section order then by name."""
    raw = yaml.safe_load(_CATALOG_PATH.read_text())
    from .catalog import info
    out = []
    for fam, _title in _SECTIONS:
        for name in sorted(k for k in raw if k.startswith(fam + "/")):
            if "alias_of" in raw[name]:
                continue
            entry = dict(raw[name])
            if entry.get("deprecated_after"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    if info(name).get("deprecated"):
                        continue
            out.append((fam, name, entry))
    return out


_PAGES = {
    "observations": (("obs",), "acmadDL catalogued datasets \u2014 observations"),
    "forecasts": (("nmme", "c3s", "chc"), "acmadDL catalogued datasets \u2014 forecast systems"),
    "all": (("obs", "nmme", "c3s", "chc"), "acmadDL catalogued datasets"),
}


def show_datasets(which="both", save=None):
    """Render the dataset catalogue.

    ``which``: ``"both"`` (default) renders TWO figures \u2014 observations and
    forecast systems \u2014 returned as a tuple (with ``save`` given, the files
    gain ``-observations`` / ``-forecasts`` suffixes). Or pass
    ``"observations"``, ``"forecasts"`` or ``"all"`` (the single long sheet)
    for one figure.

    Per product: id, source, variables (units), resolution and cadence, plus
    a coverage bar over a shared 1980-2028 year axis (the catalog
    ``hindcast_range``; forecast products carry an arrow cap for their ongoing
    forecast stream; live/rolling products draw an open marker).
    """
    if which == "both":
        figs = []
        for page in ("observations", "forecasts"):
            target = None
            if save:
                sp = Path(save)
                target = sp.with_name(sp.stem + "-" + page + (sp.suffix or ".png"))
            figs.append(show_datasets(which=page, save=target))
        return tuple(figs)
    if which not in _PAGES:
        raise ValueError(
            f"which must be 'both', 'observations', 'forecasts' or 'all'; got {which!r}")
    fams, title = _PAGES[which]

    import matplotlib.pyplot as plt

    entries = [e for e in _collect() if e[0] in fams]

    HEADER, ROW = 0.40, 0.335
    heights, rows = [], []
    last = None
    for fam, name, entry in entries:
        if fam != last:
            rows.append(("header", dict(_SECTIONS)[fam]))
            heights.append(HEADER)
            last = fam
        rows.append(("row", (fam, name, entry)))
        heights.append(ROW)
    total = sum(heights) + 1.0
    fig = plt.figure(figsize=(10.5, total))
    fig.suptitle(title, fontsize=13, fontweight="bold",
                 y=1 - 0.12 / total)
    fig.text(0.695, 1 - 0.42 / total, "temporal coverage (hindcast range)",
             fontsize=7.5, color="0.35", ha="center")

    TL_X0, TL_W = 0.46, 0.47                    # timeline column
    yr = lambda v: (v - _YEAR_MIN) / (_YEAR_MAX - _YEAR_MIN)

    y = total - 0.55
    for (rtype, payload), h in zip(rows, heights):
        y -= h
        if rtype == "header":
            fig.text(0.03, (y + 0.09) / total, payload, fontsize=11,
                     fontweight="bold")
            fig.lines.append(plt.Line2D([0.03, 0.97], [(y + 0.03) / total] * 2,
                                        transform=fig.transFigure,
                                        color="0.75", linewidth=0.8))
            continue
        fam, name, entry = payload
        fig.text(0.045, (y + h - 0.15) / total, name, fontsize=8.5,
                 fontweight="bold")
        fig.text(0.045, (y + h - 0.28) / total, _meta_line(entry), fontsize=6.5,
                 color="0.4")
        ax = fig.add_axes([TL_X0, (y + 0.075) / total, TL_W, 0.16 / total])
        ax.set_xlim(_YEAR_MIN, _YEAR_MAX)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        for decade in range(1980, _YEAR_MAX, 10):
            ax.axvline(decade, color="0.88", linewidth=0.6, zorder=0)
        ax.set_xticks(list(range(1980, _YEAR_MAX, 10)))
        ax.set_xticklabels([str(d) for d in range(1980, _YEAR_MAX, 10)],
                           fontsize=5.5, color="0.55")
        ax.tick_params(length=0, pad=1)
        for s in ax.spines.values():
            s.set_visible(False)
        color = _SECTION_COLOR[fam]
        rng = (entry.get("grid") or {}).get("hindcast_range")
        is_fcst = fam in ("nmme", "c3s", "chc")
        if rng:
            y0, y1 = rng
            ax.axvspan(y0, y1 + 1, ymin=0.28, ymax=0.92, color=color, alpha=0.85)
            if is_fcst:                          # ongoing forecast stream
                ax.plot([_YEAR_MAX - 1.2], [0.60], marker=">", ms=4,
                        color=color, clip_on=False)
                ax.plot([y1 + 1, _YEAR_MAX - 1.2], [0.60, 0.60],
                        color=color, linewidth=1.0, linestyle=(0, (2, 2)))
        else:                                    # live / rolling window
            ax.plot([_YEAR_MAX - 1.2], [0.60], marker=">", ms=4, color=color,
                    clip_on=False)
            ax.text(_YEAR_MAX - 2.6, 0.58, "live / rolling", fontsize=6,
                    color=color, ha="right", va="center", style="italic")
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


def main(argv=None):
    import sys
    args = sys.argv[1:] if argv is None else argv
    if args:
        show_datasets(save=args[0])
        sp = Path(args[0])
        suffix = sp.suffix or ".png"
        for page in ("observations", "forecasts"):
            print("wrote " + str(sp.with_name(sp.stem + "-" + page + suffix)))
    else:
        import matplotlib.pyplot as plt
        show_datasets()
        plt.show()


if __name__ == "__main__":
    main()
