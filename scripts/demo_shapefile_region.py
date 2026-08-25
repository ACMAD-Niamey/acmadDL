#!/usr/bin/env python3
"""Demo: fetch seasonal data clipped to a country shapefile (acmaddl-plan §5).

Shows the two boundary rules for shapefile region input side by side:

    boundary="center" (default) -> a cell is kept only if its centre is inside
                                   the border (xarray/CDO/rasterio convention)
    boundary="cover"            -> every cell the border touches is kept, so the
                                   country is covered to its true edges

Fetches NMME CFSv2 MAM accumulated precipitation in mm (OPeNDAP — needs
network, no credentials), plots both panels with the country outline overlaid,
and writes a PNG.

Usage:
    scripts/demo_shapefile_region.py                 # Kenya
    scripts/demo_shapefile_region.py ethiopia        # by name (data/shapefiles/<name>.shp)
    scripts/demo_shapefile_region.py path/to/any.shp # any shapefile

Requires the `geo` extra + matplotlib:
    pip install 'acmadDL[geo]' matplotlib
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")  # headless: write PNG, no display needed
import matplotlib.pyplot as plt
import numpy as np

import acmaddl

REPO = Path(__file__).resolve().parents[1]
SHP_DIR = REPO / "data" / "shapefiles"
OUT_DIR = REPO / "output" / "demos"

PRODUCT = "nmme/cfsv2"
VARIABLE = "precip"
INIT = "2010-02"      # within cfsv2's hindcast range (1982–2010)
TARGET = "MAM"
HINDCAST = (2010, 2010)


def _resolve_shp(arg: str) -> Path:
    """Accept a bare country name (kenya) or an explicit shapefile path."""
    p = Path(arg)
    if p.suffix == ".shp" and p.exists():
        return p
    candidate = SHP_DIR / f"{arg.lower()}.shp"
    if candidate.exists():
        return candidate
    raise SystemExit(
        f"No shapefile for {arg!r}. Tried {p} and {candidate}.\n"
        f"Run scripts/fetch_country_shapefiles.py first, or pass a .shp path."
    )


def _to_2d(da):
    """Collapse forecast dims (member/lead/init) to a plottable lat/lon field."""
    for dim in ("member", "M", "lead_time", "L", "init_time"):
        if dim in da.dims:
            da = da.mean(dim, keep_attrs=True)
    return da


def main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    name = argv[0] if argv else "kenya"
    shp = _resolve_shp(name)
    gdf = gpd.read_file(shp).to_crs("EPSG:4326")

    print(f"[demo] shapefile : {shp}")
    print(f"[demo] fetching  : {PRODUCT} {VARIABLE} init={INIT} target={TARGET} "
          f"(both boundary modes)")

    # Same shapefile region, the two boundary rules side by side.
    center_ds = acmaddl.fetch(PRODUCT, VARIABLE, init=INIT, target=TARGET,
                              hindcast=HINDCAST, region=str(shp),
                              boundary="center", verbose=False, progress=False)
    cover_ds = acmaddl.fetch(PRODUCT, VARIABLE, init=INIT, target=TARGET,
                             hindcast=HINDCAST, region=str(shp),
                             boundary="cover", verbose=False, progress=False)

    center = _to_2d(center_ds[VARIABLE])
    cover = _to_2d(cover_ds[VARIABLE])
    print(f"[demo] valid cells: center={int(center.notnull().sum())}  "
          f"cover={int(cover.notnull().sum())}")

    # ── plot ──────────────────────────────────────────────────────────────
    vmax = float(np.nanmax(cover)) or 1.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    panels = (
        (axes[0], center, "boundary='center' (default)\ncell kept if its centre is inside"),
        (axes[1], cover, "boundary='cover'\nany cell the border touches"),
    )
    for ax, field, title in panels:
        mesh = ax.pcolormesh(field.lon, field.lat, field, cmap="YlGnBu",
                             vmin=0, vmax=vmax, shading="auto")
        gdf.boundary.plot(ax=ax, edgecolor="crimson", linewidth=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        fig.colorbar(mesh, ax=ax, shrink=0.8,
                     label=f"{VARIABLE} ({cover.attrs.get('units','')})")
    fig.suptitle(f"{PRODUCT}  {VARIABLE}  {TARGET} {INIT}  —  {name}", fontsize=12)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / f"shapefile_region_{name.lower()}.png"
    fig.savefig(out_png, dpi=130)
    print(f"[demo] wrote {out_png}")


if __name__ == "__main__":
    main()
