#!/usr/bin/env python3
"""Download country boundary shapefiles for region-input demos/tests.

Pulls ADM0 (national) boundaries from geoBoundaries (gbOpen, CC BY 4.0) and
writes them as EPSG:4326 shapefiles under ``data/shapefiles/``. These are the
real polygons used by ``scripts/demo_shapefile_region.py`` and the shapefile
integration test.

Usage:
    scripts/fetch_country_shapefiles.py                 # Kenya, Nigeria, Ethiopia
    scripts/fetch_country_shapefiles.py KEN NGA ETH TZA # custom ISO3 list

Requires the `geo` extra: pip install 'acmadDL[geo]'
Source: https://www.geoboundaries.org/  (gbOpen, ADM0, simplified)
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import geopandas as gpd

# ISO3 -> output filename stem. Defaults are the East/West-Africa countries
# most relevant to ACMAD seasonal workflows.
DEFAULT_COUNTRIES = {"KEN": "kenya", "NGA": "nigeria", "ETH": "ethiopia"}

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "shapefiles"
URL = ("https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/"
       "gbOpen/{iso}/ADM0/geoBoundaries-{iso}-ADM0_simplified.geojson")


def fetch_one(iso: str, name: str, out_dir: Path) -> Path:
    url = URL.format(iso=iso)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (trusted host)
        geojson = resp.read()
    tmp = out_dir / f"_{iso}.geojson"
    tmp.write_bytes(geojson)

    gdf = gpd.read_file(tmp).to_crs("EPSG:4326")
    # Keep just geometry + a friendly name; drop geoBoundaries' metadata columns.
    out = gpd.GeoDataFrame({"name": [name]}, geometry=[gdf.union_all()],
                           crs="EPSG:4326")
    shp = out_dir / f"{name}.shp"
    out.to_file(shp)
    tmp.unlink()
    s, n, w, e = (out.total_bounds[1], out.total_bounds[3],
                  out.total_bounds[0], out.total_bounds[2])
    print(f"{name:10s} -> {shp.relative_to(out_dir.parents[1])}  "
          f"bbox=[{s:.2f}, {n:.2f}, {w:.2f}, {e:.2f}]")
    return shp


def main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if argv:
        countries = {iso.upper(): iso.lower() for iso in argv}
    else:
        countries = DEFAULT_COUNTRIES

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for iso, name in countries.items():
        fetch_one(iso, name, OUT_DIR)
    print(f"\nWrote {len(countries)} shapefile(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
