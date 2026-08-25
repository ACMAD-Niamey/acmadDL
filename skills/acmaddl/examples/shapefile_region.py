"""Clip fetches to a country boundary from a shapefile.

Requires the geo extra: pip install 'acmadDL[geo]'

The bbox derived from the shapefile drives the upstream request (and the
cache key); the dissolved polygon is applied as a final NaN mask. Two
boundary modes control which edge cells survive:
  - "center" (default): keep cells whose centre falls inside the polygon
  - "cover": keep every cell the polygon touches (all_touched=True),
    padded upstream by region_buffer degrees so edge cells aren't clipped
"""

from pathlib import Path

from acmaddl import fetch

# Country ADM0 shapefiles can be downloaded from geoBoundaries with the
# repo script: python scripts/fetch_country_shapefiles.py  (Kenya/Nigeria/
# Ethiopia by default, written to data/shapefiles/<name>.shp)
SHP = Path("data/shapefiles/kenya.shp")

common = dict(
    product="nmme/cfsv2",
    variable="precip",
    init="2010-02",
    target="MAM",
    hindcast=(2010, 2010),
    region=str(SHP),  # any path ending .shp; also accepts shapely/geopandas geometry
)

ds_center = fetch(**common, boundary="center")
ds_cover = fetch(**common, boundary="cover", region_buffer=1.5)
# Because this is targeted NMME precipitation, both datasets are in mm.

# Cells outside the polygon are NaN; "cover" keeps a wider fringe than "center".
print("center cells:", int(ds_center["precip"].notnull().sum()))
print("cover cells: ", int(ds_cover["precip"].notnull().sum()))

# Notes:
# - Multi-feature shapefiles are dissolved (union_all) and clip as one region.
# - Shapefiles are reprojected to EPSG:4326 automatically.
# - Polygons crossing the antimeridian are NOT handled — split them first.
# - Same-bbox fetches share cached raw data regardless of polygon shape.
