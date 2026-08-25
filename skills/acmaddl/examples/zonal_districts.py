"""Reduce a gridded field to one value per district with acmaddl.zonal.

Requires the geo extra: pip install 'acmadDL[geo]'

fetch(region=shp) dissolves a shapefile's features into ONE mask and answers
"give me the data over this area". Reporting asks the other question — "give me
one number PER district" — which is what zonal() answers: it rasterizes all the
geometries once and reduces the grid with a single groupby, so a thousand
woredas cost one pass, not a thousand.
"""

import geopandas as gpd

from acmaddl import fetch, zonal

# One dekadal precip field over the whole shapefile's extent (dissolved bbox).
districts = gpd.read_file("data/shapefiles/eth_woredas.shp")
rain = fetch("obs/chirps-v3-dekad-tif", "precip", region="data/shapefiles/eth_woredas.shp")
# rain: (time, lat, lon)

# One series per district. Index by a UNIQUE code column; carry the (possibly
# repeating) human-readable name as a label coordinate. Admin datasets routinely
# have a unique code and a non-unique name — passing a non-unique column to `by`
# raises ValueError, on purpose (an index has to be unambiguous for .sel).
per_district = zonal(
    rain,
    districts,
    by="shapeID",        # unique id -> the `region` index
    label="ADM3_EN",      # repeatable name -> a companion `region_label` coord
    stat="mean",          # mean/sum/min/max/median/std/count
    weights="area",       # cos(lat) area weighting (default); only mean/sum use it
)
# per_district: (time, region), with region indexed by shapeID and a
# region_label coord carrying ADM3_EN. Output mirrors the input geometry
# order/count: a district with no valid grid cell is NaN (count -> 0), not
# dropped.

print(dict(per_district.sizes))
print(per_district["region_label"].values[:5])

# Notes / gotchas:
# - CHIRPS dekads are 0.05-degree, so most woredas cover many cells. On a COARSE
#   grid a district smaller than one cell captures no cell centre and yields NaN
#   — pass all_touched=True to include every touched cell.
# - all_touched=True can make coverage WORSE where districts share a boundary:
#   rasterize is last-one-wins, so a shared boundary cell goes to whichever
#   feature burned last. Use it for the small/coarse case, not reflexively.
# - NaN cells are excluded from the reduction: a half-ocean district is averaged
#   over its land only.
# - The output keeps every non-spatial dim (here `time`), so it drops straight
#   into anything that reduces along time (e.g. accumulation curves).
