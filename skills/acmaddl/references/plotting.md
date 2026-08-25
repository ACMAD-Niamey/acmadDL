# Plotting fetched data (quick-look maps, facets, time series)

How to visualize what `fetch()`/`assemble()` return: quick-look maps, ensemble facets, regional time series, and region-overlay sanity checks. acmadDL ships no plotting API of its own — these are plain xarray/matplotlib recipes that work *because* every fetch is normalized to the canonical schema (lat ascending, lon in [-180, 180], EPSG:4326, canonical units). For downstream forecast graphics (tercile-probability maps, skill maps, reliability diagrams, verification PDFs) use deepscale's plotting functions instead — see the last section.

## Dependencies

`pip install acmadDL[demo]` installs matplotlib + cartopy + rasterio. Cartopy is only needed for coastline/border basemaps; every recipe below degrades to plain matplotlib if you drop the `projection=`/feature lines.

## Getting to a 2-D (lat, lon) slice

Fetched arrays carry extra dims you must select/reduce before mapping:

| You fetched | Shape | Typical 2-D slice |
|---|---|---|
| Obs/reanalysis | `(time, lat, lon)` | `da.sel(time="2020-03")` or `da.mean("time")` (climatology) |
| Obs with `seasonal="mean"` | `(year, lat, lon)` | `da.sel(year=2020)` |
| Forecast | `(init_time, lead_time, member, lat, lon)` | `da.isel(init_time=0, lead_time=0).mean("member")` |
| Forecast with `year_index=True` / `assemble()` | `(year, member, lat, lon)` | `da.sel(year=2016).mean("member")` (ensemble mean) |

## Quick-look map (cartopy)

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from acmaddl import fetch

da = fetch("obs/era5", "precip", region=[-5, 15, 33, 48],
           hindcast=(1993, 2016))["precip"].mean("time")

ax = plt.axes(projection=ccrs.PlateCarree())   # data are EPSG:4326 — no transform gymnastics
da.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="YlGnBu",
        cbar_kwargs={"label": "precip (mm/day)"})
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=0.4)
plt.savefig("precip_clim.png", dpi=200, bbox_inches="tight")
```

Because normalization guarantees lat ascending and lon in [-180, 180], maps come out right-side-up with no `origin=` or longitude-wrapping fixes. If a map looks flipped or split at the dateline, the data did not come through `fetch()` — normalize it first.

## Colormap and units conventions

Canonical units are guaranteed by the normalize pipeline (see [data-conventions.md](data-conventions.md)) — label colorbars accordingly.

| Variable | Units | Full field | Anomaly |
|---|---|---|---|
| `precip` | collapsed seasonal (`year_index=True`/`assemble`): mm; lead-resolved/daily: usually mm/day | sequential (`YlGnBu`, `viridis`), `vmin=0` | diverging `BrBG` (brown=dry, green=wet), `center=0` |
| `temp` | °C | `RdYlBu_r` | `RdBu_r`, `center=0` |
| `sst` | K (mostly) | `RdYlBu_r` | `RdBu_r`, `center=0` |

xarray's `.plot(center=0)` picks symmetric limits automatically — use it for anomalies so white means zero.

## Faceting members, years, leads

```python
hc = fetch("nmme/cfsv2", "precip", init="2024-02", target="MAM",
           region=[-5, 15, 33, 48], hindcast=(1993, 2016),
           year_index=True)["precip"]                 # (year, member, lat, lon), mm

hc.sel(year=2016).plot(col="member", col_wrap=4, cmap="YlGnBu")  # ensemble spread at a glance
hc.mean("member").plot(col="year", col_wrap=6, cmap="YlGnBu")    # interannual variability
```

Faceted plots create their own figure; save via the grid object: `g = da.plot(col=...); g.fig.savefig("facets.png")`.

## Regional-mean time series (area-weighted)

Don't take a plain `.mean(["lat", "lon"])` over a sizable domain — grid cells shrink with latitude. Weight by cos(lat):

```python
import numpy as np
w = np.cos(np.deg2rad(da.lat))
ts = da.weighted(w).mean(["lat", "lon"])   # dims: time or year
ts.plot(marker="o")
```

## Overlaying the region you asked for

Visual sanity check that clipping did what you meant:

```python
# bbox [lat_s, lat_n, lon_w, lon_e] — note lat first
lat_s, lat_n, lon_w, lon_e = -5, 15, 33, 48
ax.plot([lon_w, lon_e, lon_e, lon_w, lon_w],
        [lat_s, lat_s, lat_n, lat_n, lat_s],
        "r--", transform=ccrs.PlateCarree())

# shapefile/geometry region: overlay the polygon boundary
import geopandas as gpd
gdf = gpd.read_file("kenya.shp").to_crs("EPSG:4326")
gdf.boundary.plot(ax=ax, edgecolor="red", linewidth=1)
```

Polygon regions arrive with NaN outside the polygon (the bbox drives the download; the mask is applied last) — blank cells outside the outline are correct, not missing data. With `boundary="cover"` the field extends slightly past the polygon/bbox because every touched cell is kept.

## Headless / scripted use

None of this needs a display: set `MPLBACKEND=Agg` (or call `matplotlib.use("Agg")` before importing pyplot) and use `plt.savefig(...)`. acmadDL itself never opens a window.

## Downstream forecast graphics live in deepscale

Tercile-probability maps, skill-metric maps, reliability diagrams, and WMO-SVSLRF verification PDFs are deepscale's job — feed it the `(year, member, lat, lon)` arrays from `assemble()` / `year_index=True` and use `deepscale.plot_terciles`, `plot_skill_maps`, `SkillReport.to_pdf`, etc. (documented in the deepscale skill's `references/plotting-reporting.md`).
