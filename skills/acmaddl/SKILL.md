---
name: acmaddl
description: Fetch and normalize seasonal/sub-seasonal climate data (NMME, C3S/Copernicus, ERA5, CHIRPS, CHIRPS-GEFS, ERSST, CMAP, TAMSAT, IMERG, S2S) into canonical CF-aligned xarray Datasets using the acmadDL Python package. Use when fetching climate model hindcasts/forecasts or observations, fetching issuance-keyed short-range forecasts (CHIRPS-GEFS), building an observed field (e.g. ERSST SST) as a CCA predictor track, assembling multi-model ensembles, reducing a gridded field to one value per district/region (zonal aggregation), clipping to regions/shapefiles, regridding, plotting quick-look maps, managing the acmaddl cache, or debugging CDS/ECDS/IRI credentials and product errors.
license: MIT
metadata:
  author: accord-research
  package: acmadDL
compatibility: Requires Python 3.12+. Network access to data providers; some products need credentials (~/.cdsapirc, ~/.pycpt_dlauth, ~/.ecmwfapirc).
---

# acmadDL — canonical climate data acquisition

acmadDL is ACCORD's data adapter layer for seasonal climate forecasting. One call — `fetch()` — retrieves data from many providers (Copernicus CDS, ECMWF Data Store, OPeNDAP/IRI, HTTP, S3, Sheerwater/GCS, CCSR) and returns a **normalized xarray Dataset** with canonical variable names, units, and coordinates. Data stays at the source; acmaddl hosts no central copy.

- Install: `pip install acmadDL` — **import name is `acmaddl`**, not `acmadDL`.
- Extras: `[geo]` (shapefile/geometry regions), `[s3]`, `[icechunk]`, `[demo]` (plotting), `[dev]`.
- Downstream: acmaddl feeds normalized xarray into **deepscale** (downscaling/forecasting); the interface is plain xarray, no hard coupling.

## Canonical output schema (what every fetch returns)

| Concept | Canonical name | Notes |
|---|---|---|
| Latitude / longitude | `lat`, `lon` | lat always **ascending**; lon in [-180, 180] |
| Obs/reanalysis time | `time` | `datetime64`, monthly/daily |
| Forecast init | `init_time` | `datetime64` (integer `year` if `year_index=True`) |
| Forecast lead | `lead_time` | numeric, source-dependent units |
| Ensemble | `member` | integer index |
| Variables | `precip`, `temp`, `sst` | collapsed seasonal precip (`year_index=True`/`assemble`) `mm`; lead-resolved/daily precip usually `mm/day`; temp `C`; sst mostly `K` |

Typical shapes: forecasts `(init_time, lead_time, member, lat, lon)`; observations `(time, lat, lon)`; `assemble()` output `(year, member, lat, lon)`.

## Quick start

```python
from acmaddl import fetch, catalog

# Observations: ERA5 temperature over the Horn of Africa (needs ~/.cdsapirc)
ds = fetch("obs/era5", "temp", region=[-5, 15, 33, 48], hindcast=(1993, 2016))

# Seasonal hindcast: CFSv2 precip, Feb inits, MAM target, 1993-2016
ds = fetch("nmme/cfsv2", "precip", init="2010-02", target="MAM",
           region=[-5, 15, 33, 48], hindcast=(1993, 2016))

# What products exist?
catalog.list_products(include_deprecated=False)
catalog.info("nmme/cfsv2")   # full config: variables, grid, streams, adapter
```

`fetch(product, variable, ...)` key parameters (full reference: [references/api.md](references/api.md)):

- `product`: catalog id string, e.g. `"nmme/cfsv2"`, `"obs/era5"`, `"c3s/ecmwf-monthly"`. Full list: [references/products.md](references/products.md).
- `variable`: canonical name — `precip`, `temp`, or `sst` (per-product availability varies).
- `init`: `"YYYY-MM"` (seasonal) or `"YYYY-MM-DD"` (S2S daily issuance).
- `target`: season string (`"MAM"`, `"OND"`, `"DJF"`, ...) or `(start_dt, end_dt)` tuple — drives lead-month selection.
- `region`: bbox `[lat_s, lat_n, lon_w, lon_e]` (**lat first**), a `.shp` path, or a shapely/geopandas geometry.
- `hindcast`: `(start_year, end_year)` — the fetch's year range.
- `year_index=True`: reshape `init_time` → integer `year` and collapse `lead_time` (the shape deepscale consumes). Targeted precipitation becomes an exact seasonal accumulation in `mm`: monthly rates are day-weighted; daily amounts are summed. Other variables use a lead mean.
- `seasonal="mean"`: average obs over the target season per calendar year (`time` → `year`); wraparound seasons (NDJ/DJF) raise `NotImplementedError`.
- `grid_res=1.0` or `regrid_to=some_da`: regrid (mutually exclusive; `grid_res` requires `region`).
- `boundary`: `"center"` (default) or `"cover"` (keep every cell the region touches).
- `cache=True` (default): nuthatch-backed local cache; `cache=False` bypasses it.
- `months=[6,7,8,9]`: restrict an **observational** fetch to those calendar months (rejected with `init`); for one-file-per-(year,month) HTTP products it prunes the download, not just the result.
- `degenerate_attempts=1` (default): the zero-fill/truncation guard on the general path is **opt-in** — pass `>1` to validate + retry a fetch you don't trust. OPeNDAP-obs products (`obs/ersst-v5`, `obs/cmap`) validate always. Do not assume universal protection.
- `init=[...]`: a **sequence** of `"YYYY-MM-DD"` dates — only for issuance-keyed products (CHIRPS-GEFS); stacks on `init_time`.

## Multi-model assembly (feeding deepscale)

```python
from acmaddl import assemble

roster = [
    # (label, product, forecast_range, hindcast_range)
    ("CFSv2",   "nmme/cfsv2",   (2011, None), (1993, 2016)),
    ("GEOSS2S", "nmme/geoss2s", (2017, None), (1993, 2016)),
]
models = assemble(roster, "precip", init="2024-02", target="MAM",
                  region=[-5, 15, 33, 48])
# -> {label: (hindcast_da, forecast_da)}, each (year, member, lat, lon),
#    member dim guaranteed even for single-member sources
```

`assemble()` **raises on the first per-model failure** — deliberate, so a roster never silently shrinks. But real-time availability varies (a model may have a hindcast and no live forecast this month), so for tolerant rosters fetch per model and degrade explicitly:

```python
models = {}
for label, product, hind in [("CanSIPS", "nmme/cansipsic4", (1993, 2016)), ...]:
    hc = fetch(product, "precip", init=init, target=target, region=region,
               hindcast=hind, year_index=True)["precip"]
    try:
        fc = fetch(product, "precip", init=init, target=target, region=region,
                   hindcast=(init_year, init_year), year_index=True)["precip"]
    except Exception:
        fc = None                       # no live forecast — keep the hindcast
    models[label] = (hc, fc)
```

## Observed field as a predictor track

`obs_predictor()` is the observations counterpart of `assemble()`: it turns an observed gridded field (e.g. ERSST SST) into a CCA predictor, returning the same canonical `(year, member, lat, lon)` `(hindcast, forecast)` pair a model would — so an observed predictor drops into deepscale's `predictor_tracks` like a model.

```python
from acmaddl import obs_predictor

hcst, fcst = obs_predictor("obs/ersst-v5", "sst",
                           months=[6],              # single-month June predictor
                           hindcast=(1991, 2020), forecast_year=2026,
                           region=[-40, 40, -60, 60])
```

Requires exactly one of `target` (a 3-month season) or `months` (explicit calendar months). Full signature: [references/api.md](references/api.md).

## Zonal reduction — one value per region

`fetch(region=...)` dissolves a shapefile's features into a single mask ("data over this area"). Reporting asks the other question — "one number *per* district" — answered by `zonal()`, which rasterizes all N geometries once and reduces with a single groupby (needs the `geo` extra):

```python
import geopandas as gpd
from acmaddl import fetch, zonal

woredas = gpd.read_file("woredas.shp")
rain = fetch("obs/chirps-v3-dekad-tif", "precip", region="woredas.shp")

# one series per district; index by a UNIQUE id, carry the (repeatable) name as a label
per_district = zonal(rain, woredas, by="shapeID", label="ADM3_EN")  # (time, region)
```

`stat` ∈ mean/sum/min/max/median/std/count; `weights` defaults to `"area"` (= `cos(lat)`; only mean/sum use it). Output mirrors input geometry order/count (empty regions → NaN, 0 for count). Gotchas: on a coarse grid a sub-cell polygon gets NaN, and `all_touched=True` can make coverage *worse* (boundary contention); admin names are often non-unique — index `by` a unique id, carry the name via `label`. Full reference: [references/api.md](references/api.md).

## Issuance-keyed short-range forecasts (CHIRPS-GEFS)

Products with an `issuance` catalog block are keyed by issuance date, not season. Pass `init="YYYY-MM-DD"` or a sequence of dates:

```python
# same 30-Jun issuance across the reforecast era, stacked on init_time
gefs = fetch("chc/chirps-gefs-daily", "precip",
             init=[f"{y}-06-30" for y in range(2001, 2020)],
             region="ethiopia.shp")
# -> (init_time, lead_time, member?, lat, lon); lead_time in days, plus a valid_time coord
```

Output carries `init_time`/`lead_time`/`valid_time` (for `chc/chirps-gefs-15day`, `valid_time` is the window START). A season `target` cannot combine with a sequence. See [references/data-conventions.md](references/data-conventions.md).

## Regions

Three forms (see [references/api.md](references/api.md) for details):
1. **bbox** `[lat_s, lat_n, lon_w, lon_e]` — note lat before lon.
2. **shapefile path** ending `.shp` — needs `[geo]` extra; reprojected to EPSG:4326 and dissolved; bbox drives the upstream request, polygon applied as a final NaN mask.
3. **geometry** — bare shapely geometry (assumed EPSG:4326) or GeoSeries/GeoDataFrame.

There are **no named built-in regions**. For country boundaries, pull ADM0 polygons from geoBoundaries (gbOpen, CC BY 4.0) and pass the geometry straight to `region=` — no shapefile on disk required (needs the `[geo]` extra):

```python
import geopandas as gpd

ISO3 = "KEN"
url = ("https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/"
       f"gbOpen/{ISO3}/ADM0/geoBoundaries-{ISO3}-ADM0_simplified.geojson")
kenya = gpd.read_file(url).to_crs("EPSG:4326")

ds = fetch("nmme/cfsv2", "precip", init="2024-02", target="MAM", region=kenya)
```

The repo ships `fetch_country_shapefiles.py` under its `scripts/` directory, which does the same thing and writes `.shp` files, if you have the checkout and want them on disk. Polygons crossing the antimeridian are not handled — split them first. Requesting a `region` on a non-gridded product raises `ValueError`.

## Notable products by family (full list: [references/products.md](references/products.md))

- **NMME seasonal** (`nmme/*`, no creds) — CFSv2, CCSM4, CESM1, GEOSS2S, SPEAR, CanSIPS. `single_year_fetch` on CCSM4/CESM1/GEOSS2S/SPEAR chunks per year (full-range CCSR requests silently zero-fill).
- **C3S seasonal** (`c3s/*`, CDS creds) and **S2S** (`c3s/ecmwf-s2s`, ECDS creds).
- **Reanalysis** (`obs/era5`, `obs/era5-land-monthly`, CDS creds).
- **NOAA PSL OPeNDAP obs** (no creds) — `obs/ersst-v5` (2° monthly SST, ~1954-present) and `obs/cmap` (2.5° monthly precip, ~1979-present). Chunked (`max_request_years`) with the always-on truncation guard.
- **CHIRPS** (CHC/UCSB, rate-limited; also Rhiza/Sheerwater mirrors) — monthly/pentad/dekad/annual, plus new `obs/chirps-v3-dekad-tif` (final) and `obs/chirps-v3-dekad-prelim` (near-real-time tail).
- **CHIRPS-GEFS short-range forecasts** (issuance-keyed, no creds) — `chc/chirps-gefs-daily` (16 daily leads; hindcast 2001-2019, forecast 2021-present, 2020 absent) and `chc/chirps-gefs-15day` (single 15-day accumulation).
- **TAMSAT** (`obs/tamsat`, JASMIN public, no creds) — 0.0375° monthly precip, kept in `mm/month`, Africa land-only.

## Caching

- Local nuthatch cache at `~/.nuthatch/caches` (override with `ACMADDL_CACHE_DIR` before import). Scratch downloads: `~/.nuthatch/acmaddl/_tmp` (`ACMADDL_TMP_DIR`).
- Cache key: `(product, variable, date_range, region-bbox, init_months, init_date)`. Polygon masks never affect the key — same-bbox requests share cached raw data.
- CLI: `acmaddl cache list`, `acmaddl cache clear [--product X] [--yes]`.

## Health checks

```python
from acmaddl import check_product, check_all_products
check_product("nmme/cfsv2")                  # config-level check
check_product("obs/era5", probe_remote=True) # also pings the live source
```

## Credentials you may need

| Products | Credential | Notes |
|---|---|---|
| `c3s/*` (not s2s), `obs/era5*` | `~/.cdsapirc` (Copernicus CDS) | Must also accept each dataset licence in the CDS web UI; 403 names the missing one |
| `c3s/ecmwf-s2s` | ECDS account (`https://ecds.ecmwf.int/api`) | Separate from CDS; two licence layers |
| `c3s/ecmwf-seas51c` | `~/.pycpt_dlauth` (IRI) | `cptdl.setup_dlauth("email")`; IRIDL sunsets ~Oct 2026 |
| S2S MARS fallback | `~/.ecmwfapirc` | reforecast-only |
| `nmme/*`, CHIRPS, sheerwater mirrors | none | public |

## Common pitfalls (full list: [references/troubleshooting.md](references/troubleshooting.md))

- CHIRPS at UCSB rate-limits hard (4-hour IP ban above ~2 req/s) — catalog sets `request_interval` floors; for routine daily data prefer `obs/chirps-v3-daily-rhiza` (0.25°, public GCS) over native `obs/chirps-v3-daily` (~23.5 GiB/file).
- `allow_partial=False` (default) raises if any file in a multi-file fetch fails; pass `allow_partial=True` for best-effort.
- Deprecated products emit `DeprecationWarning` and alias to successors — check `catalog.info(p)["deprecated"]`.
- S3 adapter shells out to the AWS **CLI**, not boto3 — `aws` must be configured.
- If nuthatch tries to reach `gs://sheerwater-datalake/...` and 401s, ambient config is shadowing acmaddl's — see troubleshooting.
- Collapsed targeted seasonal precipitation (`year_index=True` and `assemble()`) is uniformly delivered in `mm` for NMME, C3S/CDS, and IRI sources. Lead-resolved fetches keep per-step units. CFSv2 preserves its 24 populated members; four all-NaN upstream member slots are removed.
- Real-time model availability drifts (hindcast present, live forecast absent) — probe with `check_product(p, probe_remote=True)` before committing a multi-model roster; `assemble()` raises on the first failing model by design.

## Runnable examples

- [examples/basic_fetch.py](examples/basic_fetch.py) — obs + hindcast fetch, save to NetCDF/GeoTIFF
- [examples/multi_model_assemble.py](examples/multi_model_assemble.py) — `assemble()` roster → deepscale-ready arrays
- [examples/shapefile_region.py](examples/shapefile_region.py) — country clipping, center vs cover
- [examples/zonal_districts.py](examples/zonal_districts.py) — `zonal()` one-value-per-district reduction
- [examples/chirps_gefs_issuance.py](examples/chirps_gefs_issuance.py) — issuance-keyed CHIRPS-GEFS (single + sequence)
- [examples/s2s_fetch.py](examples/s2s_fetch.py) — sub-seasonal issuance + on-the-fly reforecasts
- [examples/quick_look.py](examples/quick_look.py) — cartopy map, ensemble facets, weighted time series

## Reference files

- [references/api.md](references/api.md) — complete API: `fetch` (incl. `months`, `degenerate_attempts`, `init` sequence), `zonal`, `assemble`, `obs_predictor`, `parse_target`/`parse_init`, `catalog`, health, `validate`, `storage`, CLI
- [references/products.md](references/products.md) — every product id with adapter, variables, resolution, credentials
- [references/data-conventions.md](references/data-conventions.md) — the normalization pipeline step by step, unit conversion table, season strings
- [references/plotting.md](references/plotting.md) — visualizing fetched data: quick-look maps (cartopy), colormap/units conventions, ensemble facets, area-weighted time series, region overlays
- [references/troubleshooting.md](references/troubleshooting.md) — errors, credentials setup, rate limits, cache issues
