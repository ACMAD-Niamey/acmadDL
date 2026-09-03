# acmadDL

An API for fetching climate, environmental, and contextual datasets

acmadDL is ACCORD's data adapter layer. A single `fetch()` call retrieves data from many different providers (Copernicus CDS, the ECMWF Data Store, OPeNDAP, HTTP, S3, and others) and returns it as a normalized xarray dataset with canonical names, units, and coordinates. Data stays at the source; acmadDL does not host a central copy of anything. Adding a new dataset or provider is a catalog and adapter change, not a rewrite of your workflow.

## Installation

```bash
pip install acmadDL
```

The distribution is published as `acmadDL`; the import name is `acmaddl`. acmadDL requires Python 3.12 or newer.

Most CDS-based products (`c3s/*`, `obs/era5`) need CDS or ECMWF Data Store credentials; see [CDS / ECDS setup](#cds--ecds-setup).

## Core API

```python
import acmaddl

ds = acmaddl.fetch(
    product="nmme/cfsv2",
    variable="precip",
    init="2025-02",
    target="MAM",
    region=[-12, 6, 28, 42],
    hindcast=(1993, 2016),
    verbose=True,
    progress=True,
)
```

You pass a canonical `product` and `variable` plus optional temporal and spatial selectors. You get back a CF-aligned xarray dataset, ready for analysis or modeling. Results are returned in memory by default, or written to a local path or S3.

Verify an install and list what is available:

```python
import acmaddl
acmaddl.check_all_products()          # config checks for every product
acmaddl.catalog.list_products()       # list product ids
```

### Normalized coordinates

Every dataset comes back with canonical coordinate names regardless of the upstream source:

| Coordinate | Applies to | Description |
|------------|-----------|-------------|
| `lat`, `lon` | all products | Spatial axes (latitude ascending) |
| `time` | observations / reanalysis | Monthly time axis (`datetime64`) |
| `init_time` | forecasts | Initialization time (`datetime64`) |
| `lead_time` | forecasts | Lead time (numeric, units vary by source) |
| `member` | forecasts | Ensemble member index |

Numeric time encodings (for example "months since 1960-01-01") are decoded to `datetime64` automatically.

### Region input

`region` accepts three forms:

| Form | Example | Behaviour |
|------|---------|-----------|
| bbox | `region=[-12, 6, 28, 42]` | `[lat_s, lat_n, lon_w, lon_e]` |
| shapefile | `region="kenya.shp"` | bounding box slices upstream; result masked to the polygon |
| geometry | `region=gdf.geometry` | shapely geometry or geopandas `GeoSeries`, same masking |

Shapefile and geometry inputs need the `geo` extra (`pip install 'acmadDL[geo]'`). The polygon is reprojected to EPSG:4326 and dissolved, so multi-feature files (for example an archipelago) clip correctly. Cells outside the polygon come back as `NaN`.

```python
import acmaddl

# Clip to a country boundary; values outside Kenya become NaN.
ds = acmaddl.fetch(
    product="nmme/cfsv2",
    variable="precip",
    init="2025-02",
    target="MAM",
    region="kenya.shp",
    hindcast=(1993, 2016),
)
```

By default a grid cell is included only if its centre lies inside the region (`boundary="center"`, the xarray/CDO/rasterio convention, and unbiased for area means). Pass `boundary="cover"` to keep every cell the region touches (matches rasterio's `all_touched=True`), which is useful for display or coarse grids where center-based selection can drop a country's thin tips. This applies to bbox and shapefile/geometry inputs alike.

```python
ds = acmaddl.fetch(..., region="kenya.shp", boundary="cover")
```

Note: polygons crossing the plus/minus 180 degree antimeridian are not yet handled (the derived bounding box spans the full longitude range). Split such geometries at the antimeridian before passing them in.

### Multiple predictor domains

Seasonal forecasting workflows often use two predictor domains from the same model, for example a large sea-surface-temperature domain and a smaller regional precipitation domain. Make one `fetch()` call per domain; there is no combined helper, because the domains differ in extent and variable.

```python
import acmaddl

# SST predictor: large tropical domain
sst_predictor = acmaddl.fetch(
    product="nmme/geoss2s", variable="sst",
    init="2025-02", target="MAM",
    region=[-20, 20, 30, 180], hindcast=(1993, 2016),
)

# Precipitation predictor: regional domain
prcp_predictor = acmaddl.fetch(
    product="nmme/geoss2s", variable="precip",
    init="2025-02", target="MAM",
    region=[-20, 20, 10, 75], hindcast=(1993, 2016),
)

# Predictand: observations
predictand = acmaddl.fetch(
    product="obs/chirps-v2-monthly", variable="precip",
    target="MAM", region=[-12, 15, 22, 52], hindcast=(1993, 2016),
)
```

### Health checks

```python
import acmaddl

acmaddl.check_product("nmme/cfsv2")             # one product, config only
acmaddl.check_all_products()                    # all products, config only
acmaddl.check_all_products(probe_remote=True)   # also probe the live source
```

Each result includes `product`, `adapter`, `healthy`, `kind`, `message`, and `checked_at`.

## Available products

How to read the tables. Hindcast is each model's fixed reforecast period, not a fetch cap: real-time forecasts run past it to the present. Forecast is the live-verified real-time availability, shown as `year–present` (ongoing) or `start–end (retired)` when the pinned system version was superseded (hindcasts still fetch, but no new forecasts issue). Members (F/H) are the real-time-forecast and reforecast ensemble sizes, which differ. `†` marks a deprecated access route. Full field conventions are documented at the top of [`src/acmaddl/catalog.yaml`](src/acmaddl/catalog.yaml).

Several C3S entries pin a system version whose real-time stream has ended (a 2026 forecast init returns no data); they still fetch hindcasts. CMCC's live stream has moved to `c3s/cmcc-sps4`. JMA and UKMO do not yet have an active-forecast entry on their current systems (JMA CPS4, UKMO 605).

Collapsed targeted seasonal precipitation (`year_index=True`, including
`assemble()`) is delivered in `mm` across NMME, C3S/CDS, and IRI products.
Lead-resolved fetches retain honest per-step units (usually `mm/day`). Monthly
rates are calendar-weighted and daily amounts are summed. Native ensemble spread is preserved;
`nmme/cfsv2` returns the 24 populated members and removes four structurally
empty trailing slots exposed by the upstream endpoint.

### Seasonal forecast, NMME

| Product | Model / system | Host | Adapter | Variables | Cadence | Members (F/H) | Hindcast | Forecast |
|---|---|---|---|---|---|---|---|---|
| `nmme/cansipsic4` | ECCC CanSIPS-IC4 | Columbia CCSR | `ccsr` | precip, temp, sst | monthly | 40 / 40 | 1990–2024 | 2024–present |
| `nmme/ccsm4` | NCAR/COLA CCSM4 | Columbia CCSR | `ccsr` | precip, temp, sst | monthly | 10 / 10 | 1982–2026 | 2014–present |
| `nmme/cesm1` | NCAR CESM1 | Columbia CCSR | `ccsr` | precip, temp, sst | monthly | 10 / 10 | 1982–2026 | 2017–present |
| `nmme/geoss2s` | NASA GEOS-S2S | Columbia CCSR | `ccsr` | precip, temp, sst | monthly | 10 / 4 | 1981–2017 | 2019–present |
| `nmme/spear` | GFDL SPEAR | Columbia CCSR | `ccsr` | precip, temp, sst | monthly | 30 / 15 | 1991–2020 | 2021–present |
| `nmme/spearb` | GFDL SPEARb | Columbia CCSR | `ccsr` | sst | monthly | 30 / 15 | 1991–2020 | 2021–present |
| `nmme/cfsv2` † | NCEP CFSv2 | IRI Data Library | `opendap` | precip, temp, sst | monthly | 28 / 28 | 1982–2010 | 2011–present |

### Seasonal forecast, C3S

| Product | Model / system | Host | Adapter | Variables | Cadence | Members (F/H) | Hindcast | Forecast |
|---|---|---|---|---|---|---|---|---|
| `c3s/cmcc` | CMCC SPSv3.5 (sys 35) | Copernicus CDS | `cds` | precip, temp, sst | monthly | 50 / 40 | 1993–2016 | 2020–2025 (retired) |
| `c3s/cmcc-daily` | CMCC SPSv3.5 (sys 35) | Copernicus CDS | `cds` | precip, temp, sst | daily | 50 / 40 | 1993–2016 | 2020–2025 (retired) |
| `c3s/cmcc-sps4` | CMCC SPS4 (sys 4) | Copernicus CDS | `cds` | precip, temp, sst | monthly | 50 / 30 | 1993–2024 | 2025–present |
| `c3s/cmcc-sps4-daily` | CMCC SPS4 (sys 4) | Copernicus CDS | `cds` | precip, temp, sst | daily | 50 / 30 | 1993–2024 | 2025–present |
| `c3s/dwd` | DWD GCFS2.2 (sys 22) | Copernicus CDS | `cds` | precip, temp, sst | monthly | 50 / 30 | 1993–2023 | 2023–present |
| `c3s/dwd-daily` | DWD GCFS2.2 (sys 22) | Copernicus CDS | `cds` | precip, temp, sst | daily | 50 / 30 | 1993–2023 | 2023–present |
| `c3s/dwd-gcfs21` | DWD GCFS2.1 (sys 21) | Copernicus CDS | `cds` | precip, temp, sst | monthly | 50 / 30 | 1993–2019 | 2020–2025 (retired) |
| `c3s/eccc-cansips` | ECCC GEM5-NEMO (sys 3) | Copernicus CDS | `cds` | precip, temp, sst | monthly | 10 / 10 | 1990–2020 | 2021–2024 (retired) |
| `c3s/eccc-cansipsv3` | ECCC CanESM5.1 (sys 4) | Copernicus CDS | `cds` | precip, temp, sst | monthly | 20 / 20 | 1980–2023 | 2024–present |
| `c3s/eccc-daily` | ECCC CanESM5.1 (sys 4) | Copernicus CDS | `cds` | precip, temp, sst | daily | 20 / 20 | 1980–2023 | 2024–present |
| `c3s/ecmwf` | ECMWF SEAS5 (sys 51) | Copernicus CDS | `cds` | precip, temp, sst | monthly | 51 / 25 | 1981–2016 | 2017–present |
| `c3s/ecmwf-monthly` | ECMWF SEAS5 (sys 51) | Copernicus CDS | `cds` | precip, temp, sst | monthly | 51 / 25 | 1981–2016 | 2017–present |
| `c3s/jma` | JMA CPS3 (sys 3) | Copernicus CDS | `cds` | precip, temp, sst | monthly | 155 / 10 | 1991–2020 | 2022–2026 (retired) |
| `c3s/jma-cps2` | JMA CPS2 (sys 2) | Copernicus CDS | `cds` | precip, temp, sst | monthly | 13 / 10 | 1981–2016 | 2015–2022 (retired) |
| `c3s/meteofrance` | Météo-France Sys 9 | Copernicus CDS | `cds` | precip, temp, sst | monthly | 51 / 31 | 1993–2024 | 2025–present |
| `c3s/meteofrance-daily` | Météo-France Sys 9 | Copernicus CDS | `cds` | precip, temp, sst | daily | 51 / 31 | 1993–2024 | 2025–present |
| `c3s/ukmo` | UKMO GloSea6 GC3.2 (sys 604) | Copernicus CDS | `cds` | precip, temp, sst | monthly | 62 / 28 | 1993–2016 | 2025–2026 (retired) |
| `c3s/ukmo-daily` | UKMO GloSea6 GC3.2 (sys 604) | Copernicus CDS | `cds` | precip, temp, sst | daily | 7 / 7 | 1993–2016 | 2025–2026 (retired) |
| `c3s/ecmwf-seas51c` † | - | IRI Data Library | `iridl` | precip, sst | monthly | 51 / 25 | 1993–2016 | - |

### Sub-seasonal forecast

| Product | Model / system | Host | Adapter | Variables | Cadence | Members (F/H) | Hindcast | Forecast |
|---|---|---|---|---|---|---|---|---|
| `c3s/ecmwf-s2s` | ECMWF S2S | ECMWF Data Store | `cds` | precip, sst | twice weekly | 50 / 11 | - | on-the-fly |

`c3s/ecmwf-s2s` is date-keyed: call it with `init="YYYY-MM-DD"` (the issuance date). Its reforecasts are generated on the fly, so there is no fixed hindcast window. It uses the ECMWF Data Store (`ecds.ecmwf.int`), a separate service from the Copernicus CDS; see [ECMWF Data Store (ECDS) setup](#ecmwf-data-store-ecds-setup).

### Reanalysis

| Product | Model / system | Host | Adapter | Variables | Cadence | Members (F/H) | Hindcast | Forecast |
|---|---|---|---|---|---|---|---|---|
| `obs/era5` | - | Copernicus CDS | `cds` | temp, precip, sst | monthly | - | 1940–2025 | - |
| `obs/era5-land-monthly` | - | Copernicus CDS | `cds` | precip, temp, pev | monthly | - | 1950–2025 | - |

### Observation

| Product | Model / system | Host | Adapter | Variables | Cadence | Members (F/H) | Hindcast | Forecast |
|---|---|---|---|---|---|---|---|---|
| `obs/chirps-live-rhiza` | - | Rhiza/Sheerwater | `sheerwater` | precip | daily | - | - | - |
| `obs/chirps-v2-annual` | - | UCSB CHC | `http` | precip | annual | - | 1981–2024 | - |
| `obs/chirps-v2-daily` | - | UCSB CHC | `http` | precip | daily | - | 1981–2025 | - |
| `obs/chirps-v2-dekad` | - | UCSB CHC | `http` | precip | dekad | - | 1981–2025 | - |
| `obs/chirps-v2-dekadal-rhiza` | - | Rhiza/Sheerwater | `sheerwater` | precip | dekadal-rolling | - | 1997–2024 | - |
| `obs/chirps-v2-monthly` | - | UCSB CHC | `http` | precip | monthly | - | 1981–2025 | - |
| `obs/chirps-v2-pentad` | - | UCSB CHC | `http` | precip | pentad | - | 1981–2025 | - |
| `obs/chirps-v3-annual` | - | UCSB CHC | `http` | precip | annual | - | 1981–2025 | - |
| `obs/chirps-v3-daily` | - | UCSB CHC | `http` | precip | daily | - | 1998–2025 | - |
| `obs/chirps-v3-daily-rhiza` | - | Rhiza/Sheerwater | `sheerwater` | precip | daily | - | 2000–2024 | - |
| `obs/chirps-v3-dekad` | - | UCSB CHC | `http` | precip | dekad | - | 1981–2025 | - |
| `obs/chirps-v3-monthly` | - | UCSB CHC | `http` | precip | monthly | - | 1981–2025 | - |
| `obs/chirps-v3-pentad` | - | UCSB CHC | `http` | precip | pentad | - | 1981–2025 | - |
| `obs/ghcn` | - | Rhiza/Sheerwater | `sheerwater` | precip | monthly | - | 1981–2024 | - |
| `obs/gpcc-first-guess` | - | DWD open data | `http` | precip | monthly | - | 2013–2026 | - |
| `obs/gpcc-monitoring-v2020` | - | DWD open data | `http` | precip | monthly | - | 1982–2026 | - |
| `obs/gpcp-v2-3` | - | NOAA NCEI | `http` | precip | monthly | - | 1979–2026 | - |
| `obs/imerg` | - | Rhiza/Sheerwater | `sheerwater` | precip | monthly | - | 2000–2024 | - |
| `obs/oisst-v2-highres` | - | NOAA PSL | `opendap` | sst | monthly | - | 1981–2026 | - |

For `nmme/cfsv2 †`, the model is still an active NMME member (live on CPC FTP); only its IRI Data Library access route is deprecated (IRIDL shutdown around October 2026), with no successor yet.

Output is always NetCDF (extensible to Zarr and GeoTIFF).

## CDS / ECDS setup

acmadDL's `cds` adapter talks to two distinct ECMWF endpoints, which are separate services with separate accounts, API keys, and licence-acceptance flows:

- Copernicus Climate Data Store (`cds.climate.copernicus.eu`): most `c3s/*` products and `obs/era5`.
- ECMWF Data Store (`ecds.ecmwf.int`): the newer service, currently used by `c3s/ecmwf-s2s`.

Each product's catalog entry selects its endpoint. You need credentials and accepted licences for each endpoint you fetch from.

### Copernicus CDS setup

For Copernicus CDS products (all `c3s/*` except `c3s/ecmwf-s2s`, plus `obs/era5`):

1. Create a CDS account and API key at <https://cds.climate.copernicus.eu>.
2. Add credentials to `~/.cdsapirc`:

   ```bash
   cat > ~/.cdsapirc << 'EOF'
   url: https://cds.climate.copernicus.eu/api
   key: <YOUR-CDS-API-KEY>
   EOF
   ```

3. Accept the required dataset licences in the CDS web UI before your first download. Each dataset page has a "Terms of use" section you tick once per account. If `acmaddl.fetch()` fails with a 403, the error names the dataset(s) still missing acceptance.

### ECMWF Data Store (ECDS) setup

Required for `c3s/ecmwf-s2s` (and any future product whose catalog entry uses `cds_url: "https://ecds.ecmwf.int/api"`). ECDS is a separate service; you cannot reuse Copernicus CDS credentials.

1. Create an ECMWF account at <https://www.ecmwf.int/>, log in to <https://ecds.ecmwf.int/>, and copy your API key from your profile settings.
2. Point `cdsapi` at the ECDS endpoint. If you only use ECDS, the simplest setup is to replace `~/.cdsapirc`:

   ```bash
   cat > ~/.cdsapirc << 'EOF'
   url: https://ecds.ecmwf.int/api
   key: <YOUR-ECDS-API-KEY>
   EOF
   ```

   If you also use the Copernicus CDS, keep one endpoint in `~/.cdsapirc` and pass the other via the `CDSAPI_URL` / `CDSAPI_KEY` environment variables per session.
3. Accept both layers of ECDS licences (ECDS returns 403 until both are accepted):
   - The site-wide Terms of Use, accepted once per account from the ECDS profile settings.
   - The dataset licence, ticked on the dataset's own page. For S2S: <https://ecds.ecmwf.int/datasets/s2s-forecasts?tab=download#manage-licences>. Other datasets follow the same `https://ecds.ecmwf.int/datasets/<dataset-id>?tab=download#manage-licences` pattern.

   As with the Copernicus CDS, a skipped licence produces a 403 that names exactly which licence is missing.
4. Test the connection:

   ```bash
   uv run pytest tests/test_integration.py::test_fetch_c3s_ecmwf_s2s_precip -v
   ```

## Cache configuration

acmadDL caches adapter downloads locally using [Nuthatch](https://github.com/rhiza-research/nuthatch). Cache files live in `~/.nuthatch/acmaddl` by default (configured in `pyproject.toml`).

```bash
acmaddl cache list                        # inspect the cache
acmaddl cache clear                       # clear everything
acmaddl cache clear --product nmme/cfsv2  # clear one product (with confirmation)
```

## Development setup

```bash
git clone https://github.com/ACMAD-Niamey/acmadDL.git
cd acmadDL
uv sync
```

## Relationship to DeepScale

acmadDL handles ingestion and normalization; [DeepScale](https://github.com/accord-research/deepscale) handles downscaling and skill evaluation. Their interface is standardized xarray.
