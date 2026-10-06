# acmadDL

An API for fetching climate, environmental, and contextual datasets

acmadDL is ACCORD's data adapter layer. A single `fetch()` call retrieves data from many different providers (Copernicus CDS, the ECMWF Data Store, OPeNDAP, HTTP, S3, and others) and returns it as a normalized xarray dataset with canonical names, units, and coordinates. Data stays at the source; acmadDL does not host a central copy of anything. Adding a new dataset or provider is a catalog and adapter change, not a rewrite of your workflow.

## Installation

```bash
pip install acmadDL
```

The distribution is published as `acmadDL`; the import name is `acmaddl`. acmadDL requires Python 3.12 or newer.

Most CDS-based products (`c3s/*`, `obs/era5`) need CDS or ECMWF Data Store credentials; see [CDS / ECDS setup](#cds--ecds-setup).

The `weather-skills/*` products run [Rhiza Research's weather-skills](https://github.com/weather-skills/weather-skills-catalog) fetchers in-process. Their packages are git-pinned, so they live in a uv dependency group rather than an extra:

```bash
uv sync --group weather-skills
```

This requires uv: the group needs acmadDL's `xarray>=2026.7` override of sheerwater's stale pin, which plain `pip` cannot apply. See [Rhiza weather-skills](#weather-skills-weather-skills).

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

A source can be healthy and still hand back a corrupt field — every cell fill, every cell zero, a few absurd values. `acmaddl.usable(field, variable="precip")` is the per-field guard for that; run it on each model's returned arrays before calibration and drop a model with a stated reason rather than let the field into the ensemble.

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

### Weather skills (`weather-skills/*`)

The `weather-skills/*` products run [Rhiza Research's weather-skills](https://github.com/weather-skills/weather-skills-catalog) fetcher scripts **unmodified, in-process**, and reshape only their output. Install the dependency group first (`uv sync --group weather-skills`; see [Installation](#installation)). What they share:

- **Forecasts are one issuance per fetch**: `init="YYYY-MM-DD"`. There is no reforecast stream, so they cannot feed `assemble()` hindcast tuples; for ECMWF S2S reforecasts keep using `c3s/ecmwf-s2s` with `reforecast=True`.
- **Member 0 is the control run** and is kept; `forecast_members` counts it.
- **`lead_time` is a `timedelta64` of native steps** (daily for ECMWF S2S, 3-6 hourly for IFS-ENS and GEFS, one step for a SubC outlook); `time` is the derived valid time. Aggregate downstream.
- **Observation windows** come from `hindcast=(y, y)` plus `months=[...]`, clipped to the skill's own published latest day; without `hindcast` you get the trailing 10 days. `weather-skills/chirps-daily` and `weather-skills/imerg-daily*` load the full global grid per day upstream, so acmadDL fetches them in 10-day chunks and crops each chunk as it loads — keep those windows short.
- **Provenance** travels with the data: their `weather_skills_history` and `weather_skills_source` attributes stay on the output (so their `provenance` skill reads our files), plus `weather_skills_name`, `weather_skills_version` and `weather_skills_pin` (the provider commit).
- `weather-skills/ifs-ens-46d` is the same ECMWF extended-range ensemble as `weather-skills/ecmwf-s2s`, served credential-free and without the 2-day embargo from the dynamical.org open catalog.
- The ICON-EU and MRMS entries are regional (Europe, CONUS); an African bounding box returns an empty selection. They are catalogued for completeness. dynamical.org's HRDPS and HRRR datasets sit on projected grids, which their fetcher refuses, so they are not catalogued.

**Forecasts**

| Product | Skill (dataset) | Variables | Cadence | Members | Credentials |
|---|---|---|---|---|---|
| `weather-skills/ecmwf-s2s` | ecmwf-fetch | precip, temp, sst, d2m, mx2t6, mn2t6, u10, v10, msl, cape, tcw | daily | 101 | ECDS |
| `weather-skills/ifs-ens-15d` | dynamical-fetch (`ecmwf-ifs-ens-forecast-15-day-0-25-degree`) | precip, temp | 3-6 hourly | 51 | none |
| `weather-skills/ifs-ens-46d` | dynamical-fetch (`ecmwf-ifs-ens-forecast-46-day-daily-1-5-degree`) | precip, temp, tmax, tmin, sst | daily | 101 | none |
| `weather-skills/ifs-ens-46d-6h` | dynamical-fetch (`ecmwf-ifs-ens-forecast-46-day-6-hourly-1-5-degree`) | precip, tmax, tmin | 6 hourly | 101 | none |
| `weather-skills/aifs-ens` | dynamical-fetch (`ecmwf-aifs-ens-forecast`) | precip, temp | 6 hourly | 51 | none |
| `weather-skills/aifs-single` | dynamical-fetch (`ecmwf-aifs-single-forecast`) | precip, temp | 6 hourly | - | none |
| `weather-skills/gefs-35d` | dynamical-fetch (`noaa-gefs-forecast-35-day`) | precip, temp | 3-6 hourly | 31 | none |
| `weather-skills/gfs` | dynamical-fetch (`noaa-gfs-forecast`) | precip, temp | 3 hourly | - | none |
| `weather-skills/icon-eu-5d` | dynamical-fetch (`dwd-icon-eu-forecast-5-day`) | precip, temp | hourly | - | none |
| `weather-skills/subc-mme-7d` | subc-mme-fetch (`7d`) | precip, temp, sst, tasmax, tasmin, tdps | 7-day outlook | - | none |
| `weather-skills/subc-mme-15d` | subc-mme-fetch (`15d`) | precip, temp, sst, tasmax, tasmin, tdps | 15-day outlook | - | none |
| `weather-skills/subc-mme-30d` | subc-mme-fetch (`30d`) | precip, temp, sst, tasmax, tasmin, tdps | 30-day outlook | - | none |

**Observations and analyses**

| Product | Skill (dataset) | Variables | Cadence | Members | Credentials |
|---|---|---|---|---|---|
| `weather-skills/gefs-analysis` | dynamical-fetch (`noaa-gefs-analysis`) | precip, temp | 3 hourly | - | none |
| `weather-skills/gfs-analysis` | dynamical-fetch (`noaa-gfs-analysis`) | precip, temp | hourly | - | none |
| `weather-skills/imerg-early-30min` | dynamical-fetch (`nasa-imerg-analysis-early`) | precip | half-hourly | - | none |
| `weather-skills/imerg-late-30min` | dynamical-fetch (`nasa-imerg-analysis-late`) | precip | half-hourly | - | none |
| `weather-skills/mrms-hourly` | dynamical-fetch (`noaa-mrms-conus-analysis-hourly`) | precip | hourly | - | none |
| `weather-skills/chirps-daily` | chirps-fetch | precip | daily | - | none |
| `weather-skills/imerg-daily` | imerg-fetch (`late`) | precip | daily | - | Earthdata |
| `weather-skills/imerg-daily-final` | imerg-fetch (`final`) | precip | daily | - | Earthdata |
| `weather-skills/era5` | arco-era5-fetch | precip, temp, sst | hourly | - | none |
| `weather-skills/oisst-daily` | oisst-fetch | sst | daily | - | none |
| `weather-skills/smap-daily` | smap-fetch | soil_moisture | daily | - | Earthdata |
| `weather-skills/cmip6` | cmip6-fetch (`ssp245`) | precip, temp | monthly | - | none |

`weather-skills/ecmwf-s2s` reads ECDS credentials from `ECMWF_DATASTORES_URL`/`ECMWF_DATASTORES_KEY`, `~/.ecmwfdatastoresrc`, or an ECDS-pointing `~/.cdsapirc` (a Copernicus CDS token is refused with instructions). IMERG daily and SMAP use a NASA Earthdata Login token, in `~/.earthdatarc` or `EARTHDATA_TOKEN` (username/password are refused), see [NASA Earthdata setup](#nasa-earthdata-setup); IMERG also needs the "NASA GESDISC DATA ARCHIVE" application approved in your Earthdata profile.

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

### NASA Earthdata setup

Required for `weather-skills/imerg-daily`, `weather-skills/imerg-daily-final` and `weather-skills/smap-daily`. Earthdata is NASA's single login for GES DISC (IMERG) and NSIDC (SMAP); it is free and separate from the ECMWF services above.

1. Create an account at <https://urs.earthdata.nasa.gov>.
2. For IMERG, approve the "NASA GESDISC DATA ARCHIVE" application once: profile → Applications → Authorized Apps → Approve More Applications. Without it IMERG downloads return 401 even with valid credentials. SMAP needs no extra approval.
3. Generate a token (profile → Generate Token; Earthdata Login user tokens last about 60 days) and put it, alone, in `~/.earthdatarc`:

   ```bash
   echo '<YOUR-EARTHDATA-TOKEN>' > ~/.earthdatarc
   chmod 600 ~/.earthdatarc
   ```

   Or set `EARTHDATA_TOKEN` per session or in CI, which takes precedence over the file. acmadDL maps the file onto that variable for the duration of each call, which is what the `earthaccess` library reads. **Only tokens are accepted**: `EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD` and a `~/.netrc` entry are refused on purpose, since a revocable, short-lived token is the safer credential to keep around.

   acmadDL checks that a token exists before running the skill and raises `WeatherSkillError` otherwise; left to itself, the IMERG script would wait on an interactive login prompt.

4. Test the connection:

   ```bash
   uv run pytest -m "integration and not cds" tests/test_weather_skills_integration.py -k "imerg_daily or smap" -v
   ```

### ECDS for `weather-skills/ecmwf-s2s`

The ECMWF S2S skill requires `ECMWF_DATASTORES_URL` / `ECMWF_DATASTORES_KEY` in the environment. acmadDL fills them for the duration of the call from the standard file `~/.ecmwfdatastoresrc` (either the bare key on one line, or `url:` and `key:` lines in the client's own cdsapirc style), or from an ECDS-pointing `~/.cdsapirc` or `CDSAPI_URL`/`CDSAPI_KEY`, so the ECDS setup above is enough; a Copernicus CDS key is refused with instructions rather than sent.

## Cache configuration

acmadDL caches adapter downloads locally using [Nuthatch](https://github.com/rhiza-research/nuthatch). Cache files live in `~/.nuthatch/acmaddl` by default (configured in `pyproject.toml`).

```bash
acmaddl cache list                        # inspect the cache
acmaddl cache clear                       # clear everything
acmaddl cache clear --product nmme/cfsv2  # clear one product (with confirmation)
```

## MCP server

acmadDL ships a [Model Context Protocol](https://modelcontextprotocol.io) server so AI agents (Claude Code, Beaker, Codex, and any other MCP client) can browse the catalog, check product health, and fetch data as tools.

```bash
pip install 'acmadDL[mcp]'
acmaddl-mcp                     # stdio transport (what MCP clients spawn)
acmaddl-mcp --transport streamable-http --port 8000 --stateless
```

The server implements MCP protocol revision 2026-07-28 (via the `mcp` 2.x SDK): stateless core, `server/discover`, cache freshness hints on list results, and Streamable HTTP as the only network transport (HTTP+SSE is deprecated). Background fetch handles (`start_fetch` job ids) are persisted under the workdir, so `--stateless` HTTP deployments and server restarts keep them valid.

Register it with a client, for example in Claude Code:

```bash
claude mcp add acmaddl -- acmaddl-mcp
```

Tools: `list_products`, `describe_product`, `check_product`, `check_all_products`, `fetch`, `start_fetch` / `fetch_status` / `list_jobs` (background fetch for slow CDS queues), `describe_dataset`, and `zonal`. Arrays never cross the protocol: `fetch` and `zonal` write NetCDF under `ACMADDL_MCP_WORKDIR` (default `~/.acmaddl/mcp`) and return the path plus a compact summary, which downstream tools such as the [africas2s](https://github.com/ACMAD-Niamey/africas2s) MCP server accept by path. The Agent Skill and catalog are also exposed as resources (`acmaddl://skill`, `acmaddl://skill/references/{name}`, `acmaddl://catalog`).

The server is a thin wrapper over the public API, so credentials, caching, and product behaviour are exactly as documented above. Inspect it interactively with `npx @modelcontextprotocol/inspector acmaddl-mcp`.

## Development setup

```bash
git clone https://github.com/ACMAD-Niamey/acmadDL.git
cd acmadDL
uv sync
```

## Relationship to DeepScale

acmadDL handles ingestion and normalization; [DeepScale](https://github.com/accord-research/deepscale) handles downscaling and skill evaluation. Their interface is standardized xarray.
