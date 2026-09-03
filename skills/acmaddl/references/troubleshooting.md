# Troubleshooting

## Credentials setup

### Copernicus CDS (`c3s/*` except s2s, `obs/era5*`)

`~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api
key: <your-key>
```
Then accept each dataset's licence once in the CDS web UI (a 403 error names the missing dataset licence). acmadDL creates the cdsapi client with `retry_max=12, sleep_max=30, timeout=60` — cdsapi's defaults (500 retries / 120 s sleep) would hide real errors behind ~17 hours of retrying.

### ECMWF Data Store / ECDS (`c3s/ecmwf-s2s`)

A **separate** service from Copernicus CDS — CDS credentials do not work. Create an ECDS account, then point credentials at `https://ecds.ecmwf.int/api` via `~/.cdsapirc` or the `CDSAPI_URL`/`CDSAPI_KEY` env vars. Two licence layers must be accepted: the site-wide Terms of Use and the per-dataset licence.

### IRI Data Library (`c3s/ecmwf-seas51c`, iridl adapter)

Needs `~/.pycpt_dlauth` — create an IRI account and run `cptdl.setup_dlauth("<email>")`. A missing file raises `FileNotFoundError`; an HTML response means auth failed. acmadDL writes a temporary `~/.dodsrc`. IRIDL is expected to sunset around Oct 2026 (`c3s/ecmwf-seas51c` is deprecated after 2026-04-30; successor `c3s/ecmwf-monthly`).

### ECMWF legacy MARS (S2S reforecast fallback)

Needs `~/.ecmwfapirc`. The adapter raises unless called with `reforecast=True` and both `hindcast` and `region`.

### S3 products

The s3 adapter shells out to the **AWS CLI** (`aws s3 ls/cp`) — configure `aws` itself; boto3 is not used for fetching (the `[s3]` extra's boto3 is only for icechunk profile handling and s3fs saving).

## Common errors

| Symptom | Cause / fix |
|---|---|
| 403 from CDS naming a dataset | Licence not accepted — accept it in the CDS web UI |
| HTTP 403 "CrowdSec Ban" from data.chc.ucsb.edu | CHIRPS rate limit tripped; IP banned ~4 h. Raise `request_interval`, reduce parallelism, or switch to `obs/chirps-*-rhiza` mirrors |
| `RuntimeError: N/M file(s) failed; refusing to return partial data` | Default `allow_partial=False`; pass `allow_partial=True` for best-effort, or retry |
| `ValueError` on `region=` | Product is non-gridded (no lat/lon dims) — spatial subset impossible |
| `ImportError` mentioning `acmadDL[geo]` | Shapefile/geometry region needs the `geo` extra (geopandas, rasterio) |
| `NotImplementedError` from `seasonal="mean"` | Wraparound season (NDJ/DJF) not supported by seasonal averaging |
| `ValueError`: `grid_res` and `regrid_to` | Mutually exclusive; `grid_res` also requires `region` |
| `KeyError` from `catalog.info` | Unknown product id — check `catalog.list_products()` |
| `DeprecationWarning` on fetch | Product is an alias or date-deprecated — see `catalog.info(p)` for `successor` |
| 401 / interactive auth prompt / fsspec `Protocol not known` touching `gs://sheerwater-datalake/...` | Ambient nuthatch config shadowing — see below |
| GRIB decode errors | MARS/S2S need `cfgrib` + `eccodeslib` (installed by default); acmaddl uses `decode_timedelta=False` to avoid an xarray non-nanosecond-timedelta assertion |
| `ERR: DAP DATADDS packet is apparently too short` printed to **stderr** | An OPeNDAP server (seen with NOAA PSL/THREDDS) truncated a large request; netCDF4 then returns a **plausible zero-filled array** — right shape/dtype, wrong data. See the silent-truncation guard section below for what is and isn't caught automatically. |
| `DegenerateResponseError` | acmaddl's guard rejected a bitwise-constant / zero-filled (or, on the OPeNDAP obs path, all-NaN) response before it reached the cache. For OPeNDAP obs (`obs/ersst-v5`, `obs/cmap`) this is always on — commonly it means a truncated chunk, or (for the ocean-only `obs/ersst-v5`) a **land-only bbox** that is legitimately all-NaN (request an ocean-containing region). On the general path it only fires when you passed `degenerate_attempts>1`; acmaddl already retried with the cache bypassed, so a persistent error means the source itself is returning bad data. |
| `AttributeError` on sheerwater's `chirps_raw_live` | Upstream sheerwater removed the near-real-time function; `obs/chirps-live-rhiza` fails until it is restored — for current-season CHIRPS use the native CHC products (subject to their rate limits) |
| Full CrowdSec 403 on **every** CHC path including the site root | The IP ban is site-wide and time-limited (hours to days) — wait it out; don't retry, it can extend the ban |

## OPeNDAP silent truncation and the degenerate-response guard

A large OPeNDAP request against NOAA PSL / THREDDS can be truncated server-side: netCDF4 prints `ERR: DAP DATADDS packet is apparently too short` to **stderr** and then hands back a **plausible zero-filled array** (right shape and dtype, wrong data) with **no exception**. Left unchecked it reaches the cache and re-serves on retry.

acmadDL now ships a guard (`reject_if_degenerate` / `DegenerateResponseError`) that detects a bitwise-constant, zero-filled, or all-NaN response. **It is not universal — know which path you are on:**

- **OPeNDAP observational chunk path** (`obs/ersst-v5`, `obs/cmap`, `obs/oisst-v2-highres`): the guard is **always on** and strict (`reject_all_nan=True`). Requests are bounded by `max_request_years` (5, or 2 for 0.25° OISST) **and** a response-size budget (4M values), and a seam-crossing region is issued as one contiguous request per segment; every piece is validated. A truncated *or* all-NaN chunk raises `DegenerateResponseError` before caching. This is why a **land-only bbox for `obs/ersst-v5` or `obs/oisst-v2-highres`** (ocean-only SST fields) raises — request ocean cells.
- **General fetch path** (every other product/adapter): the guard is **opt-in**. With the default `degenerate_attempts=1` acmaddl does **not** validate — a truncated response can still poison the cache. Pass `degenerate_attempts>1` on a fetch you don't trust to turn on validation (cache-miss responses validated before caching, cache hits re-validated to catch pre-existing poison) plus that many cache-bypass retries.
- **CCSR NMME** models use a related mechanism: entries with `single_year_fetch: true` chunk per year because the full-range request overflows the CCSR server and silently zero-fills.

If you suspect a poisoned entry from before you enabled the guard, purge it with `acmaddl cache clear --product X` and re-fetch with `degenerate_attempts>1`. Independent of the guard, sanity-check any large remote pull (non-zero variance, expected land/ocean mask) before trusting it.

## Cache issues

- Cache root: `~/.nuthatch/caches` (override with `ACMADDL_CACHE_DIR` **before importing acmaddl**; power users: `NUTHATCH_ROOT_FILESYSTEM` / `NUTHATCH_LOCAL_FILESYSTEM`, which acmaddl only `setdefault`s). Scratch downloads: `~/.nuthatch/acmaddl/_tmp` (`ACMADDL_TMP_DIR`) — deliberately not the system tempdir, because macOS reaps `/var/folders/.../T/` and breaks pickled lazy datasets.
- Cache key: `(product, variable, date_range, region-bbox, init_months, init_date, target_months)` under namespace `acmaddl`, versioned by `_CACHE_VERSION` (currently **9**). A version bump invalidates all cached entries. Version 9 landed with the fix below, because it changes adapter output for an existing key. Polygon geometries never enter the key.
- `cache=False` bypasses nuthatch entirely (straight to adapter).
- Inspect / clear: `acmaddl cache list`, `acmaddl cache clear [--product X] [--yes]` (wraps `python -m nuthatch list/delete --namespace acmaddl`).
- **Sheerwater shadowing gotcha:** when `sheerwater` is co-installed, nuthatch's upward config search can escape `site-packages/acmaddl/` and adopt sheerwater's `nuthatch.toml`, which points the cache root at a private GCS bucket (`gs://sheerwater-datalake/...`) → 401s or fsspec crashes for users without those credentials. acmadDL defends twice: it ships a shadow `src/acmaddl/nuthatch.toml` (`filesystem = "file://~/.nuthatch/caches"`) and pins the env vars at import. If you still see GCS errors, check for an ambient `~/.nuthatch.toml` or env vars pointing at GCS. If nuthatch tries the private bucket first and **prompts interactively** (fatal in scripts/CI), skip it explicitly so reads fall through to the public anonymous mirror:

  ```toml
  # ~/.nuthatch.toml
  [tool.nuthatch]
  skipped_filesystems = ["gs://sheerwater-datalake/caches"]
  ```
- Datasets are **eagerly loaded** (`.load()`) before caching on purpose: pickling a lazy dataset stores only a "reopen this temp file" recipe that breaks across sessions.

## HTTP-product gotchas

- **Longitude convention on a 0..360 `http` source (fixed in `_CACHE_VERSION` 9).** The `http` adapter crops to the region inside each per-file download, *before* normalize sees the data. That crop used a plain `slice(lon_w, lon_e)`, so a bbox written in -180..180 against a 0..360 grid kept only its eastern half — a West African `[-8, 8]` request returned three longitudes instead of six — and normalize could not recover cells already dropped. It now selects longitude through `normalize.select_lon`, the same convention-aware helper the OPeNDAP adapter uses, and matches `obs/cmap` cell-for-cell. `obs/gpcp-v2-3` is the first `http` product on a 0..360 grid (CHIRPS and TAMSAT are both -180..180), which is why this surfaced now. **Any cache entry written before version 9 for a seam-crossing bbox holds the truncated domain** — the version bump retires them.
- **`FileNotFoundError: no file matching 'gpcp_v02r03_monthly_d2026NN_*.nc'`.** The requested month exists only as NCEI's `-preliminary` stream; `obs/gpcp-v2-3` deliberately matches the final product only. Request an earlier month, or add a `-prelim` entry.
- **`DegenerateResponseError` from a NOAA PSL product (what used to cause it, and what to check now).** Two separate defects made PSL zero-fill a request; both are fixed, and the always-on guard meant neither could ever return silent zeros.

  1. **A seam-crossing region was one malformed request.** `select_lon` covers a box spanning the source's longitude seam by concatenating two slices; lazily, over DAP, that became a single request PSL zero-filled **at every size** — an 80°×90° OISST box truncated even at 230k values, while the identical cells loaded fine (up to 2.76M values) as one contiguous request. The adapter now plans segments with `normalize.lon_selection_bounds` and issues one contiguous request per segment. This was the real cause of "large OISST requests fail"; region *size* was a red herring.
  2. **A year cap cannot bound a response.** One time step's size depends on the region's cell count, so `max_request_years: 2` is ~25M values globally at 0.25°. Requests are now also bounded by a size budget of 4M values — about half the measured ceiling (6.2M/24.9 MB loaded, 8.3M/33.2 MB truncated: a ~32 MiB server cap).

  If you still see this: check the region isn't land-only for an SST product (legitimately all-NaN, see above), and that the source itself isn't serving a bad period.
- **GPCC time axes.** Both `obs/gpcc-*` entries set `time_from_pattern`, taking time from the filename because their in-file axes are unusable (`day as %Y%m%d.%f`, and an unpadded `months since` that raises on decode). If you point a *new* product at that key, remember it assumes exactly one time step per file and raises otherwise.

## Install notes

- `pip install acmadDL`; `import acmaddl`. Python ≥ 3.12.
- Under **uv**, `[tool.uv] override-dependencies = ["zarr>=3.1.0"]` resolves the sheerwater (`zarr==2.18.3` pin) vs icechunk (`zarr>=3`) conflict. Under plain pip, the `icechunk` extra is opt-in for the same reason.
- Extras: `geo` (shapefile/geometry regions, geotiff band descriptions), `s3`, `icechunk`, `demo` (matplotlib, cartopy, rasterio), `dev` (pytest).
- Tests: `pytest` runs unit tests; markers `integration`, `cds`, `network` gate live-network suites.
