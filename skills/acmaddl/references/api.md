# acmadDL API reference

Import name is `acmaddl` (distribution: `acmadDL`). Top-level exports: `catalog`, `fetch`, `zonal`, `parse_target`, `parse_init`, `assemble`, `obs_predictor`, `check_product`, `check_all_products`, `VariableNotSupported`.

At import time acmaddl pins the nuthatch cache root via `os.environ.setdefault("NUTHATCH_ROOT_FILESYSTEM", ...)` / `NUTHATCH_LOCAL_FILESYSTEM` to `file://<ACMADDL_CACHE_DIR or ~/.nuthatch/caches>`. Set `ACMADDL_CACHE_DIR` **before** importing acmaddl to relocate the cache.

## `fetch()`

```python
def fetch(product, variable, init=None, target=None, region=None,
          hindcast=None, destination=None, format="netcdf", verbose=True,
          progress=True, cache=True, allow_partial=False,
          max_retries=3, retry_backoff=1.0, request_interval=0.0,
          reforecast=False, boundary="center", region_buffer=1.5,
          year_index=False, seasonal=None, grid_res=None, regrid_to=None,
          months=None, degenerate_attempts=1)
    -> xr.Dataset
```

| Param | Type / default | Meaning |
|---|---|---|
| `product` | str, required | Catalog id, e.g. `"nmme/cfsv2"`, `"obs/era5"`. See `references/products.md`. |
| `variable` | str, required | Canonical variable: `precip`, `temp`, or `sst` (availability per product). |
| `init` | str \| datetime \| sequence \| None | `"YYYY-MM"` = seasonal monthly init (sets `init_months`); `"YYYY-MM-DD"` = daily issuance (also sets the internal `_init_date`). A **sequence** of `YYYY-MM-DD` dates/datetimes is allowed **only** for products with an `issuance` catalog block (CHIRPS-GEFS): the result stacks on `init_time` with `lead_time`/`valid_time`. A sequence raises for non-issuance products and cannot combine with `target`. |
| `target` | str \| (datetime, datetime) \| None | Season string (see season table below) or explicit start/end datetimes. Drives lead-month selection. |
| `region` | list \| str \| geometry \| None | bbox `[lat_s, lat_n, lon_w, lon_e]`, `.shp` path, or shapely/geopandas geometry. |
| `hindcast` | (int, int) \| None | `(start_year, end_year)` year range for the fetch. If omitted with `init`: defaults to `(init_year, init_year)`, or `(init_year-20, init_year-1)` when `reforecast=True`. |
| `destination` | str \| None | Local path or `s3://...` to also write the result. |
| `format` | `"netcdf"` (default) \| `"geotiff"` | Output format for `destination`. GeoTIFF has shape restrictions (see `storage.save` below). |
| `verbose` | bool = True | Logging (also sets nuthatch logger level). |
| `progress` | bool = True | tqdm progress bars. |
| `cache` | bool = True | Route through the nuthatch cache. `False` calls the adapter directly, never touching the cache. |
| `allow_partial` | bool = False | Multi-file adapters: `False` raises `RuntimeError` on any file failure; `True` returns what succeeded. |
| `max_retries` | int = 3 | Transient-failure retries (HTTP, OPeNDAP, CCSR adapters). |
| `retry_backoff` | float = 1.0 | Backoff multiplier between retries. |
| `request_interval` | float = 0.0 | Seconds between requests. This is a **floor** — a per-product catalog value can raise it, a caller cannot undercut it. |
| `reforecast` | bool = False | Fetch the on-the-fly hindcast suite (S2S: switches to the `s2s-reforecasts` collection). |
| `boundary` | `"center"` (default) \| `"cover"` | `"center"`: keep cells whose centre is inside the region (xarray/CDO convention). `"cover"`: keep every cell the region touches (`all_touched=True`). |
| `region_buffer` | float = 1.5 (degrees) | Only used with `boundary="cover"`; pads the fetched bbox so edge cells aren't clipped. Must exceed half the grid spacing. |
| `year_index` | bool = False | Reshape forecast `init_time` into integer `year` and collapse `lead_time`. For targeted precipitation, monthly rates are calendar-weighted and daily amounts are summed to seasonal `mm`; other variables use a lead mean. |
| `seasonal` | None \| `"mean"` | `"mean"`: subset the target season's months on `time` and average to one value per calendar year (`time` -> `year`). Requires `target`. Wraparound seasons (NDJ, DJF) raise `NotImplementedError`. |
| `grid_res` | float \| None | Regrid onto a regular lat/lon grid at this resolution spanning `region` (via `.interp`). Requires `region`. Mutually exclusive with `regrid_to`. |
| `regrid_to` | xr.DataArray \| None | Regrid onto this array's `lat`/`lon` coordinates. |
| `months` | list[int] \| None | Restrict an **observational** fetch to these calendar months (1-12). Rejected alongside `init` (which already names the issuance). For HTTP products stored one file per `(year, month)` it prunes the **download** (a 4-month season over 45 years is 180 files, not 540); for whole-file OPeNDAP it prunes only the result. Reuses the `init_months` cache slot, so it does not invalidate existing cache. Can pair with `seasonal="mean"` (average the selected months) in place of `target`. |
| `degenerate_attempts` | int = 1 | **Opt-in** zero-fill/truncation guard for the general fetch path. Default `1` = **no validation** (behaviour unchanged). `>1` enables it: a cache-miss response is validated in `_fetch_raw` before being memoized, a cache **hit** is re-validated (catching entries poisoned before the guard existed), and a `DegenerateResponseError` triggers a cache-bypass retry up to this many attempts. See the caveat below — most fetches are **not** auto-protected. |

**`degenerate_attempts` — what is and isn't protected.** The guard on the *general* path is opt-in: with the default `degenerate_attempts=1` acmaddl does **not** validate any adapter's response against silent zero-fill/truncation. Pass `degenerate_attempts>1` to enable validation + cache-bypass retries for a fetch you don't trust (e.g. a large OPeNDAP pull). The **exception** is the OPeNDAP *observational chunk* path (`obs/ersst-v5`, `obs/cmap`): there the guard is **always on** and stricter (rejects all-NaN chunks too), independent of `degenerate_attempts`. Do not assume universal protection.

Behavioral notes:

- Requesting a variable the product's catalog entry does not declare raises `VariableNotSupported` **before any network I/O** (see Exceptions below).
- Requesting `region` on a product whose normalized output has no `lat`/`lon` dims raises `ValueError` (station/tabular data cannot be spatially subset).
- Polygon geometries are applied as a **final NaN mask** in normalization; the upstream request and the cache key only see the bbox. Two requests with the same bbox but different polygons share cached raw data.
- Deprecated products and aliases emit `DeprecationWarning` when resolved.

## `parse_target(target, year=None) -> (datetime, datetime)`

Tuples pass through. Season strings map via `SEASON_MONTHS` (start_month, end_month):

| Season | Months | Season | Months |
|---|---|---|---|
| DJF | (12, 2) | JJA | (6, 8) |
| JFM | (1, 3) | JAS | (7, 9) |
| FMA | (2, 4) | ASO | (8, 10) |
| MAM | (3, 5) | SON | (9, 11) |
| AMJ | (4, 6) | OND | (10, 12) |
| MJJ | (5, 7) | NDJ | (11, 1) |

Current year is used when `year` is None; the end date is the last day of the end month; wraparound seasons (end < start) roll into the next year.

## `parse_init(init) -> datetime`

10-char strings parse as `%Y-%m-%d`; otherwise the first 7 chars parse as `%Y-%m`. A `datetime` passes through.

## `assemble()`

```python
def assemble(roster, variable, *, init, target, region=None, grid_res=None,
             regrid_to=None, seasonal=None, range_index=2, cache=True,
             verbose=True, boundary="cover")
    -> dict[str, tuple[xr.DataArray, xr.DataArray]]
```

Multi-model fan-out over `fetch`, always with `year_index=True`.

- `roster`: iterable of rows `(label, product, *ranges)` where each range is a `(start, end)` tuple; `range_index` (default 2 = third element) selects which range column is the hindcast window.
- Returns `{label: (hindcast_da, forecast_da)}`. Each DataArray is canonicalized: a `member` dim is guaranteed (expanded to size 1 if absent — downstream code like deepscale does `hindcast.mean("member")`), stray non-dim coords (`number`/`member`/`spatial_ref`) are dropped, and dims are transposed to `(year, member, lat, lon)`.
- Raises on the first per-row failure.

## `obs_predictor()`

```python
def obs_predictor(product, variable, *, target=None, months=None, hindcast,
                  forecast_year, region=None, grid_res=None, regrid_to=None,
                  seasonal="mean", boundary="center", cache=True, verbose=True)
    -> tuple[xr.DataArray, xr.DataArray]
```

The observations counterpart of `assemble()`: use an **observed** gridded field (e.g. ERSST SST) as a seasonal predictor track for a CCA. Returns the SAME canonical `(year, member, lat, lon)` `(hindcast, forecast)` pair a model would, so an observed predictor drops into `deepscale.seasonal_mme`'s `predictor_tracks` exactly like a model forecast.

- Requires **exactly one** of `target` (a 3-month season string) or `months` (an explicit calendar-month list, e.g. `[6]` for a single-month June predictor) — passing both or neither raises `ValueError`.
- `hindcast`: `(start_year, end_year)` training window. `forecast_year`: the single year to predict from.
- `seasonal="mean"` (default) averages the selected month(s) per calendar year, so the training series and the single forecast year come from the same observed product. Observations carry no init/lead.
- The returned `forecast` is the `forecast_year` slice: `year=[forecast_year]`, `member=[0]`.

## `zonal()`

```python
def zonal(data, geometries, *, by=None, label=None, stat="mean",
          weights="area", all_touched=False, dim="region",
          lat="lat", lon="lon")
    -> xr.DataArray | xr.Dataset
```

Reduce a gridded field over **many geometries at once** — one number per feature (a district, a woreda), the reporting question, as opposed to `fetch(region=...)` which dissolves features into one mask (the "data over this area" question). Every geometry is rasterized into a single integer label grid and the reduction is one `groupby` over that grid — one pass regardless of the number of regions. Needs the `geo` extra (geopandas + rasterio).

| Param | Type / default | Meaning |
|---|---|---|
| `data` | DataArray \| Dataset, required | Must carry the `lat` and `lon` dims (names configurable). Every other dim is preserved. Needs a grid ≥ 2×2 to infer cell size. |
| `geometries` | shapefile path \| GeoDataFrame \| GeoSeries \| list[shapely] | Reprojected to EPSG:4326 if a CRS is attached. **Not** dissolved — one output element per feature. |
| `by` | str \| None | Column to index the output `dim` by (e.g. a unique admin code). **Must be unique** — raises `ValueError` on duplicate values. Defaults to the positional index. |
| `label` | str \| None | Column carried as a companion `{dim}_label` coordinate (for display). **May repeat** — this is the clean way to keep human-readable names on a uniquely-indexed axis. |
| `stat` | mean/sum/min/max/median/std/count | The reduction. `count` returns the number of contributing cells. |
| `weights` | `"area"` (default) \| `"cos_lat"` \| None \| DataArray | Cell weighting. `"area"` and `"cos_lat"` are the same `cos(lat)` weighting under two names (a cell's area on a regular grid falls off as `cos(lat)`). **Only `mean` and `sum` use weights**; order statistics ignore them. |
| `all_touched` | bool = False | Include every cell a geometry touches, not only those whose centre it contains. |
| `dim` | str = `"region"` | Name of the new output axis that replaces `lat`/`lon`. |
| `lat`, `lon` | str = `"lat"`/`"lon"` | Names of the spatial dims in `data`. |

Returns the same type as `data` with `lat`/`lon` replaced by `dim`; output **mirrors the input geometry order and count** (a region with no valid cell yields NaN, or `0` for `count`, rather than being dropped). NaN cells are excluded from the reduction (a half-ocean district is averaged over its land).

**Gotchas:**
- On a **coarse grid**, a polygon smaller than one cell captures no cell centre and yields NaN — pass `all_touched=True` (though see next).
- `all_touched=True` can make coverage **worse**, not better: neighbouring boundary cells contend, and `rasterio` uses last-one-wins, so a cell on a shared boundary is assigned to whichever feature rasterized last. Use it for small/coarse cases, not reflexively.
- Admin **names are often non-unique** (two woredas share a name). Index `by` a unique id column (e.g. `shapeID`) and carry the name via `label=` — passing a non-unique column to `by` raises.
- If no grid cell falls inside any geometry, raises `ValueError` (grid and geometries may not overlap, or regions are sub-cell — try `all_touched=True`).

## `catalog` module

- `catalog.list_products(include_deprecated=True) -> list[str]` — all catalog product ids; `include_deprecated=False` filters aliases and date-deprecated entries.
- `catalog.variables(product_name) -> list[str]` — the variable names a product declares, in catalog order.
- `catalog.require_variable(product_name, variable, config=None)` — raises `VariableNotSupported` unless the product declares `variable`. Pass an already-resolved `config` to reuse it (and avoid re-emitting an alias `DeprecationWarning`). Used by `fetch()` and by `assemble()`'s roster precheck.
- `catalog.info(product_name) -> dict` (alias: `catalog.get`) — resolved config for a product. Follows `alias_of` chains (emitting `DeprecationWarning`), adds a computed `deprecated` bool (from `deprecated_after` vs today). Raises `KeyError` for unknown products. Config keys of interest: `adapter`, `variables` (each with `native_name`, `units`, `target_units`, optional `accumulated`, `fill_value`, `season_total_conversion`), `grid`, `hindcast`/`forecast` ranges, `drop_empty_members`, `member_reduce`, `request_interval`, `max_workers`.

## Exceptions (`acmaddl.errors`)

- `VariableNotSupported(ValueError)` — the product's catalog entry does not declare the requested variable. Attributes: `product`, `variable`, `available` (list of the variables it does serve). Raised by `fetch()` before any network I/O, and by `assemble()` for the **whole roster** before the first fetch — so one unsupported model does not cost the earlier models' downloads.

  It exists so callers can separate a **capability gap** (permanent; this product will never serve this variable — deselect it) from an **availability gap** (transient; this init/target isn't published yet — retry or pick another season), without string-matching a bare `KeyError`:

  ```python
  try:
      ds = acmaddl.fetch(pid, "precip", init="2025-09", target="OND")
  except acmaddl.VariableNotSupported as e:
      log.info("%s cannot serve %s (serves %s) - skipping", e.product, e.variable, e.available)
  ```

  A `ValueError` subclass, since it reports a statically-answerable request error rather than an I/O failure. Note `assemble()`'s precheck is a *capability* check only: a product id absent from the catalog is still reported by `fetch()` as `KeyError`, unchanged.

## Health checks (`health.py`)

- `check_product(product, probe_remote=False, variable=None) -> dict` — keys: `product`, `adapter`, `checked_at` (UTC ISO), `healthy` (bool), `kind` (`"config" | "remote" | "transient" | "runtime" | "capability"`), `message`, plus adapter-specific fields. Returns early with a `pending_url` message when the catalog marks a product as not yet live. `probe_remote=True` also opens/pings the live source.
  - `variable=<name>` also checks the product can serve it. A variable the catalog does not declare returns `kind="capability"`, `healthy=False`, plus `variable` and `available` keys — reported *distinctly* from a `config` problem or a `remote`/`transient` outage, which is the distinction between "deselect this product" and "retry later". Answerable from the catalog alone, so it never costs a remote probe even with `probe_remote=True`. Unlike `fetch()`, this **returns** the mismatch rather than raising `VariableNotSupported` — health checks report, they do not throw.
- `check_all_products(probe_remote=False, variable=None) -> list[dict]` — iterates the whole catalog, converting exceptions into `kind="runtime"` failures. `variable` is threaded through, so this doubles as a capability sweep ("which products can serve precip?").

## `validate` module (not re-exported; `from acmaddl import validate`)

For checking acmaddl output against independent references:

- `ValidationResult` dataclass — fields include `product`, `variable`, `reference`, `r_timeseries`, `r_spatial`, `status`, `threshold=0.95`, `structural_checks`, `error`, `timestamp`; methods `passed()`, `to_report_entry()`.
- `check_structure(ds, product, variable, product_config=None) -> dict` — variable presence, units attr, spatial dims, not-all-NaN, unit-aware plausible value ranges (precip rates up to 500 mm/day; accumulated precip up to 5000 mm; temp -90-70 C; sst -3-45 C; ±50 tolerance), member count vs catalog, hindcast range.
- `compare(acmaddl_da, reference_da, threshold=0.95) -> (r_ts, r_spatial, status)` — regrids to a common grid, computes temporal and spatial Pearson correlations; status `PASS`/`CHECK`/`ERROR`.
- `regrid_to_common(da1, da2)`, `compute_correlations(ros_da, ref_da)` (needs `(init_time, lat, lon)`), `validate_product(...)`, `write_report(results, path=None)` / `read_report(path=None)` (default `output/validation_report.json`).

## `storage.save(ds, destination, format="netcdf")`

- `"netcdf"`: `ds.to_netcdf`; `s3://` destinations go through `s3fs`.
- `"geotiff"`: reduces to `(bands, lat, lon)` — means over `member`, squeezes length-1 `init_time`; the one remaining non-spatial dim becomes bands (values become band descriptions). Requires exactly one data var, `lat`/`lon` dims, and at most one extra dim after reduction. S3 GeoTIFF is `NotImplementedError`. Needs the `geo` extra for band descriptions.

## CLI (`acmaddl`, Click)

```
acmaddl cache list                      # python -m nuthatch list --namespace acmaddl
acmaddl cache clear [--product X] [--yes]
```

`clear` prompts for confirmation unless `--yes`; `--product` maps to a nuthatch `--cache-key` filter.

## Region resolution (`region.resolve_region(region) -> (bbox, geometry)`)

1. **bbox** — list/tuple/np.array of exactly 4 real numbers `[lat_s, lat_n, lon_w, lon_e]` (lat before lon; south, north, west, east). Returned as floats with `geometry=None`.
2. **shapefile** — string/`Path` ending `.shp` (anything else raises `ValueError`). Read with geopandas, reprojected to EPSG:4326, dissolved via `union_all()` (so multi-feature files clip as one region); bbox from `.bounds`.
3. **geometry** — bare shapely `BaseGeometry` (assumed EPSG:4326) or geopandas `GeoSeries`/`GeoDataFrame` (reprojected + dissolved).

`None` -> `(None, None)`. Shapefile/geometry input requires the `geo` extra (raises `ImportError` with a `pip install 'acmadDL[geo]'` hint). Antimeridian-crossing polygons are not special-cased — split them first.

## Adapters (internal, for debugging)

Registry (`acmaddl.adapters.get_adapter(name)`): `cds`, `opendap`, `http`, `iridl`, `mars`, `ncei`, `s3`, `sheerwater`, `ccsr`, `icechunk`. All subclass `AdapterBase` with abstract `fetch_data(product_config, variable, date_range=None, region=None)`, plus `_resolve_streams` (hindcast/forecast split-stream stitching) and `health_check`. Caching happens **only** in `fetch._fetch_raw_cached`, never inside adapters.
