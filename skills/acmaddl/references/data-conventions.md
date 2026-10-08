# Data conventions and the normalization pipeline

"acmadDL" = translation: every adapter's raw output passes through `normalize.normalize(ds, product_config, variable, region=None, geometry=None, boundary="center", year_index=False)`, which applies the following steps **in order**:

1. **Decode numeric times** — any dim-coordinate with CF `"<unit> since <epoch>"` units becomes `datetime64`. Handles `months since` (e.g. NMME's `S` = "months since 1960-01-01"), `days since`, `hours since`.
2. **S2S scalar-time handling** — a scalar (non-dim) `time` coord is pre-renamed to `init_time` so the later `valid_time -> time` rename cannot collide.
3. **`hdate` handling** — S2S reforecasts carry an `hdate` dim (calendar issuance dates); converted to integer years and renamed `year`.
4. **Coordinate renames** to canonical names:
   - Spatial: `latitude`/`LAT`/`Y` -> `lat`; `longitude`/`LON`/`X` -> `lon`
   - Obs time: `forecast_time`, `valid_time`, `T`, `TIME` -> `time`
   - Forecast init: `S`, `forecast_reference_time`, `indexing_time` -> `init_time`
   - Forecast lead: `L`, `forecastMonth`, `forecast_period`, `step` -> `lead_time`
   - Ensemble: `M`, `number` -> `member`
5. **Empty-member removal / `member_reduce`** (optional, catalog-driven) — products preserve populated native members. CFSv2 removes four structurally all-NaN trailing slots and returns its 24 usable members without averaging them. A separate `member_reduce` knob remains available for products whose public contract explicitly requests a subset reduction.
6. **Variable rename** — `native_name` (or `short_name` fallback) -> canonical `precip`/`temp`/`sst`.
7. **Deaccumulation** — if the catalog marks the variable `accumulated: true` and `lead_time` exists: `.diff("lead_time")` (CDS accumulated precip).
8. **Unit conversion** (`_CONVERSIONS`):

   | From | To | Operation |
   |---|---|---|
   | K | C | subtract 273.15 |
   | kg m-2 s-1 | mm/day | × 86400 |
   | m s-1 | mm/day | × 1000 × 86400 |
   | m | mm/day | × 1000 |
   | m/s | mm/day | × 86,400,000 |
   | mm/month | mm/day | ÷ 30.0 (**no catalog entry uses this** — see below) |
   | mm | mm/day | identity (already mm per 24 h) |

   The `mm/month → mm/day` row is a flat 30, which no calendar month has, and
   the result is not recoverable: multiplying back by the real month length
   lands 3.3% high in January and 6.7% low in February. Every monthly
   observational precip entry therefore declares `target_units: mm/month` and
   keeps the source's own totals. Do not reach for this conversion when adding
   an entry.

   Targeted CFSv2 precipitation uses a calendar-aware path: the selected
   `mm/day` lead mean is multiplied by the exact target-season day count and
   labeled `mm`. The count is computed separately for every initialization
   year, so leap years are respected. Without `target=`, CFSv2 remains
   `mm/day` because there is no accumulation window.

   **Verify units in practice — conversion is catalog-driven.** Targeted NMME
   products use `mm`; general daily/monthly observation and forecast products
   may remain `mm/day`, while native pentad/dekad/annual accumulations remain
   `mm`.
9. **Fill-value masking** — catalog `fill_value` (e.g. -9999) -> NaN.
10. **`scale`** (optional, catalog-driven) — a per-variable multiplier, applied last of the value transforms. It expresses a **sign** convention that differs from acmaddl's, not a unit change: `obs/era5-land-monthly` `pev` declares `scale: -1.0` because ERA5 serves potential evaporation as a negative upward flux, so with the `m -> mm/day` conversion it comes out as a positive rate directly comparable to `precip` in the same entry. Deliberately **after** step 9 — scaling first would turn a `-999` sentinel into plausible-looking `+999` data. The `units` label from step 8 is preserved.
11. **Latitude ascending** — `ds.sortby("lat")`. Canonical convention: lat always ascending.
12. **Spatial selection** — polygon clip (`clip_to_geometry`, rioxarray `.rio.clip`, `all_touched = (boundary == "cover")`) when a geometry was given; otherwise bbox `.sel(lat=slice, lon=slice)` (cover mode expands by half a grid cell).
13. **CF axis attributes** — `lat.axis="Y"`, `lon.axis="X"`, `time`/`init_time` `.axis="T"`.
14. **`year_index`** — if requested and `init_time` is a dim, replace it with integer `year` and collapse `lead_time`. Targeted precipitation has a uniform seasonal-total contract (`mm`): monthly `mm/day` rates are multiplied by each target month’s exact day count and summed; daily amounts are summed; server-averaged rates are multiplied by the exact season length. Other variables retain the lead-mean behavior.

## Canonical output schema

- Coordinates: `lat`, `lon` (ascending lat, lon in [-180, 180]); `time` (obs, `datetime64`); `init_time` (forecasts, `datetime64`); `lead_time` (numeric for seasonal products; a `timedelta64` for issuance-keyed and `weather-skills/*` forecasts); `member` (integer ensemble index). With `year_index=True`: integer `year` replaces `init_time`.
- Typical dims — forecasts: `(init_time, lead_time, member, lat, lon)`; observations: `(time, lat, lon)`; `assemble()` output: `(year, member, lat, lon)`.
- Units: collapsed targeted seasonal forecasts (`year_index=True`/`assemble`) use precip `mm` across adapter families. Lead-resolved and daily precipitation generally remains `mm/day`; native CHIRPS pentad/dekad/annual products keep `mm` totals, and every monthly observational precip product — `obs/chirps-v2-monthly`, `obs/chirps-v3-monthly`, `obs/chirps-v3-monthly-prelim`, `obs/tamsat`, `obs/gpcc-*` — keeps `mm/month`. Temp is `C`; sst is mostly `K` (ERA5 sst is `C`); `pev` is `mm/day`.

  **The `mm` seasonal-total contract is a FORECAST contract.** It lives in the `year_index` branch (step 14), which is gated on `init_time` being a dim — observations have none, so it never fires for them. An observational fetch with `seasonal="mean"` returns the **mean** of the season's months in the entry's own `target_units`, never a seasonal total: `obs/cmap` JAS 2015 over the Sahel is ~4.7 `mm/day`, where the same season as a total would be ~436 `mm`. Multiply by the season's day count (or its month count, for a `mm/month` product) before comparing an obs field against an `assemble()` forecast.

## Season strings

`SEASON_MONTHS` (start_month, end_month): DJF (12,2), JFM (1,3), FMA (2,4), MAM (3,5), AMJ (4,6), MJJ (5,7), JJA (6,8), JAS (7,9), ASO (8,10), SON (9,11), OND (10,12), NDJ (11,1). Wraparound seasons (end < start) roll into the following year; `seasonal="mean"` does not support them.

### Year labeling at multi-month leads (wraparound bookkeeping)

At an operational lead (e.g. 2 months: MAM initialized in January), early-year targets put the init in the **previous calendar year** — JFM targeted from November, FMA from December. `year_index=True` labels years from `init_time`, so when aligning such forecasts with obs labeled by *target* year, add +1 to the forecast's year for those seasons (compute the init month mod-12 and offset when it wraps). Two further rules from downstream use:

- **Rectangular multi-season cubes:** when stacking several seasons into one `(season, year, ...)` cube (e.g. for deepscale's `seasonal_coefficients`), *intersect* the years available across all seasons rather than union them — wraparound seasons otherwise NaN-pad the cube.
- The `init_time → year` rename is what `year_index=True` does for you; for obs use `seasonal="mean"` (which yields a `year` dim directly). Prefer these over hand-rolled `.dt.year` renames.

## Issuance-keyed forecasts (`init_time` / `lead_time` / `valid_time`)

Short-range products with an `issuance` catalog block (CHIRPS-GEFS: `chc/chirps-gefs-daily`, `chc/chirps-gefs-15day`) are addressed by issuance date, not by season. Fetch them with `init="YYYY-MM-DD"` (one issuance) or a **sequence** of `YYYY-MM-DD` dates — the hindcast-skill case, where you want the same calendar issuance across many years. A sequence stacks the result on `init_time`.

Output layout:

- `init_time` — the issuance date(s) (`datetime64`).
- `lead_time` — a `timedelta64` (so it carries its own units; `lead_units` only sets the step the catalog's integer `leads` count in). `chc/chirps-gefs-daily` carries 16 daily leads (0-15, where lead 0 is the issuance day); `chc/chirps-gefs-15day` carries a single lead (the 15-day accumulation window).
- `valid_time` — the target date each `(init_time, lead_time)` pair verifies against. **For `chc/chirps-gefs-15day`, `valid_time` marks the window's START** ([init, init+15d)), not its end.

A season `target` cannot combine with an issuance sequence (target selects leads relative to one init); fetch the leads and select the target window afterwards. A sequence passed to a non-issuance product raises `ValueError`.

## Rhiza weather-skills products (`weather-skills/*`)

The `weather-skills` adapter runs a Rhiza weather-skills fetcher script in-process and reads the Zarr it writes. Their raw shapes and what acmaddl makes of them:

| Their output | acmaddl output |
|---|---|
| forecast `(number, step, latitude, longitude)` + scalar `time` (the init for ecmwf-fetch / dynamical-fetch, the **valid date** for SubC) | `(init_time, lead_time, member, lat, lon)`; `init_time` is the requested init (any scalar `time` is dropped), `lead_time` the native `step` as `timedelta64`, `time` = `init_time + lead_time` |
| deterministic forecast (no `number`) | no `member` dimension |
| analysis / observation `(time, latitude, longitude)` | `(time, lat, lon)` |

- Member 0 is the control run and is kept; `grid.forecast_members` counts it.
- Native steps are preserved (3-6 hourly for IFS-ENS/GEFS/AIFS, hourly for GFS/analyses, daily for ECMWF S2S and IFS-ENS 46 d, half-hourly for IMERG). Aggregate downstream.
- Units are declared per variable in the catalog and converted by the usual `(units, target_units)` table: their precipitation rates are `mm/day`, temperatures and SST `C`; SubC precipitation is a window **sum** in `mm`, with each `<var>_anomaly` field passed through unchanged.
- Attributes kept: `weather_skills_history` (JSON), `weather_skills_source`; added: `weather_skills_name`, `weather_skills_version`, `weather_skills_pin`. Dict-valued attributes some sources stamp on coordinates are JSON-encoded so the result writes to NetCDF.
- Observation windows: `hindcast=(y0, y1)` × `months=` collapse into contiguous month runs, clipped to the skill's own `--probe-latest` day (or yesterday), chunked by `window_days` for the global fetchers (each chunk cropped to the bbox before concatenation); no `hindcast` = trailing `window_days` (default 10) days. `allow_future` (CMIP6) disables the clip.
- `year_index=True` is not blocked but, with a single real-time issuance and no hindcast, collapses to a one-year axis and means nothing.

### Outbound mapping (ours → theirs, `acmaddl.weather_skills.to_standard_dataset`)

| acmadDL / africas2s | weather-skills standard dataset |
|---|---|
| `lat`, `lon`, `time`, `member` | unchanged (their vocabulary accepts these spellings) |
| `lead_time` as `timedelta64` | unchanged |
| `lead_time` numeric, `units` hours/days | `timedelta64` |
| `lead_time` numeric in months | refused (`ValueError`): collapse the lead first |
| `(init_time=1, lead_time, …)` | their fetcher layout: scalar `time` (the init) + `step` |
| `(init_time=n>1, lead_time, …)` | unchanged |
| `(year, member, lat, lon)` | `init_time` at `init_month`/`init_day` per year + one `lead_time` from `lead=` |
| `(member, lat, lon)` / `(lat, lon)` | `init_time` from `init=` + one `lead_time` from `lead=` |
| 2-D valid `time` coordinate | dropped (their `step-to-time` derives it) |
| units `C`, `K`, `mm/day`, `mm/month`, `kg/m2`, `m3/m3`, `%` | `degree_Celsius`, `kelvin`, `mm day-1`, `mm month-1`, `kg m-2`, `1`, `percent` |
| `precip`/`temp`-like variable without units | refused (their classifier requires units for those kinds) |
| dict-valued attrs | JSON strings |
| `weather_skills_history` | appended with an `acmaddl` entry (`record=False`: unchanged); `weather_skills_source` = `source` or `acmaddl:<product>` |

`from_standard_dataset` is the inverse: their names and pint spellings back to ours (checked by unit equality, so `millimeter / day` → `mm/day`), latitude ascending, the forecast reshape, and the valid time as acmadDL's 2-D `time` coordinate.

## Longitude-convention helpers (`normalize.py`, `fetch.py`)

The 0-360 vs -180..180 longitude footgun is now handled by named helpers rather than ad-hoc slicing:

- `normalize.select_lon(ds, lon_w, lon_e, lon_name="lon")` — convention-aware longitude subselection. Short-circuits to "return everything" on a full-globe request (span ≥ 359°), translates the requested bounds into the source's convention, and handles a seam-crossing box (west > east after translation) by selecting both sides and concatenating. Shared by the `http` adapter and `normalize` (dim `lon`), because the source convention is only known once the data is opened.
- `normalize.lon_selection_bounds(lon_values, lon_w, lon_e)` — the same convention/seam arithmetic, exposed as the 1-2 contiguous `(start, stop)` pairs rather than a selected dataset, so a caller can plan *requests* with it. `select_lon` is built on it. The OPeNDAP adapter uses it directly: a lazily seam-concatenated DAP selection is one malformed request that NOAA PSL answers with zeros at any size, so each segment must be requested contiguously and joined after loading.
- `fetch._match_lon_convention(obj, target_lons)` — used by regrid/interp (`grid_res`/`regrid_to`): rolls the source's longitude into the target grid's convention (and re-sorts) before `.interp`, so interpolating a 0-360 source onto a grid with negative longitudes doesn't silently NaN the western band.
- `normalize.sanitize_for_netcdf(ds)` — rebuilds a dataset from fresh arrays (dropping CF bounds vars and stale inherited encoding) so OPeNDAP/CF results round-trip through `to_netcdf` instead of raising `NetCDF: String match to name in use`. Applied to every `fetch` result.

## Downstream contract (deepscale)

Deepscale consumes `(year, member, lat, lon)` hindcasts and `(member, lat, lon)` forecasts and calls `hindcast.mean("member")` — which is why `assemble()` guarantees a `member` dim (size 1 if the source has none) and transposes to `(year, member, lat, lon)`. Observations feed deepscale as `(year, lat, lon)` — get there with `fetch(..., seasonal="mean", target=...)` (obs) or `year_index=True` (forecasts).
