# Product catalog

The catalog lives at `src/acmaddl/catalog.yaml`. Enumerate at runtime with `catalog.list_products()`; inspect any entry with `catalog.info("<product>")`. Canonical variables are always `precip`, `temp`, `sst` (availability varies per product). Below is the full catalog as of v0.1.0.

## NMME seasonal models (`nmme/*`) — no credentials

| Product | Adapter | Variables | Notes |
|---|---|---|---|
| `nmme/cfsv2` | opendap (IRI DL) | precip, temp, sst | 1° global; split+append hindcast [1982-2011] / forecast [2011-now]; targeted precip rate converted to seasonal `mm`; 24 populated members preserved (4 all-NaN trailing slots removed) |
| `nmme/ccsm4` | ccsr | precip, temp, sst | `single_year_fetch`; native names prcp/t2m/sst |
| `nmme/cesm1` | ccsr | precip, temp, sst | `single_year_fetch`; hindcast [1982-2026], forecast [2017-now] |
| `nmme/geoss2s` | ccsr | precip, temp, sst | split streams, `single_year_fetch`; forecast M=10, hindcast M=4 (ends 2017) |
| `nmme/spear` | ccsr | precip, temp, sst | split streams, `single_year_fetch` |
| `nmme/spearb` | ccsr | sst | |
| `nmme/cansipsic4` | ccsr | precip, temp, sst | |

Deprecated aliases (emit `DeprecationWarning`, resolve to base products): `nmme/cfsv2-forecast`, `nmme/ccsm4-iri`, `nmme/ccsm4-hindcast`, `nmme/geoss2s-forecast`, `nmme/geoss2s-hindcast`, `nmme/cesm1-hindcast`, `nmme/spear-hindcast`, `nmme/spearb-hindcast`, `nmme/cansipsic4-hindcast`.

The CCSR adapter also normalizes longitude from 0-360 to [-180, 180]. Entries marked `single_year_fetch: true` (`nmme/ccsm4`, `nmme/cesm1`, `nmme/geoss2s`, `nmme/spear`) issue one DAP request per year: the full-range hindcast request overflows the CCSR server and silently zero-fills, so acmaddl chunks it. This is the CCSR analogue of the OPeNDAP `max_request_years` guard (see the NOAA PSL products below).

CanSIPS-IC4 precipitation uses different CCSR dataset paths by stream: `hindcast/prcp` and `forecast/pr` (both contain the native variable `pr`). The catalog records this with a stream-specific `path_name` mapping.

Two practical caveats from operational use:

- **Real-time vs hindcast availability differ per model and drift over time** — a model can have a full 1991-2020 hindcast but publish no current-month forecast (SPEAR in mid-2026), or a hindcast too short for a 30-year baseline (GEOSS2S ends 2017). Probe before committing a roster: `check_product(p, probe_remote=True)` or a cheap single-year fetch of the current init.
- **Collapsed targeted precip is delivered in `mm`** — this is uniform across NMME, C3S/CDS, and IRI when using `year_index=True` or `assemble()`. Lead-resolved fetches retain per-step units. CFSv2 without `target=` remains `mm/day` because no accumulation window exists.

## C3S seasonal models (`c3s/*`) — Copernicus CDS credentials (`~/.cdsapirc`)

All fetch precip/temp/sst, monthly or daily cadence via the `cds` adapter:

`c3s/ecmwf`, `c3s/ecmwf-monthly`, `c3s/eccc-cansips`, `c3s/eccc-cansipsv3`, `c3s/eccc-daily`, `c3s/meteofrance`, `c3s/meteofrance-daily`, `c3s/cmcc`, `c3s/cmcc-daily`, `c3s/cmcc-sps4`, `c3s/cmcc-sps4-daily`, `c3s/dwd`, `c3s/dwd-daily`, `c3s/dwd-gcfs21`, `c3s/ukmo`, `c3s/ukmo-daily`, `c3s/jma`, `c3s/jma-cps2`.

Notes:
- `year_index=True` / `assemble()` calendar-weights monthly rates or sums deaccumulated daily leads, returning seasonal precipitation in `mm`. A lead-resolved fetch remains `mm/day`.
- Each dataset's licence must be accepted once in the CDS web UI; a 403 error names the missing licence.
- Some entries have retired real-time streams (hindcasts still fetch; no new forecasts) — check `catalog.info(p)` for `deprecated_after` / `deprecation_note`.
- `c3s/ecmwf-seas51c` uses the **iridl** adapter (needs `~/.pycpt_dlauth`), is deprecated after 2026-04-30 with successor `c3s/ecmwf-monthly`, and carries precip/sst only.
- CDS accumulated-precip variables are deaccumulated over `lead_time` automatically (`accumulated: true` in the catalog).

## Sub-seasonal (S2S)

| Product | Adapter | Credentials | Notes |
|---|---|---|---|
| `c3s/ecmwf-s2s` | cds (ECDS endpoint `https://ecds.ecmwf.int/api`) | ECDS account — **separate from Copernicus CDS** | precip, sst; 1.5°; date-keyed via `init="YYYY-MM-DD"`; `reforecast=True` switches to the `s2s-reforecasts` collection (on-the-fly hindcasts) |

A legacy MARS adapter exists as an S2S reforecast-only fallback (needs `~/.ecmwfapirc`); it raises unless used with `reforecast=True` plus `hindcast` and `region`.

## Reanalysis (`obs/era5*`) — Copernicus CDS credentials

| Product | CDS dataset | Variables | Resolution | Range |
|---|---|---|---|---|
| `obs/era5` | reanalysis-era5-single-levels-monthly-means | temp, precip, sst | 0.25° | 1940-2025 |
| `obs/era5-land-monthly` | reanalysis-era5-land-monthly-means | precip, temp, **pev** | 0.1° | 1950-2025 (no sst) |

## CHIRPS native (UCSB, `http` adapter) — no credentials, heavily rate-limited

All precip-only, 0.05°, `fill_value: -9999`:

| Product | Format | Cadence |
|---|---|---|
| `obs/chirps-v2-monthly` / `obs/chirps-v3-monthly` | COG | monthly |
| `obs/chirps-v2-daily` / `obs/chirps-v3-daily` | NetCDF | daily (**v3 daily ≈ 23.5 GiB/file** — downloaded whole, then cropped) |
| `obs/chirps-v2-pentad` / `obs/chirps-v3-pentad` | COG | pentad |
| `obs/chirps-v2-dekad` / `obs/chirps-v3-dekad` | COG | dekad |
| `obs/chirps-v3-dekad-tif` | TIF (range-readable) | dekad — **final** gauge-corrected stream, 1981-~2025 |
| `obs/chirps-v3-dekad-prelim` | TIF | dekad — **preliminary** near-real-time (~1-dekad lag), rolling `forecast_range: [2024, null]` |
| `obs/chirps-v2-annual` / `obs/chirps-v3-annual` | TIF | annual |

`obs/chirps-v3-dekad-tif` and `-prelim` are 0.05° calendar-dekad GeoTIFF streams (`request_interval: 3.0`, `fill_value: -9999`, precip in `mm` per 10-day dekad). The tif stream is the completed, gauge-corrected archive; the prelim stream fills the current-season tail before the final product catches up. Splice prelim onto the tif stream (prelim for the trailing dekads, tif for everything settled) for a continuous near-real-time record.

`data.chc.ucsb.edu` sits behind CrowdSec: exceeding ~2 req/s earns a **4-hour IP ban** (HTTP 403 "CrowdSec Ban"). The catalog sets `request_interval` floors (3.0 s for COG/TIF, 1.0 s for NetCDF) and `max_workers: 2` caps; for large/continental pulls raise `request_interval` further (rule of thumb: range-reads-per-file / 2 seconds). Native pentad/dekad/annual keep `mm` totals rather than `mm/day`.

## Rhiza / Sheerwater mirrors (`sheerwater` adapter, public GCS Zarr, 0.25°) — no credentials

| Product | Source | Notes |
|---|---|---|
| `obs/chirps-v3-daily-rhiza` | chirps_v3 | preferred over native v3 daily for routine use. 0.25° (coarsened from native 0.05°); archive covers ~2000 to mid-2024 — no current-season data |
| `obs/chirps-v2-dekadal-rhiza` | chirps_v2 (`agg_days: 10`) | 10-day **rolling** aggregate — NOT calendar dekads (native `obs/chirps-v2-dekad` is calendar); used by the deepscale S2S testbed |
| `obs/chirps-live-rhiza` | chirps.chirps_raw_live | 0.05°, near-real-time (~8-day lag), `recompute`/`cache_mode: write`; used by the deepscale S2S testbed. **Known issue:** upstream sheerwater removed `chirps_raw_live` (AttributeError) — see troubleshooting |
| `obs/imerg` | imerg_final | satellite precip |
| `obs/ghcn` | ghcn_avg | station-derived temp |

Sheerwater reads public GCS anonymously (a benign gcsfs "Could not determine bucket type" warning is filtered). Bbox regions become a global lazy fetch + client-side crop. The sheerwater adapter adds no cache of its own (sheerwater is already nuthatch-cached upstream).

## NOAA PSL OPeNDAP observations (`opendap` adapter) — no credentials

Whole-file NetCDF served over NOAA PSL's THREDDS/OpenDAP endpoint. The PSL server silently zero-fills (and prints `ERR: DAP DATADDS packet is apparently too short` to stderr) rather than erroring when a request is one it can't answer, so the adapter requests the record in pieces and validates each. This obs-chunk path runs the **always-on** degenerate-response guard (`reject_all_nan=True`) — a zero-filled *or* all-NaN chunk raises `DegenerateResponseError` before it can be cached.

Two things bound a request, and both are needed (probe-verified 2026-09-02):

- **`max_request_years`** (catalog, 5 for the 2°/2.5° products, 2 for 0.25° OISST) — an upper bound on years per request.
- **A response-size budget** (4M values, ~half the observed ~32 MiB PSL cap) — because the size of one time step depends on how many cells the region covers, the same year count that is safe at 2° is hopeless at 0.25°. A wide region is therefore split into sub-year requests automatically, so any region size works, global included.

**A seam-crossing region becomes two separate requests.** `select_lon` answers a box spanning the source's longitude seam (e.g. an Africa box -30..60 on a 0..360 grid) by concatenating two slices; done lazily that is a single malformed DAP request, which PSL zero-fills **at every size** — an 80°×90° OISST box truncated even at 230k values, while the same cells loaded fine as one contiguous request. The adapter now plans the segments with `normalize.lon_selection_bounds` and issues one contiguous request per segment, joining the loaded results.

| Product | Variable | Native → canonical | Resolution | Coverage | Notes |
|---|---|---|---|---|---|
| `obs/ersst-v5` | sst | `sst` degC → C | 2.0° monthly | ~1954-present | NOAA Extended Reconstructed SST v5. Reference SST for the 1991-2020 WMO baseline (ONI/RONI/IOD verification). Latitude stored N→S upstream; the adapter sorts it ascending. |
| `obs/cmap` | precip | `precip` mm/day (native) | 2.5° monthly | ~1979-present | CPC Merged Analysis of Precipitation (gauge+satellite), the natively-served NetCDF sibling of the IRIDL-only CAMS-OPI. |
| `obs/oisst-v2-highres` | sst | `sst` degC → C | 0.25° monthly | Sep 1981-present | NOAA OISST v2.1 (AVHRR-only high-res), the fine-grid sibling of `obs/ersst-v5`. PSL also publishes the whole record as one ~2.2 GB HTTP file; this entry deliberately uses OPeNDAP so requests are subset server-side. `max_request_years: 2`, further tightened by the response-size budget — a global month is ~1.04M cells, 64× ERSST's. Any region size works, global included. |

**GOTCHA (SST land mask):** because the obs-chunk guard rejects all-NaN chunks, requesting a **land-only** bbox for `obs/ersst-v5` or `obs/oisst-v2-highres` (ocean-only SST fields) raises `DegenerateResponseError` — the chunk is legitimately all-NaN over land. Request an ocean-containing region.

## TAMSAT (`http` adapter, JASMIN public) — no credentials

| Product | Variable | Native → canonical | Resolution | Coverage | Notes |
|---|---|---|---|---|---|
| `obs/tamsat` | precip | `rfe` → precip, **mm/month** | 0.0375° monthly | ~1983-2025 | TAMSAT v3.1 monthly rainfall estimate (Reading), one NetCDF per (year, month) on JASMIN's public server (`request_interval: 0.5`, `fill_value: -999`). **Africa land-only.** |

TAMSAT keeps `mm/month` totals (`target_units: mm/month`) rather than converting to `mm/day` — check `attrs["units"]` before mixing with rate-based products. The native grid is ~3.7M cells; coarsen (`grid_res=`/`regrid_to=`) before CCA or any dense operation.

## GPCC gauge analyses (`http` adapter, DWD open data) — no credentials

Monthly **gauge-only** 1° analyses from the Global Precipitation Climatology Centre, served as gzipped NetCDF on DWD's open-data server. Gauge-only, so they complement the satellite-merged products (`obs/cmap`, `obs/gpcp-v2-3`) rather than duplicating them.

| Product | Variable | Native → canonical | Resolution | Coverage | Notes |
|---|---|---|---|---|---|
| `obs/gpcc-monitoring-v2020` | precip | `p` → precip, **mm/month** | 1.0° monthly | 1982-present (~2-month lag) | The quality-controlled monitoring analysis. Carries seven further fields beside `p` (gauge count `s`, interpolation/gauge errors, solid+liquid fractions) which pass through unused. |
| `obs/gpcc-first-guess` | precip | `p` → precip, **mm/month** | 1.0° monthly | **2013**-present (~1-month lag) | Near-real-time first guess from far fewer stations; superseded by the monitoring product once it catches up. |

Both keep native monthly **totals** as `mm/month` (the `obs/tamsat` convention) rather than dividing by a flat 30 into a fake daily rate. Latitude is stored north-to-south upstream; normalize sorts it ascending.

**COVERAGE CAVEAT (`obs/gpcc-first-guess`):** the year directories on DWD go back to 2004, but 2004-2012 hold **only** legacy GrADS binaries (`gpcc_first_guess_MM_YYYY.gz`). The `.nc.gz` files this entry fetches begin in **2013**, which is where `hindcast_range` starts. Use `obs/gpcc-monitoring-v2020` for anything earlier.

**Why both declare `time_from_pattern`:** neither in-file time axis is usable as served. `monitoring_v2020` writes `20240301.0` with units `day as %Y%m%d.%f` — a GrADS convention, not CF `"<unit> since <epoch>"`, so nothing decodes it. `first_guess` writes `0` with units `months since 2024-3-1` — unpadded, so `decode_times=True` raises outright, and since every file says `0` against its own reference month, concatenating them without per-file stamping collapses the whole record onto a single date. The adapter therefore takes the `(year, month)` that built each URL as the authority. The files are also gzipped, which the adapter expands before opening.

## GPCP v2.3 (`http` adapter, NOAA NCEI) — no credentials

| Product | Variable | Native → canonical | Resolution | Coverage | Notes |
|---|---|---|---|---|---|
| `obs/gpcp-v2-3` | precip | `precip` mm/day (native) | 2.5° monthly | 1979-present (~3-month lag) | Satellite-gauge merged precipitation from the producing archive (NCEI). Native `mm/day`, matching `obs/cmap`. Files carry `precip_error` and CF bounds variables alongside `precip`; the bounds are dropped on the way out. |

NCEI stamps every filename with its processing date (`gpcp_v02r03_monthly_d202403_c20240607.nc`), which no static pattern can predict, so the entry's `file_pattern` ends in a `*` and the adapter resolves it against the directory listing (one listing per year, so a 45-year pull costs 45 listings, not 540). If a month was reprocessed and appears twice, the later `_c<date>` wins.

**GOTCHA (preliminary stream):** the same directory holds a `gpcp_v02r03-preliminary_monthly_…` stream for months the final product hasn't reached yet. The literal part of the pattern excludes it, so `obs/gpcp-v2-3` is always the settled product — a request for a month that only exists as preliminary raises `FileNotFoundError` rather than quietly returning a provisional estimate. Add a separate `-prelim` entry if the current-season tail is ever needed, the way `obs/chirps-v3-dekad-tif` / `-prelim` are paired.

## CHC CHIRPS-GEFS short-range forecasts (`http` adapter, issuance-keyed) — no credentials

These carry an `issuance` catalog block: they are keyed by an issuance date, not a season. Fetch with `init="YYYY-MM-DD"` (a single issuance) or a **sequence** of `YYYY-MM-DD` dates (many issuances stacked on `init_time`). A season `target` cannot be combined with them; select the target window from `lead_time` afterwards. See `references/data-conventions.md` for the `init_time`/`lead_time`/`valid_time` output layout.

| Product | Variable | Resolution | Leads | Coverage | Notes |
|---|---|---|---|---|---|
| `chc/chirps-gefs-daily` | precip mm/day | 0.05° daily | 16 daily leads (`leads: [0, 15]`, lead 0 = issuance day) | hindcast 2001-2019, forecast 2021-present — **2020 absent** | CHIRPS3-GEFS daily precip forecast, issued daily. Aggregate to dekads/pentads downstream rather than fetching CHC's separately-keyed dekad product. |
| `chc/chirps-gefs-15day` | precip mm | 0.05°, 15-day accumulation | single lead (`leads: [0, 0]`) | hindcast 2001-2019, forecast 2021-present | One accumulated 15-day total per issuance. **GOTCHA:** `valid_time` marks the window's **start** (the [init, init+15d) window), not its end. |

Both share the CHC CrowdSec constraint (`request_interval: 3.0`, 4-hour IP ban above ~2 req/s). The 2001-2019 window is the GEFSv12 reforecast era.

## Catalog entry anatomy

Each variable block in a product declares:

```yaml
variables:
  precip:
    native_name: prate      # upstream name (short_name optional fallback)
    units: kg m-2 s-1       # source units
    target_units: mm/day    # canonical output units
    accumulated: true       # optional: deaccumulate over lead_time
    fill_value: -9999       # optional: masked to NaN
    scale: -1.0             # optional: multiplier applied AFTER the conversion
                            # and AFTER fill_value masking. For a source whose
                            # SIGN convention differs from acmaddl's — ERA5-Land
                            # pev is a negative upward flux.
```

Product-level keys of interest: `adapter`, `grid`, `hindcast`/`forecast` year ranges, split-stream config, `member_reduce`, `request_interval`, `max_workers`, `alias_of`, `deprecated_after`, `successor`, `deprecation_note`, `pending_url`.

Two `http`-adapter keys are worth knowing:

- **`time_from_pattern: true`** — take the time coordinate from the `(year, month)` that built each filename instead of the file's own axis, stamped per file **before** the concat, and skip CF time decoding entirely. For sources whose in-file encoding is undecodable or not self-contained (both GPCC streams). A file holding more than one time step raises rather than being mislabelled.
- **`*` in `file_pattern`** — resolved against the server's directory listing, for filenames carrying something unpredictable (NCEI's processing-date suffix). No match raises `FileNotFoundError`; several matches take the lexically greatest name.

A `.gz` URL is gunzipped automatically — no key needed, since the suffix is unambiguous.
