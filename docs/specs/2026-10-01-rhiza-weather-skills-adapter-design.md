# Rhiza weather-skills fetchers as acmadDL products

**Status:** proposed
**Date:** 2026-10-01
**Motivating case:** Emmett's request (2026-09-30) to bring Rhiza Research's weather-skills catalog into the ACCORD stack as its subseasonal component, using their code unchanged.

## The problem

Rhiza Research publishes a catalog of "weather skills": standalone Python scripts, one per capability, built on the `@weather_skill` decorator from `weather-skills-core`. The fetchers in that catalog cover the subseasonal and medium range that acmadDL is thin on: ECMWF S2S and IFS-ENS, GEFS, AIFS, the SubC multi-model outlooks, plus credential-free ERA5, OISST, SMAP, CMIP6, CHIRPS and IMERG.

The constraint is that we run their code, not a port of it. The skills must therefore be executed as they are, and only their *output* is reshaped into the dataset acmadDL promises its callers.

Two facts make this workable in-process rather than as a subprocess:

- `@weather_skill` returns a wrapper that takes an argument list (`fetch(["--date", ...])`), parses it exactly as the CLI would, runs the skill, and writes the Zarr to the `-o` path. There is no in-memory return to intercept, so the adapter always round-trips through a temporary Zarr.
- The provider packages (`weather-skills`, `chc-skills`) build into wheels that bundle every `skills/<name>/scripts/<file>.py` as a data file. The script can be located through `importlib.metadata.files()` and loaded by path.

Both were verified in a throwaway environment on 2026-10-01: three fetchers imported by path and answered `--help`; a real IFS-ENS fetch over Kenya completed in six seconds through the in-process wrapper.

## Decisions already taken

| Question | Decision |
|---|---|
| Run their code how | In-process import of the pinned provider packages, not a subprocess. |
| Which fetchers | Every gridded fetcher in both provider packages: 9 skills, 27 products (table below). |
| Control member | Kept. Their `number=0` is the control run; the catalog states the full member count and documents that member 0 is the control. |
| Family prefix | `rhiza/`, naming the organisation that serves the route, as `chc/` and `c3s/` do. |
| Lead time | Native steps, kept as `timedelta64`, with `valid_time` derived. This is already acmadDL's convention for issuance-keyed products (`adapters/_issuance.lead_timedelta`); the data-conventions reference that says "numeric" is stale and is corrected here. |
| Reforecasts | Not in this round. Their ECMWF S2S fetcher has no reforecast mode; acmadDL's existing `c3s/ecmwf-s2s` reforecast path is untouched. |

## Dependencies

### The one conflict, and the override

`sheerwater` pins `xarray==2025.1.0` (still true in its latest release, 0.2.10, 2026-05-12). `weather-skills-core` needs `xarray>=2026.7` and `dynamical-catalog==0.5.0` needs `xarray>=2025.1.2`. acmadDL already overrides sheerwater's equally stale `zarr==2.18.3` pin; the same mechanism takes the xarray pin:

```toml
[tool.uv]
override-dependencies = ["zarr>=3.1.0", "xarray>=2026.7"]
```

With that override the full set resolves (263 packages, Python 3.12). acmadDL's offline CI subset passed under the moved versions (xarray 2026.9, numpy 2.5, pandas 3.0.6, sheerwater 0.2.10): 206 passed, 1 failure that was an install-layout artifact of the probe and not a regression.

The override is a uv feature. A plain `pip install` cannot co-install sheerwater with the Rhiza packages. This is already the situation for the icechunk extra and is documented the same way.

### A dependency group, not an extra

acmadDL publishes to PyPI, and PyPI rejects metadata whose requirements are direct git URLs. The Rhiza packages therefore live in a uv dependency group:

```toml
[dependency-groups]
rhiza = [
    "weather-skills-core",
    "weather-skills",
    "chc-skills",
    # Per-skill runtime needs declared in the scripts' PEP 723 blocks.
    "ecmwf-datastores-client==0.4.2",
    "dynamical-catalog==0.5.0",
    "pint-xarray>=0.6",
    "cf-xarray>=0.11",
    "aiohttp",
    "dask",
    "gcsfs",
    "h5netcdf",
    "h5py",
]

[tool.uv.sources]
weather-skills-core = { git = "https://github.com/rhiza-research/weather-skills-core", rev = "e118b9531181224cb1459efbd9f2b114117f0a79" }
weather-skills      = { git = "https://github.com/rhiza-research/weather-skills",      rev = "4f95364b323e601f5cbd43ee3af39d2d77617596" }
chc-skills          = { git = "https://github.com/rhiza-research/chc-skills",          rev = "20bce784a77a10a1fe7a368e29669a5d673eb176" }
```

Pins are commits, not `main`: their `main` is their release branch and auto-bumps skill versions on every merge. Moving a pin is a deliberate change that:

1. re-captures the test fixtures (below), and
2. bumps `fetch._CACHE_VERSION`, because the raw-fetch cache key does not see the pin and a re-pinned skill may write a different dataset for the same arguments.

Install: `uv sync --group rhiza`. Without the group, every `rhiza/*` fetch raises an error that names that command. CI (`test.yml`, `publish.yml`) adds the group to its sync step.

The scripts' PEP 723 headers say `requires-python = ">=3.12,<3.13"`. That ceiling governs `uv run --script`, which we bypass; `weather-skills-core` itself has no ceiling. CI runs 3.12, matching their target.

## The adapter

`src/acmaddl/adapters/rhiza.py`, class `RhizaAdapter(AdapterBase)`, registered as `"rhiza"` in `adapters/__init__.py`. One class for every skill; adding a fetcher is a catalog entry.

### Catalog grammar

```yaml
rhiza/ifs-ens-15d:
  adapter: rhiza
  provider: weather-skills            # distribution to look in; default weather-skills
  skill: dynamical-fetch              # skills/<skill>/scripts/*.py
  entrypoint: fetch                   # decorated function name; default fetch
  argv: ["--dataset", "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
         "--date", "{init}", "--bbox", "{bbox}", "-v", "{variable}"]
  requires_region: false              # true -> fetch() without region= is an error
  probe_latest: true                  # remote health probe may call --probe-latest
  credentials: null                   # or "ecds" (see Credentials)
  window_days: null                   # observation fetchers only (see Windows)
  variables:
    precip: {native_name: precipitation_surface, units: "mm/day", target_units: "mm/day"}
    temp:   {native_name: temperature_2m,        units: C,        target_units: C}
  grid:
    lat_res: 0.25
    lon_res: 0.25
    hindcast_range: null
    forecast_range: [2024, null]
    forecast_members: 51              # member 0 is the control
    temporal: 3-6 hourly
  notes: >-
    Rhiza weather-skills dynamical-fetch. Member 0 is the control run. One
    issuance per fetch; no reforecast stream.
```

Template fields available to `argv`:

| Field | Value | Source |
|---|---|---|
| `{init}` | `YYYY-MM-DD` | `config["_init_date"]`, set by `fetch(init=...)` |
| `{bbox}` | `N/W/S/E` | `region` bbox `[lat_s, lat_n, lon_w, lon_e]` reordered |
| `{start}`, `{end}` | `YYYY-MM-DD` | observation window (see Windows) |
| `{variable}` | native name | `variables[<canonical>].native_name` |

If `region` is `None` and the template contains a `--bbox {bbox}` pair, the pair is dropped unless `requires_region` is true, in which case `fetch()` raises before anything runs (ecmwf-fetch makes `--bbox` mandatory).

### Fetch path

1. **Resolve the script.** `importlib.metadata.files(provider)` is scanned for `skills/<skill>/scripts/*.py`; exactly one match is required. The module is loaded with `importlib.util.spec_from_file_location` under a private name and memoised per process. `getattr(module, entrypoint)` must be a `@weather_skill` wrapper (it carries a `.parser` attribute). A missing distribution raises `RhizaNotInstalled` with the `uv sync --group rhiza` hint.
2. **Render `argv`** from the template and append `-o <tmpdir>/out.zarr`.
3. **Map credentials** (below), then **run** `wrapper(argv)` with stderr captured. The decorator turns `SkillError`/`UsageError` into `sys.exit(code)` after printing the message; the adapter catches `SystemExit` with a non-zero code and raises `RhizaSkillError(message)`, a subclass of acmadDL's fetch error, carrying the captured text. Any other exception propagates unchanged.
4. **Open** the Zarr (`consolidated=True`), `.load()` it, remove the temporary directory.
5. **Reshape** (below) and return the raw dataset to `_fetch_raw_cached`.

### Reshape rules

Deliberately small; everything else is left to `normalize()`.

- **Forecast products** (the entry rendered `{init}`): any scalar `time` or `valid_time` coordinate is dropped, `init_time` is added as a length-1 dimension from the requested init, and `valid_time = init_time + step` is derived, as `adapters/http.py` does for CHIRPS-GEFS. This single rule covers ecmwf-fetch and dynamical-fetch (scalar `time` is the init) and SubC (scalar `time` is the valid date, so it must not be mistaken for the init). `normalize()` then renames `step -> lead_time`, `number -> member`, `latitude -> lat`, `longitude -> lon`, and maps `valid_time` onto the canonical `time` name.
- **Observation products** (`{start}`/`{end}` rendered): `time` is already a dimension; nothing to do.
- The `step_bounds` helper coordinate and its `nv` dimension are dropped.
- Attributes kept on the dataset: `weather_skills_history` and `weather_skills_source` (so their `provenance` skill still reads our files), plus three of ours: `rhiza_skill`, `rhiza_skill_version` (the script's `_SKILL_VERSION`), `rhiza_pin` (the provider commit). `sanitize_for_netcdf` already serialises string attributes; the history is a JSON string.
- `lead_time` stays `timedelta64[ns]`. Native steps are preserved: daily for ECMWF S2S, 3- to 6-hourly for IFS-ENS and GEFS, one step for a SubC outlook.

Units are declared in the catalog and converted by the existing `(units, target_units)` table. Their precipitation rates are already `mm day-1` (declared `mm/day`), temperatures `degree_Celsius` (declared `C`), SST `degree_Celsius` (declared and delivered `C`; the catalog note says so because most acmadDL SST is `K`). SubC precipitation is a window **sum** in `mm`, declared as such.

### Windows (observation fetchers)

acmadDL's observation API is year-based (`hindcast=(y0, y1)`) with an optional `months=` filter, which `fetch()` stores as `config["init_months"]`. The adapter turns that into one or more `[start, end]` windows: the first day of the first selected month to the last day of the last, per year. `window_days` splits each window into chunks of at most that many days; each chunk is fetched, cropped to the bbox as soon as it loads, and the chunks are concatenated on `time`.

This exists because `chirps-fetch` and `imerg-fetch` have no `--bbox` flag and build the full global grid in memory (about 104 MB and 26 MB per day respectively). Their entries set `window_days: 10`. Fetchers that accept `--bbox` set no `window_days` and pass the box through.

### Credentials

`credentials: ecds` on `rhiza/ecmwf-s2s`. Their script requires `ECMWF_DATASTORES_URL` and `ECMWF_DATASTORES_KEY` in the environment (or `~/.ecmwfdatastoresrc`, which their client reads itself). If both variables are unset, the adapter reads `~/.cdsapirc` (or `CDSAPI_URL`/`CDSAPI_KEY`), and only if that URL points at `ecds.ecmwf.int` exports them for the duration of the call, restoring the environment afterwards. A cdsapirc pointing at the Copernicus CDS carries a different token, so in that case the adapter raises with instructions instead of sending the wrong key.

IMERG and SMAP authenticate through `earthaccess`, which reads `EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD` or `.netrc` itself. `earthaccess` is already in acmadDL's dependency tree, so there is nothing to map.

### Health check

`health_check(config, probe_remote=False)` follows the existing adapters:

- config: the group is installed, the distribution has exactly one script for `skill`, `entrypoint` resolves to a decorated wrapper, and `wrapper(["--help"])` exits 0.
- `probe_remote=True` and `probe_latest: true`: run the skill's own `--probe-latest` (with `--dataset`/`--outlook` where the skill needs them) and report the date it prints.

### Caching

No new cache arguments. A `rhiza/*` forecast is keyed like any S2S product (`init_date` is the issuance), an observation like any other (`date_range`, `init_months`, `region`). New products start with empty caches, so `_CACHE_VERSION` does not change in this PR; it changes whenever a pin moves.

## Catalogued

Nine skills, 27 products. Native variable names are taken from each skill's documented output and confirmed by the integration tests.

### Forecasts

| Product | Skill (dataset) | Grid, leads, members | Credentials |
|---|---|---|---|
| `rhiza/ecmwf-s2s` | ecmwf-fetch | 1.5°, daily to 46 d, 101 (0 = control) | ECDS; 2-day embargo |
| `rhiza/ifs-ens-15d` | dynamical-fetch (`ecmwf-ifs-ens-forecast-15-day-0-25-degree`) | 0.25°, 3-6 h to 15 d, 51 | none |
| `rhiza/ifs-ens-46d` | dynamical-fetch (`ecmwf-ifs-ens-forecast-46-day-daily-1-5-degree`) | 1.5°, daily to 46 d, 51 | none |
| `rhiza/ifs-ens-46d-6h` | dynamical-fetch (`ecmwf-ifs-ens-forecast-46-day-6-hourly-1-5-degree`) | 1.5°, 6 h to 46 d, 51 | none |
| `rhiza/aifs-ens` | dynamical-fetch (`ecmwf-aifs-ens-forecast`) | AIFS ensemble | none |
| `rhiza/aifs-single` | dynamical-fetch (`ecmwf-aifs-single-forecast`) | AIFS deterministic (no member dimension) | none |
| `rhiza/gefs-35d` | dynamical-fetch (`noaa-gefs-forecast-35-day`) | 0.25°, to 35 d, 31 | none |
| `rhiza/gfs` | dynamical-fetch (`noaa-gfs-forecast`) | deterministic | none |
| `rhiza/icon-eu-5d` | dynamical-fetch (`dwd-icon-eu-forecast-5-day`) | Europe only | none |
| `rhiza/hrdps` | dynamical-fetch (`eccc-hrdps-forecast`) | Canada only | none |
| `rhiza/hrrr-48h` | dynamical-fetch (`noaa-hrrr-forecast-48-hour`) | CONUS only | none |
| `rhiza/subc-mme-7d`, `-15d`, `-30d` | subc-mme-fetch (`--outlook 7d\|15d\|30d`, chc-skills) | 1°, one lead, MME mean + anomaly | none |

ecmwf-fetch's variables: `tp -> precip`, `t2m -> temp`, `sst -> sst`; the other single-level fields (`d2m`, `mx2t6`, `mn2t6`, `u10`, `v10`, `msl`, `cape`, `tcw`) are exposed under their native names. Pressure-level fields are not catalogued (control-only, `vertical` dimension; round two). dynamical-fetch's variables: `precipitation_surface -> precip`, `temperature_2m -> temp`. SubC's: `pr -> precip` (window sum, `mm`), `tas -> temp`, `ts -> sst`; `tasmax`, `tasmin`, `tdps` under their native names; each `<var>_anomaly` field passes through unchanged beside the mean.

### Observations and analyses

| Product | Skill (dataset) | Output | Credentials |
|---|---|---|---|
| `rhiza/chirps-daily` | chirps-fetch | 0.05° daily `precip`, `window_days: 10`, cropped post-load | none |
| `rhiza/imerg-daily` | imerg-fetch (`--version late`) | 0.1° daily `precip`, `window_days: 10` | Earthdata |
| `rhiza/imerg-daily-final` | imerg-fetch (`--version final`) | as above | Earthdata |
| `rhiza/imerg-early-30min` | dynamical-fetch (`nasa-imerg-analysis-early`) | half-hourly `precip` | none |
| `rhiza/imerg-late-30min` | dynamical-fetch (`nasa-imerg-analysis-late`) | half-hourly `precip` | none |
| `rhiza/gefs-analysis` | dynamical-fetch (`noaa-gefs-analysis`) | analysis | none |
| `rhiza/gfs-analysis` | dynamical-fetch (`noaa-gfs-analysis`) | analysis | none |
| `rhiza/hrrr-analysis` | dynamical-fetch (`noaa-hrrr-analysis`) | CONUS only | none |
| `rhiza/mrms-hourly` | dynamical-fetch (`noaa-mrms-conus-analysis-hourly`) | CONUS only | none |
| `rhiza/era5` | arco-era5-fetch | ERA5 reanalysis, `total_precipitation -> precip`, `2m_temperature -> temp`, `sea_surface_temperature -> sst` | none |
| `rhiza/oisst-daily` | oisst-fetch | 0.25° daily `sst` (°C, land NaN) | none |
| `rhiza/smap-daily` | smap-fetch (`--overpass AM`) | 9 km daily `soil_moisture` | Earthdata |
| `rhiza/cmip6` | cmip6-fetch (`--model GFDL-CM4 --experiment ssp245 --table Amon`) | monthly `pr -> precip`, `tas -> temp` | none |

`rhiza/cmip6` pins one model and scenario because they are skill arguments; other combinations are a copied catalog block with different `argv`. The five regional products (ICON-EU, HRDPS, HRRR, MRMS) are catalogued for completeness; a fetch over an African bbox returns an empty selection and the catalog note says so.

## Round-one limits

Stated in each entry's `notes` and in the docs:

- One issuance per fetch. `fetch(init=[...])` sequences stay a follow-up (the entries declare no `issuance` block, so `fetch()` rejects a sequence with its existing message).
- No reforecasts. These products cannot feed `assemble()` hindcast tuples or the training leg of the africas2s S2S testbed. Round two decides between contributing a reforecast flag upstream and adding historical products to acmadDL.
- Global observation fetchers (`rhiza/chirps-daily`, `rhiza/imerg-daily*`) should be called with `months=` or a one-year `hindcast`; a decade of daily global CHIRPS is not a reasonable request through them.
- `year_index=True` is not blocked on `rhiza/*` forecasts, but with a single real-time issuance and no hindcast it collapses to a one-year axis and means nothing; the docs say so.

## Testing

### Offline (bare `pytest`, in CI)

`tests/test_rhiza_adapter.py`, no network. Fixtures under `tests/fixtures/rhiza/`: real outputs captured once per output shape at the pinned commits by `tools/capture_rhiza_fixtures.py` over a tiny bbox and subset to at most 3 members, 4 steps, 3×3 cells: `ensemble_forecast.zarr` (dynamical IFS-ENS), `single_forecast.zarr` (GFS), `s2s_forecast.zarr` (ecmwf-fetch), `analysis.zarr` (GEFS analysis), `daily_obs.zarr` (CHIRPS), `subc_envelope.zarr`. Re-capturing them is part of moving a pin.

- Script resolution: found in the distribution's file list; `RhizaNotInstalled` when the group is absent (simulated by patching `importlib.metadata.files`).
- `argv` rendering for every catalogued entry: `{init}`, `{bbox}` ordering, `{start}`/`{end}`, `{variable}`, the `--bbox` drop when `region=None`, the `requires_region` error.
- Credential mapping: variables set only when absent, environment restored, refusal when `~/.cdsapirc` points at Copernicus CDS.
- Run path with a fake wrapper that copies a fixture to the `-o` path: returned dataset, temporary directory removed, the three `rhiza_*` attributes present, their two attributes preserved.
- Error path: fake wrapper printing to stderr and raising `SystemExit(2)` becomes `RhizaSkillError` containing the message; a `RuntimeError` propagates untouched.
- Windows: `window_days` chunking, per-chunk crop, concatenation on `time`, year × months enumeration.
- Reshape + `normalize()` on each fixture: dims `(init_time, lead_time, member, lat, lon)` or `(time, lat, lon)`, `lead_time` dtype `timedelta64[ns]`, derived `time` equals init plus lead, member 0 present, latitude ascending, declared units; SubC's valid-date `time` not mistaken for the init.
- Catalog contract: every `rhiza/*` entry has `adapter: rhiza`, `skill`, `argv`, complete `variables` and `grid`; every `skill` resolves in the pinned distribution (skipped when the group is not installed).
- Health check config path, including the `--help` probe.
- `tests/test_mcp_server.py`: `list_products` includes `rhiza/*`.

### Real data (`tests/test_rhiza_integration.py`)

Marked `integration` and `network`, following `tests/test_integration.py` and `tests/test_sheerwater_integration.py`; skipped in the default run; run on demand:

```
uv run pytest -m "integration and not cds" tests/test_rhiza_integration.py
```

At least one test per skill, and one per output shape for dynamical-fetch, each a real fetch through `acmaddl.fetch` over the shared small East-Africa bbox with the smallest sensible request, asserting the normalized contract (`_check_dataset`-style helper plus dims, `timedelta64` lead, member 0, units, finite values, provenance attributes):

| Test | Request | Marker beyond integration+network |
|---|---|---|
| IFS-ENS 15 d | latest init via `--probe-latest`, `precip` | — |
| GFS | latest init, `temp` | — |
| GEFS analysis | one day | — |
| IMERG late half-hourly | one hour | — |
| ECMWF S2S | an init 3 days old, `precip` | `cds` (ECDS key) |
| SubC 7 d | latest init via probe, `precip`, checks window-sum units and anomaly passthrough | — |
| CHIRPS daily | two days with `window_days` forced to 1, checks chunk concatenation | — |
| IMERG daily | two days | skipped without Earthdata credentials |
| ERA5 | two days, `temp` | — |
| OISST | two days | — |
| SMAP | two days | skipped without Earthdata credentials |
| CMIP6 | one month of `tas` | — |

Plus one cross-check: `tests/test_sheerwater_integration.py` is run under the moved xarray to confirm the sheerwater override is still only stale, not load-bearing.

### CI

`test.yml` and `publish.yml`: `uv sync --extra dev --extra mcp --extra geo --group rhiza`; `test.yml` adds `tests/test_rhiza_adapter.py` to its explicit file list. Integration tests stay manual, as today.

## Documentation (AGENTS.md sync table)

- `README.md`: "Rhiza weather-skills products (`rhiza/*`)" section under *Available products* with the 27-row table; install note (`uv sync --group rhiza`, the xarray override, uv-only); credentials; round-one limits.
- `skills/acmaddl/SKILL.md`: the family in the product-families list and the one-issuance / no-reforecast limits.
- `skills/acmaddl/references/products.md`: the `rhiza/*` section.
- `skills/acmaddl/references/data-conventions.md`: `lead_time` is a `timedelta64` for issuance-keyed and `rhiza/*` forecasts (correcting "numeric"); member 0 is the control for `rhiza/*`; `valid_time` derivation; the `rhiza_*` and `weather_skills_*` attributes.
- `skills/acmaddl/references/troubleshooting.md`: `RhizaNotInstalled`; pip cannot co-install sheerwater with the group; memory for the global observation fetchers; errors surfaced from their scripts (embargo, missing credentials); a pin bump requires fixture re-capture and `_CACHE_VERSION` bump.
- `skills/acmaddl/examples/rhiza_fetch.py`: IFS-ENS over Kenya plus a 10-day CHIRPS window.
- `src/acmaddl/catalog.yaml` header: the `rhiza` adapter knobs.
- `AGENTS.md`: a sync-table row for `adapters/rhiza.py`, the pins, and the fixtures.
- MCP server: no code change; `list_products`/`fetch` read the catalog at runtime. `README.md` "MCP server" is unchanged.

## Out of scope (round two candidates)

- Reforecasts: a `--reforecast` flag contributed upstream to ecmwf-fetch, or historical products added to acmadDL.
- Multi-issuance sequences for `rhiza/*` products.
- Station-shaped skills (ghcn-daily-fetch, tahmo-fetch, openaq-fetch): `(time, station_id)` is a new data shape for acmadDL's `normalize` and region handling.
- ecmwf-fetch pressure-level fields (`vertical` dimension); kenya-forecast-fetch (unstable archive contents); the CHC image fetchers (PNG, not data).
- Pointing `scripts/s2s` in africas2s at `rhiza/ecmwf-s2s` or `rhiza/ifs-ens-46d` for its real-time leg: a separate change in that repo.
- Their transform and plot skills: acmadDL's `normalize` and africas2s already cover them.
- The outbound provider repo (ACCORD functionality packaged as weather skills): its own sub-project.

## Verification

```
uv sync --extra dev --extra mcp --extra geo --group rhiza
uv run pytest -q -m "not network and not integration and not cds"      # offline, must stay green
uv run pytest -m "integration and not cds" tests/test_rhiza_integration.py tests/test_sheerwater_integration.py
uv run pytest -m "integration and cds" tests/test_rhiza_integration.py     # with ECDS credentials
uv run python skills/acmaddl/examples/rhiza_fetch.py
uv run acmaddl-mcp --help                                                 # server still imports
```
