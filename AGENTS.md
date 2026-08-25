# Agent instructions for acmaddl

Instructions for any AI coding agent (Claude Code, Codex, OpenCode, etc.) working in this repository.

## Before you start

Read [`skills/acmaddl/SKILL.md`](skills/acmaddl/SKILL.md) before writing or modifying code that uses acmaddl. It is the authoritative quick reference for the public API, product catalog, data conventions, credentials, and caching behavior, with deeper detail in `skills/acmaddl/references/`. If your harness supports Agent Skills natively, load the skill instead of re-deriving the API from source.

## Keep the docs and skill in sync — this is a hard requirement

The skill is a snapshot of the source. Any change that alters observable behavior MUST update the matching documentation **in the same PR**:

| If you change... | You must update... |
|---|---|
| `fetch()` / `assemble()` signatures, defaults, or behavior | `skills/acmaddl/SKILL.md` + `skills/acmaddl/references/api.md` |
| `fetch()` params `months`, `degenerate_attempts`, `init` sequence / issuance handling | `skills/acmaddl/references/api.md` (params) + `data-conventions.md` (issuance `init_time`/`lead_time`/`valid_time`) + `SKILL.md` |
| `zonal.py` (`acmaddl.zonal` signature, stats, weights, gotchas) | `skills/acmaddl/references/api.md` + `SKILL.md` (+ `examples/zonal_districts.py`) |
| `assemble.py` `obs_predictor` (obs-as-predictor signature/behavior) | `skills/acmaddl/references/api.md` + `SKILL.md` |
| `catalog.yaml` (add/remove/deprecate products, variables, adapters) | `skills/acmaddl/references/products.md` (+ `SKILL.md` product families if a new family) |
| Issuance-keyed catalog entries (CHIRPS-GEFS) or OPeNDAP obs entries (ERSST/CMAP, `max_request_years`) | `skills/acmaddl/references/products.md` + `data-conventions.md` (issuance layout) + `troubleshooting.md` (truncation guard) + `examples/chirps_gefs_issuance.py` |
| `normalize.py` (renames, unit conversions, pipeline order, `select_lon`/`sanitize_for_netcdf`) | `skills/acmaddl/references/data-conventions.md` (+ `plotting.md` if dims/units/orientation guarantees change) |
| Degenerate-response guard (`_robust.reject_if_degenerate`, `DegenerateResponseError`, opt-in vs always-on) | `skills/acmaddl/references/troubleshooting.md` + `api.md` (`degenerate_attempts`) |
| Caching (`_CACHE_VERSION`, cache keys, env vars, nuthatch config) | `skills/acmaddl/references/troubleshooting.md` |
| Credentials/adapters (new source, auth flow, rate limits) | `skills/acmaddl/references/products.md` + `troubleshooting.md` |
| Anything user-facing | `README.md` if it covers the topic |

Also update `skills/acmaddl/examples/` if a change breaks or obsoletes an example. If you are unsure whether a change is documented, grep `skills/` and `README.md` for the function, product id, or parameter you touched — stale docs are treated as bugs.

## Repo conventions

- Package layout: `src/acmaddl/` (import name `acmaddl`, distribution `acmadDL`). Python ≥ 3.12, `uv` for dependency management.
- Tests: `pytest` for the unit suite; markers `integration`, `cds`, `network` gate live-network tests. New behavior needs tests.
- Caching lives only in `fetch._fetch_raw_cached` — never add `@cache` decorators inside adapters. Bump `_CACHE_VERSION` when adapter logic or normalization changes output.
- The catalog is declarative: prefer adding/adjusting `catalog.yaml` entries over special-casing code paths.
