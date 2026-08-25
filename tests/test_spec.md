# acmadDL — Test Specification

Tests are organized in layers: unit tests that run fast with no network, integration tests that hit real APIs, and end-to-end tests that verify the full `fetch()` pipeline.

---

## 1. Unit tests (no network, no credentials)

These test internal logic using fixtures and mocks. They should run in <5 seconds total.

### 1.1 Catalog

```
test_catalog_loads
  catalog.yaml parses without error.
  Every product entry has required fields: adapter, variables, grid.

test_catalog_list_products
  catalog.list_products() returns a non-empty list of strings.

test_catalog_info
  catalog.info("nmme/cfsv2") returns a dict with expected keys.
  catalog.info("nonexistent/product") raises KeyError or returns None.

test_catalog_variable_mapping
  For each product, every variable entry has native_name, units, target_units.
```

### 1.2 Normalize

```
test_normalize_renames_variables
  Given an xarray Dataset with native_name "prate", normalize() renames it to "precip".

test_normalize_standardizes_coords
  Input with "latitude"/"longitude" dims → output has "lat"/"lon".
  Input with "forecast_time" → output has "time".

test_normalize_converts_units
  K → C: input value 300.0 → output value 26.85.
  kg/m²/s → mm/day: input value 1.0 → output ≈ 86400.0.
  mm/day → mm/day: no change.

test_normalize_subsets_region
  Given a global dataset and region [-12, 6, 28, 42], output lat/lon are within bounds.

test_normalize_cf_compliance
  Output has 'units' attribute on each variable.
  Coordinates have 'axis' attributes (lat→Y, lon→X, time→T).
```

### 1.3 Storage

```
test_save_netcdf_local
  save(ds, "/tmp/test.nc", format="netcdf") creates a readable NetCDF file.
  xr.open_dataset("/tmp/test.nc") round-trips the data.

test_save_returns_none_when_no_destination
  fetch() with destination=None returns xarray but writes nothing.

test_save_unsupported_format_raises
  save(ds, "out.xyz", format="xyz") raises ValueError.
```

### 1.4 Adapter base

```
test_adapter_base_is_abstract
  Instantiating AdapterBase directly raises TypeError.

test_get_adapter_returns_correct_class
  get_adapter("cds") returns CDSAdapter instance.
  get_adapter("opendap") returns OPeNDAPAdapter instance.
  get_adapter("unknown") raises KeyError.
```

### 1.5 Date/season parsing

```
test_parse_target_season
  "MAM" → (March 1, May 31) of the relevant year.
  "OND" → (October 1, December 31).
  (datetime(2025,3,1), datetime(2025,5,31)) passes through unchanged.

test_parse_init
  "2025-02" → datetime(2025, 2, 1).
  datetime(2025, 2, 1) passes through unchanged.
```

---

## 2. Integration tests (network required, real APIs)

These hit actual data sources. They are slow and require credentials for CDS. Mark with `@pytest.mark.integration` so they can be skipped in CI.

### 2.1 OPeNDAP adapter — NMME/CFSv2

```
test_opendap_fetch_cfsv2_hindcast
  Fetch CFSv2 precipitation hindcast for:
    region: East Africa [–12, 6, 28, 42]
    hindcast: (2010, 2012)  # just 3 years, fast
    init: February
    target: MAM
  Assert:
    - Returns xr.Dataset
    - Has dimension "year" with 3 values
    - Has dimension "member"
    - Has dimensions "lat", "lon" within requested bbox
    - Variable "prate" (or native name) is present
    - No NaN-only slices

test_opendap_fetch_cfsv2_realtime
  Fetch CFSv2 real-time forecast for:
    init: most recent available month
    target: next season
    region: East Africa
  Assert:
    - Returns xr.Dataset
    - Has "member" dimension
    - Spatial extent matches requested region
```

### 2.2 CDS adapter — C3S/ECMWF

```
test_cds_fetch_ecmwf_hindcast
  Fetch ECMWF SEAS5 temperature hindcast for:
    region: East Africa [–12, 6, 28, 42]
    hindcast: (2014, 2016)  # just 3 years
    init: February
    target: MAM
  Assert:
    - Returns xr.Dataset
    - Has expected dimensions (year, member, lat, lon)
    - Temperature values are physically plausible (200-330 K range)

test_cds_fetch_era5
  Fetch ERA5 monthly temperature for:
    region: East Africa
    years: (2015, 2016)
  Assert:
    - Returns xr.Dataset with (time, lat, lon)
    - 0.25° resolution
```

### 2.3 HTTP adapter — CHIRPS

```
test_http_fetch_chirps
  Fetch CHIRPS v2 monthly precipitation for:
    region: East Africa [–12, 6, 28, 42]
    years: (2015, 2016)
  Assert:
    - Returns xr.Dataset with (time, lat, lon)
    - ~0.05° resolution
    - Precipitation values ≥ 0
```

---

## 3. End-to-end tests (full `fetch()` pipeline)

These test the complete flow: catalog lookup → adapter → normalize → storage.

### 3.1 Fetch and normalize

```
test_e2e_fetch_cfsv2_normalized
  ds = acmaddl.fetch(
      product="nmme/cfsv2",
      variable="precip",
      init="2015-02",
      target="MAM",
      region=[-12, 6, 28, 42],
      hindcast=(2010, 2012),
  )
  Assert:
    - Variable name is "precip" (not "prate")
    - Coordinates are "lat", "lon" (not "latitude", "longitude")
    - Units are mm/day
    - Region is correctly subsetted

test_e2e_fetch_chirps_normalized
  ds = acmaddl.fetch(
      product="obs/chirps",
      variable="precip",
      target="MAM",
      region=[-12, 6, 28, 42],
      hindcast=(2010, 2012),
  )
  Assert:
    - Fine-resolution grid (~0.05°)
    - Variable name is "precip"
    - Time dimension spans requested years
```

### 3.2 Fetch and write

```
test_e2e_fetch_and_save_netcdf
  ds = acmaddl.fetch(
      product="nmme/cfsv2",
      variable="precip",
      init="2015-02",
      target="MAM",
      region=[-12, 6, 28, 42],
      hindcast=(2010, 2012),
      destination="/tmp/test_cfsv2.nc",
      format="netcdf",
  )
  Assert:
    - File exists at /tmp/test_cfsv2.nc
    - xr.open_dataset reads it back with matching data
    - Returned ds matches file contents
```

### 3.3 Fetch for DeepScale compatibility

This is the "contract test" — it verifies that acmadDL output is ready for DeepScale.

```
test_e2e_deepscale_contract_gcm
  ds = acmaddl.fetch(product="nmme/cfsv2", variable="precip", ...)
  Assert:
    - ds has dims: {year, member, lat, lon} (hindcast) or {member, lat, lon} (forecast)
    - ds.lat and ds.lon are float64, monotonically increasing
    - ds["precip"] is a DataArray with no object-type coords

test_e2e_deepscale_contract_obs
  ds = acmaddl.fetch(product="obs/chirps", variable="precip", ...)
  Assert:
    - ds has dims: {year, lat, lon} or {time, lat, lon}
    - Grid resolution is finer than the GCM product
    - ds["precip"] values are finite and ≥ 0
```

---

## Running tests

```bash
# Unit tests only (fast, no network)
pytest acmaddl/tests/ -m "not integration"

# All tests including integration (slow, needs network + CDS credentials)
pytest acmaddl/tests/ -m ""

# Just the contract tests
pytest acmaddl/tests/ -k "deepscale_contract"
```
