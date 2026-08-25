"""Basic acmaddl fetches: observations and a seasonal hindcast.

Requires ~/.cdsapirc (Copernicus CDS) for the ERA5 fetch; the NMME fetch
needs no credentials. Results are cached under ~/.nuthatch/caches, so
re-runs are fast.
"""

from acmaddl import catalog, fetch

HORN_OF_AFRICA = [-5, 15, 33, 48]  # [lat_south, lat_north, lon_west, lon_east]

# --- Discover what's available -------------------------------------------
print(catalog.list_products(include_deprecated=False))
print(catalog.info("nmme/cfsv2")["variables"].keys())

# --- Observations: ERA5 monthly temperature ------------------------------
# (time, lat, lon), temp in C, lat ascending, lon in [-180, 180]
obs = fetch(
    "obs/era5",
    "temp",
    region=HORN_OF_AFRICA,
    hindcast=(1993, 2016),
)
print(obs)

# Season-averaged obs: one value per year for the MAM season -> (year, lat, lon)
obs_mam = fetch(
    "obs/era5",
    "temp",
    region=HORN_OF_AFRICA,
    hindcast=(1993, 2016),
    target="MAM",
    seasonal="mean",
)

# --- Seasonal hindcast: CFSv2 precip, Feb inits targeting MAM ------------
# (init_time, lead_time, member, lat, lon), targeted seasonal precip in mm
hcst = fetch(
    "nmme/cfsv2",
    "precip",
    init="2010-02",
    target="MAM",
    region=HORN_OF_AFRICA,
    hindcast=(1993, 2016),
)

# Same fetch, reshaped for downstream forecasting (e.g. deepscale):
# init_time -> integer year, meaned over lead_time -> (year, member, lat, lon)
hcst_yearly = fetch(
    "nmme/cfsv2",
    "precip",
    init="2010-02",
    target="MAM",
    region=HORN_OF_AFRICA,
    hindcast=(1993, 2016),
    year_index=True,
)

# --- Save while fetching -------------------------------------------------
fetch(
    "obs/era5",
    "precip",
    region=HORN_OF_AFRICA,
    hindcast=(2016, 2016),
    destination="era5_precip_2016.nc",   # or "s3://bucket/key.nc"
    format="netcdf",                     # "geotiff" also supported (geo extra)
)
