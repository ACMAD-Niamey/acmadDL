"""Sub-seasonal (S2S) fetches: a dated issuance plus its on-the-fly reforecasts.

c3s/ecmwf-s2s uses the ECMWF Data Store (ECDS) — a SEPARATE service from
Copernicus CDS. Point ~/.cdsapirc (or CDSAPI_URL/CDSAPI_KEY) at
https://ecds.ecmwf.int/api and accept both the site Terms of Use and the
dataset licence.
"""

from acmaddl import fetch

KENYA_BBOX = [-5, 5, 33, 42]

# --- Real-time forecast for a single issuance date -----------------------
# S2S inits are daily, so init is a full date (YYYY-MM-DD).
fcst = fetch(
    "c3s/ecmwf-s2s",
    "precip",
    init="2026-07-17",
    region=KENYA_BBOX,
)
# -> (init_time, lead_time, member, lat, lon) at 1.5 degrees, precip mm/day
# (accumulated source precip is deaccumulated over lead_time automatically)

# --- On-the-fly reforecasts for the same issuance ------------------------
# reforecast=True switches to the s2s-reforecasts collection. Without an
# explicit hindcast range it defaults to (init_year - 20, init_year - 1).
# Reforecast issuance dates arrive as an `hdate` dim, normalized to an
# integer `year` dim.
rfcst = fetch(
    "c3s/ecmwf-s2s",
    "precip",
    init="2026-07-17",
    region=KENYA_BBOX,
    reforecast=True,
    hindcast=(2006, 2025),
)

print(dict(fcst.sizes))
print(dict(rfcst.sizes))

# Pair with dekadal observations for verification (public, no credentials):
obs = fetch(
    "obs/chirps-v2-dekadal-rhiza",  # 0.25-degree, 10-day rolling aggregate
    "precip",
    region=KENYA_BBOX,
    hindcast=(2006, 2025),
)
