"""Rhiza weather-skills products: a credential-free IFS-ENS forecast and a CHIRPS window.

Requires the rhiza dependency group (`uv sync --group rhiza`) and network access.
"""
from datetime import date

import acmaddl

KENYA = [-5, 5.5, 33.5, 42]   # [lat_south, lat_north, lon_west, lon_east]

# Latest published IFS-ENS init (dynamical.org open catalog, no credentials).
latest = acmaddl.check_product("rhiza/ifs-ens-15d", probe_remote=True)["latest"]

# (init_time, lead_time, member, lat, lon); lead_time is a timedelta of native
# 3-6 hourly steps; member 0 is the control run; precip is a rate in mm/day.
fc = acmaddl.fetch("rhiza/ifs-ens-15d", "precip", init=latest, region=KENYA)
print(fc)
print(fc.attrs["rhiza_skill"], fc.attrs["rhiza_skill_version"], fc.attrs["rhiza_pin"][:8])

# Ensemble-mean rate, averaged per lead day (timedelta -> whole days), first 10 days.
daily = (fc["precip"].mean("member").squeeze("init_time")
         .groupby(fc["lead_time"].dt.days).mean())
print(daily.isel(days=slice(0, 10)).mean(("lat", "lon")).values)

# CHIRPS daily for the current month to date: fetched in 10-day chunks upstream
# (their skill has no --bbox), cropped to Kenya as each chunk loads, clipped to
# the day CHIRPS has actually published.
today = date.today()
obs = acmaddl.fetch("rhiza/chirps-daily", "precip", hindcast=(today.year, today.year),
                    months=[today.month], region=KENYA)
print(obs)
