"""Rhiza weather-skills products: a credential-free IFS-ENS forecast and a CHIRPS window.

Requires the rhiza dependency group (`uv sync --group weather-skills`) and network access.
"""
import acmaddl

KENYA = [-5, 5.5, 33.5, 42]   # [lat_south, lat_north, lon_west, lon_east]

# Latest published IFS-ENS init (dynamical.org open catalog, no credentials).
latest = acmaddl.check_product("weather-skills/ifs-ens-15d", probe_remote=True)["latest"]

# (init_time, lead_time, member, lat, lon); lead_time is a timedelta of native
# 3-6 hourly steps; member 0 is the control run; precip is a rate in mm/day.
fc = acmaddl.fetch("weather-skills/ifs-ens-15d", "precip", init=latest, region=KENYA)
print(fc)
print(fc.attrs["weather_skills_name"], fc.attrs["weather_skills_version"], fc.attrs["weather_skills_pin"][:8])

# Ensemble-mean rate, averaged per lead day (timedelta -> whole days), first 10 days.
daily = (fc["precip"].mean("member").squeeze("init_time")
         .groupby(fc["lead_time"].dt.days).mean())
print(daily.isel(days=slice(0, 10)).mean(("lat", "lon")).values)

# CHIRPS daily, the last 10 published days: no hindcast= means the trailing
# window ending on the day CHIRPS has actually published (it lags about a
# week). Their skill has no --bbox, so acmaddl fetches in 10-day chunks and
# crops each chunk to Kenya as it loads. For a specific month use
# hindcast=(year, year), months=[m]; a month with nothing published yet is
# refused with "not published yet".
obs = acmaddl.fetch("weather-skills/chirps-daily", "precip", region=KENYA)
print(obs)
