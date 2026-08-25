"""Fetch issuance-keyed CHIRPS-GEFS short-range forecasts.

CHIRPS-GEFS products carry an `issuance` catalog block: they are addressed by
issuance DATE, not by season. No credentials, but they live on CHC's UCSB server
behind CrowdSec — the catalog pins request_interval: 3.0 (a 4-hour IP ban waits
above ~2 req/s).
"""

from acmaddl import fetch

ETHIOPIA = [3, 15, 33, 48]  # [lat_s, lat_n, lon_w, lon_e]

# --- Single issuance ------------------------------------------------------
# init is a full YYYY-MM-DD date. chc/chirps-gefs-daily carries 16 daily leads
# (0-15, where lead 0 is the issuance day itself).
fcst = fetch("chc/chirps-gefs-daily", "precip", init="2026-06-30", region=ETHIOPIA)
# -> dims include init_time, lead_time (days), lat, lon; plus a valid_time coord
#    mapping each (init_time, lead_time) to the date it verifies against.
print(dict(fcst.sizes))

# --- Many issuances at once (hindcast-skill case) -------------------------
# Pass a SEQUENCE of dates to stack the same calendar issuance across years on
# init_time. This is allowed ONLY for issuance-keyed products; a sequence to a
# non-issuance product raises ValueError, and a season `target` cannot be
# combined with a sequence (fetch the leads, select the target window after).
hindcast_issuances = fetch(
    "chc/chirps-gefs-daily",
    "precip",
    init=[f"{y}-06-30" for y in range(2001, 2020)],  # GEFSv12 reforecast era
    region=ETHIOPIA,
)
print(dict(hindcast_issuances.sizes))

# --- 15-day accumulation --------------------------------------------------
# chc/chirps-gefs-15day is a single accumulated 15-day total per issuance (one
# lead), in mm (not mm/day). GOTCHA: valid_time marks the window's START —
# the [init, init+15d) window — not its end.
acc = fetch("chc/chirps-gefs-15day", "precip", init="2026-06-30", region=ETHIOPIA)
print(dict(acc.sizes), acc["precip"].attrs.get("units"))

# Notes:
# - 2020 is deliberately absent from chc/chirps-gefs-daily (hindcast 2001-2019,
#   forecast 2021-present) — don't request it.
# - Aggregate the daily product to dekads/pentads downstream rather than fetching
#   CHC's separate dekad forecast, which is keyed by target dekad, not issuance.
