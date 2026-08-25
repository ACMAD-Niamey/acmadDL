"""Quick-look visualization of fetched data: maps, ensemble facets, time series.

Needs the demo extra: pip install acmadDL[demo]  (matplotlib + cartopy).
See references/plotting.md for the full recipe set.
"""

import matplotlib

matplotlib.use("Agg")  # headless-safe; drop for interactive use

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

from acmaddl import fetch

REGION = [-5, 15, 33, 48]  # Horn of Africa: [lat_s, lat_n, lon_w, lon_e]

# --- 1. Observation climatology map --------------------------------------
obs = fetch("obs/era5", "precip", region=REGION, hindcast=(1993, 2016))["precip"]

ax = plt.axes(projection=ccrs.PlateCarree())
obs.mean("time").plot(ax=ax, transform=ccrs.PlateCarree(), cmap="YlGnBu",
                      cbar_kwargs={"label": "precip (mm/day)"})
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=0.4)
ax.set_title("ERA5 precip climatology 1993-2016")
plt.savefig("obs_climatology.png", dpi=200, bbox_inches="tight")
plt.close()

# --- 2. Hindcast ensemble members for one year ----------------------------
hc = fetch("nmme/cfsv2", "precip", init="2024-02", target="MAM", region=REGION,
           hindcast=(1993, 2016), year_index=True)["precip"]  # (year, member, lat, lon), mm

g = hc.sel(year=2016).plot(col="member", col_wrap=4, cmap="YlGnBu")
g.fig.suptitle("CFSv2 MAM 2016 hindcast by member", y=1.02)
g.fig.savefig("members_2016.png", dpi=150, bbox_inches="tight")
plt.close(g.fig)

# --- 3. Area-weighted regional-mean time series ---------------------------
w = np.cos(np.deg2rad(obs.lat))
fig, ax = plt.subplots(figsize=(8, 3))
obs.weighted(w).mean(["lat", "lon"]).plot(ax=ax)
ax.set_ylabel("precip (mm/day)")
ax.set_title("Regional-mean monthly precip")
fig.savefig("regional_series.png", dpi=150, bbox_inches="tight")
