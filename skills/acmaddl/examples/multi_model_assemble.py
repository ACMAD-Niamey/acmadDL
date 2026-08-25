"""Assemble a multi-model roster into deepscale-ready arrays.

assemble() fans out over fetch() with year_index=True and returns
{label: (hindcast, forecast)} where each DataArray is (year, member, lat, lon)
with a guaranteed member dim — the exact shape deepscale's methods consume
(they call hindcast.mean("member")).
"""

from acmaddl import assemble

HORN_OF_AFRICA = [-5, 15, 33, 48]

# Roster rows: (label, product, *ranges). range_index (default 2) picks which
# range column is the hindcast window — here the third element.
roster = [
    ("CFSv2",   "nmme/cfsv2",   (2011, None), (1993, 2016)),
    ("GEOSS2S", "nmme/geoss2s", (2017, None), (1993, 2016)),
    ("SPEAR",   "nmme/spear",   (2021, None), (1993, 2016)),
]

models = assemble(
    roster,
    "precip",
    init="2024-02",       # forecast issuance month
    target="MAM",         # target season -> lead selection
    region=HORN_OF_AFRICA,
    grid_res=1.0,         # put every model on a common 1-degree grid
    boundary="cover",     # assemble's default: keep every touched cell
)
# All targeted precipitation tracks use the seasonal-total contract (mm),
# regardless of whether their source is NMME, C3S/CDS, or IRI.
# CFSv2 contributes its 24 populated members (empty upstream slots are dropped).

for label, (hindcast, forecast) in models.items():
    print(f"{label}: hindcast {dict(hindcast.sizes)}, forecast {dict(forecast.sizes)}, "
          f"units={hindcast.attrs.get('units')}")

# Hand straight to deepscale, e.g.:
#   import deepscale as ds
#   best = ds.optimize(models["CFSv2"][0], obs, methods=["bcsd", "cca"])
