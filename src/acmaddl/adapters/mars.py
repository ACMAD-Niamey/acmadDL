"""Legacy-MARS adapter using ecmwf-api-client.

Used for ECMWF S2S reforecasts, where the modern ECDS endpoint
(`s2s-reforecasts`) currently has a broken MARS routing on its
`marsth-ecmwf` partition. The legacy ECMWF Web API hits MARS
production directly with the documented S2S reforecast request shape
(`stream=enfh`, `type=pf`, explicit `hdate` list).

Forecast retrieval continues to flow through the CDS adapter — this
adapter is reforecast-only on day 1.

Auth: requires ~/.ecmwfapirc with url/key/email. Register at
https://api.ecmwf.int/v1/key/ if you don't have one.
"""

from __future__ import annotations

import os
import tempfile

import xarray as xr

from .base import AdapterBase


def _open_downloaded(path: str) -> xr.Dataset:
    """Open a GRIB file produced by ECMWFDataServer.retrieve.

    We deliberately leave MARS to return GRIB (not NetCDF): MARS's
    server-side ``grib_to_netcdf`` flattens the ``hdate × step`` axes
    into a single ``time`` axis and silently drops ``hdate``, which
    destroys the per-year structure deepscale needs. cfgrib preserves
    both axes.

    ``decode_timedelta=False`` matches what the CDS adapter does for
    S2S NetCDFs — recent xarray versions raise an assertion on
    non-nanosecond timedeltas (the ``step`` coordinate is in hours).
    """
    return xr.open_dataset(path, engine="cfgrib", decode_timedelta=False)


class MARSAdapter(AdapterBase):
    """Hits the legacy ECMWF Web API via ecmwf-api-client.

    Currently reforecast-only. Forecast support could be added later by
    relaxing the `_reforecast` gate and switching `stream`/`type`.
    """

    def fetch_data(self, config, variable, date_range=None, region=None):
        if not config.get("_reforecast"):
            raise ValueError(
                "MARSAdapter is reforecast-only in v1; got _reforecast=False. "
                "Forecasts should be fetched via the CDS adapter."
            )
        if date_range is None:
            raise ValueError(
                "MARSAdapter requires date_range=(start_year, end_year) to "
                "build the hdate list."
            )

        init_date = config["_init_date"]  # 'YYYY-MM-DD'
        _, mm, dd = init_date.split("-")

        hy_start, hy_end = int(date_range[0]), int(date_range[1])
        hdate = "/".join(f"{yr}-{mm}-{dd}" for yr in range(hy_start, hy_end + 1))

        var_cfg = config["variables"][variable]
        mars_param = var_cfg.get("mars_param", "228228")  # default to total precip

        # Region: acmaddl's convention is [lat_s, lat_n, lon_w, lon_e].
        # MARS area is N/W/S/E slash-delimited.
        if region is None:
            raise ValueError("MARSAdapter requires a region.")
        lat_s, lat_n, lon_w, lon_e = region
        area = f"{lat_n}/{lon_w}/{lat_s}/{lon_e}"

        origin = config.get("cds_model", "ecmf")  # same value the CDS path uses
        # MARS step: default the same 0..1104 by 24 hours window as the
        # forecast path. Catalog can override via mars_step.
        step = config.get("mars_step", "0/to/1104/by/24")
        # ECMWF S2S perturbed reforecasts publish 10 perturbed members
        # (numbers 1..10). MARS rejects an unspecified `number` field on
        # type=pf reforecast requests with "expected N, got 10*N" because
        # its internal field-count check assumes one member. Catalog can
        # override via mars_number to subset members.
        number = config.get("mars_number", "1/to/10")

        req = {
            "class": "s2",
            "dataset": "s2s",
            "expver": "prod",
            "date": init_date,
            "hdate": hdate,
            "stream": "enfh",
            "type": "pf",
            "origin": origin,
            "levtype": "sfc",
            "param": mars_param,
            "step": step,
            "number": number,
            "time": "00:00:00",
            "area": area,
        }

        from ecmwfapi import ECMWFDataServer
        server = ECMWFDataServer()

        tmpdir = tempfile.mkdtemp(prefix="acmaddl_mars_")
        target = os.path.join(tmpdir, "reforecast.grib")
        req["target"] = target

        if config.get("_verbose"):
            print(f"[acmaddl:mars] reforecast request date={init_date} "
                  f"hdate={hdate.split('/')[0]}..{hdate.split('/')[-1]} area={area}")

        server.retrieve(req)
        ds = _open_downloaded(target)
        # cfgrib labels the historical issuance axis as `time` (the hdate
        # values). Rename to `hdate` so the acmaddl normalize layer's
        # `hdate → year` transform fires, and so `valid_time → time` can
        # then rename without conflict.
        if "time" in ds.dims:
            ds = ds.rename({"time": "hdate"})
        return ds
