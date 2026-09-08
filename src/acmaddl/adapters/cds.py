import os
import re
import shutil
import tempfile
import zipfile
import xarray as xr
from .base import AdapterBase
from .._paths import get_tmpdir



class CDSAdapter(AdapterBase):
    def health_check(self, product_config, probe_remote=False):
        required = ["cds_dataset", "variables"]
        missing = [k for k in required if k not in product_config]
        if missing:
            return {
                "healthy": False,
                "kind": "config",
                "message": f"Missing required config keys: {missing}",
                "probe_remote": bool(probe_remote),
            }

        if not probe_remote:
            return {
                "healthy": True,
                "kind": "config",
                "message": "CDS adapter config is valid.",
                "probe_remote": False,
            }

        try:
            import cdsapi

            cdsapi.Client()
            return {
                "healthy": True,
                "kind": "remote",
                "message": "CDS client initialized successfully.",
                "probe_remote": True,
            }
        except Exception as e:
            return {
                "healthy": False,
                "kind": "remote",
                "message": f"CDS client init failed: {e}",
                "probe_remote": True,
            }

    def fetch_data(self, product_config, variable, date_range=None, region=None):
        verbose = product_config.get("_verbose", True)
        import cdsapi
        # cdsapi's defaults are retry_max=500, sleep_max=120 — designed for long
        # production queues, but they hide real errors behind ~17h of silent retry.
        # 12 × 30s = 6min max covers legit queue waits while failing fast enough to
        # surface the underlying CDS response.
        # Allow products to override the CDS endpoint URL — ECDS
        # (ecds.ecmwf.int) carries S2S data on a different endpoint than the
        # standard Copernicus CDS.
        client_kwargs = dict(retry_max=12, sleep_max=30, timeout=60,
                             quiet=not verbose)
        if "cds_url" in product_config:
            client_kwargs["url"] = product_config["cds_url"]
        client = cdsapi.Client(**client_kwargs)
        var_cfg = product_config["variables"][variable]
        dataset = product_config["cds_dataset"]
        # Reforecast mode: ECDS exposes the on-the-fly hindcast suite as a
        # separate collection (`s2s-reforecasts`, plural). The realtime
        # `s2s-forecasts` collection's `forecast_type=reforecast` option is
        # broken upstream (MARS routing error class=s2 ambiguous). Switching
        # the collection routes through ECDS's working reforecast pipeline.
        if product_config.get("_reforecast") and dataset == "s2s-forecasts":
            dataset = "s2s-reforecasts"

        request = {
            "variable": var_cfg["native_name"],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        if dataset in ("s2s-forecasts", "s2s-reforecasts"):
            # Sub-seasonal forecasts have a different request shape than the
            # seasonal datasets handled below: single-day issuance, forecast_type
            # selector, Mars-style leadtime range, `origin` not `originating_centre`.
            init_date = product_config.get("_init_date")
            if not init_date:
                raise ValueError(
                    "s2s-forecasts requires _init_date (YYYY-MM-DD) in product "
                    "config; pass it via fetch(..., init='YYYY-MM-DD')."
                )
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(init_date)):
                raise ValueError(
                    f"_init_date must be YYYY-MM-DD format, got {init_date!r}. "
                    f"Example: '2026-05-12'."
                )
            y, m, d = init_date.split("-")
            request["year"] = y
            request["month"] = m
            request["day"] = d
            if dataset == "s2s-reforecasts":
                # Reforecast collection uses a different forecast_type vocabulary;
                # `perturbed_reforecast` returns the 10-member ensemble we need
                # for BCSD/CCA/rank-analog training. Catalog can override.
                request["forecast_type"] = product_config.get(
                    "reforecast_type", "perturbed_reforecast"
                )
            else:
                request["forecast_type"] = product_config.get(
                    "forecast_type", "perturbed_forecast"
                )
            request["time"] = product_config.get("forecast_time", "00:00")
            if "cds_model" in product_config:
                request["origin"] = product_config["cds_model"]
            # Leadtime spec, most specific first: per-variable, then per-product,
            # then the MARS range shorthand.
            #
            # The two forms are NOT interchangeable, and each fails on the other's
            # variables (verified 2026-09-04 against s2s-forecasts, init 2026-08-31):
            #   * Accumulated fields (precip) need the shorthand "0/to/1104/by/24".
            #     Passing explicit "0_24"-style periods makes MARS return no data
            #     at all (MarsNoDataError).
            #   * Instantaneous fields (sst) need the explicit list. With the
            #     shorthand ECDS silently returns only step 24 — one lead instead
            #     of 46, no error — so it presents as missing data rather than a
            #     malformed request.
            # Hence the per-variable override, set on sst in the catalog.
            leadtime = var_cfg.get(
                "leadtime_hour",
                product_config.get("leadtime_hour", ["0/to/1104/by/24"]),
            )
            request["leadtime_hour"] = [str(h) for h in leadtime]
            # level_type is required by ECDS — without it the MARS request lands
            # in a test partition (marsth-ecmwf) with no data, producing
            # MarsNoDataError. Surface variables use "single_level"; pressure
            # variables would override with "pressure" via the catalog.
            request["level_type"] = product_config.get("level_type", "single_level")
            if region:
                lat_s, lat_n, lon_w, lon_e = region
                request["area"] = [lat_n, lon_w, lat_s, lon_e]
            if verbose:
                print(f"[acmaddl:cds] s2s request dataset={dataset} "
                      f"init={init_date} forecast_type={request['forecast_type']}")
            tmpdir = get_tmpdir()
            with tempfile.NamedTemporaryFile(
                suffix=".nc", delete=False, dir=tmpdir
            ) as tmp:
                download_path = tmp.name
            extract_dir = None
            try:
                client.retrieve(dataset, request, download_path)
                if verbose:
                    print("[acmaddl:cds] download complete")
                if zipfile.is_zipfile(download_path):
                    extract_dir = tempfile.mkdtemp(prefix="acmaddl_cds_", dir=tmpdir)
                    with zipfile.ZipFile(download_path) as zf:
                        zf.extractall(extract_dir)
                    ncs = sorted(
                        os.path.join(extract_dir, f)
                        for f in os.listdir(extract_dir)
                        if f.endswith(".nc")
                    )
                    if not ncs:
                        raise RuntimeError(
                            f"CDS returned a zip with no .nc files: "
                            f"{os.listdir(extract_dir)}"
                        )
                    if len(ncs) == 1:
                        with xr.open_dataset(ncs[0], decode_timedelta=False) as opened:
                            ds = opened.load()
                    else:
                        with xr.open_mfdataset(
                            ncs, decode_timedelta=False, combine="by_coords"
                        ) as opened:
                            ds = opened.load()
                else:
                    with xr.open_dataset(download_path, decode_timedelta=False) as opened:
                        ds = opened.load()
                # ECDS reforecasts label the historical-issuance axis as `time`
                # (datetime64 values, one per hindcast year). Rename to `hdate`
                # so the normalize layer's existing `hdate → year` transform
                # converts to integer years, and so `valid_time → time` can
                # rename downstream without colliding.
                if dataset == "s2s-reforecasts" and "time" in ds.dims:
                    ds = ds.rename({"time": "hdate"})
                return ds
            finally:
                try:
                    os.unlink(download_path)
                except OSError:
                    pass
                if extract_dir is not None:
                    shutil.rmtree(extract_dir, ignore_errors=True)

        for key in ("product_type", "system"):
            if key in product_config:
                request[key] = product_config[key]

        if date_range:
            years = [str(y) for y in range(date_range[0], date_range[1] + 1)]
        else:
            from datetime import datetime
            years = [str(datetime.now().year)]
        request["year"] = years

        if "init_months" in product_config:
            request["month"] = [f"{m:02d}" for m in product_config["init_months"]]
        else:
            request["month"] = [f"{m:02d}" for m in range(1, 13)]

        if product_config.get("product_type") == "monthly_averaged_reanalysis":
            request["time"] = "00:00"

        if "cds_model" in product_config:
            request["originating_centre"] = product_config["cds_model"]

        if dataset == "seasonal-original-single-levels":
            request["day"] = "01"

        if "leadtime_hour" in product_config:
            request["leadtime_hour"] = [str(h) for h in product_config["leadtime_hour"]]
        elif "leadtime_month" in product_config:
            request["leadtime_month"] = [str(m) for m in product_config["leadtime_month"]]
        elif "cds_model" in product_config:
            request["leadtime_hour"] = [str(h) for h in range(24, 24 * 215, 24)]

        if region:
            lat_s, lat_n, lon_w, lon_e = region
            request["area"] = [lat_n, lon_w, lat_s, lon_e]

        if verbose:
            print(f"[acmaddl:cds] request dataset={dataset} years={len(request['year'])}")
        tmpdir = get_tmpdir()
        with tempfile.NamedTemporaryFile(
            suffix=".nc", delete=False, dir=tmpdir
        ) as tmp:
            download_path = tmp.name
        extract_dir = None
        try:
            client.retrieve(dataset, request, download_path)
            if verbose:
                print("[acmaddl:cds] download complete")
            # CDS silently delivers a zip for datasets that don't support
            # `download_format: unarchived` (e.g. seasonal-original-single-levels
            # daily requests). Detect by magic bytes, extract, and combine.
            if zipfile.is_zipfile(download_path):
                extract_dir = tempfile.mkdtemp(prefix="acmaddl_cds_", dir=tmpdir)
                with zipfile.ZipFile(download_path) as zf:
                    zf.extractall(extract_dir)
                ncs = sorted(
                    os.path.join(extract_dir, f)
                    for f in os.listdir(extract_dir)
                    if f.endswith(".nc")
                )
                if not ncs:
                    raise RuntimeError(
                        f"CDS returned a zip with no .nc files: {os.listdir(extract_dir)}"
                    )
                if verbose:
                    print(f"[acmaddl:cds] extracted {len(ncs)} netCDF file(s) from zip")
                # decode_timedelta=False works around an xarray 2025.1.0 assertion
                # that fails when pandas returns timedelta64[us] instead of [ns].
                if len(ncs) == 1:
                    open_kwargs = {"decode_timedelta": False}
                    with xr.open_dataset(ncs[0], **open_kwargs) as opened:
                        ds = opened.load()
                else:
                    with xr.open_mfdataset(
                        ncs, decode_timedelta=False, combine="by_coords"
                    ) as opened:
                        ds = opened.load()
            else:
                with xr.open_dataset(download_path, decode_timedelta=False) as opened:
                    ds = opened.load()
            # Eager load + close detaches ds from any on-disk file. Required for
            # the nuthatch cache: pickling a lazy ds stores only a reopen recipe
            # pointing at this temp file, which gets cleaned up by macOS and
            # breaks the cache across sessions (issue #24).
            return ds
        finally:
            try:
                os.unlink(download_path)
            except OSError:
                pass
            if extract_dir is not None:
                shutil.rmtree(extract_dir, ignore_errors=True)
