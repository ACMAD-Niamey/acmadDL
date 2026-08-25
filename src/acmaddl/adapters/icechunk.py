import xarray as xr
from .base import AdapterBase


class IcechunkAdapter(AdapterBase):
    """Adapter for reading from Icechunk versioned Zarr stores."""

    def _open_repo(self, product_config):
        import icechunk

        source_url = product_config["source_url"]
        repo_config = None
        if "icechunk_config" in product_config:
            repo_config = icechunk.RepositoryConfig.default()
            cfg = product_config["icechunk_config"]
            if "cache_bytes" in cfg:
                repo_config.caching = icechunk.CachingConfig(
                    num_bytes_chunks=cfg["cache_bytes"]
                )

        if source_url.startswith("s3://"):
            parts = source_url.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            creds = product_config.get("credentials", {})
            aws_profile = product_config.get("aws_profile")
            region = product_config.get("aws_region", "us-east-1")
            if aws_profile:
                import boto3
                session = boto3.Session(profile_name=aws_profile)
                frozen = session.get_credentials().get_frozen_credentials()
                storage = icechunk.s3_storage(
                    bucket=bucket,
                    prefix=prefix,
                    access_key_id=frozen.access_key,
                    secret_access_key=frozen.secret_key,
                    session_token=frozen.token,
                    region=region,
                )
            elif creds:
                storage = icechunk.s3_storage(
                    bucket=bucket,
                    prefix=prefix,
                    access_key_id=creds.get("access_key_id"),
                    secret_access_key=creds.get("secret_access_key"),
                    region=creds.get("region"),
                    endpoint_url=creds.get("endpoint_url"),
                )
            else:
                storage = icechunk.s3_storage(
                    bucket=bucket, prefix=prefix, from_env=True
                )
        elif source_url.startswith("gs://"):
            parts = source_url.replace("gs://", "").split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            storage = icechunk.gcs_storage(
                bucket=bucket, prefix=prefix, from_env=True
            )
        else:
            storage = icechunk.local_filesystem_storage(source_url)

        if repo_config:
            return icechunk.Repository.open(storage, config=repo_config)
        return icechunk.Repository.open(storage)

    def _get_session(self, repo, product_config):
        version = product_config.get("version")
        if version is None:
            return repo.readonly_session(branch="main")
        if version.startswith("tag:"):
            return repo.readonly_session(tag=version[4:])
        if version.startswith("branch:"):
            return repo.readonly_session(branch=version[7:])
        # Treat as snapshot ID
        return repo.readonly_session(snapshot_id=version)

    def health_check(self, product_config, probe_remote=False):
        url = product_config.get("source_url")
        if not url:
            return {
                "healthy": False,
                "kind": "config",
                "message": "Missing source_url in product config.",
                "probe_remote": bool(probe_remote),
            }

        if not probe_remote:
            return {
                "healthy": True,
                "kind": "config",
                "message": "Icechunk adapter config is valid.",
                "probe_remote": False,
            }

        try:
            repo = self._open_repo(product_config)
            session = self._get_session(repo, product_config)
            ds = xr.open_zarr(session.store, zarr_format=3, consolidated=False)
            ds.close()
            return {
                "healthy": True,
                "kind": "remote",
                "message": "Icechunk repository opened successfully.",
                "probe_remote": True,
            }
        except Exception as e:
            return {
                "healthy": False,
                "kind": "remote",
                "message": f"Icechunk probe failed: {e}",
                "probe_remote": True,
            }

    def fetch_data(self, product_config, variable, date_range=None, region=None):
        verbose = product_config.get("_verbose", True)
        source_url = product_config["source_url"]
        if verbose:
            version = product_config.get("version", "main")
            print(f"[acmaddl:icechunk] opening repository: {source_url} @ {version}")

        repo = self._open_repo(product_config)
        session = self._get_session(repo, product_config)

        group = product_config.get("group")
        ds = xr.open_zarr(
            session.store,
            group=group,
            zarr_format=3,
            consolidated=False,
        )

        # Select the variable of interest
        var_cfg = product_config["variables"][variable]
        native_name = var_cfg["native_name"]
        if native_name in ds:
            ds = ds[[native_name]]
        elif variable in ds:
            ds = ds[[variable]]

        # Temporal filtering
        if date_range:
            y0, y1 = date_range
            if "time" in ds.coords:
                ds = ds.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
            elif "init_time" in ds.coords:
                ds = ds.sel(init_time=slice(f"{y0}-01-01", f"{y1}-12-31"))

        # Spatial subsetting
        if region:
            lat_s, lat_n, lon_w, lon_e = region
            lat_name = next(
                (n for n in ("lat", "latitude", "Y") if n in ds.dims), None
            )
            lon_name = next(
                (n for n in ("lon", "longitude", "X") if n in ds.dims), None
            )
            if lat_name and lon_name:
                # Handle descending lat by swapping slice bounds
                lat_vals = ds[lat_name].values
                if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
                    lat_slice = slice(lat_n, lat_s)
                else:
                    lat_slice = slice(lat_s, lat_n)
                ds = ds.sel(
                    {lat_name: lat_slice, lon_name: slice(lon_w, lon_e)}
                )

        if verbose:
            print(f"[acmaddl:icechunk] loaded {list(ds.data_vars)}, dims={dict(ds.dims)}")

        return ds
