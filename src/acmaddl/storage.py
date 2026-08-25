def save(ds, destination, format="netcdf"):
    if format == "netcdf":
        _save_netcdf(ds, destination)
    elif format == "geotiff":
        _save_geotiff(ds, destination)
    else:
        raise ValueError(
            f"Unsupported format: {format}. Use 'netcdf' or 'geotiff'."
        )


def _save_netcdf(ds, destination):
    if destination.startswith("s3://"):
        import s3fs
        fs = s3fs.S3FileSystem()
        with fs.open(destination, "wb") as f:
            ds.to_netcdf(f)
    else:
        ds.to_netcdf(destination)


def _save_geotiff(ds, destination):
    """Write a Dataset to GeoTIFF (EPSG:4326).

    Reductions applied so the result fits a (bands, lat, lon) raster:
      - `member` dim -> ensemble mean
      - length-1 `init_time` -> squeezed
    The remaining non-spatial dim (typically `lead_time` for forecasts or
    `time` for observations) becomes bands; its coord values are written as
    band descriptions so GIS tools can identify them.

    Raises ValueError if the dataset can't be reduced to at most one
    non-spatial dim, or doesn't have (lat, lon).
    """
    if destination.startswith("s3://"):
        raise NotImplementedError(
            "GeoTIFF export to S3 is not yet supported."
        )

    import rioxarray  # noqa: F401  registers the .rio accessor

    data_vars = list(ds.data_vars)
    if len(data_vars) != 1:
        raise ValueError(
            f"GeoTIFF export expects exactly one data variable, "
            f"found {len(data_vars)}: {data_vars}"
        )
    da = ds[data_vars[0]]

    if "member" in da.dims:
        da = da.mean(dim="member", keep_attrs=True)
    if "init_time" in da.dims and da.sizes["init_time"] == 1:
        da = da.squeeze("init_time", drop=False)

    if not {"lat", "lon"}.issubset(set(da.dims)):
        raise ValueError(
            f"GeoTIFF export requires lat and lon dims; got {da.dims}"
        )

    extra_dims = [d for d in da.dims if d not in {"lat", "lon"}]
    if len(extra_dims) > 1:
        raise ValueError(
            f"Cannot write GeoTIFF: {len(extra_dims)} non-spatial dims "
            f"remain after reduction ({extra_dims}). Select a single "
            f"slice (e.g. .isel(lead_time=0)) before export."
        )

    da = da.sortby("lat", ascending=False)
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    da = da.rio.write_crs("EPSG:4326", inplace=False)
    da.rio.to_raster(destination)

    if extra_dims:
        band_dim = extra_dims[0]
        labels = [f"{band_dim}={v}" for v in da[band_dim].values]
        import rasterio
        with rasterio.open(destination, "r+") as dst:
            for i, label in enumerate(labels, start=1):
                dst.set_band_description(i, label)
