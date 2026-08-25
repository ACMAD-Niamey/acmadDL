"""Zonal aggregation: reduce a gridded field over many geometries at once.

``fetch(region=...)`` dissolves a shapefile's features into one mask, because a
fetch answers "give me the data over this area". Reporting asks the other
question — "give me one number *per* district" — and answering it with a Python
loop over a thousand woreda polygons means a thousand rasterizations of the same
grid.

:func:`zonal` rasterizes once. Every geometry is burned into a single integer
label grid, and the reduction is a `groupby` over that grid: one pass, whatever
the number of regions.

    districts = geopandas.read_file("woredas.shp")
    rain = acmaddl.fetch("obs/chirps-v3-dekad", "precip", region="woredas.shp")
    per_district = zonal(rain, districts, by="ADM3_EN")   # (time, region)

The output adds a ``region`` dim and keeps every other dim untouched, so it
drops straight into anything that reduces along time — including
:func:`deepscale.completion.complete`, which is why per-district accumulation
curves need no separate code path.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

_STATS = ("mean", "sum", "min", "max", "median", "std", "count")
_WEIGHTED_STATS = ("mean", "sum")

_GEO_HINT = (
    "Zonal aggregation requires geopandas and rasterio. "
    "Install the geo extra:  pip install 'acmadDL[geo]'"
)


def _require_geo():
    try:
        import geopandas as gpd
        from rasterio import features
        from rasterio.transform import from_bounds
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_GEO_HINT) from e
    return gpd, features, from_bounds


def _as_geodataframe(geometries, gpd):
    if isinstance(geometries, gpd.GeoDataFrame):
        return geometries
    if isinstance(geometries, gpd.GeoSeries):
        return gpd.GeoDataFrame(geometry=geometries)
    if isinstance(geometries, (str,)) or hasattr(geometries, "__fspath__"):
        return gpd.read_file(geometries)
    try:
        return gpd.GeoDataFrame(geometry=list(geometries), crs="EPSG:4326")
    except Exception as e:
        raise TypeError(
            f"Unsupported geometries type {type(geometries).__name__}. Pass a "
            "shapefile path, a GeoDataFrame/GeoSeries, or a list of shapely "
            "geometries."
        ) from e


def _labels_for(gdf, lat, lon, features, from_bounds, all_touched):
    """Burn each geometry's positional index into a (lat, lon) integer grid.

    Cells covered by no geometry get -1. Later geometries win where they
    overlap, which matches ``rasterio.features.rasterize``'s documented
    last-one-wins behaviour; administrative units rarely overlap, and when they
    do there is no non-arbitrary answer.
    """
    n_lat, n_lon = lat.size, lon.size
    if n_lat < 2 or n_lon < 2:
        raise ValueError(
            "zonal needs a grid at least 2x2 to infer cell size; "
            f"got lat={n_lat}, lon={n_lon}"
        )

    d_lat = float(lat[1] - lat[0])
    d_lon = float(lon[1] - lon[0])
    # Pixel edges, from centre coordinates. `from_bounds` expects north-up, so
    # the transform is built with the latitude axis descending and the label
    # grid is flipped back afterwards.
    west, east = float(lon[0]) - d_lon / 2, float(lon[-1]) + d_lon / 2
    south, north = float(lat[0]) - d_lat / 2, float(lat[-1]) + d_lat / 2
    transform = from_bounds(west, south, east, north, n_lon, n_lat)

    shapes = [(geom, i) for i, geom in enumerate(gdf.geometry) if geom is not None]
    if not shapes:
        raise ValueError("geometries contained no usable shapes")

    burned = features.rasterize(
        shapes, out_shape=(n_lat, n_lon), transform=transform,
        fill=-1, dtype="int32", all_touched=all_touched,
    )
    return np.flipud(burned)  # back to ascending latitude


def _weights_for(weights, lat_name, lat_values, template):
    if weights is None:
        return None
    if isinstance(weights, xr.DataArray):
        return weights
    if weights in ("area", "cos_lat"):
        # On a regular lat/lon grid a cell's area is proportional to cos(lat);
        # "area" and "cos_lat" are the same weighting, named two ways.
        cos = np.cos(np.deg2rad(lat_values)).clip(min=0.0)
        return xr.DataArray(cos, dims=lat_name, coords={lat_name: lat_values})
    raise ValueError(
        f"weights must be None, 'area', 'cos_lat' or a DataArray, got {weights!r}"
    )


def zonal(
    data,
    geometries,
    *,
    by: str | None = None,
    label: str | None = None,
    stat: str = "mean",
    weights: str | xr.DataArray | None = "area",
    all_touched: bool = False,
    dim: str = "region",
    lat: str = "lat",
    lon: str = "lon",
):
    """Reduce ``data`` over each geometry, returning a new ``dim`` axis.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Must carry the ``lat`` and ``lon`` dims. Every other dim is preserved.
    geometries : shapefile path, GeoDataFrame, GeoSeries, or list of geometries
        Reprojected to EPSG:4326 if a CRS is attached. **Not** dissolved — one
        output element per feature, which is the difference from
        ``fetch(region=...)``.
    by : str, optional
        Column of ``geometries`` to index the output ``dim`` by (e.g. a unique
        admin code like ``"shapeID"``). Defaults to the positional index. Must
        be unique — an index has to be, for ``.sel`` to be unambiguous. Real
        admin datasets routinely have a unique *code* column and a non-unique
        *name* column; index by the code, and pass the name as ``label``.
    label : str, optional
        Column carried as a companion ``{dim}_label`` coordinate for display.
        Unlike ``by`` it may repeat — two woredas can share a name. This is the
        clean way to keep human-readable names on a uniquely-indexed axis.
    stat : {"mean", "sum", "min", "max", "median", "std", "count"}
        The reduction. ``"count"`` returns the number of contributing cells.
    weights : {"area", "cos_lat", None} or xr.DataArray
        Cell weighting. Defaults to ``"area"``: on a regular lat/lon grid a
        cell's area falls off as ``cos(lat)``, so an unweighted mean over a tall
        region over-counts its poleward cells. Only ``"mean"`` and ``"sum"``
        use weights; the order statistics ignore them.
    all_touched : bool
        Include every cell a geometry touches, rather than only those whose
        centre it contains. Necessary when a region is small relative to the
        grid, or it will pick up no cells at all.

    Returns
    -------
    Same type as ``data``, with ``lat``/``lon`` replaced by ``dim``.

    Notes
    -----
    NaN cells are excluded from the reduction, so a district that is half ocean
    is averaged over its land. A district containing no valid cell yields NaN
    (and ``count`` 0) rather than being dropped, so the output always has one
    element per input geometry.
    """
    if stat not in _STATS:
        raise ValueError(f"stat must be one of {_STATS}, got {stat!r}")
    if lat not in data.dims or lon not in data.dims:
        raise ValueError(
            f"data must have {lat!r} and {lon!r} dims; got {tuple(data.dims)}"
        )

    gpd, features, from_bounds = _require_geo()
    gdf = _as_geodataframe(geometries, gpd)
    if gdf.crs is not None:
        gdf = gdf.to_crs("EPSG:4326")
    if len(gdf) == 0:
        raise ValueError("geometries is empty")

    if by is None:
        names = np.arange(len(gdf))
    else:
        if by not in gdf.columns:
            raise ValueError(
                f"column {by!r} not found in geometries; have {list(gdf.columns)}"
            )
        names = gdf[by].to_numpy()
        if len(set(names.tolist())) != len(names):
            dupes = sorted({n for n in names.tolist() if names.tolist().count(n) > 1})
            raise ValueError(
                f"column {by!r} has duplicate values ({dupes[:5]}...), so it "
                f"cannot index the {dim!r} axis. Admin datasets usually have a "
                "unique code column (e.g. 'shapeID') and a repeating name column "
                "— index by the code and pass the name as label=. Or pass "
                "by=None for a positional index."
            )

    label_values = None
    if label is not None:
        if label not in gdf.columns:
            raise ValueError(
                f"label column {label!r} not found in geometries; "
                f"have {list(gdf.columns)}"
            )
        label_values = gdf[label].to_numpy()

    data = data.sortby([lat, lon])
    lat_values = data[lat].values
    lon_values = data[lon].values

    labels = _labels_for(gdf, lat_values, lon_values, features, from_bounds, all_touched)
    label_da = xr.DataArray(
        labels, dims=(lat, lon), coords={lat: lat_values, lon: lon_values},
        name="_zone",
    )

    weight_da = _weights_for(weights, lat, lat_values, data)
    stacked = data.stack(_cell=(lat, lon))
    zone = label_da.stack(_cell=(lat, lon))
    stacked = stacked.where(zone >= 0, drop=True)
    zone = zone.where(zone >= 0, drop=True)

    if stacked.sizes.get("_cell", 0) == 0:
        raise ValueError(
            "no grid cell falls inside any geometry. The grid and the geometries "
            "may not overlap, or the regions may be smaller than one cell — try "
            "all_touched=True."
        )

    grouped = stacked.groupby(zone.rename("_zone"))
    if stat == "count":
        result = grouped.count()
    elif stat in _WEIGHTED_STATS and weight_da is not None:
        cell_weights = weight_da.broadcast_like(label_da).stack(_cell=(lat, lon))
        cell_weights = cell_weights.sel(_cell=stacked._cell)
        result = _weighted_group(stacked, cell_weights, zone, stat)
    else:
        result = getattr(grouped, stat)()

    result = result.rename({"_zone": dim})
    # groupby yields the sorted set of labels present; a region with no cells is
    # absent. Reindex so the output always mirrors the input geometries. The
    # reindex is done against the positional index (groupby's own labels), then
    # the requested names are assigned — so a `by` value that never won a cell
    # still appears, rather than being silently dropped.
    result = result.reindex({dim: np.arange(len(gdf))})
    if stat == "count":
        result = result.fillna(0)
    result = result.assign_coords({dim: names})
    if label_values is not None:
        result = result.assign_coords({f"{dim}_label": (dim, label_values)})
    return result


def _weighted_group(stacked, cell_weights, zone, stat):
    """Weighted mean/sum per zone, with NaN cells contributing nothing."""
    valid = stacked.notnull()
    weighted = (stacked.fillna(0) * cell_weights).groupby(zone.rename("_zone")).sum()
    if stat == "sum":
        return weighted
    denominator = (cell_weights * valid).groupby(zone.rename("_zone")).sum()
    return weighted / denominator.where(denominator > 0)
