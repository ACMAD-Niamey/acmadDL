"""Region resolution for ``acmaddl.fetch(region=...)``.

A region may be supplied three ways:

    * a bbox ``[lat_s, lat_n, lon_w, lon_e]`` (list/tuple/array of 4 numbers);
    * a path to a ``.shp`` shapefile;
    * a shapely geometry or geopandas ``GeoSeries`` / ``GeoDataFrame``.

``resolve_region`` collapses all three into a ``(bbox, geometry)`` pair:

    bbox      ``[lat_s, lat_n, lon_w, lon_e]`` — used for upstream slicing by
              adapters and the bbox crop in :mod:`acmaddl.normalize`. ``None``
              only when ``region`` itself is ``None``.
    geometry  a shapely geometry in EPSG:4326 for the final polygon mask in
              :func:`acmaddl.normalize.clip_to_geometry`, or ``None`` for plain
              bbox input (nothing to mask beyond the bbox).

geopandas is an optional dependency (the ``geo`` extra) and is imported lazily,
so bbox-only callers never need it installed.
"""

from __future__ import annotations

from numbers import Real
from pathlib import Path

_GEO_HINT = (
    "Shapefile and geometry region inputs require geopandas. "
    "Install the geo extra:  pip install 'acmadDL[geo]'"
)


def _require_geopandas():
    try:
        import geopandas as gpd
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_GEO_HINT) from e
    return gpd


def _is_bbox(region) -> bool:
    """True if ``region`` is a sequence of exactly four real numbers."""
    if isinstance(region, (str, bytes, Path)):
        return False
    try:
        seq = list(region)
    except TypeError:
        return False
    return len(seq) == 4 and all(
        isinstance(v, Real) and not isinstance(v, bool) for v in seq
    )


def _bbox_from_geom(geom) -> list:
    """shapely ``.bounds`` (minx, miny, maxx, maxy) -> [lat_s, lat_n, lon_w, lon_e].

    NOTE: geometries crossing the ±180° antimeridian are not special-cased —
    ``.bounds`` returns lon_w≈-180, lon_e≈+180, so the bbox spans the whole
    globe in longitude. Split such polygons at the antimeridian upstream.
    """
    lon_w, lat_s, lon_e, lat_n = geom.bounds
    return [lat_s, lat_n, lon_w, lon_e]


def _resolve_gdf(gdf):
    """Reproject to EPSG:4326, dissolve to one geometry, return (bbox, geom)."""
    if gdf.crs is not None:
        gdf = gdf.to_crs("EPSG:4326")
    # union_all() dissolves multi-feature / disjoint geometries so the bbox is
    # correct for e.g. a multi-island country (avoids a bbox spanning the gaps
    # of unrelated features). geopandas >= 0.14; replaces deprecated unary_union.
    geom = gdf.union_all()
    return _bbox_from_geom(geom), geom


def resolve_region(region):
    """Resolve ``region`` to ``(bbox, geometry)``. See module docstring."""
    if region is None:
        return None, None

    if _is_bbox(region):
        return [float(v) for v in region], None

    if isinstance(region, (str, Path)):
        if not str(region).endswith(".shp"):
            raise ValueError(
                f"Unsupported region string {region!r}: "
                "expected a .shp path or a [lat_s, lat_n, lon_w, lon_e] bbox."
            )
        gpd = _require_geopandas()
        return _resolve_gdf(gpd.read_file(region))

    # Bare shapely geometry — no CRS attached, assume EPSG:4326.
    try:
        from shapely.geometry.base import BaseGeometry
    except ImportError:
        BaseGeometry = ()
    if isinstance(region, BaseGeometry):
        return _bbox_from_geom(region), region

    # geopandas GeoSeries / GeoDataFrame — reproject + dissolve.
    gpd = _require_geopandas()
    if isinstance(region, gpd.GeoSeries):
        region = gpd.GeoDataFrame(geometry=region)
    if isinstance(region, gpd.GeoDataFrame):
        return _resolve_gdf(region)

    raise TypeError(
        f"Unsupported region type: {type(region).__name__}. "
        "Use a bbox, a .shp path, or a shapely/geopandas geometry."
    )
