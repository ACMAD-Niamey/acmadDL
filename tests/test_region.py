"""Tests for shapefile / geometry region input (acmaddl-plan §5, issue #12).

Three input forms must work for ``acmaddl.fetch(region=...)``:
    * a bbox ``[lat_s, lat_n, lon_w, lon_e]`` (passes through unchanged);
    * a path to a ``.shp`` shapefile (bbox + polygon extracted);
    * a shapely geometry / geopandas GeoSeries (bbox + polygon extracted).

``resolve_region`` returns ``(bbox, geometry)``; the bbox is used for upstream
slicing and ``normalize.clip_to_geometry`` applies the final polygon mask.

No network required — synthetic shapefiles are written into ``tmp_path``.
"""

import sys

import numpy as np
import pytest
import xarray as xr

from acmaddl.region import resolve_region
from acmaddl.normalize import clip_to_geometry, normalize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_shapefile(path, geom, crs="EPSG:4326"):
    """Write a single-geometry shapefile and return its path as a string."""
    import geopandas as gpd

    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs=crs)
    gdf.to_file(path)
    return str(path)


def _fake_env(monkeypatch, raw):
    """Patch fetch's adapter + catalog to serve `raw`; return (module, seen)."""
    import importlib

    fetch_mod = importlib.import_module("acmaddl.fetch")
    seen = {}

    class FakeAdapter:
        def fetch_data(self, config, variable, date_range=None, region=None):
            seen["region"] = region
            return raw

    monkeypatch.setattr(fetch_mod, "get_adapter", lambda name: FakeAdapter())
    monkeypatch.setattr(fetch_mod.catalog, "get", lambda product: {
        "adapter": "fake",
        "variables": {"precip": {"native_name": "prate",
                                 "units": "mm/day", "target_units": "mm/day"}},
    })
    return fetch_mod, seen


def _ones_grid(lat, lon):
    data = np.ones((len(lat), len(lon)), dtype=np.float32)
    return xr.Dataset({"prate": (["latitude", "longitude"], data)},
                      coords={"latitude": lat, "longitude": lon})


@pytest.fixture
def square_shp(tmp_path):
    """A square covering lon 30..40, lat -5..5 in EPSG:4326."""
    from shapely.geometry import box

    return _write_shapefile(tmp_path / "square.shp", box(30, -5, 40, 5))


# ---------------------------------------------------------------------------
# resolve_region — bbox passthrough
# ---------------------------------------------------------------------------

class TestResolveBbox:
    def test_bbox_list_passthrough(self):
        bbox, geom = resolve_region([-20, 20, 20, 55])
        assert bbox == [-20, 20, 20, 55]
        assert geom is None

    def test_bbox_tuple_passthrough(self):
        bbox, geom = resolve_region((-2, 2, 36, 40))
        assert bbox == [-2, 2, 36, 40]
        assert geom is None

    def test_bbox_numpy_floats(self):
        bbox, geom = resolve_region(np.array([-2.0, 2.0, 36.0, 40.0]))
        assert bbox == [-2.0, 2.0, 36.0, 40.0]
        assert geom is None

    def test_none_returns_none(self):
        assert resolve_region(None) == (None, None)


# ---------------------------------------------------------------------------
# resolve_region — shapefile path
# ---------------------------------------------------------------------------

class TestResolveShapefile:
    def test_extracts_bbox_and_geometry(self, square_shp):
        bbox, geom = resolve_region(square_shp)
        lat_s, lat_n, lon_w, lon_e = bbox
        assert (lat_s, lat_n, lon_w, lon_e) == pytest.approx((-5, 5, 30, 40))
        assert geom is not None
        assert geom.bounds == pytest.approx((30, -5, 40, 5))

    def test_reprojects_to_epsg4326(self, tmp_path):
        # A square defined in Web Mercator (metres) must come back as lat/lon
        # degrees, not metres.
        from shapely.geometry import box

        # ~ lon 30..31, lat 0..1 expressed in EPSG:3857 metres.
        merc = box(3_339_584, 0, 3_450_853, 111_325)
        shp = _write_shapefile(tmp_path / "merc.shp", merc, crs="EPSG:3857")
        bbox, geom = resolve_region(shp)
        lat_s, lat_n, lon_w, lon_e = bbox
        assert lon_w == pytest.approx(30, abs=0.5)
        assert lon_e == pytest.approx(31, abs=0.5)
        assert lat_s == pytest.approx(0, abs=0.5)
        assert lat_n == pytest.approx(1, abs=0.5)

    def test_multipolygon_dissolved_bbox(self, tmp_path):
        # Two disjoint squares -> bbox spans both, geometry is their union.
        from shapely.geometry import box
        import geopandas as gpd

        gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
            crs="EPSG:4326",
        )
        shp = str(tmp_path / "multi.shp")
        gdf.to_file(shp)

        bbox, geom = resolve_region(shp)
        lat_s, lat_n, lon_w, lon_e = bbox
        assert (lon_w, lat_s, lon_e, lat_n) == pytest.approx((0, 0, 11, 11))

    def test_non_shp_string_raises(self):
        with pytest.raises(ValueError, match=r"\.shp"):
            resolve_region("/tmp/not_a_shapefile.txt")

    def test_missing_geopandas_raises_clear_error(self, monkeypatch):
        # Simulate geopandas not being installed: importing it fails.
        monkeypatch.setitem(sys.modules, "geopandas", None)
        with pytest.raises(ImportError, match=r"acmadDL\[geo\]"):
            resolve_region("region.shp")


# ---------------------------------------------------------------------------
# resolve_region — shapely / geopandas geometry objects
# ---------------------------------------------------------------------------

class TestResolveGeometry:
    def test_shapely_geometry_input(self):
        from shapely.geometry import box

        bbox, geom = resolve_region(box(30, -5, 40, 5))
        assert bbox == pytest.approx([-5, 5, 30, 40])
        assert geom.bounds == pytest.approx((30, -5, 40, 5))

    def test_geoseries_input(self):
        from shapely.geometry import box
        import geopandas as gpd

        gs = gpd.GeoSeries([box(0, 0, 1, 1), box(2, 2, 3, 3)], crs="EPSG:4326")
        bbox, geom = resolve_region(gs)
        lat_s, lat_n, lon_w, lon_e = bbox
        assert (lon_w, lat_s, lon_e, lat_n) == pytest.approx((0, 0, 3, 3))


# ---------------------------------------------------------------------------
# clip_to_geometry — final polygon mask
# ---------------------------------------------------------------------------

def _grid(lat=np.arange(-5, 6, 1.0), lon=np.arange(30, 41, 1.0)):
    data = np.ones((len(lat), len(lon)), dtype=np.float32)
    ds = xr.Dataset({"precip": (["lat", "lon"], data)},
                    coords={"lat": lat, "lon": lon})
    ds["precip"].attrs["units"] = "mm/day"
    return ds


class TestClipToGeometry:
    def test_outside_polygon_is_nan(self):
        from shapely.geometry import Polygon

        ds = _grid()  # lat -5..5, lon 30..40
        # Triangle so the corners of its bbox fall outside the polygon.
        tri = Polygon([(31, -4), (39, -4), (35, 4)])
        clipped = clip_to_geometry(ds, tri)

        # An interior cell keeps its data...
        assert float(clipped["precip"].sel(lat=-3, lon=35, method="nearest")) == pytest.approx(1.0)
        # ...while a cell inside the bbox but outside the triangle is NaN.
        assert bool(np.isnan(clipped["precip"].sel(lat=3, lon=31, method="nearest")))

    def test_cropped_to_polygon_bbox(self):
        from shapely.geometry import box

        ds = _grid()  # 11x11 grid, integer centres
        # Edges at .5 so cell membership is unambiguous (centres 33..37 inside).
        clipped = clip_to_geometry(ds, box(32.5, -2.5, 37.5, 2.5))
        # drop=True crops to the polygon's bbox (smaller than the input grid)...
        assert clipped.sizes["lon"] < ds.sizes["lon"]
        assert clipped.sizes["lat"] < ds.sizes["lat"]
        # ...covering exactly the enclosed cells, with no holes.
        assert float(clipped.lon.min()) == 33.0 and float(clipped.lon.max()) == 37.0
        assert not bool(clipped["precip"].isnull().any())

    def test_units_preserved(self):
        from shapely.geometry import box

        ds = _grid()
        clipped = clip_to_geometry(ds, box(33, -2, 37, 2))
        assert clipped["precip"].attrs.get("units") == "mm/day"

    def test_center_based_default_masks_grazed_cell(self):
        # Default (all_touched=False) is center-based: the cell centred at 36 is
        # only grazed by the polygon (which reaches 35.6), so its centre is
        # outside and it's excluded. Matches xarray/CDO/rasterio defaults.
        from shapely.geometry import box

        ds = _grid()  # lat -5..5, lon 30..40, integer centres
        clipped = clip_to_geometry(ds, box(31, -3, 35.6, 3))
        assert float(clipped["precip"].lon.max()) == 35.0

    def test_all_touched_keeps_grazed_cell(self):
        # all_touched=True keeps any cell the polygon touches: cell 36 (spans
        # 35.5–36.5) is grazed by the polygon edge at 35.6, so it's retained.
        from shapely.geometry import box

        ds = _grid()
        clipped = clip_to_geometry(ds, box(31, -3, 35.6, 3), all_touched=True)
        assert float(clipped["precip"].lon.max()) == 36.0


# ---------------------------------------------------------------------------
# Integration: fetch() -> normalize() -> clip wiring with a real shapefile.
# Uses a triangle so that bbox-cropping alone cannot explain the masking:
# cells inside the bbox but outside the triangle must be NaN.
# ---------------------------------------------------------------------------

class TestFetchWiring:
    def test_normalize_applies_geometry_clip(self):
        from shapely.geometry import box

        # native-named raw dataset, as an adapter would return it
        lat = np.arange(-5, 6, 1.0)
        lon = np.arange(30, 41, 1.0)
        data = np.ones((len(lat), len(lon)), dtype=np.float32)
        raw = xr.Dataset({"prate": (["latitude", "longitude"], data)},
                         coords={"latitude": lat, "longitude": lon})
        cfg = {"variables": {"precip": {"native_name": "prate",
                                        "units": "mm/day", "target_units": "mm/day"}}}

        bbox, geom = resolve_region(box(33, -2, 37, 2))
        clean = normalize(raw, cfg, "precip", region=bbox, geometry=geom)

        # bbox slice cropped to lon 33..37, lat -2..2
        assert float(clean.lon.min()) >= 33
        assert float(clean.lon.max()) <= 37
        # within the bbox the rectangle fully covers it, so nothing is masked
        assert not bool(clean["precip"].isnull().any())

    def test_fetch_threads_shapefile_through_to_clip(self, tmp_path, monkeypatch):
        import importlib
        from shapely.geometry import Polygon

        # `acmaddl.fetch` the attribute is the function (re-exported in
        # __init__), so reach the module object explicitly to monkeypatch it.
        fetch_mod = importlib.import_module("acmaddl.fetch")

        # Triangle inside lon 30..40 / lat -5..5: corners of the bbox lie
        # outside it, so a correct geometry mask NaNs them.
        triangle = Polygon([(30, -5), (40, -5), (35, 5)])
        shp = _write_shapefile(tmp_path / "tri.shp", triangle)

        lat = np.arange(-5, 6, 1.0)
        lon = np.arange(30, 41, 1.0)
        data = np.ones((len(lat), len(lon)), dtype=np.float32)
        raw = xr.Dataset({"prate": (["latitude", "longitude"], data)},
                         coords={"latitude": lat, "longitude": lon})

        class FakeAdapter:
            def fetch_data(self, config, variable, date_range=None, region=None):
                # adapter receives the resolved bbox, never the shapefile path
                assert region is None or len(region) == 4
                return raw

        monkeypatch.setattr(fetch_mod, "get_adapter", lambda name: FakeAdapter())
        monkeypatch.setattr(fetch_mod.catalog, "get", lambda product: {
            "adapter": "fake",
            "variables": {"precip": {"native_name": "prate",
                                     "units": "mm/day", "target_units": "mm/day"}},
        })

        ds = fetch_mod.fetch("fake/product", "precip", region=shp,
                             cache=False, verbose=False, progress=False)

        # apex region (near lat 5, lon 35) retained; bottom-left corner masked
        apex = ds["precip"].sel(lat=4, lon=35, method="nearest")
        corner = ds["precip"].sel(lat=5, lon=30, method="nearest")
        assert not bool(np.isnan(apex))
        assert bool(np.isnan(corner))

    def test_shapefile_edges_fully_covered(self, tmp_path, monkeypatch):
        # Regression: a polygon whose bounds fall BETWEEN coarse grid-cell
        # centres must still be 100% covered. A tight (centre-based) bbox slice
        # drops the boundary cells, leaving the country's tips empty.
        import importlib
        from shapely.geometry import box
        fetch_mod = importlib.import_module("acmaddl.fetch")

        # Coarse 1° grid, integer-degree centres.
        lat = np.arange(0, 17, 1.0)
        lon = np.arange(30, 51, 1.0)
        data = np.ones((len(lat), len(lon)), dtype=np.float32)
        raw = xr.Dataset({"prate": (["latitude", "longitude"], data)},
                         coords={"latitude": lat, "longitude": lon})

        # Ethiopia-bbox-like: east 47.96 sits in the cell centred at 48, which
        # a tight slice on the polygon bounds would drop.
        poly = box(33.0, 3.4, 47.96, 14.85)
        shp = _write_shapefile(tmp_path / "poly.shp", poly)

        seen = {}

        class FakeAdapter:
            def fetch_data(self, config, variable, date_range=None, region=None):
                seen["region"] = region
                return raw

        monkeypatch.setattr(fetch_mod, "get_adapter", lambda name: FakeAdapter())
        monkeypatch.setattr(fetch_mod.catalog, "get", lambda product: {
            "adapter": "fake",
            "variables": {"precip": {"native_name": "prate",
                                     "units": "mm/day", "target_units": "mm/day"}},
        })

        ds = fetch_mod.fetch("fake/product", "precip", region=shp,
                             boundary="cover", cache=False,
                             verbose=False, progress=False)
        f = ds["precip"]
        res = 1.0

        # In cover mode the adapter is asked for a bbox padded beyond the bounds.
        assert seen["region"][3] > 47.96 and seen["region"][2] < 33.0

        # The returned grid's cell edges must enclose the whole polygon — no
        # missing eastern tip / northern / western / southern slivers.
        assert float(f.lon.max()) + res / 2 >= 47.96, "east tip uncovered"
        assert float(f.lon.min()) - res / 2 <= 33.0, "west edge uncovered"
        assert float(f.lat.max()) + res / 2 >= 14.85, "north edge uncovered"
        assert float(f.lat.min()) - res / 2 <= 3.40, "south edge uncovered"

    def test_bbox_cover_covers_full_requested_box(self, monkeypatch):
        # A bbox whose bounds fall between coarse cell centres: cover mode must
        # extend the grid so its cell-edges enclose the full requested box.
        lat = np.arange(0, 17, 1.0)
        lon = np.arange(30, 51, 1.0)
        fetch_mod, _ = _fake_env(monkeypatch, _ones_grid(lat, lon))
        req = [3.4, 14.85, 33.0, 47.96]

        ds = fetch_mod.fetch("fake/product", "precip", region=req,
                             boundary="cover", cache=False,
                             verbose=False, progress=False)
        f = ds["precip"]
        res = 1.0
        assert float(f.lon.max()) + res / 2 >= 47.96 and float(f.lon.min()) - res / 2 <= 33.0
        assert float(f.lat.max()) + res / 2 >= 14.85 and float(f.lat.min()) - res / 2 <= 3.40

    def test_bbox_center_is_clamped_by_default(self, monkeypatch):
        # Default (center) is the ecosystem-standard: keep only cells whose
        # centre falls inside the requested box.
        lat = np.arange(0, 17, 1.0)
        lon = np.arange(30, 51, 1.0)
        fetch_mod, _ = _fake_env(monkeypatch, _ones_grid(lat, lon))
        req = [3.4, 14.85, 33.0, 47.96]

        ds = fetch_mod.fetch("fake/product", "precip", region=req,
                             cache=False, verbose=False, progress=False)
        f = ds["precip"]
        assert float(f.lon.min()) == 33.0 and float(f.lon.max()) == 47.0
        assert float(f.lat.min()) == 4.0 and float(f.lat.max()) == 14.0

    def test_region_on_non_griddable_product_raises_clearly(self, monkeypatch):
        # A non-gridded source (e.g. station/tabular data) can't honour a bbox
        # or shapefile. Requesting a region must fail loudly, not silently
        # return unsubset data.
        raw = xr.Dataset({"prate": (["time"], np.ones(4, dtype=np.float32))},
                         coords={"time": [0, 1, 2, 3]})
        fetch_mod, _ = _fake_env(monkeypatch, raw)
        with pytest.raises(ValueError, match="(?i)region|spatial|grid"):
            fetch_mod.fetch("fake/product", "precip", region=[-2, 2, 36, 40],
                            cache=False, verbose=False, progress=False)

    def test_non_griddable_product_without_region_is_fine(self, monkeypatch):
        # No region requested -> no spatial expectation -> no error.
        raw = xr.Dataset({"prate": (["time"], np.ones(4, dtype=np.float32))},
                         coords={"time": [0, 1, 2, 3]})
        fetch_mod, _ = _fake_env(monkeypatch, raw)
        ds = fetch_mod.fetch("fake/product", "precip",
                             cache=False, verbose=False, progress=False)
        assert "precip" in ds
