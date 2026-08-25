"""Zonal aggregation over many geometries at once.

`fetch(region=...)` dissolves a shapefile into one mask. Reporting needs the
other reduction: one number per district. These tests pin the geometry-to-cell
assignment (the part that is easy to get subtly wrong, and silently) and the
weighting, then check that the result drops into deepscale's completion engine
without a bespoke code path.

No network: synthetic grids and shapely boxes.
"""
import numpy as np
import pytest
import xarray as xr

pytest.importorskip("geopandas")
pytest.importorskip("rasterio")

import geopandas as gpd  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402

from acmaddl.zonal import zonal  # noqa: E402


@pytest.fixture
def grid():
    """A 4x4 grid of 1-degree cells centred on 0.5..3.5, valued by column."""
    lat = np.array([0.5, 1.5, 2.5, 3.5])
    lon = np.array([0.5, 1.5, 2.5, 3.5])
    values = np.tile(np.arange(4.0), (4, 1))  # value == lon index
    return xr.DataArray(values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


@pytest.fixture
def halves():
    """Two 2x4 boxes: 'west' covers lon 0-2, 'east' covers lon 2-4."""
    return gpd.GeoDataFrame(
        {"name": ["west", "east"]},
        geometry=[box(0, 0, 2, 4), box(2, 0, 4, 4)],
        crs="EPSG:4326",
    )


# --- the core reduction ----------------------------------------------------


def test_zonal_returns_one_element_per_geometry(grid, halves):
    got = zonal(grid, halves, by="name")
    assert got.dims == ("region",)
    assert list(got.region.values) == ["west", "east"]


def test_zonal_assigns_each_cell_to_the_geometry_containing_its_centre(grid, halves):
    """West holds lon centres 0.5 and 1.5 (values 0, 1); east holds 2.5 and 3.5
    (values 2, 3). Getting the pixel-edge transform wrong shifts this by a cell."""
    got = zonal(grid, halves, by="name", weights=None)
    assert float(got.sel(region="west")) == pytest.approx(0.5)
    assert float(got.sel(region="east")) == pytest.approx(2.5)


def test_zonal_does_not_dissolve_the_features(grid, halves):
    """The difference from fetch(region=...): two features stay two regions."""
    got = zonal(grid, halves, by="name")
    assert got.sizes["region"] == 2
    assert float(got.sel(region="west")) != float(got.sel(region="east"))


def test_zonal_preserves_every_other_dim(halves):
    lat = lon = np.array([0.5, 1.5, 2.5, 3.5])
    da = xr.DataArray(
        np.ones((5, 4, 4)), dims=("time", "lat", "lon"),
        coords={"time": np.arange(5), "lat": lat, "lon": lon},
    )
    got = zonal(da, halves, by="name")
    assert got.dims == ("time", "region") or got.dims == ("region", "time")
    assert got.sizes["time"] == 5


def test_zonal_works_on_a_dataset(grid, halves):
    ds = grid.to_dataset(name="precip")
    got = zonal(ds, halves, by="name")
    assert "precip" in got and got.sizes["region"] == 2


# --- statistics ------------------------------------------------------------


@pytest.mark.parametrize(
    "stat, expected_west",
    [("sum", 1.0), ("min", 0.0), ("max", 1.0), ("median", 0.5), ("count", 8.0)],
)
def test_zonal_supports_the_named_statistics(grid, halves, stat, expected_west):
    got = zonal(grid, halves, by="name", stat=stat, weights=None)
    if stat == "sum":
        # 8 cells: four at 0, four at 1.
        assert float(got.sel(region="west")) == pytest.approx(4.0)
    else:
        assert float(got.sel(region="west")) == pytest.approx(expected_west)


def test_zonal_rejects_an_unknown_stat(grid, halves):
    with pytest.raises(ValueError, match="stat must be one of"):
        zonal(grid, halves, by="name", stat="mode")


# --- weighting -------------------------------------------------------------


def test_area_weighting_is_the_default_and_downweights_poleward_cells():
    """An unweighted mean over a tall region over-counts its poleward cells,
    whose true area is smaller by cos(lat)."""
    lat = np.array([5.0, 55.0])
    lon = np.array([0.5, 1.5])
    values = np.array([[0.0, 0.0], [10.0, 10.0]])  # the poleward row is the hot one
    da = xr.DataArray(values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    tall = gpd.GeoDataFrame({"name": ["all"]}, geometry=[box(0, 0, 2, 60)],
                            crs="EPSG:4326")

    unweighted = float(zonal(da, tall, by="name", weights=None).sel(region="all"))
    weighted = float(zonal(da, tall, by="name").sel(region="all"))  # "area" default

    assert unweighted == pytest.approx(5.0)
    w = np.cos(np.deg2rad(lat))
    assert weighted == pytest.approx(10.0 * w[1] / (w[0] + w[1]))
    assert weighted < unweighted


def test_cos_lat_is_an_alias_for_area(grid, halves):
    a = zonal(grid, halves, by="name", weights="area")
    b = zonal(grid, halves, by="name", weights="cos_lat")
    np.testing.assert_allclose(a.values, b.values)


def test_weights_are_ignored_by_the_order_statistics(grid, halves):
    weighted = zonal(grid, halves, by="name", stat="max", weights="area")
    unweighted = zonal(grid, halves, by="name", stat="max", weights=None)
    np.testing.assert_allclose(weighted.values, unweighted.values)


def test_an_explicit_weight_array_is_honoured(grid, halves):
    """Weight the lon=0.5 column ten times as heavily as the rest."""
    w = xr.DataArray([10.0, 1.0, 1.0, 1.0], dims="lon", coords={"lon": grid.lon})
    got = zonal(grid, halves, by="name", weights=w)
    # West holds values 0 (weight 10) and 1 (weight 1) -> 1/11.
    assert float(got.sel(region="west")) == pytest.approx(1.0 / 11.0)


def test_unknown_weights_are_rejected(grid, halves):
    with pytest.raises(ValueError, match="weights must be"):
        zonal(grid, halves, by="name", weights="equal")


# --- missing data ----------------------------------------------------------


def test_nan_cells_are_excluded_from_the_reduction(grid, halves):
    """A district that is half ocean is averaged over its land."""
    holed = grid.copy()
    holed[0, 0] = np.nan  # drop one of west's four zero-valued cells
    got = zonal(holed, halves, by="name", weights=None)
    assert float(got.sel(region="west")) == pytest.approx(4.0 / 7.0)


def test_a_region_with_no_valid_cells_yields_nan_not_a_dropped_row(grid):
    """The output must always mirror the input geometries, or the caller's
    region list silently stops lining up with the results."""
    regions = gpd.GeoDataFrame(
        {"name": ["real", "empty"]},
        geometry=[box(0, 0, 2, 4), box(100, 0, 102, 4)],
        crs="EPSG:4326",
    )
    got = zonal(grid, regions, by="name")
    assert list(got.region.values) == ["real", "empty"]
    assert np.isfinite(float(got.sel(region="real")))
    assert np.isnan(float(got.sel(region="empty")))


def test_count_of_an_empty_region_is_zero(grid):
    regions = gpd.GeoDataFrame(
        {"name": ["real", "empty"]},
        geometry=[box(0, 0, 2, 4), box(100, 0, 102, 4)],
        crs="EPSG:4326",
    )
    got = zonal(grid, regions, by="name", stat="count")
    assert float(got.sel(region="empty")) == 0.0


def test_no_overlap_at_all_raises(grid):
    far = gpd.GeoDataFrame({"name": ["x"]}, geometry=[box(100, 0, 102, 4)],
                           crs="EPSG:4326")
    with pytest.raises(ValueError, match="no grid cell falls inside"):
        zonal(grid, far, by="name")


def test_all_touched_rescues_a_region_smaller_than_one_cell(grid):
    """A sub-cell district contains no cell centre. Centre-in gives it nothing;
    all_touched gives it the cell it sits in."""
    tiny = gpd.GeoDataFrame(
        {"name": ["dot"]}, geometry=[Point(2.6, 2.6).buffer(0.05)], crs="EPSG:4326"
    )
    with pytest.raises(ValueError, match="no grid cell falls inside"):
        zonal(grid, tiny, by="name")
    got = zonal(grid, tiny, by="name", all_touched=True)
    assert float(got.sel(region="dot")) == pytest.approx(2.0)


# --- geometry inputs -------------------------------------------------------


def test_zonal_accepts_a_shapefile_path(grid, halves, tmp_path):
    path = tmp_path / "halves.shp"
    halves.to_file(path)
    got = zonal(grid, str(path), by="name")
    assert sorted(got.region.values.tolist()) == ["east", "west"]


def test_zonal_accepts_a_geoseries_and_a_bare_geometry_list(grid, halves):
    from_series = zonal(grid, halves.geometry)
    from_list = zonal(grid, list(halves.geometry))
    np.testing.assert_allclose(from_series.values, from_list.values)
    assert list(from_series.region.values) == [0, 1]


def test_zonal_reprojects_geometries_to_epsg_4326(grid, halves):
    projected = halves.to_crs("EPSG:3857")
    got = zonal(grid, projected, by="name")
    reference = zonal(grid, halves, by="name")
    np.testing.assert_allclose(got.values, reference.values, rtol=1e-6)


def test_zonal_rejects_an_unsupported_geometry_type(grid):
    with pytest.raises(TypeError, match="Unsupported geometries type"):
        zonal(grid, 42)


def test_zonal_rejects_empty_geometries(grid):
    empty = gpd.GeoDataFrame({"name": []}, geometry=[], crs="EPSG:4326")
    with pytest.raises(ValueError, match="geometries is empty"):
        zonal(grid, empty, by="name")


# --- labelling -------------------------------------------------------------


def test_by_defaults_to_the_positional_index(grid, halves):
    got = zonal(grid, halves)
    assert list(got.region.values) == [0, 1]


def test_duplicate_labels_are_rejected(grid):
    dupes = gpd.GeoDataFrame(
        {"name": ["same", "same"]},
        geometry=[box(0, 0, 2, 4), box(2, 0, 4, 4)], crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="duplicate values"):
        zonal(grid, dupes, by="name")


def test_a_missing_by_column_is_rejected(grid, halves):
    with pytest.raises(ValueError, match="not found in geometries"):
        zonal(grid, halves, by="ADM3_EN")


def test_the_output_dim_can_be_renamed(grid, halves):
    got = zonal(grid, halves, by="name", dim="woreda")
    assert "woreda" in got.dims


def test_a_grid_without_lat_lon_dims_is_rejected(halves):
    da = xr.DataArray(np.ones((2, 2)), dims=("y", "x"))
    with pytest.raises(ValueError, match="must have 'lat' and 'lon' dims"):
        zonal(da, halves, by="name")


def test_a_degenerate_grid_is_rejected(halves):
    da = xr.DataArray([[1.0]], dims=("lat", "lon"),
                      coords={"lat": [0.5], "lon": [0.5]})
    with pytest.raises(ValueError, match="at least 2x2"):
        zonal(da, halves, by="name")


def test_descending_latitude_input_gives_the_same_answer(grid, halves):
    flipped = grid.isel(lat=slice(None, None, -1))
    np.testing.assert_allclose(
        zonal(flipped, halves, by="name").values,
        zonal(grid, halves, by="name").values,
    )


# --- integration with the completion engine --------------------------------


def test_zonal_output_feeds_the_completion_engine_unchanged(halves):
    """The payoff: `complete()` reduces along time and touches nothing else, so a
    (time, region) array from zonal needs no admin-unit code path."""
    deepscale = pytest.importorskip("deepscale")
    import pandas as pd

    stamps = pd.DatetimeIndex(
        [pd.Timestamp(y, m, d) for y in range(1991, 2027)
         for m in range(1, 13) for d in (1, 11, 21)]
    )
    lat = lon = np.array([0.5, 1.5, 2.5, 3.5])
    values = np.broadcast_to(
        (stamps.year.to_numpy() - 1990)[:, None, None], (len(stamps), 4, 4)
    ).astype(float)
    da = xr.DataArray(values, dims=("time", "lat", "lon"),
                      coords={"time": stamps, "lat": lat, "lon": lon})

    per_region = zonal(da, halves, by="name").transpose("time", "region")
    assert per_region.dims == ("time", "region")

    clim = deepscale.seasonal_stack(per_region, "JJAS", years=range(1991, 2026))
    obs = per_region.sel(time=slice("2026-06-01", "2026-07-31"))
    result = deepscale.complete(
        obs, deepscale.analogs_from_years([2000, 2010]),
        climatology=clim, season="JJAS",
    )
    assert result.totals.dims == ("scenario", "region")
    assert list(result.percentile.region.values) == ["west", "east"]


# --- unique index + repeating display label --------------------------------
# Real admin datasets (GeoBoundaries, GADM, OCHA) carry a unique code column and
# a non-unique name column: two districts can share a name across parent zones.


def _coded_regions():
    import geopandas as gpd
    from shapely.geometry import box
    return gpd.GeoDataFrame(
        {"code": ["A1", "A2"], "name": ["Gursum", "Gursum"]},  # same name, twice
        geometry=[box(0, 0, 2, 4), box(2, 0, 4, 4)], crs="EPSG:4326",
    )


def test_zonal_indexes_by_a_unique_code_and_carries_a_repeating_label(grid):
    got = zonal(grid, _coded_regions(), by="code", label="name")
    assert list(got.region.values) == ["A1", "A2"]
    assert list(got.region_label.values) == ["Gursum", "Gursum"]
    # the label rides along and stays selectable by the unique index
    assert str(got.sel(region="A1").region_label.values) == "Gursum"


def test_zonal_duplicate_index_error_points_at_the_code_plus_label_pattern(grid):
    with pytest.raises(ValueError, match="unique code column"):
        zonal(grid, _coded_regions(), by="name")


def test_zonal_label_column_must_exist(grid, halves):
    with pytest.raises(ValueError, match="label column 'nope' not found"):
        zonal(grid, halves, by="name", label="nope")


def test_zonal_label_is_optional(grid, halves):
    got = zonal(grid, halves, by="name")
    assert "region_label" not in got.coords
