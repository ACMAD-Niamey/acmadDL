import numpy as np
import pandas as pd
import pytest
import tempfile
import warnings
import xarray as xr

# ---------------------------------------------------------------------------
# 1. Catalog tests
# ---------------------------------------------------------------------------

def test_catalog_loads():
    from acmaddl import catalog
    products = catalog.list_products()
    assert len(products) > 0
    for p in products:
        cfg = catalog.info(p)
        assert "adapter" in cfg
        assert "variables" in cfg
        assert "grid" in cfg


def test_catalog_list_products():
    from acmaddl import catalog
    products = catalog.list_products()
    assert isinstance(products, list)
    assert all(isinstance(p, str) for p in products)
    assert "nmme/cfsv2" in products
    assert "obs/chirps-v3-monthly" in products


def test_catalog_info():
    from acmaddl import catalog
    info = catalog.info("nmme/cfsv2")
    assert "adapter" in info
    assert "variables" in info
    assert info["adapter"] == "opendap"


def test_catalog_info_missing():
    from acmaddl import catalog
    with pytest.raises(KeyError):
        catalog.info("nonexistent/product")


def test_catalog_variable_mapping():
    from acmaddl import catalog
    for product_name in catalog.list_products():
        cfg = catalog.info(product_name)
        for var_name, var_cfg in cfg["variables"].items():
            assert "native_name" in var_cfg
            assert "units" in var_cfg
            assert "target_units" in var_cfg


# ---------------------------------------------------------------------------
# 2. Normalize tests
# ---------------------------------------------------------------------------

def test_normalize_renames_variables(synthetic_precip_ds):
    from acmaddl.normalize import normalize
    config = {
        "variables": {
            "precip": {"native_name": "prate", "units": "kg m-2 s-1", "target_units": "mm/day"}
        }
    }
    result = normalize(synthetic_precip_ds, config, "precip")
    assert "precip" in result
    assert "prate" not in result


def test_normalize_standardizes_coords(synthetic_global_ds):
    from acmaddl.normalize import normalize
    config = {
        "variables": {
            "precip": {"native_name": "prate", "units": "kg m-2 s-1", "target_units": "mm/day"}
        }
    }
    result = normalize(synthetic_global_ds, config, "precip")
    assert "lat" in result.dims
    assert "lon" in result.dims
    assert "latitude" not in result.dims
    assert "longitude" not in result.dims


def test_normalize_converts_units_k_to_c(synthetic_temp_ds):
    from acmaddl.normalize import normalize
    config = {
        "variables": {
            "temp": {"native_name": "2m_temperature", "units": "K", "target_units": "C"}
        }
    }
    result = normalize(synthetic_temp_ds, config, "temp")
    np.testing.assert_allclose(result["temp"].values.mean(), 26.85, atol=0.01)


def test_normalize_converts_units_precip(synthetic_precip_ds):
    from acmaddl.normalize import normalize
    config = {
        "variables": {
            "precip": {"native_name": "prate", "units": "kg m-2 s-1", "target_units": "mm/day"}
        }
    }
    result = normalize(synthetic_precip_ds, config, "precip")
    np.testing.assert_allclose(result["precip"].values.mean(), 86400.0, atol=1.0)


def test_normalize_no_change_same_units():
    from acmaddl.normalize import normalize
    lat = np.arange(-5, 5, 1.0)
    lon = np.arange(30, 40, 1.0)
    ds = xr.Dataset(
        {"precip": (["lat", "lon"], np.full((10, 10), 5.0))},
        coords={"lat": lat, "lon": lon},
    )
    config = {
        "variables": {
            "precip": {"native_name": "precip", "units": "mm/day", "target_units": "mm/day"}
        }
    }
    result = normalize(ds, config, "precip")
    np.testing.assert_allclose(result["precip"].values.mean(), 5.0)


def test_normalize_masks_fill_value():
    """Catalog `fill_value` converts sentinel values to NaN after unit
    conversion. Mirrors the chirps_raw_live case where -9999 marks ocean
    cells and we don't want it polluting downstream means."""
    from acmaddl.normalize import normalize
    lat = np.arange(-5, 5, 1.0)
    lon = np.arange(30, 40, 1.0)
    vals = np.full((10, 10), 5.0)
    vals[0, 0] = -9999.0     # sentinel
    vals[1, 1] = -9999.0     # sentinel
    ds = xr.Dataset(
        {"precip": (["lat", "lon"], vals)},
        coords={"lat": lat, "lon": lon},
    )
    config = {
        "variables": {
            "precip": {
                "native_name": "precip",
                "units": "mm/day",
                "target_units": "mm/day",
                "fill_value": -9999,
            }
        }
    }
    result = normalize(ds, config, "precip")
    # Sentinels become NaN; real values pass through unchanged.
    assert np.isnan(result["precip"].values[0, 0])
    assert np.isnan(result["precip"].values[1, 1])
    np.testing.assert_allclose(result["precip"].values[2, 2], 5.0)
    # Nanmean skips sentinels, matches the real-value mean.
    np.testing.assert_allclose(np.nanmean(result["precip"].values), 5.0)


def test_normalize_no_fill_value_is_no_op():
    """Variables without `fill_value` in the catalog leave data untouched."""
    from acmaddl.normalize import normalize
    lat = np.arange(-5, 5, 1.0)
    lon = np.arange(30, 40, 1.0)
    vals = np.full((10, 10), 5.0)
    vals[0, 0] = -9999.0
    ds = xr.Dataset(
        {"precip": (["lat", "lon"], vals)},
        coords={"lat": lat, "lon": lon},
    )
    config = {
        "variables": {
            "precip": {"native_name": "precip", "units": "mm/day", "target_units": "mm/day"}
        }
    }
    result = normalize(ds, config, "precip")
    # -9999 remains because no fill_value was declared
    assert result["precip"].values[0, 0] == -9999.0


def test_normalize_subsets_region(synthetic_global_ds):
    from acmaddl.normalize import normalize
    config = {
        "variables": {
            "precip": {"native_name": "prate", "units": "kg m-2 s-1", "target_units": "mm/day"}
        }
    }
    region = [-12, 6, 28, 42]
    result = normalize(synthetic_global_ds, config, "precip", region=region)
    assert float(result.lat.min()) >= -12
    assert float(result.lat.max()) <= 6
    assert float(result.lon.min()) >= 28
    assert float(result.lon.max()) <= 42


def test_normalize_cf_compliance(synthetic_precip_ds):
    from acmaddl.normalize import normalize
    config = {
        "variables": {
            "precip": {"native_name": "prate", "units": "kg m-2 s-1", "target_units": "mm/day"}
        }
    }
    result = normalize(synthetic_precip_ds, config, "precip")
    assert result["precip"].attrs["units"] == "mm/day"
    assert result["lat"].attrs["axis"] == "Y"
    assert result["lon"].attrs["axis"] == "X"


def test_year_index_reshapes_init_time_to_year(synthetic_nmme_ds):
    from acmaddl.normalize import normalize
    config = {"variables": {"precip": {"native_name": "prec", "units": "mm/day",
                                        "target_units": "mm/day"}}}
    out = normalize(synthetic_nmme_ds, config, "precip", year_index=True)
    assert "year" in out.dims and "init_time" not in out.dims
    assert out["year"].dtype.kind == "i"
    assert list(out["year"].values) == [2010, 2011, 2012]


def test_year_index_default_keeps_init_time(synthetic_nmme_ds):
    from acmaddl.normalize import normalize
    config = {"variables": {"precip": {"native_name": "prec", "units": "mm/day",
                                        "target_units": "mm/day"}}}
    out = normalize(synthetic_nmme_ds, config, "precip")
    assert "init_time" in out.dims and "year" not in out.dims


def test_year_index_preserves_units_attr(synthetic_nmme_ds):
    """The year_index path averages over lead_time; that reduction must keep the
    units attribute set by normalize (xarray's mean drops attrs by default), so a
    year-indexed dataset carries the same units as the default init_time path."""
    from acmaddl.normalize import normalize
    config = {"variables": {"precip": {"native_name": "prec", "units": "mm/day",
                                        "target_units": "mm/day"}}}
    out = normalize(synthetic_nmme_ds, config, "precip", year_index=True)
    assert out["precip"].attrs.get("units") == "mm/day"


# ---------------------------------------------------------------------------
# 3. Storage tests
# ---------------------------------------------------------------------------

def test_save_netcdf_local():
    from acmaddl.storage import save
    ds = xr.Dataset({"temp": (["lat", "lon"], np.random.rand(5, 5))})
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        path = f.name
    save(ds, path, format="netcdf")
    loaded = xr.open_dataset(path)
    xr.testing.assert_equal(ds, loaded)
    loaded.close()


def test_save_unsupported_format_raises():
    from acmaddl.storage import save
    ds = xr.Dataset({"temp": (["lat", "lon"], np.random.rand(5, 5))})
    with pytest.raises(ValueError):
        save(ds, "out.xyz", format="xyz")


# ---------------------------------------------------------------------------
# 4. Adapter base tests
# ---------------------------------------------------------------------------

def test_adapter_base_is_abstract():
    from acmaddl.adapters.base import AdapterBase
    with pytest.raises(TypeError):
        AdapterBase()


def test_get_adapter_returns_correct_class():
    from acmaddl.adapters import get_adapter
    from acmaddl.adapters.cds import CDSAdapter
    from acmaddl.adapters.opendap import OPeNDAPAdapter
    from acmaddl.adapters.http import HTTPAdapter
    assert isinstance(get_adapter("cds"), CDSAdapter)
    assert isinstance(get_adapter("opendap"), OPeNDAPAdapter)
    assert isinstance(get_adapter("http"), HTTPAdapter)


def test_get_adapter_unknown_raises():
    from acmaddl.adapters import get_adapter
    with pytest.raises(KeyError):
        get_adapter("unknown")


def test_get_adapter_sheerwater():
    from acmaddl.adapters import get_adapter
    from acmaddl.adapters.sheerwater import SheerwaterAdapter
    adapter = get_adapter("sheerwater")
    assert isinstance(adapter, SheerwaterAdapter)


# ---------------------------------------------------------------------------
# 5. Date/season parsing tests
# ---------------------------------------------------------------------------

def test_parse_target_season():
    from acmaddl.fetch import parse_target
    from datetime import datetime
    start, end = parse_target("MAM", year=2025)
    assert start == datetime(2025, 3, 1)
    assert end == datetime(2025, 5, 31)


def test_parse_target_ond():
    from acmaddl.fetch import parse_target
    from datetime import datetime
    start, end = parse_target("OND", year=2025)
    assert start == datetime(2025, 10, 1)
    assert end == datetime(2025, 12, 31)


def test_parse_target_passthrough():
    from acmaddl.fetch import parse_target
    from datetime import datetime
    pair = (datetime(2025, 3, 1), datetime(2025, 5, 31))
    assert parse_target(pair) == pair


def test_parse_init():
    from acmaddl.fetch import parse_init
    from datetime import datetime
    assert parse_init("2025-02") == datetime(2025, 2, 1)
    dt = datetime(2025, 2, 1)
    assert parse_init(dt) == dt


# ---------------------------------------------------------------------------
# 6. Adapter health checks
# ---------------------------------------------------------------------------

def test_check_product_config_health():
    import acmaddl
    status = acmaddl.check_product("nmme/cfsv2")
    assert status["product"] == "nmme/cfsv2"
    assert status["adapter"] == "opendap"
    assert status["healthy"] is True
    assert status["kind"] == "config"
    assert "checked_at" in status


def test_check_all_products_returns_all():
    import acmaddl
    statuses = acmaddl.check_all_products()
    products = set(acmaddl.catalog.list_products())
    assert len(statuses) == len(products)
    assert {s["product"] for s in statuses} == products
    assert all("healthy" in s for s in statuses)


# ---------------------------------------------------------------------------
# 7. Time coordinate normalization
# ---------------------------------------------------------------------------

def test_decode_months_since(synthetic_nmme_ds):
    """Numeric 'months since 1960-01-01' coords are decoded to datetime64."""
    from acmaddl.normalize import _decode_numeric_times
    result = _decode_numeric_times(synthetic_nmme_ds)
    assert np.issubdtype(result["S"].dtype, np.datetime64)
    dates = pd.DatetimeIndex(result["S"].values)
    assert list(dates.year) == [2010, 2011, 2012]
    assert list(dates.month) == [2, 2, 2]


def test_decode_days_since():
    """Numeric 'days since' coords are decoded to datetime64."""
    from acmaddl.normalize import _decode_numeric_times
    ds = xr.Dataset(
        {"x": (["time"], [1.0, 2.0])},
        coords={"time": [0.0, 365.0]},
    )
    ds["time"].attrs["units"] = "days since 2000-01-01"
    result = _decode_numeric_times(ds)
    assert np.issubdtype(result["time"].dtype, np.datetime64)
    dates = pd.DatetimeIndex(result["time"].values)
    assert dates[0] == pd.Timestamp("2000-01-01")
    assert dates[1] == pd.Timestamp("2000-12-31")


def test_decode_skips_non_numeric_coords():
    """Coords that are already datetime64 are left unchanged."""
    from acmaddl.normalize import _decode_numeric_times
    times = pd.to_datetime(["2020-01-01", "2020-02-01"])
    ds = xr.Dataset(
        {"x": (["time"], [1.0, 2.0])},
        coords={"time": times},
    )
    result = _decode_numeric_times(ds)
    assert np.issubdtype(result["time"].dtype, np.datetime64)
    result_times = pd.DatetimeIndex(result["time"].values)
    assert list(result_times.year) == [2020, 2020]
    assert list(result_times.month) == [1, 2]
    assert list(result_times.day) == [1, 1]


def test_decode_skips_coords_without_since():
    """Coords with plain units (no 'since') are left as-is."""
    from acmaddl.normalize import _decode_numeric_times
    ds = xr.Dataset(
        {"x": (["L"], [0.5, 1.5])},
        coords={"L": [0.5, 1.5]},
    )
    ds["L"].attrs["units"] = "months"
    result = _decode_numeric_times(ds)
    np.testing.assert_array_equal(result["L"].values, [0.5, 1.5])


def test_normalize_nmme_renames_dims(synthetic_nmme_ds):
    """NMME dims S/L/M/Y/X are renamed to init_time/lead_time/member/lat/lon."""
    from acmaddl.normalize import normalize
    config = {
        "variables": {
            "precip": {"native_name": "prec", "units": "mm/day",
                       "target_units": "mm/day"}
        }
    }
    result = normalize(synthetic_nmme_ds, config, "precip")
    assert "init_time" in result.dims
    assert "lead_time" in result.dims
    assert "member" in result.dims
    assert "lat" in result.dims
    assert "lon" in result.dims
    for old in ("S", "L", "M", "Y", "X"):
        assert old not in result.dims


def test_normalize_nmme_decodes_init_time(synthetic_nmme_ds):
    """NMME init_time coordinate is decoded to datetime64 during normalize."""
    from acmaddl.normalize import normalize
    config = {
        "variables": {
            "precip": {"native_name": "prec", "units": "mm/day",
                       "target_units": "mm/day"}
        }
    }
    result = normalize(synthetic_nmme_ds, config, "precip")
    assert np.issubdtype(result["init_time"].dtype, np.datetime64)
    dates = pd.DatetimeIndex(result["init_time"].values)
    assert list(dates.month) == [2, 2, 2]


def test_normalize_cds_forecast_renames_dims(synthetic_cds_forecast_ds):
    """CDS forecast dims are renamed to init_time/lead_time/member/lat/lon."""
    from acmaddl.normalize import normalize
    config = {
        "variables": {
            "temp": {"native_name": "t2m", "units": "K", "target_units": "C"}
        }
    }
    result = normalize(synthetic_cds_forecast_ds, config, "temp")
    assert "init_time" in result.dims
    assert "lead_time" in result.dims
    assert "member" in result.dims
    assert "lat" in result.dims
    assert "lon" in result.dims
    for old in ("forecast_reference_time", "forecastMonth", "number"):
        assert old not in result.dims


def test_normalize_cds_forecast_preserves_datetime(synthetic_cds_forecast_ds):
    """CDS init_time stays as datetime64 (already decoded by xarray)."""
    from acmaddl.normalize import normalize
    config = {
        "variables": {
            "temp": {"native_name": "t2m", "units": "K", "target_units": "C"}
        }
    }
    result = normalize(synthetic_cds_forecast_ds, config, "temp")
    assert np.issubdtype(result["init_time"].dtype, np.datetime64)


def test_normalize_obs_time_unchanged(synthetic_obs_monthly_ds):
    """Obs datasets keep their time dim as 'time' with datetime64 values."""
    from acmaddl.normalize import normalize
    config = {
        "variables": {
            "precip": {"native_name": "precip", "units": "mm/day",
                       "target_units": "mm/day"}
        }
    }
    result = normalize(synthetic_obs_monthly_ds, config, "precip")
    assert "time" in result.dims
    assert np.issubdtype(result["time"].dtype, np.datetime64)
    assert result["time"].attrs["axis"] == "T"


def test_normalize_init_time_cf_axis(synthetic_nmme_ds):
    """init_time gets CF axis='T' attribute."""
    from acmaddl.normalize import normalize
    config = {
        "variables": {
            "precip": {"native_name": "prec", "units": "mm/day",
                       "target_units": "mm/day"}
        }
    }
    result = normalize(synthetic_nmme_ds, config, "precip")
    assert result["init_time"].attrs["axis"] == "T"


# ---------------------------------------------------------------------------
# 8. OPeNDAP S-coordinate filtering (regression for year-mismatch fix)
# ---------------------------------------------------------------------------

def _make_nmme_remote_ds():
    """Synthetic NMME-like dataset as returned by xr.open_dataset (undecoded).

    S values (months since 1960-01-01):
      601 = Feb 2010, 602 = Mar 2010,
      613 = Feb 2011, 614 = Mar 2011,
      625 = Feb 2012, 626 = Mar 2012
    """
    s_vals = np.array([601.0, 602.0, 613.0, 614.0, 625.0, 626.0])
    lat = np.arange(-2, 3, 1.0)
    lon = np.arange(36, 41, 1.0)
    data = np.random.rand(len(s_vals), len(lat), len(lon)).astype(np.float32)
    ds = xr.Dataset(
        {"prec": (["S", "Y", "X"], data)},
        coords={"S": s_vals, "Y": lat, "X": lon},
    )
    ds["S"].attrs["units"] = "months since 1960-01-01"
    return ds


def _nmme_product_config(**overrides):
    config = {
        "adapter": "opendap",
        "source_url": "http://fake",
        "variables": {
            "precip": {"native_name": "prec", "units": "mm/day",
                       "target_units": "mm/day"}
        },
        "_verbose": False,
    }
    config.update(overrides)
    return config


def test_opendap_filters_s_by_date_range(monkeypatch):
    """S coordinate is filtered to only years within date_range."""
    from acmaddl.adapters.opendap import OPeNDAPAdapter
    fake_ds = _make_nmme_remote_ds()
    monkeypatch.setattr(xr, "open_dataset", lambda *a, **kw: fake_ds.copy(deep=True))

    adapter = OPeNDAPAdapter()
    result = adapter.fetch_data(
        _nmme_product_config(), "precip", date_range=(2010, 2011),
    )
    # 601=Feb2010, 602=Mar2010, 613=Feb2011, 614=Mar2011 -> 4 values kept
    assert len(result.S) == 4
    assert float(result.S.min()) == 601.0
    assert float(result.S.max()) == 614.0


def test_opendap_filters_s_by_init_months(monkeypatch):
    """S coordinate is filtered to only matching init months."""
    from acmaddl.adapters.opendap import OPeNDAPAdapter
    fake_ds = _make_nmme_remote_ds()
    monkeypatch.setattr(xr, "open_dataset", lambda *a, **kw: fake_ds.copy(deep=True))

    adapter = OPeNDAPAdapter()
    config = _nmme_product_config(init_months=[2])
    result = adapter.fetch_data(config, "precip")
    # Only Feb entries: 601, 613, 625
    assert len(result.S) == 3
    kept = result.S.values.tolist()
    assert kept == [601.0, 613.0, 625.0]


def test_opendap_filters_s_by_date_range_and_init_months(monkeypatch):
    """Both date_range and init_months are applied together."""
    from acmaddl.adapters.opendap import OPeNDAPAdapter
    fake_ds = _make_nmme_remote_ds()
    monkeypatch.setattr(xr, "open_dataset", lambda *a, **kw: fake_ds.copy(deep=True))

    adapter = OPeNDAPAdapter()
    config = _nmme_product_config(init_months=[3])
    result = adapter.fetch_data(config, "precip", date_range=(2011, 2012))
    # Mar 2011 (614) and Mar 2012 (626)
    assert len(result.S) == 2
    kept = result.S.values.tolist()
    assert kept == [614.0, 626.0]


def test_opendap_no_filter_keeps_all_s(monkeypatch):
    """Without date_range or init_months, all S values are retained."""
    from acmaddl.adapters.opendap import OPeNDAPAdapter
    fake_ds = _make_nmme_remote_ds()
    monkeypatch.setattr(xr, "open_dataset", lambda *a, **kw: fake_ds.copy(deep=True))

    adapter = OPeNDAPAdapter()
    result = adapter.fetch_data(_nmme_product_config(), "precip")
    assert len(result.S) == 6


def test_opendap_filters_lead_months(monkeypatch):
    """target_lead_months selects and averages the correct L values."""
    from acmaddl.adapters.opendap import OPeNDAPAdapter
    l_vals = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    lat = np.arange(-2, 3, 1.0)
    lon = np.arange(36, 41, 1.0)
    data = np.random.rand(len(l_vals), len(lat), len(lon)).astype(np.float32)
    ds = xr.Dataset(
        {"prec": (["L", "Y", "X"], data)},
        coords={"L": l_vals, "Y": lat, "X": lon},
    )
    monkeypatch.setattr(xr, "open_dataset", lambda *a, **kw: ds.copy(deep=True))

    adapter = OPeNDAPAdapter()
    # target_lead_months=[1,2,3] → L=[0.5, 1.5, 2.5]
    config = _nmme_product_config(target_lead_months=[1, 2, 3])
    result = adapter.fetch_data(config, "precip")
    assert "L" not in result.dims  # averaged out
    expected = ds["prec"].sel(L=[0.5, 1.5, 2.5]).mean("L")
    np.testing.assert_allclose(result["prec"].values, expected.values)


def test_s3_filters_lead_months(monkeypatch, tmp_path):
    """S3 adapter filters and averages L dimension using target_lead_months."""
    from acmaddl.adapters.s3 import S3Adapter

    # Build a synthetic S3-style file (single init, with S/L/M/Y/X)
    s_val = 601.0  # Feb 2010
    l_vals = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    m_vals = np.arange(4)
    lat = np.arange(-2, 3, 1.0)
    lon = np.arange(36, 41, 1.0)
    shape = (len(l_vals), len(m_vals), len(lat), len(lon))
    data = np.random.rand(*shape).astype(np.float32)
    ds = xr.Dataset(
        {"precip": (["L", "M", "Y", "X"], data)},
        coords={"S": s_val, "L": l_vals, "M": m_vals, "Y": lat, "X": lon},
    )
    ds["S"].attrs["units"] = "months since 1960-01-01"
    nc_path = str(tmp_path / "test.nc")
    ds.to_netcdf(nc_path)

    # Mock S3 to serve our local file
    monkeypatch.setattr(
        "acmaddl.adapters.s3._s3_exists", lambda p: True,
    )
    monkeypatch.setattr(
        "acmaddl.adapters.s3._s3_download",
        lambda s3_path, local: __import__("shutil").copy(nc_path, local),
    )

    config = {
        "adapter": "s3",
        "s3_bucket": "fake",
        "s3_prefix": "fake",
        "file_template": "test_{year}-{month}.nc",
        "variables": {
            "precip": {"native_name": "precip", "units": "mm/day",
                       "target_units": "mm/day"}
        },
        "init_months": [2],
        "target_lead_months": [1, 2, 3],
        "_verbose": False,
        "_progress": False,
    }

    adapter = S3Adapter()
    result = adapter.fetch_data(config, "precip", date_range=(2010, 2010))

    # L should be averaged out
    assert "L" not in result.dims
    assert "init_time" in result.dims
    # Check the mean was computed over the correct leads
    expected = ds["precip"].sel(L=[0.5, 1.5, 2.5]).mean("L")
    np.testing.assert_allclose(
        result["precip"].squeeze("init_time").values, expected.values,
    )


def test_s3_no_lead_filter_keeps_all(monkeypatch, tmp_path):
    """Without target_lead_months, S3 adapter returns all leads."""
    from acmaddl.adapters.s3 import S3Adapter

    s_val = 601.0
    l_vals = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    m_vals = np.arange(4)
    lat = np.arange(-2, 3, 1.0)
    lon = np.arange(36, 41, 1.0)
    shape = (len(l_vals), len(m_vals), len(lat), len(lon))
    data = np.random.rand(*shape).astype(np.float32)
    ds = xr.Dataset(
        {"precip": (["L", "M", "Y", "X"], data)},
        coords={"S": s_val, "L": l_vals, "M": m_vals, "Y": lat, "X": lon},
    )
    ds["S"].attrs["units"] = "months since 1960-01-01"
    nc_path = str(tmp_path / "test.nc")
    ds.to_netcdf(nc_path)

    monkeypatch.setattr("acmaddl.adapters.s3._s3_exists", lambda p: True)
    monkeypatch.setattr(
        "acmaddl.adapters.s3._s3_download",
        lambda s3_path, local: __import__("shutil").copy(nc_path, local),
    )

    config = {
        "adapter": "s3",
        "s3_bucket": "fake",
        "s3_prefix": "fake",
        "file_template": "test_{year}-{month}.nc",
        "variables": {
            "precip": {"native_name": "precip", "units": "mm/day",
                       "target_units": "mm/day"}
        },
        "init_months": [2],
        "_verbose": False,
        "_progress": False,
    }

    adapter = S3Adapter()
    result = adapter.fetch_data(config, "precip", date_range=(2010, 2010))
    assert "L" in result.dims
    assert len(result.L) == 6


def test_opendap_year_filter_without_s_coord(monkeypatch):
    """Datasets with a 'year' coord (no S) still use the original year filter path."""
    from acmaddl.adapters.opendap import OPeNDAPAdapter
    years = np.array([2008, 2009, 2010, 2011, 2012])
    lat = np.arange(-2, 3, 1.0)
    lon = np.arange(36, 41, 1.0)
    data = np.random.rand(len(years), len(lat), len(lon)).astype(np.float32)
    ds = xr.Dataset(
        {"prec": (["year", "Y", "X"], data)},
        coords={"year": years, "Y": lat, "X": lon},
    )
    monkeypatch.setattr(xr, "open_dataset", lambda *a, **kw: ds.copy(deep=True))

    adapter = OPeNDAPAdapter()
    result = adapter.fetch_data(
        _nmme_product_config(), "precip", date_range=(2009, 2011),
    )
    assert list(result.year.values) == [2009, 2010, 2011]


# ---------------------------------------------------------------------------
# 9. Deprecation tests
# ---------------------------------------------------------------------------

def test_catalog_deprecated_entries():
    from acmaddl import catalog
    deprecated = [p for p in catalog.list_products() if catalog.info(p).get("deprecated")]
    assert len(deprecated) >= 5


def test_catalog_info_deprecated():
    # nmme/cfsv2 was de-deprecated (R5); use c3s/cmcc which retains deprecated_after+successor
    # c3s/cmcc is deprecated, so a DeprecationWarning is expected here
    from acmaddl import catalog
    info = catalog.info("c3s/cmcc")
    assert info["deprecated"] is True
    assert "deprecated_after" in info
    assert "successor" in info


def test_catalog_info_not_deprecated():
    from acmaddl import catalog
    info = catalog.info("c3s/ecmwf")
    assert info.get("deprecated", False) is False


def test_health_check_deprecated_emits_warning():
    # nmme/cfsv2 was de-deprecated (R5); c3s/cmcc is still deprecated and serves as the test subject
    from acmaddl import health
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = health.check_product("c3s/cmcc")
    assert any("deprecated" in str(warning.message).lower() for warning in w), \
        f"Expected deprecation warning, got: {[str(x.message) for x in w]}"


def test_health_check_non_deprecated_no_warning():
    from acmaddl import health
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        health.check_product("c3s/ecmwf")
    deprecation_warnings = [x for x in w if "deprecated" in str(x.message).lower()]
    assert len(deprecation_warnings) == 0


def test_cfsv2_use_emits_no_deprecation_warning():
    """nmme/cfsv2 was de-deprecated in R5 (append_streams + notes replace the
    sunset warning). Resolving it must NOT emit a DeprecationWarning."""
    import warnings
    from acmaddl import catalog
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        catalog.info("nmme/cfsv2")
    deps = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert not deps, f"nmme/cfsv2 is no longer deprecated; expected no DeprecationWarning but got: {[str(x.message) for x in deps]}"


# ---------------------------------------------------------------------------
# 10. Placeholder entries
# ---------------------------------------------------------------------------

def test_c3s_entries_have_sst():
    from acmaddl import catalog
    c3s_products = [p for p in catalog.list_products()
                    if p.startswith("c3s/") and not catalog.info(p).get("deprecated")
                    and not catalog.info(p).get("pending_url")]
    assert len(c3s_products) > 0, "No non-deprecated C3S products found"
    for product in c3s_products:
        cfg = catalog.info(product)
        assert "sst" in cfg["variables"], f"{product} is missing sst variable block"
        sst = cfg["variables"]["sst"]
        assert sst["native_name"] == "sea_surface_temperature"
        assert sst["units"] == "K"
        assert sst["target_units"] == "K"

def test_ncei_entries_have_sst():
    # These CCSR-served products renamed sst -> tos in the server's July 2026
    # CMIP renaming (and switched to Celsius); the delivered contract stays K
    # via normalize's (C, K) conversion.
    from acmaddl import catalog
    for product in ["nmme/ccsm4", "nmme/geoss2s"]:
        cfg = catalog.info(product)
        assert "sst" in cfg["variables"], f"{product} is missing sst variable block"
        sst = cfg["variables"]["sst"]
        assert sst["native_name"] == "tos"
        assert sst["units"] == "C"
        assert sst["target_units"] == "K"

def test_normalize_sst_preserves_nan():
    import numpy as np
    import xarray as xr
    from acmaddl.normalize import normalize
    data = np.array([[np.nan, 300.0], [301.0, np.nan]], dtype=np.float32)
    ds = xr.Dataset(
        {"sst": (["latitude", "longitude"], data)},
        coords={"latitude": [0.0, 1.0], "longitude": [30.0, 31.0]},
    )
    config = {"variables": {"sst": {"native_name": "sst", "units": "K", "target_units": "K"}}}
    result = normalize(ds, config, "sst")
    assert "sst" in result
    assert np.isnan(result["sst"].values[0, 0])
    assert np.isnan(result["sst"].values[1, 1])
    assert not np.isnan(result["sst"].values[0, 1])


def test_pycpt_reference_coverage():
    """Every PyCPT reference GCM maps to a resolvable acmadDL catalog entry with the correct variable."""
    from acmaddl import catalog
    from tests.conftest import PYCPT_REFERENCE_GCMS

    missing_products = []
    missing_variables = []

    for pycpt_name, (product, variable) in PYCPT_REFERENCE_GCMS.items():
        try:
            cfg = catalog.info(product)
        except KeyError:
            missing_products.append(f"{pycpt_name} -> {product} (not in catalog)")
            continue
        if variable not in cfg["variables"]:
            missing_variables.append(
                f"{pycpt_name} -> {product}.{variable} (variable not defined)"
            )

    assert not missing_products, "Missing catalog entries:\n" + "\n".join(missing_products)
    assert not missing_variables, "Missing variable definitions:\n" + "\n".join(missing_variables)


def test_all_non_pending_entries_pass_config_health_check():
    """Every catalog entry that isn't pending_url should pass a config-level health check."""
    import warnings
    from acmaddl import health, catalog

    failures = []
    for product in catalog.list_products():
        cfg = catalog.info(product)
        if cfg.get("pending_url") or cfg.get("deprecated"):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = health.check_product(product, probe_remote=False)
        if not result["healthy"]:
            failures.append(f"{product}: {result['message']}")

    assert not failures, "Config health check failures:\n" + "\n".join(failures)


def _http_product_config(**overrides):
    config = {
        "adapter": "http",
        "source_url": "https://fake/",
        "file_pattern": "fake-{year}.{month:02d}.cog",
        "format": "cog",
        "variables": {
            "precip": {"native_name": "precip", "units": "mm/month",
                       "target_units": "mm/day"},
        },
        "_verbose": False,
        "_progress": False,
    }
    config.update(overrides)
    return config


def _fake_cog_ds(year, month):
    ts = pd.Timestamp(f"{year}-{month:02d}-01")
    da = xr.DataArray(
        np.ones((2, 2), dtype="float32"),
        dims=("latitude", "longitude"),
        coords={"latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
    )
    return da.to_dataset(name="precip").expand_dims(time=[ts])


def test_http_adapter_strict_raises_on_any_failure(monkeypatch):
    """Default (allow_partial=False) must abort if any per-file open fails."""
    import re
    from acmaddl.adapters import http as http_mod

    def fake_open(url, region, variable=None, fill_value=None, timestamp=None):
        m = re.search(r"(\d{4})\.(\d{2})", url)
        year, month = int(m.group(1)), int(m.group(2))
        # Pretend the second month of the second year is poisoned.
        if year == 2011 and month == 2:
            raise RuntimeError("not recognized as being in a supported file format.")
        return _fake_cog_ds(year, month)

    monkeypatch.setattr(http_mod, "_open_cog_subset", fake_open)

    adapter = http_mod.HTTPAdapter()
    with pytest.raises(RuntimeError, match="1/24 file"):
        adapter.fetch_data(_http_product_config(), "precip", date_range=(2010, 2011))


def test_http_adapter_allow_partial_returns_partial(monkeypatch):
    """Opt-in best-effort path concatenates whatever succeeded."""
    import re
    from acmaddl.adapters import http as http_mod

    def fake_open(url, region, variable=None, fill_value=None, timestamp=None):
        m = re.search(r"(\d{4})\.(\d{2})", url)
        year, month = int(m.group(1)), int(m.group(2))
        if year == 2011 and month == 2:
            raise RuntimeError("synthetic failure")
        return _fake_cog_ds(year, month)

    monkeypatch.setattr(http_mod, "_open_cog_subset", fake_open)

    adapter = http_mod.HTTPAdapter()
    config = _http_product_config(_allow_partial=True)
    result = adapter.fetch_data(config, "precip", date_range=(2010, 2011))
    # 24 requested, 1 failed → 23 time steps survive.
    assert result.sizes["time"] == 23


def test_http_adapter_strict_succeeds_when_all_files_load(monkeypatch):
    """Strict mode passes through cleanly when nothing fails."""
    import re
    from acmaddl.adapters import http as http_mod

    def fake_open(url, region, variable=None, fill_value=None, timestamp=None):
        m = re.search(r"(\d{4})\.(\d{2})", url)
        return _fake_cog_ds(int(m.group(1)), int(m.group(2)))

    monkeypatch.setattr(http_mod, "_open_cog_subset", fake_open)

    adapter = http_mod.HTTPAdapter()
    result = adapter.fetch_data(_http_product_config(), "precip", date_range=(2010, 2010))
    assert result.sizes["time"] == 12


def test_http_adapter_retries_recover_from_transient_failure(monkeypatch):
    """Per-file retries should absorb transient errors."""
    import re
    from acmaddl.adapters import http as http_mod

    # Each (year, month) fails twice, then succeeds. With max_retries=3 we
    # should still end up with all 12 months.
    attempts = {}
    def flaky_open(url, region, variable=None, fill_value=None, timestamp=None):
        m = re.search(r"(\d{4})\.(\d{2})", url)
        key = (int(m.group(1)), int(m.group(2)))
        attempts[key] = attempts.get(key, 0) + 1
        if attempts[key] <= 2:
            raise RuntimeError("transient: not a supported file format.")
        return _fake_cog_ds(*key)

    monkeypatch.setattr(http_mod, "_open_cog_subset", flaky_open)
    # Speed the test up — no real sleeps.
    monkeypatch.setattr(http_mod.time, "sleep", lambda _s: None)

    adapter = http_mod.HTTPAdapter()
    config = _http_product_config(_max_retries=3, _retry_backoff=0.0)
    result = adapter.fetch_data(config, "precip", date_range=(2010, 2010))
    assert result.sizes["time"] == 12
    # Each file: 2 failures + 1 success = 3 attempts.
    assert all(v == 3 for v in attempts.values())


def test_http_adapter_retries_exhaust_then_raise(monkeypatch):
    """After the retry budget is spent, strict mode raises for the run."""
    from acmaddl.adapters import http as http_mod

    def always_fail(url, region, variable=None, fill_value=None, timestamp=None):
        raise RuntimeError("persistent: not a supported file format.")

    monkeypatch.setattr(http_mod, "_open_cog_subset", always_fail)
    monkeypatch.setattr(http_mod.time, "sleep", lambda _s: None)

    adapter = http_mod.HTTPAdapter()
    config = _http_product_config(_max_retries=2, _retry_backoff=0.0)
    with pytest.raises(RuntimeError, match="12/12 file"):
        adapter.fetch_data(config, "precip", date_range=(2010, 2010))


def test_http_adapter_does_not_retry_http_4xx(monkeypatch):
    """4xx errors are permanent — file doesn't exist, retries can't help."""
    from acmaddl.adapters import http as http_mod

    attempts = {"n": 0}
    def four_oh_four(url, region, variable=None, fill_value=None, timestamp=None):
        attempts["n"] += 1
        raise RuntimeError("HTTP response code: 404")

    monkeypatch.setattr(http_mod, "_open_cog_subset", four_oh_four)
    monkeypatch.setattr(http_mod.time, "sleep", lambda _s: None)

    adapter = http_mod.HTTPAdapter()
    # max_retries=5: if we wrongly retried 4xx, we'd see ~6 attempts per file
    # × 12 files = 72. Correct behaviour: one attempt per file, no retries.
    config = _http_product_config(_max_retries=5, _retry_backoff=0.0)
    with pytest.raises(RuntimeError, match="12/12 file"):
        adapter.fetch_data(config, "precip", date_range=(2010, 2010))
    assert attempts["n"] == 12  # one attempt per file, no retries


def test_http_adapter_rate_limiter_enforces_minimum_interval(monkeypatch):
    """request_interval gates successive opens at the configured rate."""
    import re
    from acmaddl.adapters import http as http_mod

    def fake_open(url, region, variable=None, fill_value=None, timestamp=None):
        m = re.search(r"(\d{4})\.(\d{2})", url)
        return _fake_cog_ds(int(m.group(1)), int(m.group(2)))

    monkeypatch.setattr(http_mod, "_open_cog_subset", fake_open)

    # Stub time.monotonic + time.sleep so we can observe sleep durations
    # without making the test wallclock-slow.
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(http_mod.time, "monotonic", lambda: clock[0])
    def fake_sleep(s):
        sleeps.append(s)
        clock[0] += s
    monkeypatch.setattr(http_mod.time, "sleep", fake_sleep)

    adapter = http_mod.HTTPAdapter()
    config = _http_product_config(_request_interval=0.25)
    adapter.fetch_data(config, "precip", date_range=(2010, 2010))
    # First call: no wait (last=0, elapsed=0 < 0.25 would still sleep 0.25 since
    # _last starts at 0; first iteration also paces). 12 files → at least 11
    # interval waits ≈ 0.25s each.
    paced = [s for s in sleeps if s >= 0.2]
    assert len(paced) >= 11


def _stub_clock(monkeypatch):
    """Replace time.monotonic/sleep so the rate limiter advances a virtual clock
    instead of sleeping for real. Returns the recorded sleep-duration list."""
    from acmaddl.adapters import http as http_mod
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(http_mod.time, "monotonic", lambda: clock[0])
    def fake_sleep(s):
        sleeps.append(s)
        clock[0] += s
    monkeypatch.setattr(http_mod.time, "sleep", fake_sleep)
    return sleeps


def test_http_adapter_catalog_request_interval_paces_without_caller_value(monkeypatch):
    """A request_interval declared in the catalog entry must pace opens even when
    the caller passes no request_interval — that's how the CHIRPS entries stay
    under the UCSB CrowdSec rate ban out of the box."""
    import re
    from acmaddl.adapters import http as http_mod

    monkeypatch.setattr(http_mod, "_open_cog_subset",
                        lambda url, region, variable=None, fill_value=None, timestamp=None:
                        _fake_cog_ds(*map(int, re.search(r"(\d{4})\.(\d{2})", url).groups())))
    sleeps = _stub_clock(monkeypatch)

    adapter = http_mod.HTTPAdapter()
    # Catalog-style key (no leading underscore), and NO caller _request_interval.
    config = _http_product_config(request_interval=0.25)
    adapter.fetch_data(config, "precip", date_range=(2010, 2010))
    paced = [s for s in sleeps if s >= 0.2]
    assert len(paced) >= 11


def test_http_adapter_catalog_interval_is_a_floor_not_a_default(monkeypatch):
    """The catalog request_interval is a floor: a caller may raise it, but a lower
    caller value must not silently undercut the product's safe minimum."""
    import re
    from acmaddl.adapters import http as http_mod

    monkeypatch.setattr(http_mod, "_open_cog_subset",
                        lambda url, region, variable=None, fill_value=None, timestamp=None:
                        _fake_cog_ds(*map(int, re.search(r"(\d{4})\.(\d{2})", url).groups())))
    sleeps = _stub_clock(monkeypatch)

    adapter = http_mod.HTTPAdapter()
    # Catalog floor 0.5 vs a lower caller value 0.25 → floor wins (0.5).
    config = _http_product_config(request_interval=0.5, _request_interval=0.25)
    adapter.fetch_data(config, "precip", date_range=(2010, 2010))
    assert len([s for s in sleeps if s >= 0.45]) >= 11


def test_resolve_max_workers_caps_concurrent_connections():
    """Per-product max_workers caps adapter concurrency (a connection limit),
    bounded by the global ceiling and the number of files."""
    from acmaddl.adapters.http import _resolve_max_workers, _MAX_WORKERS

    # Catalog cap honoured even with many files / high global ceiling.
    assert _resolve_max_workers({"max_workers": 2}, 8) == 2
    # No catalog cap → bounded by the global ceiling, then by file count.
    assert _resolve_max_workers({}, 100) == _MAX_WORKERS
    assert _resolve_max_workers({}, 3) == 3
    # A catalog value can't exceed the global ceiling.
    assert _resolve_max_workers({"max_workers": 100}, 50) == _MAX_WORKERS
    # Always at least one worker, never more than the work available.
    assert _resolve_max_workers({"max_workers": 4}, 1) == 1


def test_ccsr_entries_wired_and_config_healthy():
    """SPEAR / SPEARb / CanSIPS-IC4 were placeholders while the IRI Data Library
    sunset left them unreachable. Issue #14 wired them to the Columbia CCSR
    successor via the `ccsr` adapter, so they are no longer pending and pass the
    config-level health check."""
    from acmaddl import health, catalog
    ccsr_entries = ["nmme/spear", "nmme/spear-hindcast", "nmme/spearb",
                    "nmme/spearb-hindcast", "nmme/cansipsic4", "nmme/cansipsic4-hindcast"]
    for product in ccsr_entries:
        cfg = catalog.info(product)
        assert cfg["adapter"] == "ccsr", f"{product} should use the ccsr adapter"
        assert cfg.get("pending_url") is not True, f"{product} is no longer pending"
        assert cfg.get("source_url"), f"{product} should have a CCSR source_url"
        assert "forecast.ccsr.columbia.edu" in cfg["source_url"]
        result = health.check_product(product)  # config-only (no remote probe)
        assert result["healthy"] is True, \
            f"{product} config health should pass once wired: {result}"


# ---------------------------------------------------------------------------
# 11. S2S reforecast plumbing tests
# ---------------------------------------------------------------------------

def test_fetch_reforecast_kwarg_plumbs_to_adapter(monkeypatch):
    """acmaddl.fetch(reforecast=True) sets config['_reforecast']=True on the adapter call."""
    import acmaddl
    from acmaddl.adapters.cds import CDSAdapter

    captured = {}

    def fake_fetch_data(self, config, variable, date_range=None, region=None):
        captured["reforecast"] = config.get("_reforecast", False)
        # Return a tiny well-formed dataset matching the ECDS s2s-reforecasts
        # response shape (hdate after the cds adapter's time→hdate rename).
        import xarray as xr, numpy as np, pandas as pd
        hdates = pd.to_datetime(["2020-05-15"])
        return xr.Dataset(
            {"tp": (["hdate", "number", "step", "latitude", "longitude"],
                    np.zeros((1, 2, 3, 2, 2), dtype="float32"))},
            coords={"hdate": hdates, "number": [0, 1],
                    "step": pd.to_timedelta([24, 48, 72], unit="h"),
                    "latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
        )

    monkeypatch.setattr(CDSAdapter, "fetch_data", fake_fetch_data)

    acmaddl.fetch(
        product="c3s/ecmwf-s2s",
        variable="precip",
        init="2026-05-15",
        region=[-2, 2, 36, 40],
        reforecast=True,
        cache=False,
        verbose=False,
    )

    assert captured["reforecast"] is True


def test_fetch_reforecast_defaults_false(monkeypatch):
    """Omitting reforecast defaults to False."""
    import acmaddl
    from acmaddl.adapters.cds import CDSAdapter

    captured = {}

    def fake_fetch_data(self, config, variable, date_range=None, region=None):
        captured["reforecast"] = config.get("_reforecast", "missing")
        import xarray as xr, numpy as np
        return xr.Dataset(
            {"tp": (["latitude", "longitude"], np.zeros((2, 2), dtype="float32"))},
            coords={"latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
        )

    monkeypatch.setattr(CDSAdapter, "fetch_data", fake_fetch_data)

    acmaddl.fetch(
        product="c3s/ecmwf-s2s",
        variable="precip",
        init="2026-05-15",
        region=[-2, 2, 36, 40],
        cache=False,
        verbose=False,
    )

    assert captured["reforecast"] is False


def test_fetch_reforecast_dispatches_to_cds_adapter(monkeypatch):
    """acmaddl.fetch(reforecast=True) routes through CDSAdapter (ECDS
    s2s-reforecasts), not MARSAdapter. ECMWF decommissioned legacy WEB-API
    access to the S2S dataset in 2026-05; CDS is the canonical path."""
    import acmaddl
    from acmaddl.adapters.cds import CDSAdapter
    from acmaddl.adapters.mars import MARSAdapter

    cds_calls = []
    mars_calls = []

    def fake_cds_fetch(self, config, variable, date_range=None, region=None):
        cds_calls.append(dict(config))
        import xarray as xr, numpy as np, pandas as pd
        hdates = pd.to_datetime(["2020-05-15", "2021-05-15"])
        return xr.Dataset(
            {"tp": (["hdate", "number", "step", "latitude", "longitude"],
                    np.zeros((2, 2, 3, 2, 2), dtype="float32"))},
            coords={"hdate": hdates, "number": [0, 1],
                    "step": pd.to_timedelta([24, 48, 72], unit="h"),
                    "latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
        )

    def fake_mars_fetch(self, config, variable, date_range=None, region=None):
        mars_calls.append(dict(config))
        import xarray as xr, numpy as np, pandas as pd
        hdates = pd.to_datetime(["2020-05-15", "2021-05-15"])
        return xr.Dataset(
            {"tp": (["hdate", "number", "step", "latitude", "longitude"],
                    np.zeros((2, 2, 3, 2, 2), dtype="float32"))},
            coords={"hdate": hdates, "number": [0, 1],
                    "step": pd.to_timedelta([24, 48, 72], unit="h"),
                    "latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
        )

    monkeypatch.setattr(CDSAdapter, "fetch_data", fake_cds_fetch)
    monkeypatch.setattr(MARSAdapter, "fetch_data", fake_mars_fetch)

    acmaddl.fetch(
        product="c3s/ecmwf-s2s",
        variable="precip",
        init="2026-05-15",
        region=[-2, 2, 36, 40],
        reforecast=True,
        hindcast=(2020, 2021),
        cache=False,
        verbose=False,
    )

    assert len(cds_calls) == 1
    assert len(mars_calls) == 0


# ---------------------------------------------------------------------------
# R5. Catalog flags for pushdown
# ---------------------------------------------------------------------------

def test_catalog_flags_for_pushdown():
    from acmaddl import catalog
    cfsv2 = catalog.info("nmme/cfsv2")
    assert cfsv2.get("append_streams") is True
    assert cfsv2.get("deprecated") is False
    assert "deprecated_after" not in cfsv2
    for p in ("nmme/cansipsic4", "nmme/ccsm4", "nmme/spear"):
        assert catalog.info(p).get("single_year_fetch") is True
