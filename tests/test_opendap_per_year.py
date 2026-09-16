"""Per-year OPeNDAP file support (NOAA PSL daily products, e.g. OISST v2.1).

PSL files high-resolution daily data one NetCDF per calendar year with no
THREDDS aggregation endpoint, so a catalog entry marks itself with a `{year}`
placeholder and the adapter opens one file per requested year.

No network — xr.open_dataset is monkeypatched throughout.
"""
import numpy as np
import pytest
import xarray as xr

from acmaddl.adapters import opendap as opendap_mod
from acmaddl.adapters.opendap import OPeNDAPAdapter, _build_url, _is_per_year


PER_YEAR_TEMPLATE = "{base}/sst.day.mean.{year}.nc"


def _config(**over):
    cfg = {
        "adapter": "opendap",
        "source_url": "https://example.test/oisst",
        "url_template": PER_YEAR_TEMPLATE,
        "decode_times": True,
        "variables": {"sst": {"native_name": "sst"}},
        "_verbose": False,
    }
    cfg.update(over)
    return cfg


def _year_ds(year, ndays=8):
    """A tiny stand-in for one year's file: time x lat x lon, ascending coords."""
    time = np.array([np.datetime64(f"{year}-01-01") + np.timedelta64(d, "D")
                     for d in range(ndays)])
    data = np.full((ndays, 3, 4), float(year), dtype="float32")
    # vary within the block so the degenerate-response guard doesn't fire
    data += np.arange(ndays, dtype="float32")[:, None, None]
    return xr.Dataset(
        {"sst": (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": [-1.0, 0.0, 1.0], "lon": [50.0, 51.0, 52.0, 53.0]},
    )


def test_is_per_year_detects_the_year_placeholder():
    assert _is_per_year(_config()) is True
    assert _is_per_year(_config(url_template="{base}")) is False


def test_build_url_substitutes_the_year():
    url = _build_url(_config(), "sst", "https://example.test/oisst", year=2019)
    assert url == "https://example.test/oisst/sst.day.mean.2019.nc"


def test_per_year_fetch_opens_one_file_per_year_and_concatenates(monkeypatch):
    opened = []

    def fake_open(url, **kw):
        opened.append(url)
        year = int(url.split(".")[-2])
        return _year_ds(year)

    monkeypatch.setattr(opendap_mod.xr, "open_dataset", fake_open)
    ds = OPeNDAPAdapter().fetch_data(_config(), "sst", date_range=(2019, 2021))

    assert len(opened) == 3
    assert opened[0].endswith("sst.day.mean.2019.nc")
    assert opened[-1].endswith("sst.day.mean.2021.nc")
    assert ds.sizes["time"] == 24                     # 3 years x 8 days, concatenated
    # ...and in ascending time order across the year boundary
    assert bool(np.all(np.diff(ds["time"].values.astype("datetime64[ns]")) > np.timedelta64(0, "ns")))


def test_per_year_fetch_skips_a_missing_year_rather_than_failing(monkeypatch):
    """A year the provider has not published must not sink the whole request."""
    def fake_open(url, **kw):
        year = int(url.split(".")[-2])
        if year == 2020:
            raise OSError("NetCDF: file not found")
        return _year_ds(year)

    monkeypatch.setattr(opendap_mod.xr, "open_dataset", fake_open)
    monkeypatch.setattr(opendap_mod, "_DEFAULT_MAX_RETRIES", 1)
    ds = OPeNDAPAdapter().fetch_data(_config(), "sst", date_range=(2019, 2021))

    assert ds.sizes["time"] == 16                      # 2019 and 2021 only
    assert 2020 not in np.unique(ds["time"].dt.year.values)


def test_per_year_fetch_raises_when_no_year_is_available(monkeypatch):
    monkeypatch.setattr(opendap_mod.xr, "open_dataset",
                        lambda url, **kw: (_ for _ in ()).throw(OSError("gone")))
    monkeypatch.setattr(opendap_mod, "_DEFAULT_MAX_RETRIES", 1)
    with pytest.raises(RuntimeError, match="no data for any year"):
        OPeNDAPAdapter().fetch_data(_config(), "sst", date_range=(2019, 2020))


def test_per_year_fetch_requires_a_date_range(monkeypatch):
    monkeypatch.setattr(opendap_mod.xr, "open_dataset", lambda url, **kw: _year_ds(2019))
    with pytest.raises(ValueError, match="one file per year"):
        OPeNDAPAdapter().fetch_data(_config(), "sst", date_range=None)


def test_per_year_fetch_loads_in_day_blocks(monkeypatch):
    """A whole year of 0.25 deg daily overruns the DAP cap, so loading is blocked."""
    monkeypatch.setattr(opendap_mod.xr, "open_dataset", lambda url, **kw: _year_ds(2019, ndays=10))
    cfg = _config(max_request_days=4)
    ds = OPeNDAPAdapter().fetch_data(cfg, "sst", date_range=(2019, 2019))
    assert ds.sizes["time"] == 10          # 4 + 4 + 2, stitched back together


def test_catalog_oisst_entry_is_per_year_and_celsius():
    from acmaddl.catalog import info

    cfg = info("obs/oisst-v2-daily")
    assert _is_per_year(cfg)
    assert cfg["variables"]["sst"]["target_units"] == "C"
    assert cfg["grid"]["temporal"] == "daily"
    assert cfg["grid"]["lat_res"] == 0.25
