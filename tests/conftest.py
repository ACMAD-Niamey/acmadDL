import numpy as np
import pandas as pd
import xarray as xr
import pytest


@pytest.fixture
def synthetic_global_ds():
    """A synthetic xarray Dataset with native naming (latitude/longitude, prate)."""
    lat = np.arange(-90, 91, 1.0)
    lon = np.arange(0, 360, 1.0)
    data = np.random.rand(len(lat), len(lon)).astype(np.float32)
    return xr.Dataset(
        {"prate": (["latitude", "longitude"], data)},
        coords={"latitude": lat, "longitude": lon},
    )


@pytest.fixture
def synthetic_temp_ds():
    """A synthetic Dataset with temperature in Kelvin."""
    lat = np.arange(-10, 10, 1.0)
    lon = np.arange(30, 50, 1.0)
    data = np.full((len(lat), len(lon)), 300.0, dtype=np.float32)
    return xr.Dataset(
        {"2m_temperature": (["latitude", "longitude"], data)},
        coords={"latitude": lat, "longitude": lon},
    )


@pytest.fixture
def synthetic_precip_ds():
    """Dataset with precip in kg m-2 s-1."""
    lat = np.arange(-20, 20, 1.0)
    lon = np.arange(20, 50, 1.0)
    data = np.ones((len(lat), len(lon)), dtype=np.float32)
    return xr.Dataset(
        {"prate": (["latitude", "longitude"], data)},
        coords={"latitude": lat, "longitude": lon},
    )


@pytest.fixture
def synthetic_nmme_ds():
    """Synthetic NMME-style dataset with numeric 'months since' time encoding.

    S encodes 'months since 1960-01-01':
      Feb 2010 = (2010-1960)*12 + 1 = 601
      Feb 2011 = 613
      Feb 2012 = 625
    """
    s_vals = np.array([601.0, 613.0, 625.0])
    l_vals = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    m_vals = np.arange(4)
    lat = np.arange(-2, 3, 1.0)
    lon = np.arange(36, 41, 1.0)
    shape = (len(s_vals), len(l_vals), len(m_vals), len(lat), len(lon))
    data = np.random.rand(*shape).astype(np.float32)
    ds = xr.Dataset(
        {"prec": (["S", "L", "M", "Y", "X"], data)},
        coords={"S": s_vals, "L": l_vals, "M": m_vals, "Y": lat, "X": lon},
    )
    ds["S"].attrs["units"] = "months since 1960-01-01"
    return ds


@pytest.fixture
def synthetic_cds_forecast_ds():
    """Synthetic CDS forecast dataset with datetime64 init times."""
    init_times = pd.to_datetime(["2010-02-01", "2011-02-01", "2012-02-01"])
    lead_months = [1, 2, 3, 4, 5, 6]
    members = [0, 1]
    lat = np.arange(-2, 3, 1.0)
    lon = np.arange(36, 41, 1.0)
    shape = (len(init_times), len(lead_months), len(members), len(lat), len(lon))
    data = np.random.rand(*shape).astype(np.float32)
    return xr.Dataset(
        {"t2m": (["forecast_reference_time", "forecastMonth", "number",
                  "latitude", "longitude"], data)},
        coords={
            "forecast_reference_time": init_times,
            "forecastMonth": lead_months,
            "number": members,
            "latitude": lat,
            "longitude": lon,
        },
    )


@pytest.fixture
def synthetic_obs_monthly_ds():
    """Synthetic observational dataset with monthly datetime time axis."""
    times = pd.date_range("2010-01-01", "2012-12-01", freq="MS")
    lat = np.arange(-2, 3, 1.0)
    lon = np.arange(36, 41, 1.0)
    data = np.random.rand(len(times), len(lat), len(lon)).astype(np.float32)
    return xr.Dataset(
        {"precip": (["time", "latitude", "longitude"], data)},
        coords={"time": times, "latitude": lat, "longitude": lon},
    )


# Mapping of PyCPT predictor names to (acmaddl_product, variable) pairs.
PYCPT_REFERENCE_GCMS = {
    # PRCP predictors
    "GEOSS2S.PRCP":       ("nmme/geoss2s",         "precip"),
    "SPEAR.PRCP":         ("nmme/spear",            "precip"),
    "CCSM4.PRCP":         ("nmme/ccsm4-iri",        "precip"),
    "CFSv2.PRCP":         ("nmme/cfsv2",            "precip"),
    "CanSIPSIC4.PRCP":    ("nmme/cansipsic4",       "precip"),
    "SEAS51c.PRCP":       ("c3s/ecmwf-seas51c",     "precip"),
    "SPSv3p5.PRCP":       ("c3s/cmcc",              "precip"),
    "GCFS2p1.PRCP":       ("c3s/dwd-gcfs21",        "precip"),
    "CPS2.PRCP":          ("c3s/jma-cps2",          "precip"),
    "METEOFRANCE9.PRCP":  ("c3s/meteofrance",       "precip"),
    # SST predictors
    "GEOSS2S.SST":        ("nmme/geoss2s",           "sst"),
    "SPEARb.SST":         ("nmme/spearb",            "sst"),
    "CCSM4.SST":          ("nmme/ccsm4",             "sst"),
    "CFSv2.SST":          ("nmme/cfsv2",             "sst"),
    "CanSIPSIC4.SST":     ("nmme/cansipsic4",        "sst"),
    "SEAS51c.SST":        ("c3s/ecmwf-seas51c",      "sst"),
    "SPSv3p5.SST":        ("c3s/cmcc",               "sst"),
    "GCFS2p1.SST":        ("c3s/dwd-gcfs21",         "sst"),
    "CPS2.SST":           ("c3s/jma-cps2",           "sst"),
    "METEOFRANCE9.SST":   ("c3s/meteofrance",        "sst"),
}
