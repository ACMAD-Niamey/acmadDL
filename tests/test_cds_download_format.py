"""`download_format` is sent only to CDS collections that offer it.

The seasonal and s2s collections expose no `download_format` option (verified
against each collection's own form.json), so sending the key makes the server
log "Download format not supported for this dataset. Defaulting to as_source."
on every live request -- noise in any notebook that fetches them, describing
nothing wrong. The ERA5 collections do offer it, and there `unarchived` saves
an extract step.
"""
import pytest

from acmaddl.adapters.cds import CDSAdapter


class _Sentinel(Exception):
    """Raised by the fake client once the request has been captured."""


def _fake_cdsapi(monkeypatch, captured):
    """Replace cdsapi.Client with one that records the request and stops."""
    import cdsapi

    class _FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        def retrieve(self, dataset, request, target):
            captured["dataset"] = dataset
            captured["request"] = dict(request)
            raise _Sentinel

    monkeypatch.setattr(cdsapi, "Client", _FakeClient)


def _seasonal_config():
    return {
        "adapter": "cds",
        "cds_dataset": "seasonal-monthly-single-levels",
        "cds_model": "jma",
        "system": "3",
        "product_type": "monthly_mean",
        "leadtime_month": [1, 2, 3],
        "variables": {"precip": {"native_name": "total_precipitation"}},
        "_verbose": False,
    }


def _era5_config():
    return {
        "adapter": "cds",
        "cds_dataset": "reanalysis-era5-single-levels-monthly-means",
        "product_type": "monthly_averaged_reanalysis",
        "variables": {"precip": {"native_name": "total_precipitation"}},
        "_verbose": False,
    }


def _capture(monkeypatch, config):
    captured = {}
    _fake_cdsapi(monkeypatch, captured)
    with pytest.raises(_Sentinel):
        CDSAdapter().fetch_data(config, "precip", date_range=(2020, 2021),
                                region=[-18.0, 24.0, 6.0, 32.0])
    return captured["request"]


def test_seasonal_request_omits_download_format(monkeypatch):
    request = _capture(monkeypatch, _seasonal_config())
    assert "download_format" not in request
    assert request["data_format"] == "netcdf"


def test_era5_request_keeps_download_format(monkeypatch):
    request = _capture(monkeypatch, _era5_config())
    assert request["download_format"] == "unarchived"
