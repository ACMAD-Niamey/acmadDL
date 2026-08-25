from acmaddl.adapters.base import AdapterBase


class _Dummy(AdapterBase):
    def fetch_data(self, product_config, variable, date_range=None, region=None):
        return None


def _cfg(**kw):
    c = {"split_streams": True, "grid": {"hindcast_range": [1982, 2010]}}
    c.update(kw); return c


def test_resolve_streams_single_hindcast():
    a = _Dummy()
    assert a._resolve_streams(_cfg(), (1993, 2010)) == [("hindcast", (1993, 2010))]


def test_resolve_streams_single_forecast():
    a = _Dummy()
    assert a._resolve_streams(_cfg(), (2011, 2015)) == [("forecast", (2011, 2015))]


def test_resolve_streams_splits_when_append_and_spanning():
    a = _Dummy()
    segs = a._resolve_streams(_cfg(append_streams=True), (1993, 2020))
    assert segs == [("hindcast", (1993, 2010)), ("forecast", (2011, 2020))]


def test_resolve_streams_no_split_without_flag():
    a = _Dummy()
    # spanning range but no append_streams -> single stream (today's behavior)
    assert a._resolve_streams(_cfg(), (1993, 2020)) == [("hindcast", (1993, 2020))]


def test_resolve_streams_non_split_product():
    a = _Dummy()
    assert a._resolve_streams({"grid": {}}, (1993, 2020)) == [("hindcast", (1993, 2020))]
