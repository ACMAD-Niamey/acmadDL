"""HTTP adapter mechanics added for the DWD GPCC and NOAA NCEI GPCP products.

No network: `urllib.request.urlretrieve` (the adapter's network boundary) is
monkeypatched to hand back a locally written file, and the directory-listing
scrape is injected. Everything else — gunzipping, time stamping, wildcard
resolution, region subsetting, concatenation — is the real adapter code.
"""
import gzip
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from acmaddl.adapters.http import HTTPAdapter


# ── Fixture builders ─────────────────────────────────────────────────────────

def _gpcc_shaped(path, value=100.0, time_units=None, time_value=None, gz=False):
    """A GPCC-shaped monthly file: var `p` in mm/month, latitude N->S.

    `time_units`/`time_value` write a NUMERIC time axis with the given CF-ish
    units attribute, reproducing DWD's two encodings verbatim. Left as None the
    file carries an ordinary datetime axis.
    """
    lat = np.arange(89.5, -90.0, -1.0)      # N->S, as DWD serves it
    lon = np.arange(-179.5, 180.0, 1.0)
    data = np.full((1, len(lat), len(lon)), value, dtype="float32")
    coords = {"lat": lat, "lon": lon}
    if time_units is None:
        coords["time"] = pd.to_datetime(["2024-03-01"]).values.astype("datetime64[ns]")
    else:
        coords["time"] = np.array([time_value], dtype="float64")
    ds = xr.Dataset({"p": (["time", "lat", "lon"], data)}, coords=coords)
    ds["p"].attrs["units"] = "mm/month"
    if time_units is not None:
        ds["time"].attrs["units"] = time_units
    target = path.with_suffix(".nc") if gz else path
    ds.to_netcdf(target)
    if gz:
        with open(target, "rb") as src, gzip.open(path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        target.unlink()
    return path


@pytest.fixture
def serve_file(monkeypatch):
    """Make every urlretrieve hand back the given local file."""
    def _install(local):
        def fake(url, filename):
            shutil.copyfile(local, filename)
            return filename, None
        monkeypatch.setattr("acmaddl.adapters.http.urllib.request.urlretrieve", fake)
    return _install


def _config(**extra):
    cfg = {
        "adapter": "http",
        "source_url": "https://opendata.example.invalid/GPCC/monitoring_v2020",
        "file_pattern": "{year}/monitoring_v2020_10_{year}_{month:02d}.nc.gz",
        "format": "netcdf",
        "init_months": [3],
        "variables": {"precip": {"native_name": "p", "units": "mm/month",
                                 "target_units": "mm/month"}},
        "_verbose": False,
        "_progress": False,
    }
    cfg.update(extra)
    return cfg


# ── Gzipped payloads ─────────────────────────────────────────────────────────

def test_a_gzipped_netcdf_url_is_decompressed_before_opening(tmp_path, serve_file):
    """DWD serves GPCC as .nc.gz; netCDF4 cannot open a gzip-wrapped file."""
    serve_file(_gpcc_shaped(tmp_path / "gpcc.nc.gz", value=123.0, gz=True))
    out = HTTPAdapter().fetch_data(_config(), "precip", date_range=(2024, 2024))
    assert "p" in out
    np.testing.assert_allclose(float(out["p"].mean()), 123.0)


def test_a_plain_netcdf_url_is_still_opened_directly(tmp_path, serve_file):
    """Regression: the gunzip path must not touch ordinary .nc products."""
    serve_file(_gpcc_shaped(tmp_path / "plain.nc", value=7.0))
    cfg = _config(file_pattern="{year}/rfe{year}_{month:02d}.v3.1.nc")
    out = HTTPAdapter().fetch_data(cfg, "precip", date_range=(2024, 2024))
    np.testing.assert_allclose(float(out["p"].mean()), 7.0)


# ── time_from_pattern ────────────────────────────────────────────────────────
# DWD's two GPCC streams encode time in ways that both defeat the default
# opener, and the (year, month) that built the URL is the only unambiguous
# statement of what a file holds:
#
#   monitoring_v2020  time = 20240301.0, units "day as %Y%m%d.%f"
#                     -> not a CF "<unit> since <epoch>", so xarray leaves it
#                        numeric and normalize's decoder skips it
#   first_guess       time = 0.0, units "months since 2024-3-1 00:00:00"
#                     -> unpadded month/day: decode_times=True RAISES, and every
#                        file says 0 with its own ref, so a concat keeps only
#                        the first file's units and collapses all months onto
#                        one date

@pytest.fixture
def serve_map(monkeypatch):
    """Serve a different local file per URL, matched on substring."""
    def _install(mapping):
        def fake(url, filename):
            for key, local in mapping.items():
                if key in url:
                    shutil.copyfile(local, filename)
                    return filename, None
            raise AssertionError(f"unexpected URL: {url}")
        monkeypatch.setattr("acmaddl.adapters.http.urllib.request.urlretrieve", fake)
    return _install


def test_grads_day_as_encoding_is_stamped_from_the_filename(tmp_path, serve_file):
    serve_file(_gpcc_shaped(tmp_path / "m.nc.gz", time_units="day as %Y%m%d.%f",
                            time_value=20240301.0, gz=True))
    out = HTTPAdapter().fetch_data(_config(time_from_pattern=True), "precip",
                                   date_range=(2024, 2024))
    assert out["time"].values[0] == np.datetime64("2024-03-01", "ns")


def test_an_unpadded_months_since_axis_no_longer_breaks_the_open(tmp_path, serve_file):
    """`months since 2024-3-1` makes decode_times=True raise; the knob skips it."""
    serve_file(_gpcc_shaped(tmp_path / "fg.nc.gz",
                            time_units="months since 2024-3-1 00:00:00",
                            time_value=0.0, gz=True))
    out = HTTPAdapter().fetch_data(_config(time_from_pattern=True), "precip",
                                   date_range=(2024, 2024))
    assert out["time"].values[0] == np.datetime64("2024-03-01", "ns")


def test_every_month_keeps_its_own_timestamp_across_a_concat(tmp_path, serve_map):
    """The trap: per-file `months since <that month>` all carry the value 0.

    Concatenating them keeps only the FIRST file's units attribute, so decoding
    after the concat maps every month onto the same date. Stamping per file,
    before the concat, is what keeps them distinct.
    """
    serve_map({
        "2024_03": _gpcc_shaped(tmp_path / "mar.nc.gz", value=10.0,
                                time_units="months since 2024-3-1 00:00:00",
                                time_value=0.0, gz=True),
        "2024_04": _gpcc_shaped(tmp_path / "apr.nc.gz", value=20.0,
                                time_units="months since 2024-4-1 00:00:00",
                                time_value=0.0, gz=True),
    })
    cfg = _config(time_from_pattern=True, init_months=[3, 4])
    out = HTTPAdapter().fetch_data(cfg, "precip", date_range=(2024, 2024))
    assert list(out["time"].values) == [np.datetime64("2024-03-01", "ns"),
                                        np.datetime64("2024-04-01", "ns")]
    # and the values didn't get shuffled onto the wrong months
    np.testing.assert_allclose(
        [float(out["p"].isel(time=i).mean()) for i in (0, 1)], [10.0, 20.0])


def test_without_the_knob_the_files_own_time_axis_is_preserved(tmp_path, serve_file):
    """Regression: obs/tamsat and friends must keep decoding their own axis."""
    serve_file(_gpcc_shaped(tmp_path / "plain.nc"))
    cfg = _config(file_pattern="{year}/rfe{year}_{month:02d}.v3.1.nc")
    out = HTTPAdapter().fetch_data(cfg, "precip", date_range=(2024, 2024))
    assert out["time"].values[0] == np.datetime64("2024-03-01", "ns")


# ── Wildcard resolution against a directory listing ──────────────────────────
# NCEI's GPCP filenames carry a processing-date suffix
# (gpcp_v02r03_monthly_d202403_c20240607.nc) that no static pattern can predict,
# so a `*` in file_pattern is resolved against the server's listing.

from acmaddl.adapters.http import _resolve_wildcards  # noqa: E402

GPCP_2024 = [
    "gpcp_v02r03_monthly_d202401_c20240407.nc",
    "gpcp_v02r03_monthly_d202402_c20240508.nc",
    "gpcp_v02r03_monthly_d202403_c20240607.nc",
]


def _listing(mapping):
    """A directory lister over a {dir_url: [names]} map, counting its calls."""
    calls = []

    def list_dir(url):
        calls.append(url)
        for key, names in mapping.items():
            if key in url:
                return names
        return []
    return list_dir, calls


def test_a_wildcard_resolves_to_the_matching_file_on_the_server():
    list_dir, _ = _listing({"2024": GPCP_2024})
    out = _resolve_wildcards(
        "https://ncei.example.invalid/access",
        [("2024/gpcp_v02r03_monthly_d202403_*.nc", None)],
        list_dir=list_dir)
    assert out == [("2024/gpcp_v02r03_monthly_d202403_c20240607.nc", None)]


def test_each_directory_is_listed_only_once_for_all_of_its_months():
    """12 monthly files in a year must cost one listing, not twelve."""
    list_dir, calls = _listing({"2024": GPCP_2024})
    entries = [(f"2024/gpcp_v02r03_monthly_d2024{m:02d}_*.nc", None)
               for m in (1, 2, 3)]
    out = _resolve_wildcards("https://ncei.example.invalid/access", entries,
                             list_dir=list_dir)
    assert calls == ["https://ncei.example.invalid/access/2024/"], calls
    assert [name for name, _ in out] == [f"2024/{n}" for n in GPCP_2024]


def test_the_preliminary_sibling_stream_is_not_matched():
    """The same directory holds gpcp_v02r03-PRELIMINARY_monthly_… files.

    Picking one up as if it were the final product would silently mix a
    provisional estimate into a settled record.
    """
    listing = ["gpcp_v02r03-preliminary_monthly_d202606_c20260710.nc",
               "gpcp_v02r03_monthly_d202606_c20260810.nc"]
    list_dir, _ = _listing({"2026": listing})
    out = _resolve_wildcards("https://ncei.example.invalid/access",
                             [("2026/gpcp_v02r03_monthly_d202606_*.nc", None)],
                             list_dir=list_dir)
    assert out == [("2026/gpcp_v02r03_monthly_d202606_c20260810.nc", None)]


def test_the_newest_reprocessing_wins_when_several_match():
    """A month reprocessed twice appears twice; the later _c<date> is current."""
    listing = ["gpcp_v02r03_monthly_d202403_c20240607.nc",
               "gpcp_v02r03_monthly_d202403_c20250101.nc"]
    list_dir, _ = _listing({"2024": listing})
    out = _resolve_wildcards("https://ncei.example.invalid/access",
                             [("2024/gpcp_v02r03_monthly_d202403_*.nc", None)],
                             list_dir=list_dir)
    assert out == [("2024/gpcp_v02r03_monthly_d202403_c20250101.nc", None)]


def test_a_wildcard_matching_nothing_raises_naming_the_pattern():
    list_dir, _ = _listing({"2024": GPCP_2024})
    with pytest.raises(FileNotFoundError, match="d209912"):
        _resolve_wildcards("https://ncei.example.invalid/access",
                           [("2024/gpcp_v02r03_monthly_d209912_*.nc", None)],
                           list_dir=list_dir)


def test_patterns_without_a_wildcard_are_passed_through_unlisted():
    list_dir, calls = _listing({})
    entries = [("2024/rfe2024_03.v3.1.nc", pd.Timestamp("2024-03-01"))]
    assert _resolve_wildcards("https://x.invalid", entries, list_dir=list_dir) == entries
    assert calls == []


def test_the_adapter_downloads_the_url_a_wildcard_resolved_to(tmp_path, monkeypatch):
    """End-to-end through fetch_data: listing -> resolved URL -> opened file."""
    monkeypatch.setattr("acmaddl.adapters.http._list_directory",
                        lambda url: GPCP_2024)
    local = _gpcc_shaped(tmp_path / "gpcp.nc", value=4.0)
    seen = []

    def fake(url, filename):
        seen.append(url)
        shutil.copyfile(local, filename)
        return filename, None
    monkeypatch.setattr("acmaddl.adapters.http.urllib.request.urlretrieve", fake)

    cfg = _config(source_url="https://ncei.example.invalid/access",
                  file_pattern="{year}/gpcp_v02r03_monthly_d{year}{month:02d}_*.nc")
    out = HTTPAdapter().fetch_data(cfg, "precip", date_range=(2024, 2024))
    assert seen == ["https://ncei.example.invalid/access/2024/"
                    "gpcp_v02r03_monthly_d202403_c20240607.nc"]
    np.testing.assert_allclose(float(out["p"].mean()), 4.0)


# ── Longitude convention on region subsetting ────────────────────────────────
# GPCP is the first http product served on a 0..360 grid (CHIRPS and TAMSAT are
# both -180..180), which exposed a plain `slice(lon_w, lon_e)` in the adapter's
# region crop: a West African bbox like [-8, 8] selected only its eastern half,
# and normalize could not recover cells the adapter had already dropped.
# obs/cmap — the same 0..360 grid over OPeNDAP — returns both sides, so that is
# the behaviour to match.

from acmaddl.adapters.http import _subset_region  # noqa: E402


def _grid(lons):
    lat = np.arange(-88.75, 90.0, 2.5)
    data = np.ones((1, len(lat), len(lons)), dtype="float32")
    return xr.Dataset(
        {"precip": (["time", "lat", "lon"], data)},
        coords={"time": pd.to_datetime(["2020-08-01"]).values.astype("datetime64[ns]"),
                "lat": lat, "lon": np.asarray(lons, dtype="float64")},
    )


def test_a_seam_crossing_bbox_keeps_both_sides_of_a_0_360_grid():
    out = _subset_region(_grid(np.arange(1.25, 360.0, 2.5)), [8, 18, -8, 8])
    lons = out["lon"].values
    assert (lons > 350).any(), f"western half dropped: {lons}"
    assert (lons < 10).any(), f"eastern half dropped: {lons}"


def test_the_same_bbox_on_a_minus180_grid_is_unaffected():
    """Regression: CHIRPS/TAMSAT grids must crop exactly as before."""
    out = _subset_region(_grid(np.arange(-178.75, 180.0, 2.5)), [8, 18, -8, 8])
    lons = out["lon"].values
    assert lons.min() >= -9.0 and lons.max() <= 9.0, lons
    assert len(lons) == 8, lons


def test_a_bbox_clear_of_the_seam_is_cropped_to_it_on_a_0_360_grid():
    """East Africa (30..45E) needs no wrapping and must not gain cells."""
    out = _subset_region(_grid(np.arange(1.25, 360.0, 2.5)), [-5, 15, 30, 45])
    lons = out["lon"].values
    assert lons.min() >= 28.0 and lons.max() <= 47.0, lons


def test_the_region_crop_survives_a_real_download(tmp_path, serve_file):
    """End-to-end: the crop happens inside the per-file download."""
    serve_file(_gpcc_shaped(tmp_path / "g.nc"))
    out = HTTPAdapter().fetch_data(
        _config(file_pattern="{year}/rfe{year}_{month:02d}.v3.1.nc"),
        "precip", date_range=(2024, 2024), region=[8, 18, -8, 8])
    assert float(out["lat"].min()) >= 7.0 and float(out["lat"].max()) <= 19.0
