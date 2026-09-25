"""Tests for the dataset reference sheet (gallery.py)."""
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import yaml

from acmaddl.gallery import _CATALOG_PATH, _classify, _collect, _meta_line, _source, show_datasets, main


def _raw():
    return yaml.safe_load(_CATALOG_PATH.read_text())


def test_collect_covers_every_live_product():
    raw = _raw()
    names = {n for _, _, n, _ in _collect()}
    aliases = {k for k, v in raw.items() if "alias_of" in v}
    assert not (names & aliases)
    # every non-alias product either appears or is date-deprecated
    for k, v in raw.items():
        if "alias_of" in v:
            continue
        if k not in names:
            assert v.get("deprecated_after"), f"{k} missing without deprecation"
    # section order + taxonomy
    secs = [s for s, _, _, _ in _collect()]
    firsts = sorted(set(secs), key=secs.index)
    assert firsts == ["Observations", "Seasonal forecasts", "Sub-seasonal forecasts"]
    assert _classify("c3s/dwd-daily") == ("Sub-seasonal forecasts", "C3S / Copernicus")
    assert _classify("c3s/ecmwf-s2s") == ("Sub-seasonal forecasts", "C3S / Copernicus")
    assert _classify("chc/chirps-gefs-15day") == ("Sub-seasonal forecasts", "CHC forecasts")
    assert _classify("nmme/ccsm4") == ("Seasonal forecasts", "NMME")
    assert _classify("obs/chirps-v3-daily", {"variables": {"precip": {}}}) == (
        "Observations", "Precipitation")
    assert _classify("obs/ersst-v5", {"variables": {"sst": {}}}) == (
        "Observations", "Sea-surface temperature")
    assert _classify("obs/era5", {"variables": {"temp": {}, "precip": {}, "sst": {}}}) == (
        "Observations", "Reanalysis (multi-variable)")
    assert _classify("c3s/ecmwf") == ("Seasonal forecasts", "C3S / Copernicus")


def test_meta_line_and_source():
    raw = _raw()
    e = raw["obs/chirps-v2-monthly"]
    line = _meta_line(e)
    assert "precip (mm/month)" in line and "0.05" in line and "monthly" in line
    assert _source(e) == "data.chc.ucsb.edu"
    e2 = raw["nmme/ccsm4"]
    assert _source(e2) == "CCSR (Columbia)"
    assert "seasonal (init/lead)" in _meta_line(e2)


def test_show_datasets_all_one_axes_per_product(tmp_path):
    fig = show_datasets(which="all", save=tmp_path / "d.png")
    try:
        assert len(fig.axes) == len(_collect())
        titles = " ".join(t.get_text() for t in fig.texts)
        for _, _, name, _ in _collect():
            assert name in titles
        assert (tmp_path / "d.png").exists()
    finally:
        plt.close(fig)


def test_default_renders_two_pages(tmp_path):
    figs = show_datasets(save=tmp_path / "d.png")
    try:
        assert len(figs) == 2
        n_obs = sum(1 for s, _, _, _ in _collect() if s == "Observations")
        n_fc = len(_collect()) - n_obs
        assert len(figs[0].axes) == n_obs
        assert len(figs[1].axes) == n_fc
        assert (tmp_path / "d-observations.png").exists()
        assert (tmp_path / "d-forecasts.png").exists()
    finally:
        plt.close("all")


def test_which_validation():
    with pytest.raises(ValueError):
        show_datasets(which="nope")


def test_main_writes_two_files(tmp_path, capsys):
    out = tmp_path / "sheet.png"
    main([str(out)])
    assert (tmp_path / "sheet-observations.png").exists()
    assert (tmp_path / "sheet-forecasts.png").exists()
    assert "sheet-forecasts.png" in capsys.readouterr().out
    plt.close("all")
