"""Tests for the dataset reference sheet (gallery.py)."""
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import yaml

from acmaddl.gallery import _CATALOG_PATH, _SECTIONS, _collect, _meta_line, _source, show_datasets, main


def _raw():
    return yaml.safe_load(_CATALOG_PATH.read_text())


def test_collect_covers_every_live_product():
    raw = _raw()
    names = {n for _, n, _ in _collect()}
    aliases = {k for k, v in raw.items() if "alias_of" in v}
    assert not (names & aliases)
    # every non-alias product either appears or is date-deprecated
    for k, v in raw.items():
        if "alias_of" in v:
            continue
        if k not in names:
            assert v.get("deprecated_after"), f"{k} missing without deprecation"
    # section order: families appear in declared order
    fams = [f for f, _, _ in _collect()]
    firsts = sorted(set(fams), key=fams.index)
    assert firsts == [f for f, _ in _SECTIONS if f in firsts]


def test_meta_line_and_source():
    raw = _raw()
    e = raw["obs/chirps-v2-monthly"]
    line = _meta_line(e)
    assert "precip (mm/month)" in line and "0.05" in line and "monthly" in line
    assert _source(e) == "data.chc.ucsb.edu"
    e2 = raw["nmme/ccsm4"]
    assert _source(e2) == "CCSR (Columbia)"
    assert "seasonal (init/lead)" in _meta_line(e2)


def test_show_datasets_one_axes_per_product(tmp_path):
    fig = show_datasets(save=tmp_path / "d.png")
    try:
        assert len(fig.axes) == len(_collect())
        titles = " ".join(t.get_text() for t in fig.texts)
        for _, name, _ in _collect():
            assert name in titles
        assert (tmp_path / "d.png").exists()
    finally:
        plt.close(fig)


def test_main_writes_file(tmp_path, capsys):
    out = tmp_path / "sheet.png"
    main([str(out)])
    assert out.exists()
    assert str(out) in capsys.readouterr().out
    plt.close("all")
