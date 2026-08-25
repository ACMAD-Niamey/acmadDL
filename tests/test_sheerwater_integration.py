"""Real-data integration test for the Sheerwater adapter.

Unlike tests/test_sheerwater.py -- which mocks ``sheerwater.data.*`` and never
touches zarr -- this opens an actual public GCS Zarr store through the real
sheerwater library. It is therefore the suite's only guard that sheerwater
functions end to end, and in particular that it reads correctly under whatever
zarr version is installed.

This matters for the icechunk/zarr dependency split: sheerwater's metadata pins
``zarr==2.18.3`` while icechunk needs ``zarr>=3``. This test is what verifies
that pin is stale -- i.e. sheerwater really does open+read fine under zarr 3 --
rather than load-bearing. Run it under both zarr majors to confirm.

Requires anonymous network read access to the public sheerwater-public-datalake
GCS bucket. Marked ``integration``+``network``; skipped in the default unit run.
"""
import numpy as np
import pytest

import acmaddl

# Small East-Africa bbox keeps the read tiny: the adapter fetches the global
# store lazily and crops to this slab before materializing.
REGION = [-2, 2, 36, 40]


@pytest.mark.integration
@pytest.mark.network
def test_sheerwater_chirps_v3_daily_real_read():
    """obs/chirps-v3-daily-rhiza opens a real GCS Zarr store via the sheerwater
    library and returns finite, non-negative precip cropped to the bbox."""
    ds = acmaddl.fetch(
        "obs/chirps-v3-daily-rhiza",
        "precip",
        hindcast=(2010, 2010),
        region=REGION,
        cache=False,
        verbose=False,
    )
    da = ds["precip"]

    # Normalized coords and the spatial crop were honoured.
    assert "lat" in ds.coords and "lon" in ds.coords
    assert float(ds.lat.min()) >= REGION[0] - 1
    assert float(ds.lat.max()) <= REGION[1] + 1

    # Real data came back: a substantial slab, mostly-finite over land, and
    # physically valid (precip is non-negative). Thresholds are loose on purpose
    # so the test guards "sheerwater read real data" without being brittle.
    assert da.size > 1000, "expected a real multi-day slab, not an empty/degenerate read"
    finite = np.isfinite(da)
    assert float(finite.mean()) > 0.3, "expected substantial finite land data in the bbox"
    assert float(da.where(finite).min()) >= 0.0, "precip must be non-negative"
    assert "units" in da.attrs
