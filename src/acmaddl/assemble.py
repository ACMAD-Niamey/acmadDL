"""Multi-model fan-out over fetch: assemble aligned {label: (hindcast, forecast)}."""
from __future__ import annotations
from . import catalog
from .fetch import fetch


def _canonicalize(da):
    """Shape a year_index=True fetch result to canonical (year, member, lat, lon).

    Mirrors the reference pipeline's `_to_year_member` tail: guarantees a
    `member` dim and drops stray non-dim coords (number/member/spatial_ref)
    that `fetch`/`normalize`'s year_index branch does not itself handle.
    """
    if "member" not in da.dims:
        da = da.expand_dims(member=[0])
    da = da.drop_vars(
        [c for c in ("number", "member", "spatial_ref") if c in da.coords and c not in da.dims],
        errors="ignore",
    )
    return da.transpose("year", "member", "lat", "lon")


def assemble(roster, variable, *, init, target, region=None, grid_res=None,
             regrid_to=None, seasonal=None, range_index=2, cache=True,
             verbose=True, boundary="cover"):
    """Fan out `fetch` across a model roster and pair up hindcast/forecast.

    roster: iterable of rows `(label, product, *ranges)` where each range is
    `(start, end)`. `range_index` selects which range tuple in the row to use
    as the hindcast window (default 2 -> the third element, i.e. the first
    range column after label/product).

    Returns `{label: (hindcast, forecast)}`. Per-row fetch failures are not
    swallowed here (caller decides); raises on first failure.
    """
    roster = list(roster)
    # Validate the whole roster before fetching anything: fan-out is sequential,
    # so checking only inside fetch() would still download every model preceding
    # an unsupported one before failing. Deliberately a *capability* check only —
    # an id that isn't in the catalog at all is left to fetch() to report, so this
    # adds no failure mode that didn't already exist.
    for row in roster:
        try:
            config = catalog.get(row[1])
        except KeyError:
            continue
        catalog.require_variable(row[1], variable, config=config)

    common = dict(init=init, target=target, region=region, grid_res=grid_res,
                  regrid_to=regrid_to, seasonal=seasonal, cache=cache,
                  verbose=verbose, boundary=boundary, year_index=True)
    out = {}
    for row in roster:
        label, product = row[0], row[1]
        h0, h1 = row[range_index]
        hcst = _canonicalize(fetch(product, variable, hindcast=(h0, h1), **common)[variable])
        fcst = _canonicalize(fetch(product, variable, **common)[variable])
        out[label] = (hcst, fcst)
    return out


def obs_predictor(product, variable, *, target=None, months=None, hindcast, forecast_year,
                  region=None, grid_res=None, regrid_to=None, seasonal="mean",
                  boundary="center", cache=True, verbose=True):
    """Use an OBSERVED gridded field as a seasonal predictor: `(hindcast, forecast)`.

    The generic "observed-field-as-predictor" step — the lagged/persistence
    predictor a CCA can be built on (e.g. observed ERSSTv5 SST predicting a later
    rainfall season, ACMAD's Exp-1). Model forecasts go through `assemble`; this is
    its observations counterpart, returning the SAME canonical
    `(year, member, lat, lon)` pair so an obs predictor drops into
    `deepscale.seasonal_mme`'s `predictor_tracks` exactly like a model.

    Give EITHER `target` (a 3-month season) OR `months` (explicit calendar months,
    e.g. `[6]` for a single-month June predictor — ACMAD's Exp-1 uses the single IC
    month). `hindcast` is the `(start, end)` training window, `forecast_year` the
    year to predict from. Observations carry no init/lead; `seasonal="mean"` averages
    the selected month(s), so the training series and the single forecast year come
    from the same observed product.

    Returns `(hindcast, forecast)`; `forecast` is the `forecast_year` slice
    (`year=[forecast_year]`, `member=[0]`).
    """
    if (target is None) == (months is None):
        raise ValueError("obs_predictor needs exactly one of `target` or `months`")
    common = dict(target=target, months=months, region=region, grid_res=grid_res,
                  regrid_to=regrid_to, seasonal=seasonal, cache=cache,
                  verbose=verbose, boundary=boundary)
    y0, y1 = hindcast
    hcst = _canonicalize(fetch(product, variable, hindcast=(y0, y1), **common)[variable])
    fcst = _canonicalize(
        fetch(product, variable, hindcast=(forecast_year, forecast_year), **common)[variable])
    return hcst, fcst
