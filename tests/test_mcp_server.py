"""Unit tests for the acmaddl MCP server (no network).

The server is a thin wrapper, so these tests pin the *contract*: tools are
registered with the documented names, outputs are JSON-safe, data-producing
tools write NetCDF and return a path + summary, library errors surface with
their message, and the background job table works.
"""

import asyncio
import json

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("mcp")

from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

import acmaddl  # noqa: E402
from acmaddl.mcp import server  # noqa: E402


EXPECTED_TOOLS = {
    "list_products", "describe_product", "check_product", "check_all_products",
    "fetch", "start_fetch", "fetch_status", "list_jobs", "describe_dataset", "zonal",
}


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("ACMADDL_MCP_WORKDIR", str(tmp_path / "work"))
    return tmp_path / "work"


@pytest.fixture
def forecast_ds():
    lat = np.arange(-4.0, 5.0, 1.0)
    lon = np.arange(30.0, 39.0, 1.0)
    years = np.arange(2000, 2005)
    members = np.arange(3)
    data = np.random.rand(len(years), len(members), len(lat), len(lon)).astype("float32")
    data[0, 0, 0, 0] = np.nan
    ds = xr.Dataset(
        {"precip": (["year", "member", "lat", "lon"], data, {"units": "mm"})},
        coords={"year": years, "member": members, "lat": lat, "lon": lon},
        attrs={"product": "test/product"},
    )
    return ds


@pytest.fixture
def fake_fetch(monkeypatch, forecast_ds):
    """Replace acmaddl.fetch with one that honours destination and records kwargs."""
    calls = []

    def _fetch(**kwargs):
        calls.append(kwargs)
        if kwargs.get("destination"):
            forecast_ds.to_netcdf(kwargs["destination"])
        return forecast_ds

    monkeypatch.setattr(acmaddl, "fetch", _fetch)
    return calls


def test_tools_registered():
    names = {t.name for t in _run(server.mcp.list_tools())}
    assert EXPECTED_TOOLS <= names


def test_tool_schemas_have_descriptions():
    for tool in _run(server.mcp.list_tools()):
        assert tool.description, f"{tool.name} has no description"
        assert tool.input_schema.get("type") == "object"


def test_list_products_is_json_safe():
    products = server.list_products()
    assert products and all("product" in p and "adapter" in p for p in products)
    json.dumps(products)
    assert not any(p["deprecated"] for p in products)
    assert len(server.list_products(include_deprecated=True)) >= len(products)


def test_describe_product_roundtrips_and_reports_unknown():
    info = server.describe_product("nmme/cfsv2")
    json.dumps(info)
    assert info["product"] == "nmme/cfsv2"
    assert "precip" in info["variables"]
    with pytest.raises(ToolError, match="Product not found"):
        server.describe_product("nope/nothing")


def test_summarize_is_compact_and_json_safe(forecast_ds):
    summary = server.summarize(forecast_ds)
    json.dumps(summary)
    assert summary["dims"] == {"year": 5, "member": 3, "lat": 9, "lon": 9}
    assert summary["coords"]["lat"] == {"size": 9, "dtype": "float64", "min": -4.0, "max": 4.0, "step": 1.0}
    var = summary["variables"]["precip"]
    assert var["units"] == "mm" and var["dims"] == ["year", "member", "lat", "lon"]
    assert 0 < var["nan_fraction"] < 0.01


def test_describe_dataset_via_protocol(tmp_path, forecast_ds):
    path = tmp_path / "f.nc"
    forecast_ds.to_netcdf(path)
    result = _run(server.mcp.call_tool("describe_dataset", {"path": str(path)}))
    assert not result.is_error, result.content
    payload = json.loads(result.content[0].text)
    assert payload["path"] == str(path)
    assert payload["variables"]["precip"]["shape"] == [5, 3, 9, 9]
    assert payload["size_bytes"] > 0


def test_describe_dataset_missing_file_is_a_tool_error(tmp_path):
    with pytest.raises(ToolError, match="No such file"):
        server.describe_dataset(str(tmp_path / "missing.nc"))


def test_fetch_writes_netcdf_and_returns_summary(fake_fetch, _workdir):
    out = server.fetch("nmme/cfsv2", "precip", init="2025-02", target="MAM",
                       region=[-4, 4, 30, 38], hindcast=[2000, 2004], year_index=True)
    assert out["path"].startswith(str(_workdir))
    assert out["path"].endswith(".nc")
    assert xr.open_dataset(out["path"]).precip.shape == (5, 3, 9, 9)
    assert out["request"]["hindcast"] == [2000, 2004]
    call = fake_fetch[0]
    assert call["verbose"] is False and call["progress"] is False
    assert call["hindcast"] == (2000, 2004) and call["region"] == [-4.0, 4.0, 30.0, 38.0]
    assert call["year_index"] is True


def test_fetch_output_name_is_stable_per_request(fake_fetch):
    a = server.fetch("nmme/cfsv2", "precip", init="2025-02", target="MAM")
    b = server.fetch("nmme/cfsv2", "precip", init="2025-02", target="MAM")
    c = server.fetch("nmme/cfsv2", "precip", init="2025-03", target="MAM")
    assert a["path"] == b["path"] != c["path"]


def test_fetch_explicit_destination_and_options(fake_fetch, tmp_path):
    dest = tmp_path / "sub" / "out.nc"
    out = server.fetch("obs/era5", "temp", months=[6, 7], destination=str(dest),
                       options={"allow_partial": True})
    assert out["path"] == str(dest) and dest.exists()
    assert fake_fetch[0]["months"] == [6, 7] and fake_fetch[0]["allow_partial"] is True


def test_fetch_bad_region_is_a_tool_error(fake_fetch):
    with pytest.raises(ToolError, match="region must be"):
        server.fetch("obs/era5", "temp", region=[1, 2, 3])


def test_fetch_library_error_keeps_message(monkeypatch):
    def _boom(**kwargs):
        raise acmaddl.VariableNotSupported("obs/era5", "sst", ["precip", "temp"])

    monkeypatch.setattr(acmaddl, "fetch", _boom)
    with pytest.raises(ToolError, match="VariableNotSupported"):
        server.fetch("obs/era5", "sst")


def test_fetch_error_via_protocol_reaches_client(monkeypatch):
    def _boom(**kwargs):
        raise ValueError("CDS says no")

    monkeypatch.setattr(acmaddl, "fetch", _boom)
    with pytest.raises(ToolError, match="CDS says no"):
        _run(server.mcp.call_tool("fetch", {"product": "obs/era5", "variable": "temp"}))


def test_start_fetch_job_lifecycle(fake_fetch):
    job = server.start_fetch("nmme/cfsv2", "precip", init="2025-02", target="MAM")
    assert job["status"] == "running"
    for _ in range(200):
        status = server.fetch_status(job["job_id"])
        if status["status"] != "running":
            break
        asyncio.run(asyncio.sleep(0.01))
    assert status["status"] == "done", status
    assert status["result"]["path"].endswith(".nc")
    assert status["finished_at"]
    listing = server.list_jobs()
    assert any(j["job_id"] == job["job_id"] for j in listing)
    assert all("result" not in j for j in listing)


def test_start_fetch_error_recorded(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("queue timeout")

    monkeypatch.setattr(acmaddl, "fetch", _boom)
    job = server.start_fetch("obs/era5", "temp")
    for _ in range(200):
        status = server.fetch_status(job["job_id"])
        if status["status"] != "running":
            break
        asyncio.run(asyncio.sleep(0.01))
    assert status["status"] == "error" and "queue timeout" in status["error"]


def test_fetch_status_unknown_job():
    with pytest.raises(ToolError, match="Unknown job_id"):
        server.fetch_status("nope")


def test_zonal_writes_region_axis(tmp_path, forecast_ds):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    grid = tmp_path / "grid.nc"
    forecast_ds.isel(year=0, member=0).to_netcdf(grid)
    gdf = gpd.GeoDataFrame(
        {"code": ["W", "E"], "name": ["west", "east"]},
        geometry=[box(30, -4, 34, 4), box(34, -4, 38, 4)], crs="EPSG:4326",
    )
    shp = tmp_path / "zones.geojson"
    gdf.to_file(shp, driver="GeoJSON")
    out = server.zonal(str(grid), str(shp), by="code", label="name")
    assert out["dims"]["region"] == 2
    ds = xr.open_dataset(out["path"])
    assert list(ds.region.values) == ["W", "E"]
    assert list(ds.region_label.values) == ["west", "east"]
    assert "lat" not in ds.dims


def test_resources_listed_and_readable():
    uris = {str(r.uri) for r in _run(server.mcp.list_resources())}
    assert {"acmaddl://skill", "acmaddl://catalog"} <= uris
    templates = {t.uri_template for t in _run(server.mcp.list_resource_templates())}
    assert "acmaddl://skill/references/{name}" in templates
    contents = list(_run(server.mcp.read_resource("acmaddl://skill")))
    assert contents and "acmaddl" in contents[0].content
    api = list(_run(server.mcp.read_resource("acmaddl://skill/references/api")))
    assert api and "fetch" in api[0].content
    catalog_text = list(_run(server.mcp.read_resource("acmaddl://catalog")))
    assert "nmme/cfsv2" in catalog_text[0].content


def test_main_parses_transport(monkeypatch):
    seen = {}
    monkeypatch.setattr(server.mcp, "run", lambda transport, **kw: seen.update(t=transport, **kw))
    server.main(["--transport", "streamable-http", "--port", "9001"])
    assert seen == {"t": "streamable-http", "host": "127.0.0.1", "port": 9001}
    server.main([])
    assert seen["t"] == "stdio"


def test_schemas_carry_field_descriptions_enums_and_outputs():
    tools = {t.name: t for t in _run(server.mcp.list_tools())}
    fetch = tools["fetch"]
    props = fetch.input_schema["properties"]
    assert all(p.get("description") for p in props.values()), [k for k, p in props.items() if not p.get("description")]
    assert set(props["variable"]["enum"]) >= {"precip", "temp", "sst"}
    assert props["boundary"]["enum"] == ["center", "cover"]
    assert "path" in fetch.output_schema["properties"]
    assert fetch.annotations.open_world_hint is True and fetch.annotations.read_only_hint is False
    assert tools["list_products"].annotations.read_only_hint is True
    assert tools["zonal"].input_schema["properties"]["stat"]["enum"][0] == "mean"
    for tool in tools.values():
        assert "Returns:" in tool.description, tool.name
        assert not tool.description.startswith(" "), tool.name


def test_enum_violation_is_rejected_at_the_protocol(fake_fetch):
    with pytest.raises(ToolError, match="variable"):
        _run(server.mcp.call_tool("fetch", {"product": "obs/era5", "variable": "rain"}))
    assert fake_fetch == []


def test_unknown_product_suggests_close_matches():
    with pytest.raises(ToolError, match="Did you mean"):
        server.describe_product("nmme/cfsv3")
