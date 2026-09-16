"""Model Context Protocol server for acmaddl (``pip install 'acmadDL[mcp]'``).

Run it with ``acmaddl-mcp`` (stdio) or ``python -m acmaddl.mcp``. The tool
surface is deliberately small and mirrors the public API: catalog lookup,
health checks, ``fetch``, and ``zonal``. Data never crosses the protocol as
arrays: every tool that produces data writes a NetCDF file and returns its
path plus a compact summary, so downstream tools (e.g. the africas2s MCP
server) can pick it up by path.
"""

from .server import mcp, main  # noqa: F401
