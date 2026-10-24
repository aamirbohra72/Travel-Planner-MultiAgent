"""FastMCP server exposing TripMate's travel-search tools."""

import argparse
import asyncio
from typing import Any, cast

from mcp.server.fastmcp import FastMCP

from tripmate.config import get_settings
from tripmate.mcp_servers.data_source import create_data_source

settings = get_settings()
data_source = create_data_source(settings)
mcp = FastMCP(
    "TripMate travel tools",
    host=settings.tripmate_mcp_host,
    port=settings.tripmate_mcp_port,
    streamable_http_path="/mcp",
)


async def _run_sync_tool(function: Any, *arguments: Any) -> Any:
    """Run synchronous provider work with the configured deadline."""
    return await asyncio.wait_for(
        asyncio.to_thread(function, *arguments),
        timeout=settings.tripmate_mcp_tool_timeout_seconds,
    )


@mcp.tool()
async def search_flights(origin: str, destination: str, date: str) -> list[dict[str, Any]]:
    """Search flight options for an origin, destination, and date."""
    return cast(
        list[dict[str, Any]],
        await _run_sync_tool(data_source.search_flights, origin, destination, date),
    )


@mcp.tool()
async def search_hotels(
    city: str,
    checkin: str,
    checkout: str,
    max_price: float | None = None,
) -> list[dict[str, Any]]:
    """Search hotels for a stay and optional total-price ceiling in INR."""
    return cast(
        list[dict[str, Any]],
        await _run_sync_tool(
            data_source.search_hotels, city, checkin, checkout, max_price
        ),
    )


@mcp.tool()
async def get_weather(city: str, date: str) -> dict[str, Any]:
    """Get weather information for a city and date."""
    return await asyncio.wait_for(
        data_source.get_weather(city, date),
        timeout=settings.tripmate_mcp_tool_timeout_seconds,
    )


@mcp.tool()
async def search_places(city: str, category: str) -> list[dict[str, Any]]:
    """Search points of interest in a city by category."""
    return cast(
        list[dict[str, Any]],
        await _run_sync_tool(data_source.search_places, city, category),
    )


def main() -> None:
    """Run the MCP server over Streamable HTTP or stdio."""
    parser = argparse.ArgumentParser(description="TripMate MCP travel tools")
    parser.add_argument(
        "--transport",
        choices=("streamable-http", "stdio"),
        default="streamable-http",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()