"""MCP client configuration, tool filtering, and timeout wrappers."""

import asyncio
from typing import Any, Protocol

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from tripmate.config import Settings


class MCPToolsClient(Protocol):
    """Tool-discovery interface shared by MCP clients and offline tests."""

    async def get_tools(self, *, server_name: str | None = None) -> list[BaseTool]:
        """Discover tools from one configured MCP server."""


def with_timeout(tool: BaseTool, timeout: float) -> BaseTool:
    """Wrap a tool call with a timeout and a model-readable fallback result."""

    async def invoke(**arguments: Any) -> Any:
        try:
            return await asyncio.wait_for(tool.ainvoke(arguments), timeout=timeout)
        except TimeoutError:
            return (
                f"Tool {tool.name} timed out after {timeout:g} seconds. "
                "Treat this data as unavailable and do not invent a result."
            )
        except Exception as error:
            return (
                f"Tool {tool.name} failed ({type(error).__name__}). "
                "Treat this data as unavailable and do not invent a result."
            )

    return StructuredTool.from_function(
        name=tool.name,
        description=tool.description,
        coroutine=invoke,
        args_schema=tool.args_schema,
    )


async def load_agent_toolsets(
    settings: Settings,
    client: MCPToolsClient | None = None,
) -> dict[str, list[BaseTool]]:
    """Load MCP tools once and expose only each agent's required subset."""
    mcp_client = client or MultiServerMCPClient(
        {
            "tripmate": {
                "transport": "streamable_http",
                "url": settings.tripmate_mcp_url,
                "timeout": settings.tripmate_mcp_tool_timeout_seconds,
                "sse_read_timeout": settings.tripmate_mcp_tool_timeout_seconds,
            }
        }
    )
    try:
        discovered = await mcp_client.get_tools(server_name="tripmate")
    except Exception as error:
        raise RuntimeError(
            f"Could not connect to the TripMate MCP server at {settings.tripmate_mcp_url} "
            f"({type(error).__name__}). Start it with `python -m tripmate.mcp_servers.server`."
        ) from error

    tools_by_name = {
        tool.name: with_timeout(tool, settings.tripmate_mcp_tool_timeout_seconds)
        for tool in discovered
    }
    required = {"search_flights", "search_hotels", "get_weather", "search_places"}
    missing = required - tools_by_name.keys()
    if missing:
        raise RuntimeError(
            "TripMate MCP server is missing required tools: "
            + ", ".join(sorted(missing))
        )
    return {
        "flight": [tools_by_name["search_flights"]],
        "hotel": [tools_by_name["search_hotels"]],
        "itinerary": [tools_by_name["get_weather"], tools_by_name["search_places"]],
        "budget": [],
    }