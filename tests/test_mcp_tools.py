"""Offline tests for MCP data providers, filtering, and timeout fallbacks."""

import asyncio
from typing import cast

from langchain_core.tools import BaseTool, StructuredTool

from tripmate.config import Settings
from tripmate.mcp_servers.data_source import MockTravelDataSource
from tripmate.mcp_tools import MCPToolsClient, load_agent_toolsets, with_timeout


def _tool_set():
    def search_flights(origin: str, destination: str, date: str) -> str:
        """Return a dummy flight result."""
        return f"{origin}-{destination}-{date}"

    def search_hotels(
        city: str, checkin: str, checkout: str, max_price: float | None = None
    ) -> str:
        """Return a dummy hotel result."""
        return f"{city}-{checkin}-{checkout}-{max_price}"

    def get_weather(city: str, date: str) -> str:
        """Return a dummy weather result."""
        return f"{city}-{date}"

    def search_places(city: str, category: str) -> str:
        """Return a dummy place result."""
        return f"{city}-{category}"

    return [
        StructuredTool.from_function(func=search_flights, name="search_flights"),
        StructuredTool.from_function(func=search_hotels, name="search_hotels"),
        StructuredTool.from_function(func=get_weather, name="get_weather"),
        StructuredTool.from_function(func=search_places, name="search_places"),
    ]


class FakeMCPClient:
    """Return local tools in place of a running MCP server."""

    async def get_tools(self, *, server_name: str | None = None) -> list[BaseTool]:
        assert server_name == "tripmate"
        return _tool_set()


def test_mock_provider_filters_hotel_options_by_total_price() -> None:
    hotels = MockTravelDataSource().search_hotels(
        "Goa", "2026-11-01", "2026-11-05", max_price=11000
    )

    assert [hotel["option_id"] for hotel in hotels] == ["hotel-1"]


def test_each_agent_gets_only_its_required_tools() -> None:
    settings = Settings(_env_file=None)
    toolsets = asyncio.run(
        load_agent_toolsets(settings, client=cast(MCPToolsClient, FakeMCPClient()))
    )

    assert [tool.name for tool in toolsets["flight"]] == ["search_flights"]
    assert [tool.name for tool in toolsets["hotel"]] == ["search_hotels"]
    assert {tool.name for tool in toolsets["itinerary"]} == {
        "get_weather",
        "search_places",
    }
    assert toolsets["budget"] == []


def test_tool_timeout_returns_graceful_fallback() -> None:
    async def slow_lookup(city: str) -> str:
        await asyncio.sleep(0.05)
        return city

    tool = StructuredTool.from_function(
        coroutine=slow_lookup,
        name="slow_lookup",
        description="A deliberately slow test tool.",
    )
    guarded = with_timeout(tool, timeout=0.001)

    result = asyncio.run(guarded.ainvoke({"city": "Goa"}))

    assert "timed out" in result
    assert "do not invent" in result