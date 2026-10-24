"""Itinerary ReAct agent backed by MCP weather and places tools."""

from collections.abc import Awaitable, Callable, Sequence
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from tripmate.state import ItineraryDraft, TripMateState

ITINERARY_SYSTEM_PROMPT = """You are TripMate's itinerary agent. Use search_places for
relevant attractions and get_weather for the destination and travel date when those
tools are available. Create a practical day-by-day plan for exactly the requested
duration. Do not claim live availability or reservations. If either tool fails, make a
reasonable draft using the other available data and disclose the unavailable lookup in
the notes. Never invent weather observations."""


def create_itinerary_node(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
) -> Callable[[TripMateState], Awaitable[dict[str, object]]]:
    """Create an itinerary node with only weather and places MCP tools."""
    agent = create_react_agent(
        model,
        list(tools),
        prompt=ITINERARY_SYSTEM_PROMPT,
        response_format=ItineraryDraft,
        name="tripmate_itinerary_agent",
    )

    async def itinerary_node(state: TripMateState) -> dict[str, object]:
        """Create a day-by-day plan and preserve tool fallback notes."""
        flight = next(
            (item for item in state.get("flights", []) if item.recommended), None
        )
        hotel = next(
            (item for item in state.get("hotels", []) if item.recommended), None
        )
        try:
            result = await agent.ainvoke(
                {
                    "messages": [
                        (
                            "user",
                            f"Destination: {state.get('destination', 'unknown')}\n"
                            f"Dates: {state.get('dates', 'not specified')}\n"
                            f"Duration in days: {state.get('duration_days', 1)}\n"
                            f"Selected flight: {flight.model_dump_json() if flight else 'unavailable'}\n"
                            f"Selected hotel: {hotel.model_dump_json() if hotel else 'unavailable'}\n"
                            f"Preferences: {state.get('user_preferences', {})}\n"
                            f"Original request: {state['user_request']}",
                        )
                    ]
                },
                config={
                    "recursion_limit": 12,
                    "run_name": "tripmate.itinerary_agent",
                    "tags": ["tripmate", "agent", "itinerary_agent"],
                },
            )
            draft = cast(ItineraryDraft, result["structured_response"])
            if not draft.days:
                return {
                    "itinerary": [],
                    "itinerary_error": draft.notes or "No itinerary was returned.",
                    "itinerary_notes": draft.notes,
                }
            return {
                "itinerary": draft.days,
                "itinerary_error": None,
                "itinerary_notes": draft.notes,
            }
        except Exception as error:
            return {
                "itinerary": [],
                "itinerary_error": (
                    "Itinerary generation is unavailable "
                    f"({type(error).__name__})."
                ),
                "itinerary_notes": "",
            }

    return itinerary_node