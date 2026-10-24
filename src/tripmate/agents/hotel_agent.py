"""Hotel search ReAct agent backed by the TripMate MCP hotel tool."""

from collections.abc import Awaitable, Callable, Sequence
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from tripmate.state import HotelRecommendation, TripMateState

HOTEL_SYSTEM_PROMPT = """You are TripMate's hotel agent. Use search_hotels with the
provided city, check-in, checkout, and maximum price. Recommend only an option returned
by the tool; never invent a hotel or alter its price. Include returned options in your
structured response. If the tool fails or returns no options, return an empty options
list and explain that hotel data is unavailable."""


def create_hotel_node(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
) -> Callable[[TripMateState], Awaitable[dict[str, object]]]:
    """Create a hotel node with only the hotel-search MCP tool."""
    agent = create_react_agent(
        model,
        list(tools),
        prompt=HOTEL_SYSTEM_PROMPT,
        response_format=HotelRecommendation,
        name="tripmate_hotel_agent",
    )

    async def hotel_node(state: TripMateState) -> dict[str, object]:
        """Search MCP hotel data and save the agent's recommendation."""
        dates = state.get("dates", "not specified")
        try:
            result = await agent.ainvoke(
                {
                    "messages": [
                        (
                            "user",
                            f"City: {state.get('destination', 'unknown')}\n"
                            f"Check-in: {dates}\nCheckout: {dates}\n"
                            f"Maximum total price in INR: {state.get('budget')}\n"
                            f"Preferences: {state.get('user_preferences', {})}\n"
                            "Use the hotel search tool before recommending.",
                        )
                    ]
                },
                config={
                    "recursion_limit": 10,
                    "run_name": "tripmate.hotel_agent",
                    "tags": ["tripmate", "agent", "hotel_agent"],
                },
            )
            response = cast(HotelRecommendation, result["structured_response"])
            options = response.options
            if not options:
                return {
                    "hotels": [],
                    "hotel_error": response.rationale
                    or "Hotel options are unavailable.",
                }
            selected_id = response.option_id
            if selected_id not in {option.option_id for option in options}:
                selected_id = options[0].option_id
            return {
                "hotels": [
                    option.model_copy(
                        update={"recommended": option.option_id == selected_id}
                    )
                    for option in options
                ],
                "hotel_error": None,
            }
        except Exception as error:
            return {
                "hotels": [],
                "hotel_error": (
                    "Hotel search is unavailable; no hotel price was included "
                    f"({type(error).__name__})."
                ),
            }

    return hotel_node