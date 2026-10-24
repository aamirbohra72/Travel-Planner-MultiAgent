"""Flight search ReAct agent backed by the TripMate MCP flight tool."""

from collections.abc import Awaitable, Callable, Sequence
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from tripmate.state import FlightRecommendation, TripMateState

FLIGHT_SYSTEM_PROMPT = """You are TripMate's flight agent. Use search_flights with the
provided origin, destination, and date. Recommend only an option returned by the tool;
never invent a flight or alter its price. Include the returned options in your
structured response. If the tool fails or returns no options, return an empty options
list and explain that flight data is unavailable."""


def create_flight_node(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    default_origin: str,
) -> Callable[[TripMateState], Awaitable[dict[str, object]]]:
    """Create a flight node with only the flight-search MCP tool."""
    agent = create_react_agent(
        model,
        list(tools),
        prompt=FLIGHT_SYSTEM_PROMPT,
        response_format=FlightRecommendation,
        name="tripmate_flight_agent",
    )

    async def flight_node(state: TripMateState) -> dict[str, object]:
        """Search MCP flight data and save the agent's recommendation."""
        try:
            result = await agent.ainvoke(
                {
                    "messages": [
                        (
                            "user",
                            f"Origin: {default_origin}\n"
                            f"Destination: {state.get('destination', 'unknown')}\n"
                            f"Date: {state.get('dates', 'not specified')}\n"
                            f"Preferences: {state.get('user_preferences', {})}\n"
                            "Use the flight search tool before recommending.",
                        )
                    ]
                },
                config={
                    "recursion_limit": 10,
                    "run_name": "tripmate.flight_agent",
                    "tags": ["tripmate", "agent", "flight_agent"],
                },
            )
            response = cast(FlightRecommendation, result["structured_response"])
            options = response.options
            if not options:
                return {
                    "flights": [],
                    "flight_error": response.rationale
                    or "Flight options are unavailable.",
                }
            selected_id = response.option_id
            if selected_id not in {option.option_id for option in options}:
                selected_id = options[0].option_id
            return {
                "flights": [
                    option.model_copy(
                        update={"recommended": option.option_id == selected_id}
                    )
                    for option in options
                ],
                "flight_error": None,
            }
        except Exception as error:
            return {
                "flights": [],
                "flight_error": (
                    "Flight search is unavailable; no flight price was included "
                    f"({type(error).__name__})."
                ),
            }

    return flight_node