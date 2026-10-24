"""Budget estimation agent with deterministic total and overspend calculation."""

from collections.abc import Awaitable, Callable
from typing import cast

from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from tripmate.state import (
    BudgetEstimates,
    BudgetSummary,
    FlightOption,
    HotelOption,
    TripMateState,
)

BUDGET_SYSTEM_PROMPT = """You are TripMate's budget agent. Estimate total meals and local
transport costs in INR for the full trip using the destination, duration, and itinerary.
Return only these estimates and notes. Do not include flight or hotel prices;
TripMate adds their tool-returned prices itself."""


def calculate_budget_summary(
    flights: list[FlightOption],
    hotels: list[HotelOption],
    estimates: BudgetEstimates,
    budget: float | None,
    unavailable: list[str] | None = None,
) -> BudgetSummary:
    """Add selected prices and flag overspending only when all costs are known."""
    flight = next((item for item in flights if item.recommended), None)
    hotel = next((item for item in hotels if item.recommended), None)
    unavailable_items = list(unavailable or [])
    if flight is None:
        unavailable_items.append("Flight price unavailable")
    if hotel is None:
        unavailable_items.append("Hotel price unavailable")
    flight_cost = flight.price_inr if flight else 0
    hotel_cost = hotel.total_cost_inr if hotel else 0
    total = flight_cost + hotel_cost + estimates.meals_inr + estimates.local_transport_inr
    notes = [estimates.notes, *unavailable_items]
    return BudgetSummary(
        flight_inr=flight_cost,
        hotel_inr=hotel_cost,
        meals_inr=estimates.meals_inr,
        local_transport_inr=estimates.local_transport_inr,
        total_estimated_inr=total,
        budget_inr=budget,
        within_budget=None if budget is None or unavailable_items else total <= budget,
        notes=" ".join(note for note in notes if note),
    )


def create_budget_node(
    model: BaseChatModel,
) -> Callable[[TripMateState], Awaitable[dict[str, object]]]:
    """Create a budget ReAct agent; no external MCP tools are needed."""
    agent = create_react_agent(
        model,
        [],
        prompt=BUDGET_SYSTEM_PROMPT,
        response_format=BudgetEstimates,
        name="tripmate_budget_agent",
    )

    async def budget_node(state: TripMateState) -> dict[str, object]:
        """Estimate variable costs and add exact MCP-returned prices."""
        unavailable = [
            message
            for message in (
                state.get("flight_error"),
                state.get("hotel_error"),
                state.get("itinerary_error"),
            )
            if message
        ]
        try:
            result = await agent.ainvoke(
                {
                    "messages": [
                        (
                            "user",
                            f"Destination: {state.get('destination', 'unknown')}\n"
                            f"Duration in days: {state.get('duration_days', 1)}\n"
                            f"Budget style and preferences: {state.get('user_preferences', {})}\n"
                            f"Itinerary: {state.get('itinerary', [])}",
                        )
                    ]
                },
                config={
                    "recursion_limit": 6,
                    "run_name": "tripmate.budget_agent",
                    "tags": ["tripmate", "agent", "budget_agent"],
                },
            )
            estimates = cast(BudgetEstimates, result["structured_response"])
            error = None
        except Exception as exception:
            estimates = BudgetEstimates(
                meals_inr=0,
                local_transport_inr=0,
                notes="Meal and transport estimates are unavailable.",
            )
            error = f"Budget agent unavailable ({type(exception).__name__})."
            unavailable.append(error)
        summary = calculate_budget_summary(
            state.get("flights", []),
            state.get("hotels", []),
            estimates,
            state.get("budget"),
            unavailable,
        )
        return {"budget_summary": summary, "budget_error": error}

    return budget_node