"""Groq-backed supervisor that routes planning work one agent at a time."""

import json
from collections.abc import Awaitable, Callable
from typing import cast

from langchain_core.language_models import BaseChatModel

from tripmate.state import SupervisorDecision, TripMateState

SUPERVISOR_SYSTEM_PROMPT = """You are TripMate's travel-planning supervisor. Choose exactly
one next step: flight_agent, hotel_agent, itinerary_agent, budget_agent, or FINISH.
Use completed_agents and existing results to avoid repeating work. Route only agents
needed to answer the user's travel request, then choose FINISH. Set request_booking to
true only when the user explicitly asks to book, reserve, or purchase something. Never
claim an action is booked; bookings require separate human approval and no booking
provider is connected."""


def create_supervisor_node(
    model: BaseChatModel,
    max_iterations: int,
) -> Callable[[TripMateState], Awaitable[dict[str, object]]]:
    """Create the structured-output supervisor with a strict iteration ceiling."""
    structured_model = model.with_structured_output(SupervisorDecision)

    async def supervisor(state: TripMateState) -> dict[str, object]:
        """Select the next specialized agent or finish the turn."""
        iterations = state.get("iteration_count", 0)
        if iterations >= max_iterations:
            return {
                "route_to": "FINISH",
                "iteration_count": iterations,
                "supervisor_note": "Stopped at the maximum agent-iteration limit.",
            }

        context = {
            "request": state["user_request"],
            "history": state.get("conversation_history", [])[-6:],
            "destination": state.get("destination"),
            "dates": state.get("dates"),
            "budget_inr": state.get("budget"),
            "completed_agents": state.get("completed_agents", []),
            "flight_options": [item.model_dump() for item in state.get("flights", [])],
            "hotel_options": [item.model_dump() for item in state.get("hotels", [])],
            "itinerary": [item.model_dump() for item in state.get("itinerary", [])],
            "budget_summary": (
                state["budget_summary"].model_dump()
                if state.get("budget_summary")
                else None
            ),
            "preferences": state.get("user_preferences", {}),
        }
        try:
            decision = cast(
                SupervisorDecision,
                await structured_model.ainvoke(
                    [
                        ("system", SUPERVISOR_SYSTEM_PROMPT),
                        ("human", json.dumps(context, ensure_ascii=False)),
                    ],
                    config={
                        "run_name": "tripmate.supervisor",
                        "tags": ["tripmate", "supervisor"],
                    },
                ),
            )
        except Exception as error:
            return {
                "route_to": "FINISH",
                "iteration_count": iterations + 1,
                "supervisor_note": (
                    "Supervisor could not choose another step; returning the "
                    f"available draft ({type(error).__name__})."
                ),
            }

        next_agent = decision.next_agent
        if next_agent in state.get("completed_agents", []):
            next_agent = "FINISH"
            note = "Stopped to avoid repeating a completed agent."
        else:
            note = decision.rationale
        return {
            "route_to": next_agent,
            "iteration_count": iterations + 1,
            "booking_requested": (
                state.get("booking_requested", False) or decision.request_booking
            ),
            "supervisor_note": note,
        }

    return supervisor