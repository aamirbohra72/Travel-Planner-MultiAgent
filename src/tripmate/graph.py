"""Supervisor, guardrails, agents, and persistence wiring for TripMate."""

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from tripmate.agents.budget_agent import create_budget_node
from tripmate.agents.flight_agent import create_flight_node
from tripmate.agents.hotel_agent import create_hotel_node
from tripmate.agents.itinerary_agent import create_itinerary_node
from tripmate.agents.supervisor import create_supervisor_node
from tripmate.approval import booking_approval_node
from tripmate.config import Settings, get_settings
from tripmate.guardrails import (
    create_input_guardrail_node,
    create_output_guardrail_node,
)
from tripmate.llm import create_chat_model
from tripmate.mcp_tools import load_agent_toolsets
from tripmate.request_parser import create_request_parser_node
from tripmate.state import BudgetSummary, TripMateContext, TripMateState


def _begin_turn(state: TripMateState) -> dict[str, object]:
    """Reset per-turn fields while retaining this thread's conversation history."""
    return {
        "conversation_history": [
            *state.get("conversation_history", []),
            state["user_request"],
        ],
        "completed_agents": [],
        "iteration_count": 0,
        "route_to": None,
        "supervisor_note": "",
        "booking_requested": False,
        "booking_proposal": {},
        "booking_decision": None,
        "booking_result_message": "",
        "destination": "",
        "dates": "",
        "duration_days": 1,
        "budget": None,
        "budget_per_person": False,
        "flights": [],
        "hotels": [],
        "itinerary": [],
        "itinerary_notes": "",
        "budget_summary": None,
        "flight_error": None,
        "hotel_error": None,
        "itinerary_error": None,
        "budget_error": None,
        "final_plan": "",
        "final_answer": "",
    }


def _prepare_final(state: TripMateState) -> dict[str, object]:
    """Build a grounded summary and, when requested, a reviewable proposal."""
    flight = next(
        (item for item in state.get("flights", []) if item.recommended), None
    )
    hotel = next(
        (item for item in state.get("hotels", []) if item.recommended), None
    )
    summary: BudgetSummary | None = state.get("budget_summary")
    budget_scope = " per person" if state.get("budget_per_person") else ""
    if summary is None:
        flight_cost = flight.price_inr if flight else 0
        hotel_cost = hotel.total_cost_inr if hotel else 0
        budget_label = (
            f"INR {state['budget']:,.0f}{budget_scope}"
            if state.get("budget") is not None
            else "Not specified"
        )
        total_line = "Budget total unavailable because the estimate could not be completed."
        status = "Budget not verified."
        budget_notes = ""
    else:
        flight_cost = summary.flight_inr
        hotel_cost = summary.hotel_inr
        budget_label = (
            f"INR {summary.budget_inr:,.0f}{budget_scope}"
            if summary.budget_inr is not None
            else "Not specified"
        )
        total_line = f"Total known estimate: INR {summary.total_estimated_inr:,.0f}"
        if summary.within_budget is True:
            status = "Within budget."
        elif summary.within_budget is False:
            status = f"Over budget by INR {summary.total_estimated_inr - (summary.budget_inr or 0):,.0f}."
        else:
            status = "Budget could not be confirmed because some information is unavailable."
        budget_notes = summary.notes

    flight_line = (
        f"{flight.airline} {flight.flight_number}, {flight.departure}, "
        f"{flight.arrival} (INR {flight.price_inr:,.0f})"
        if flight
        else f"Unavailable: {state.get('flight_error') or 'no option returned.'}"
    )
    hotel_line = (
        f"{hotel.name}, {hotel.area} (INR {hotel.total_cost_inr:,.0f} for the stay)"
        if hotel
        else f"Unavailable: {state.get('hotel_error') or 'no option returned.'}"
    )
    itinerary = "\n".join(
        f"Day {day.day}: {day.title}\n"
        + "\n".join(f"  - {activity}" for activity in day.activities)
        for day in state.get("itinerary", [])
    ) or "Itinerary unavailable."
    issues = [
        state.get("flight_error"),
        state.get("hotel_error"),
        state.get("itinerary_error"),
        state.get("budget_error"),
        state.get("supervisor_note"),
        state.get("itinerary_notes"),
        budget_notes,
    ]
    fallback_notes = "\n".join(f"- {issue}" for issue in issues if issue)
    planning_notes = (
        f"\nPlanning notes\n{fallback_notes}" if fallback_notes else ""
    )
    plan = f"""TripMate Plan
Destination: {state.get('destination', 'Unknown')}
Dates: {state.get('dates', 'Not specified')}
Budget: {budget_label}

Recommended flight: {flight_line}
Recommended hotel: {hotel_line}

Itinerary
{itinerary}

Estimated budget
Flight: INR {flight_cost:,.0f}
Hotel: INR {hotel_cost:,.0f}
{total_line}
Status: {status}
{f'Notes: {budget_notes}' if budget_notes else ''}
{planning_notes}

These are recommendations, not bookings. Verify details with providers before travel."""
    plan += (
        "\n\nData note: Flight, hotel, and place results are mock samples, not live "
        "quotes or availability. Verify actual prices, schedules, and availability "
        "with providers."
    )

    booking_requested = state.get("booking_requested", False)
    proposal: dict[str, Any] = {}
    if booking_requested:
        proposal = {
            "description": "Request to book the recommended trip options",
            "flight": flight.model_dump() if flight else None,
            "hotel": hotel.model_dump() if hotel else None,
            "estimated_total_inr": (
                summary.total_estimated_inr if summary else None
            ),
            "execution": "Approval only; no booking provider is connected.",
        }
        plan += "\nA booking request needs your approval. No booking has been placed."
    return {"final_plan": plan, "booking_proposal": proposal}


def _final_answer(state: TripMateState) -> dict[str, str]:
    """Expose the reviewed answer to API and UI callers."""
    return {"final_answer": state.get("final_plan", "")}


def _route_input(state: TripMateState) -> str:
    return "parse_request" if state.get("input_classification") == "travel" else "output_guardrail"


def _route_supervisor(state: TripMateState) -> str:
    route = state.get("route_to")
    if route in {"flight_agent", "hotel_agent", "itinerary_agent", "budget_agent"}:
        return route
    return "prepare_final"


def _route_booking(state: TripMateState) -> str:
    return "booking_approval" if state.get("booking_requested") else "output_guardrail"


def _track_completion(
    agent_name: str,
    node: Callable[[TripMateState], Any],
) -> Callable[[TripMateState], Any]:
    """Record each completed specialist so the supervisor can prevent repeats."""

    async def tracked_node(state: TripMateState) -> dict[str, object]:
        update = await node(state)
        return {
            **update,
            "completed_agents": [*state.get("completed_agents", []), agent_name],
        }

    return tracked_node


def _build_workflow(
    *,
    input_guardrail: Callable[..., Any],
    parse_request: Callable[..., Any],
    supervisor: Callable[..., Any],
    flight_agent: Callable[..., Any],
    hotel_agent: Callable[..., Any],
    itinerary_agent: Callable[..., Any],
    budget_agent: Callable[..., Any],
    output_guardrail: Callable[..., Any],
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
) -> CompiledStateGraph:
    """Build the state machine; injected nodes keep control-flow tests offline."""
    graph = StateGraph(TripMateState, context_schema=TripMateContext)
    graph.add_node("input_guardrail", input_guardrail)
    graph.add_node("begin_turn", _begin_turn)
    graph.add_node("parse_request", parse_request)
    graph.add_node("supervisor", supervisor)
    graph.add_node("flight_agent", flight_agent)
    graph.add_node("hotel_agent", hotel_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("budget_agent", budget_agent)
    graph.add_node("prepare_final", _prepare_final)
    graph.add_node("booking_approval", booking_approval_node)
    graph.add_node("output_guardrail", output_guardrail)
    graph.add_node("final_answer", _final_answer)

    graph.add_edge(START, "begin_turn")
    graph.add_edge("begin_turn", "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail",
        _route_input,
        {"parse_request": "parse_request", "output_guardrail": "output_guardrail"},
    )
    graph.add_edge("parse_request", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_supervisor,
        {
            "flight_agent": "flight_agent",
            "hotel_agent": "hotel_agent",
            "itinerary_agent": "itinerary_agent",
            "budget_agent": "budget_agent",
            "prepare_final": "prepare_final",
        },
    )
    for agent_name in (
        "flight_agent",
        "hotel_agent",
        "itinerary_agent",
        "budget_agent",
    ):
        graph.add_edge(agent_name, "supervisor")
    graph.add_conditional_edges(
        "prepare_final",
        _route_booking,
        {"booking_approval": "booking_approval", "output_guardrail": "output_guardrail"},
    )
    graph.add_edge("booking_approval", "output_guardrail")
    graph.add_edge("output_guardrail", "final_answer")
    graph.add_edge("final_answer", END)
    return graph.compile(checkpointer=checkpointer, store=store)


async def build_graph(
    *,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """Load MCP tools and compile a guarded, checkpointed supervisor workflow."""
    app_settings = settings or get_settings()
    chat_model = model if model is not None else create_chat_model(app_settings)
    toolsets = await load_agent_toolsets(app_settings)
    return _build_workflow(
        input_guardrail=create_input_guardrail_node(chat_model),
        parse_request=create_request_parser_node(chat_model),
        supervisor=create_supervisor_node(
            chat_model, app_settings.tripmate_max_agent_iterations
        ),
        flight_agent=_track_completion(
            "flight_agent",
            create_flight_node(
                chat_model,
                toolsets["flight"],
                app_settings.tripmate_default_origin,
            ),
        ),
        hotel_agent=_track_completion(
            "hotel_agent", create_hotel_node(chat_model, toolsets["hotel"])
        ),
        itinerary_agent=_track_completion(
            "itinerary_agent",
            create_itinerary_node(chat_model, toolsets["itinerary"]),
        ),
        budget_agent=_track_completion(
            "budget_agent", create_budget_node(chat_model)
        ),
        output_guardrail=create_output_guardrail_node(chat_model),
        checkpointer=checkpointer,
        store=store,
    )