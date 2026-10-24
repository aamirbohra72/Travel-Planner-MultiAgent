"""Offline tests for Stage 3 routing and deterministic budget behavior."""

from tripmate.agents.budget_agent import calculate_budget_summary
from tripmate.graph import (
    _begin_turn,
    _route_booking,
    _route_input,
    _route_supervisor,
)
from tripmate.state import BudgetEstimates, FlightOption, HotelOption


def _options() -> tuple[list[FlightOption], list[HotelOption]]:
    flights = [
        FlightOption(
            option_id="f1",
            airline="IndiGo",
            flight_number="6E 512",
            departure="Delhi to Goa",
            arrival="Nonstop",
            price_inr=9500,
            recommended=True,
        )
    ]
    hotels = [
        HotelOption(
            option_id="h1",
            name="Palm Grove",
            area="Goa",
            nightly_rate_inr=2500,
            total_cost_inr=10000,
            rating=4.0,
            recommended=True,
        )
    ]
    return flights, hotels


def test_budget_helper_totals_and_flags_overspending() -> None:
    flights, hotels = _options()
    estimates = BudgetEstimates(
        meals_inr=8000,
        local_transport_inr=2500,
        notes="Sample variable-cost estimates.",
    )

    within = calculate_budget_summary(flights, hotels, estimates, budget=40000)
    over = calculate_budget_summary(flights, hotels, estimates, budget=25000)

    assert within.total_estimated_inr == 30000
    assert within.within_budget is True
    assert over.within_budget is False


def test_input_guardrail_routes_only_travel_requests_to_parser() -> None:
    assert _route_input({"input_classification": "travel"}) == "parse_request"
    assert _route_input({"input_classification": "off_topic"}) == "output_guardrail"
    assert _route_input({"input_classification": "unsafe"}) == "output_guardrail"


def test_supervisor_routes_agents_or_finishes_safely() -> None:
    assert _route_supervisor({"route_to": "hotel_agent"}) == "hotel_agent"
    assert _route_supervisor({"route_to": "FINISH"}) == "prepare_final"
    assert _route_supervisor({"route_to": "unknown"}) == "prepare_final"


def test_booking_approval_is_conditional() -> None:
    assert _route_booking({"booking_requested": True}) == "booking_approval"
    assert _route_booking({"booking_requested": False}) == "output_guardrail"


def test_new_turn_clears_previous_results_and_keeps_history() -> None:
    update = _begin_turn(
        {
            "user_request": "Change the destination to Pune",
            "conversation_history": ["Plan a trip to Goa"],
            "completed_agents": ["flight_agent"],
            "iteration_count": 4,
            "booking_decision": "approve",
            "budget": 25000,
        }
    )

    assert update["conversation_history"] == [
        "Plan a trip to Goa",
        "Change the destination to Pune",
    ]
    assert update["completed_agents"] == []
    assert update["iteration_count"] == 0
    assert update["booking_decision"] is None
    assert update["budget"] is None