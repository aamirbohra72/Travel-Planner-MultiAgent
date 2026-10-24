"""Pydantic schemas and typed state shared by TripMate graph nodes."""

from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field


class TripRequestDetails(BaseModel):
    """Travel details extracted from a free-text request."""

    destination: str = Field(description="The primary destination city or region.")
    dates: str = Field(
        description="Travel date range, or a duration if exact dates are not given."
    )
    duration_days: int = Field(
        ge=1,
        description="Number of days in the trip; use 1 when no duration is stated.",
    )
    budget: float | None = Field(
        description="Total trip budget in INR, or null if the user gave no budget."
    )


class FlightOption(BaseModel):
    """A flight option returned by the MCP flight search tool."""

    option_id: str
    airline: str
    flight_number: str
    departure: str
    arrival: str
    price_inr: float = Field(ge=0)
    recommended: bool = False


class HotelOption(BaseModel):
    """A hotel option returned by the MCP hotel search tool."""

    option_id: str
    name: str
    area: str
    nightly_rate_inr: float = Field(ge=0)
    total_cost_inr: float = Field(ge=0)
    rating: float = Field(ge=0, le=5)
    recommended: bool = False


class ItineraryDay(BaseModel):
    """Activities for one day of the proposed trip."""

    day: int = Field(ge=1)
    title: str
    activities: list[str]


class FlightRecommendation(BaseModel):
    """Structured flight-agent response containing tool results and selection."""

    options: list[FlightOption] = Field(default_factory=list)
    option_id: str | None = None
    rationale: str


class HotelRecommendation(BaseModel):
    """Structured hotel-agent response containing tool results and selection."""

    options: list[HotelOption] = Field(default_factory=list)
    option_id: str | None = None
    rationale: str


class ItineraryDraft(BaseModel):
    """Structured day-by-day itinerary returned by the itinerary agent."""

    days: list[ItineraryDay]
    notes: str = ""


class BudgetEstimates(BaseModel):
    """Variable trip costs estimated by the budget agent."""

    meals_inr: float = Field(ge=0)
    local_transport_inr: float = Field(ge=0)
    notes: str


class BudgetSummary(BaseModel):
    """Deterministic total and budget comparison for the proposed trip."""

    flight_inr: float
    hotel_inr: float
    meals_inr: float
    local_transport_inr: float
    total_estimated_inr: float
    budget_inr: float | None
    within_budget: bool | None
    notes: str


class InputGuardrailResult(BaseModel):
    """Classification of the user's request before it reaches planning agents."""

    classification: Literal["travel", "off_topic", "unsafe"]
    reason: str


class SupervisorDecision(BaseModel):
    """One supervisor routing decision for the current planning turn."""

    next_agent: Literal[
        "flight_agent",
        "hotel_agent",
        "itinerary_agent",
        "budget_agent",
        "FINISH",
    ]
    request_booking: bool = False
    rationale: str


class OutputGuardrailResult(BaseModel):
    """Review outcome for a proposed final answer."""

    action: Literal["pass", "rewrite", "reject"]
    reviewed_response: str
    reason: str


class BookingApprovalResponse(BaseModel):
    """Human decision returned when resuming a booking approval interrupt."""

    decision: Literal["approve", "edit", "reject"]
    edited_action: str | None = None


@dataclass
class TripMateContext:
    """Per-user runtime context used to namespace long-term preferences."""

    user_id: str


class TripMateState(TypedDict):
    """Typed data passed between TripMate's graph nodes."""

    user_request: str
    destination: NotRequired[str]
    dates: NotRequired[str]
    duration_days: NotRequired[int]
    budget: NotRequired[float | None]
    budget_per_person: NotRequired[bool]
    flights: NotRequired[list[FlightOption]]
    hotels: NotRequired[list[HotelOption]]
    itinerary: NotRequired[list[ItineraryDay]]
    budget_summary: NotRequired[BudgetSummary | None]
    flight_error: NotRequired[str | None]
    hotel_error: NotRequired[str | None]
    itinerary_error: NotRequired[str | None]
    itinerary_notes: NotRequired[str]
    budget_error: NotRequired[str | None]
    conversation_history: NotRequired[list[str]]
    user_preferences: NotRequired[dict[str, Any]]
    input_classification: NotRequired[Literal["travel", "off_topic", "unsafe"]]
    guardrail_reason: NotRequired[str]
    completed_agents: NotRequired[list[str]]
    iteration_count: NotRequired[int]
    route_to: NotRequired[str | None]
    supervisor_note: NotRequired[str]
    booking_requested: NotRequired[bool]
    booking_proposal: NotRequired[dict[str, Any]]
    booking_decision: NotRequired[Literal["approve", "edit", "reject"] | None]
    booking_result_message: NotRequired[str]
    output_guardrail_action: NotRequired[Literal["pass", "rewrite", "reject"]]
    final_answer: NotRequired[str]
    final_plan: NotRequired[str]