"""Human approval interrupt for proposed booking actions."""

from typing import Any

from langgraph.types import interrupt

from tripmate.state import BookingApprovalResponse, TripMateState


def booking_approval_node(state: TripMateState) -> dict[str, Any]:
    """Pause for approve/edit/reject before any booking integration could run."""
    answer = interrupt(
        {
            "title": "Review booking proposal",
            "proposed_action": state.get("booking_proposal", {}),
            "choices": ["approve", "edit", "reject"],
            "notice": "No booking provider is connected; approval will not place a booking.",
        },
        response_schema=BookingApprovalResponse,
    )
    decision = (
        answer
        if isinstance(answer, BookingApprovalResponse)
        else BookingApprovalResponse.model_validate(answer)
    )
    proposal = dict(state.get("booking_proposal", {}))
    if decision.decision == "edit" and decision.edited_action:
        proposal["description"] = decision.edited_action
    elif decision.decision == "edit":
        return {
            "booking_decision": "reject",
            "booking_result_message": "The edited proposal was empty and was rejected.",
        }

    messages = {
        "approve": "Approval recorded; no booking was placed because no provider is connected.",
        "edit": "The edited proposal was approved; no booking was placed because no provider is connected.",
        "reject": "The booking proposal was rejected; no booking was placed.",
    }
    return {
        "booking_decision": decision.decision,
        "booking_proposal": proposal,
        "booking_result_message": messages[decision.decision],
    }