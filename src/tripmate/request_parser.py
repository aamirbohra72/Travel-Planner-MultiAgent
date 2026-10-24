"""Structured extraction of destination, dates, and budget from a request."""

import re
from collections.abc import Callable
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from tripmate.state import TripMateState, TripRequestDetails

PARSER_SYSTEM_PROMPT = """Extract travel details from the user's request. Return the
destination, the stated dates or duration, duration_days, and total budget in INR.
For relative durations such as '5-day trip', set dates to '5 days (dates not
specified)' and duration_days to 5. Convert a stated INR budget to a number. If no
budget is stated, use null. If a budget range is given, use its upper end as the budget
cap. Do not invent calendar dates."""

_BUDGET_RANGE = re.compile(
    r"(?P<first>\d[\d,]*(?:\.\d+)?)\s*(?P<first_unit>thousand|lakhs?|k|l)?\s*"
    r"(?:to|[-\u2013\u2014])\s*(?P<second>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<second_unit>thousand|lakhs?|k|l)?(?![a-z])",
    re.IGNORECASE,
)
_BUDGET_CONTEXT = re.compile(
    r"\b(?:budget|affordable|price|cost|inr|rs|rupees?|per\s+person)\b|₹",
    re.IGNORECASE,
)
_PER_PERSON = re.compile(r"\bper\s+(?:person|travell?er|head)\b", re.IGNORECASE)


def _budget_range_upper_bound(request: str) -> float | None:
    """Return the upper INR limit when a request states a recognizable budget range."""
    if not _BUDGET_CONTEXT.search(request):
        return None
    request_without_dates = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", request)
    match = _BUDGET_RANGE.search(request_without_dates)
    if match is None:
        return None

    first_unit = (match.group("first_unit") or match.group("second_unit") or "").casefold()
    second_unit = (match.group("second_unit") or match.group("first_unit") or "").casefold()

    def amount(value: str, unit: str) -> float:
        multiplier = 100_000 if unit in {"l", "lakh", "lakhs"} else 1_000 if unit in {"k", "thousand"} else 1
        return float(value.replace(",", "")) * multiplier

    return amount(match.group("second"), second_unit)


def create_request_parser_node(
    model: BaseChatModel,
) -> Callable[[TripMateState], dict[str, object]]:
    """Create a node that extracts request fields using Pydantic structured output."""
    chain = ChatPromptTemplate.from_messages(
        [
            ("system", PARSER_SYSTEM_PROMPT),
            ("human", "Travel request: {user_request}"),
        ]
    ) | model.with_structured_output(TripRequestDetails)

    def parse_request(state: TripMateState) -> dict[str, object]:
        """Extract the typed trip details from the user's free text."""
        details = cast(
            TripRequestDetails,
            chain.invoke(
                {"user_request": state["user_request"]},
                config={
                    "run_name": "tripmate.request_parser",
                    "tags": ["tripmate", "request_parser"],
                },
            ),
        )
        budget = _budget_range_upper_bound(state["user_request"])
        return {
            "destination": details.destination,
            "dates": details.dates,
            "duration_days": details.duration_days,
            "budget": details.budget if budget is None else budget,
            "budget_per_person": bool(_PER_PERSON.search(state["user_request"])),
        }

    return parse_request