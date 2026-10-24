"""Budget range parsing tests."""

from tripmate.request_parser import _budget_range_upper_bound


def test_budget_range_uses_upper_bound_and_shared_k_suffix() -> None:
    request = "Thailand, 7 days, budget hotels, 50k to 70k per person"

    assert _budget_range_upper_bound(request) == 70_000


def test_budget_range_accepts_comma_separated_rupee_amounts() -> None:
    request = "Affordable trip in INR 50,000-70,000 per person"

    assert _budget_range_upper_bound(request) == 70_000


def test_budget_range_accepts_en_dash_from_user_prompt() -> None:
    request = (
        "Plan a 7-day Thailand trip from India. My total budget is INR "
        "50,000\u201370,000 per person, including flights, hotel, food, "
        "transport, and sightseeing."
    )

    assert _budget_range_upper_bound(request) == 70_000


def test_budget_range_ignores_iso_trip_dates() -> None:
    request = (
        "Depart 2026-12-10 and return 2026-12-16. "
        "Budget INR 50,000-70,000 per person."
    )

    assert _budget_range_upper_bound(request) == 70_000


def test_dates_are_not_reported_as_budget_when_budget_is_missing() -> None:
    request = "Budget not specified. Depart 2026-12-10 and return 2026-12-16."

    assert _budget_range_upper_bound(request) is None


def test_non_budget_number_range_is_not_treated_as_a_budget() -> None:
    request = "Visit Bangkok from 10 to 17 December"

    assert _budget_range_upper_bound(request) is None