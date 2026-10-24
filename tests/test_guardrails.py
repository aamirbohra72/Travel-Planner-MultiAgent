"""Regression tests for input and output safety checks."""

import asyncio
from typing import Any

from tripmate.guardrails import create_output_guardrail_node
from tripmate.state import OutputGuardrailResult


class FakeOutputModel:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def with_structured_output(self, schema: type[Any]) -> "FakeOutputRunnable":
        return FakeOutputRunnable(self, schema)


class FakeOutputRunnable:
    def __init__(self, model: FakeOutputModel, schema: type[Any]) -> None:
        self.model = model
        self.schema = schema

    async def ainvoke(
        self, messages: list[tuple[str, str]], **_: Any
    ) -> OutputGuardrailResult:
        self.model.messages = messages
        return self.schema(
            action="pass",
            reviewed_response="",
            reason="Safe plan; budget is clearly unspecified.",
        )


def test_output_guardrail_allows_safe_plan_without_numeric_budget() -> None:
    model = FakeOutputModel()
    output_guardrail = create_output_guardrail_node(model)  # type: ignore[arg-type]

    result = asyncio.run(
        output_guardrail({"final_plan": "A safe seven-day Thailand itinerary."})
    )

    assert result["output_guardrail_action"] == "pass"
    assert "Do not reject an otherwise safe and useful plan solely" in model.messages[0][1]
    assert "Budget in INR: None" in model.messages[1][1]
    assert "Budget was not specified" in result["final_plan"]
    assert "affordability cannot be confirmed" in result["final_plan"]