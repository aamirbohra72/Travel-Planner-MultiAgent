"""Persistence and human-approval tests for Stage 3."""

import asyncio

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from tripmate.approval import booking_approval_node
from tripmate.persistence import SQLitePreferenceStore
from tripmate.state import TripMateState


def _approval_graph(checkpointer: AsyncSqliteSaver):
    builder = StateGraph(TripMateState)
    builder.add_node("booking_approval", booking_approval_node)
    builder.add_edge(START, "booking_approval")
    builder.add_edge("booking_approval", END)
    return builder.compile(checkpointer=checkpointer)


def test_preferences_survive_store_recreation(tmp_path) -> None:
    database = str(tmp_path / "preferences.sqlite")
    namespace = ("test-user", "preferences")

    asyncio.run(
        SQLitePreferenceStore(database).aput(
            namespace,
            "profile",
            {"budget_style": "budget-conscious", "food": "vegetarian", "pace": "relaxed"},
        )
    )
    saved = asyncio.run(SQLitePreferenceStore(database).aget(namespace, "profile"))

    assert saved is not None
    assert saved.value["food"] == "vegetarian"
    assert saved.value["pace"] == "relaxed"


@pytest.mark.parametrize(
    ("resume", "expected_decision", "expected_description"),
    [
        ({"decision": "approve"}, "approve", "Book the recommended flight and hotel"),
        (
            {"decision": "edit", "edited_action": "Use the earlier flight instead"},
            "edit",
            "Use the earlier flight instead",
        ),
        ({"decision": "reject"}, "reject", "Book the recommended flight and hotel"),
    ],
)
def test_sqlite_checkpoint_resumes_approval_after_reopen(
    tmp_path,
    resume: dict[str, str],
    expected_decision: str,
    expected_description: str,
) -> None:
    database = str(tmp_path / f"{expected_decision}.sqlite")
    config = {"configurable": {"thread_id": "approval-thread"}}
    initial_state = {
        "user_request": "Book the recommended trip",
        "booking_proposal": {
            "description": "Book the recommended flight and hotel",
            "estimated_total_inr": 30000,
        },
    }

    async def exercise_checkpoint() -> dict[str, object]:
        async with AsyncSqliteSaver.from_conn_string(database) as checkpointer:
            await checkpointer.setup()
            graph = _approval_graph(checkpointer)
            paused = await graph.ainvoke(initial_state, config=config)
            assert paused["__interrupt__"]

        async with AsyncSqliteSaver.from_conn_string(database) as checkpointer:
            await checkpointer.setup()
            graph = _approval_graph(checkpointer)
            return await graph.ainvoke(Command(resume=resume), config=config)

    resumed = asyncio.run(exercise_checkpoint())

    assert resumed["booking_decision"] == expected_decision
    assert resumed["booking_proposal"]["description"] == expected_description
    assert "no booking was placed" in resumed["booking_result_message"].lower()