"""CLI for checkpointed TripMate conversations and booking approvals."""

import argparse
import asyncio
from uuid import uuid4

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from tripmate.config import get_settings
from tripmate.graph import build_graph
from tripmate.persistence import SQLitePreferenceStore
from tripmate.state import TripMateContext


def _approval_payload(decision: str, edited_action: str | None) -> dict[str, str | None]:
    return {"decision": decision, "edited_action": edited_action}


async def _run(
    request: str | None,
    thread_id: str,
    user_id: str,
    checkpointer: AsyncSqliteSaver,
    store: SQLitePreferenceStore,
    resume: str | None = None,
    edited_action: str | None = None,
) -> None:
    """Build the MCP workflow and invoke or resume one persistent thread."""
    settings = get_settings()
    app = await build_graph(
        checkpointer=checkpointer,
        store=store,
        settings=settings,
    )
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": "tripmate.cli",
        "tags": ["tripmate", "cli", "conversation"],
    }
    context = TripMateContext(user_id=user_id)

    if resume:
        result = await app.ainvoke(
            Command(resume=_approval_payload(resume, edited_action)),
            config=config,
            context=context,
        )
    elif request:
        result = await app.ainvoke(
            {"user_request": request}, config=config, context=context
        )
    else:
        raise ValueError("A request or approval resume action is required.")

    pending = result.get("__interrupt__", [])
    if pending:
        interrupt_value = pending[0].value
        print(f"\n{interrupt_value.get('title', 'Approval required')}")
        print(interrupt_value.get("proposed_action", {}))
        print(interrupt_value.get("notice", ""))
        print(f"Resume thread with ID: {thread_id}")
        try:
            decision = input("Approve, edit, or reject? [a/e/r] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("Paused. Resume later with the thread ID shown above.")
            return
        if decision.startswith("a"):
            payload = _approval_payload("approve", None)
        elif decision.startswith("e"):
            edited = input("Edited proposal: ").strip()
            payload = _approval_payload("edit", edited)
        else:
            payload = _approval_payload("reject", None)
        result = await app.ainvoke(
            Command(resume=payload), config=config, context=context
        )
    print(result.get("final_answer", "The request did not produce a final answer."))


def main() -> None:
    """Run a request or resume a saved approval thread."""
    parser = argparse.ArgumentParser(description="Plan a trip with TripMate.")
    parser.add_argument("request", nargs="?", help="Travel request in free text")
    parser.add_argument("--thread-id", help="Existing thread ID to resume")
    parser.add_argument(
        "--user-id", default="cli-user", help="Preference profile namespace"
    )
    parser.add_argument(
        "--resume", choices=("approve", "edit", "reject"), help="Resume an approval"
    )
    parser.add_argument("--edit-action", help="Edited booking proposal for --resume edit")
    args = parser.parse_args()
    if args.resume and args.request:
        parser.error("do not provide a new request when resuming an approval")
    if not args.resume and not args.request:
        parser.error("provide a travel request or an approval resume action")
    if args.resume == "edit" and not args.edit_action:
        parser.error("--resume edit requires --edit-action")

    settings = get_settings()
    thread_id = args.thread_id or str(uuid4())
    store = SQLitePreferenceStore(settings.tripmate_database_path)

    async def run() -> None:
        async with AsyncSqliteSaver.from_conn_string(
            settings.tripmate_database_path
        ) as checkpointer:
            await checkpointer.setup()
            await _run(
                args.request,
                thread_id,
                args.user_id,
                checkpointer,
                store,
                args.resume,
                args.edit_action,
            )

    try:
        asyncio.run(run())
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()