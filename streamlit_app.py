"""Streamlit chat UI for guarded, persistent TripMate conversations."""

import asyncio
from typing import Any
from uuid import uuid4

import streamlit as st
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from tripmate.config import get_settings
from tripmate.graph import build_graph
from tripmate.persistence import SQLitePreferenceStore
from tripmate.state import TripMateContext

APP_STYLE = """
<style>
[data-testid="stAppViewContainer"] {
    background:
        repeating-linear-gradient(135deg, rgba(102, 191, 184, 0.025) 0 1px, transparent 1px 42px),
        linear-gradient(118deg, rgba(17, 89, 112, 0.38), transparent 43%),
        linear-gradient(300deg, rgba(119, 72, 88, 0.13), transparent 52%),
        linear-gradient(180deg, #07141e 0%, #0a1724 58%, #0c1b25 100%);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppViewBlockContainer"] {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}
.tripmate-brand {
    color: #7fe1d2;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.tripmate-hero {
    animation: arrive 480ms ease-out both;
    padding: 2.2rem 0 1.5rem;
}
.tripmate-headline {
    color: #f2f7f5;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.65rem;
    font-weight: 700;
    line-height: 1.04;
    margin: 0.8rem 0 0.65rem;
    max-width: 820px;
}
.tripmate-subhead {
    color: #a9bac3;
    font-size: 1.08rem;
    line-height: 1.55;
    margin: 0;
    max-width: 650px;
}
.tripmate-status {
    background: rgba(245, 166, 107, 0.09);
    border: 1px solid rgba(245, 166, 107, 0.24);
    border-radius: 999px;
    color: #ffc28e;
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 700;
    margin-top: 1.1rem;
    padding: 0.38rem 0.72rem;
}
.tripmate-section-label {
    color: #9caeb8;
    font-size: 0.78rem;
    font-weight: 700;
    margin: 0.45rem 0 0.7rem;
    text-transform: uppercase;
}
div[data-testid="stForm"] {
    background: rgba(8, 19, 31, 0.76);
    border: 1px solid rgba(129, 177, 187, 0.24);
    border-radius: 14px;
    padding: 1.15rem 1.2rem 0.8rem;
}
div[data-testid="stTextArea"] textarea {
    background: #08121e;
    border-color: #29404d;
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    line-height: 1.55;
}
div[data-testid="stFormSubmitButton"] button {
    min-height: 3.2rem;
    font-weight: 700;
}
div[data-testid="stButton"] button {
    border-color: rgba(133, 171, 181, 0.34);
    background: rgba(15, 34, 46, 0.7);
}
div[data-testid="stButton"] button:hover {
    border-color: #65d7c8;
    color: #a4f2e5;
}
[data-testid="stChatMessage"] {
    background: rgba(12, 27, 39, 0.72);
    border: 1px solid rgba(129, 177, 187, 0.16);
    border-radius: 12px;
}
@keyframes arrive {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
@media (max-width: 640px) {
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .tripmate-hero { padding-top: 1.3rem; }
    .tripmate-headline { font-size: 2.55rem; }
    .tripmate-subhead { font-size: 0.98rem; }
    div[data-testid="stForm"] { padding: 0.85rem; }
}
</style>
"""

QUICK_PROMPTS = {
        "Thailand on a budget": (
                "Plan a 7-day Thailand trip from Delhi under INR 70,000 per person, "
                "including flights, budget hotels, food, local transport, and sightseeing."
        ),
        "Japan city break": "Plan a 7-day Japan trip with a moderate budget and public transit.",
        "Dubai getaway": "Plan a 5-day Dubai trip with affordable hotels and sightseeing.",
        "Beach escape": "Suggest a relaxed 6-day beach trip with budget stays and local food.",
}


def _set_trip_prompt(prompt: str) -> None:
        st.session_state.trip_request = prompt


async def _stream_graph(
    graph: Any,
    graph_input: dict[str, Any] | Command,
    config: dict[str, Any],
    context: TripMateContext,
    progress: Any,
) -> dict[str, Any]:
    """Drive graph execution and report completed nodes to the UI."""
    async for update in graph.astream(
        graph_input,
        config=config,
        context=context,
        stream_mode="updates",
    ):
        for node_name in update:
            if node_name != "__interrupt__":
                progress.write(f"Completed: {node_name.replace('_', ' ').title()}")
    return (await graph.aget_state(config)).values


async def _pending_approval(
    graph: Any, config: dict[str, Any]
) -> dict[str, Any] | None:
    """Read a persisted interrupt payload from the current thread snapshot."""
    snapshot = await graph.aget_state(config)
    for task in snapshot.tasks:
        for pending in task.interrupts:
            if isinstance(pending.value, dict):
                return pending.value
    return None


async def _run_with_progress(
    graph: Any,
    graph_input: dict[str, Any] | Command,
    config: dict[str, Any],
    context: TripMateContext,
) -> dict[str, Any]:
    with st.status("TripMate is planning…", expanded=True) as progress:
        values = await _stream_graph(graph, graph_input, config, context, progress)
        pending = await _pending_approval(graph, config)
        if pending:
            progress.update(label="Waiting for your approval", state="running")
        else:
            progress.update(label="Planning complete", state="complete")
    return values


async def _main() -> None:
    """Render chat, saved preferences, progress, and approval controls."""
    st.set_page_config(
        page_title="TripMate | Travel planner",
        page_icon=":material/travel_explore:",
        layout="wide",
    )
    st.markdown(APP_STYLE, unsafe_allow_html=True)
    settings = get_settings()

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = st.query_params.get("thread_id", str(uuid4()))
        st.query_params["thread_id"] = st.session_state.thread_id
    if "user_id" not in st.session_state:
        st.session_state.user_id = settings.tripmate_profile_id
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "trip_request" not in st.session_state:
        st.session_state.trip_request = ""

    store = SQLitePreferenceStore(settings.tripmate_database_path)
    namespace = (st.session_state.user_id, "preferences")
    saved_item = await store.aget(namespace, "profile")
    preferences = saved_item.value if saved_item else {}

    with st.sidebar:
        st.subheader("Travel preferences")
        with st.form("preferences_form"):
            budget_style = st.selectbox(
                "Budget style",
                ["balanced", "budget-conscious", "comfort-first"],
                index=["balanced", "budget-conscious", "comfort-first"].index(
                    preferences.get("budget_style", "balanced")
                )
                if preferences.get("budget_style", "balanced")
                in ["balanced", "budget-conscious", "comfort-first"]
                else 0,
            )
            food = st.text_input("Food preferences", preferences.get("food", ""))
            pace = st.selectbox(
                "Travel pace",
                ["relaxed", "balanced", "packed"],
                index=["relaxed", "balanced", "packed"].index(
                    preferences.get("pace", "balanced")
                )
                if preferences.get("pace", "balanced") in ["relaxed", "balanced", "packed"]
                else 1,
            )
            save_preferences = st.form_submit_button("Save preferences")
        if save_preferences:
            await store.aput(
                namespace,
                "profile",
                {
                    "budget_style": budget_style,
                    "food": food.strip(),
                    "pace": pace,
                },
            )
            st.success("Preferences saved.")
        st.caption(f"Profile: {st.session_state.user_id}")
        st.caption(f"Thread: {st.session_state.thread_id}")
        if st.button("New conversation"):
            st.session_state.thread_id = str(uuid4())
            st.query_params["thread_id"] = st.session_state.thread_id
            st.session_state.messages = []
            st.session_state.trip_request = ""
            st.rerun()

    st.markdown(
        """
        <div class="tripmate-hero">
          <div class="tripmate-brand">TripMate <span style="color:#637d88">/</span> travel planner</div>
          <h1 class="tripmate-headline">Where to next?</h1>
          <p class="tripmate-subhead">A thoughtful itinerary, shaped around your budget and pace.</p>
          <span class="tripmate-status">SAMPLE SEARCH DATA · VERIFY LIVE PRICES</span>
        </div>
        <div class="tripmate-section-label">Start with an idea</div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(horizontal=True, wrap=True):
        for label, prompt in QUICK_PROMPTS.items():
            st.button(
                label,
                key=f"prompt-{label}",
                on_click=_set_trip_prompt,
                args=(prompt,),
            )

    request: str | None = None
    with st.form("trip_request_form"):
        input_column, submit_column = st.columns(
            [5, 1], vertical_alignment="bottom"
        )
        with input_column:
            st.text_area(
                "Trip request",
                key="trip_request",
                height=132,
                label_visibility="collapsed",
                placeholder=(
                    "Tell us your destination, departure city, trip length, "
                    "budget, and what you like to do..."
                ),
            )
        with submit_column:
            submitted = st.form_submit_button(
                "Generate plan",
                type="primary",
                icon=":material/arrow_forward:",
                width="stretch",
            )
        if submitted:
            request = st.session_state.trip_request.strip()
            if not request:
                st.warning("Add a few trip details before generating a plan.")

    config = {
        "configurable": {"thread_id": st.session_state.thread_id},
        "run_name": "tripmate.streamlit",
        "tags": ["tripmate", "streamlit", "conversation"],
    }
    context = TripMateContext(user_id=st.session_state.user_id)

    try:
        async with AsyncSqliteSaver.from_conn_string(
            settings.tripmate_database_path
        ) as checkpointer:
            await checkpointer.setup()
            graph = await build_graph(
                checkpointer=checkpointer,
                store=store,
                settings=settings,
            )

            if st.session_state.messages:
                st.markdown('<div class="tripmate-section-label">Your trip</div>', unsafe_allow_html=True)
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            pending = await _pending_approval(graph, config)
            if pending:
                st.subheader(pending.get("title", "Approval required"))
                st.json(pending.get("proposed_action", {}))
                st.info(pending.get("notice", "Review the proposed action."))
                edited_action = st.text_area(
                    "Edit proposed action", key=f"edit-{st.session_state.thread_id}"
                )
                approve_col, edit_col, reject_col = st.columns(3)
                resume_value: dict[str, str | None] | None = None
                if approve_col.button("Approve", type="primary"):
                    resume_value = {"decision": "approve", "edited_action": None}
                elif edit_col.button("Save edit"):
                    if edited_action.strip():
                        resume_value = {
                            "decision": "edit",
                            "edited_action": edited_action.strip(),
                        }
                    else:
                        st.warning("Enter an edited proposal before saving.")
                elif reject_col.button("Reject"):
                    resume_value = {"decision": "reject", "edited_action": None}
                if resume_value:
                    result = await _run_with_progress(
                        graph, Command(resume=resume_value), config, context
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result.get("final_answer", ""),
                        }
                    )
                    st.rerun()
            else:
                if request:
                    st.session_state.messages.append(
                        {"role": "user", "content": request}
                    )
                    result = await _run_with_progress(
                        graph,
                        {"user_request": request},
                        config,
                        context,
                    )
                    pending = await _pending_approval(graph, config)
                    if not pending:
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": result.get("final_answer", ""),
                            }
                        )
                    st.rerun()
    except (RuntimeError, ValueError) as error:
        st.error(str(error))


if __name__ == "__main__":
    asyncio.run(_main())