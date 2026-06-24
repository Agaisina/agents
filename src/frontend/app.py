import os
import json
import uuid

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://agaisina-agents.hf.space")

_NODE_LABELS: dict[str, str] = {
    "researcher":          "🔍 Researcher — extracting corporate policies",
    "planner":             "📅 Planner — designing the itinerary",
    "planner_tools":       "🛠️ Planner — running logistics tools",
    "planner_extractor":   "📋 Planner — structuring itinerary data",
    "operator":            "🧮 Operator — auditing costs & compliance",
    "operator_tools":      "🛠️ Operator — running financial tools",
    "operator_extractor":  "📊 Operator — generating compliance verdict",
    "writer":              "✍️ Writer — drafting the final report",
    "human_review":        "⏸️ Human Review — awaiting manager approval",
}

_ROLES = ["Employee", "Manager (Approver)"]


def _get_headers() -> dict:
    token = os.getenv("HF_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Veritas Corporate Travel", page_icon="✈️", layout="centered")
st.title("✈️ Veritas Corporate Multi-Agent System")

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.awaiting_approval = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Controls")
    user_role = st.selectbox("Simulate Role:", _ROLES)
    st.caption(f"Thread: `{st.session_state.thread_id}`")

    if st.button("🗑️ New Chat"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.awaiting_approval = False
        st.rerun()

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Role-based interaction
# ---------------------------------------------------------------------------

user_input = None

if st.session_state.awaiting_approval:
    if user_role == "Employee":
        st.info(
            "⏳ Your trip request is pending manager approval. "
            "Switch to the **Manager (Approver)** role in the sidebar to review it."
        )

    elif user_role == "Manager (Approver)":
        st.warning(
            "⚠️ **Trip Under Review** — A policy violation or budget exception was detected. "
            "Please review and take action."
        )

        manager_feedback = st.text_area(
            "Manager comment (optional)",
            value="",
            placeholder="Briefly describe the reason for approval or rejection.",
        )
        col1, col2 = st.columns(2)
        action_triggered = False
        approve_action = False

        with col1:
            if st.button("✅ Approve Executive Exception", use_container_width=True):
                approve_action = True
                action_triggered = True
                note = manager_feedback or "No additional comment."
                st.session_state.messages.append(
                    {"role": "user", "content": f"*Manager approved this trip. Note: {note}*"}
                )

        with col2:
            if st.button("❌ Reject Trip", type="primary", use_container_width=True):
                approve_action = False
                action_triggered = True
                note = manager_feedback or "No additional comment."
                st.session_state.messages.append(
                    {"role": "user", "content": f"*Manager rejected this trip. Reason: {note}*"}
                )

        if action_triggered:
            st.session_state.awaiting_approval = False

            with st.chat_message("assistant"):
                status_box = st.status("Processing manager decision...", expanded=True)

                try:
                    payload = {
                        "thread_id": st.session_state.thread_id,
                        "approve": approve_action,
                        "feedback": manager_feedback,
                    }
                    response = requests.post(
                        f"{API_BASE_URL}/v1/chat/resume",
                        json=payload,
                        headers=_get_headers(),
                        timeout=60,
                    )
                    response.raise_for_status()
                    final_reply = response.json().get("answer", "No response generated.")
                    st.session_state.messages.append({"role": "assistant", "content": final_reply})
                    status_box.update(label="Workflow Resumed — Final Report Ready", state="complete")
                    st.rerun()

                except requests.Timeout:
                    status_box.update(label="Request timed out", state="error")
                    st.error("The server took too long to respond. Please try again.")
                except Exception as e:
                    status_box.update(label="Error processing decision", state="error")
                    st.error(f"Communication error: {e}")

else:
    user_input = st.chat_input("E.g.: I need to fly to London next Tuesday for 3 days...")

# ---------------------------------------------------------------------------
# Employee query processing
# ---------------------------------------------------------------------------

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status_box = st.status("Initializing Multiagent system...", expanded=True)
        message_placeholder = st.empty()
        full_response = ""
        was_interrupted = False

        try:
            payload = {"query": user_input, "thread_id": st.session_state.thread_id}
            response = requests.post(
                f"{API_BASE_URL}/v1/chat/stream",
                json=payload,
                headers=_get_headers(),
                stream=True,
                timeout=120,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    decoded = decoded[6:]

                data = json.loads(decoded)
                event_type = data.get("type")

                if event_type == "node_update":
                    node = data.get("node", "")
                    label = _NODE_LABELS.get(node, f"⚙️ Agent '{node}' processing...")
                    status_box.update(label=label)
                    status_box.write(f"✔️ {node} completed.")

                elif event_type == "token":
                    full_response += data.get("content", "")
                    message_placeholder.markdown(full_response + "▌")

                elif event_type == "interrupted":
                    was_interrupted = True
                    st.session_state.awaiting_approval = True
                    status_box.update(
                        label="⏸️ Suspended — Awaiting manager approval",
                        state="error",
                        expanded=False,
                    )
                    break

                elif event_type == "error":
                    st.error(f"Agent error: {data.get('content')}")
                    break

            if not was_interrupted:
                message_placeholder.markdown(full_response)
                status_box.update(label="✅ Workflow Complete", state="complete", expanded=False)
                if full_response:
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

        except requests.Timeout:
            status_box.update(label="Request timed out", state="error")
            st.error("The server took too long to respond (>120s). Please try again.")
        except Exception as e:
            status_box.update(label="Execution error", state="error")
            st.error(f"System error: {e}")

        st.rerun()
