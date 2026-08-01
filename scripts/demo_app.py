from __future__ import annotations

import streamlit as st

from scripts.run_demo import run_demo


st.set_page_config(page_title="DriveScene Agent Demo", page_icon="🚗", layout="wide")
st.title("DriveScene Agent")
st.caption("Minimal offline demo · no API key · no Argoverse 2 download required")

question = st.chat_input("Ask about the demo event index")
if "demo_result" not in st.session_state:
    st.session_state["demo_result"] = run_demo("汇总当前事件索引")
if question:
    st.session_state["demo_result"] = run_demo(question)

result = st.session_state["demo_result"]
st.chat_message("user").write(result["question"])
st.chat_message("assistant").write(result["final_answer"])

with st.expander("Execution details", expanded=True):
    columns = st.columns(3)
    digest = result["execution_digest"]
    columns[0].metric("Events found", digest.get("events_found", 0))
    columns[1].metric("Valid events", (digest.get("event_summary") or {}).get("num_valid", "—"))
    columns[2].metric("Evaluation", result["evaluation"].get("status", "—"))
    st.write("Plan")
    st.dataframe(result["plan"], use_container_width=True, hide_index=True)
    st.write("Execution digest")
    st.json(digest)
