from __future__ import annotations

import os

import streamlit as st

from scripts.run_demo import run_demo


st.set_page_config(page_title="DriveScene Agent Demo", page_icon="🚗", layout="wide")
st.title("DriveScene Agent")

with st.sidebar:
    st.subheader("运行模式")
    online = st.toggle(
        "使用在线 LLM Planner",
        value=bool(os.environ.get("OPENAI_API_KEY")),
        help="开启后，Planner 与 Reporter 会调用 model config 中的真实模型。",
    )
    model_config = st.text_input("Model config", "config/model.yml")
    if online and not os.environ.get("OPENAI_API_KEY"):
        st.warning("当前进程未读取到 OPENAI_API_KEY。")

mode_key = (online, model_config)
if st.session_state.get("demo_mode_key") != mode_key:
    st.session_state["demo_mode_key"] = mode_key
    st.session_state.pop("demo_result", None)

st.caption(
    "Online LLM planning · built-in sample indexes · no Argoverse 2 download required"
    if online
    else "Offline deterministic planning · no API key · no Argoverse 2 download required"
)

question = st.chat_input("Ask about the demo event index")
if "demo_result" not in st.session_state:
    try:
        st.session_state["demo_result"] = run_demo(
            "汇总当前事件索引",
            online=online,
            model_config=model_config,
        )
    except Exception as exc:
        st.error(f"运行失败：{exc}")
        st.stop()
if question:
    try:
        st.session_state["demo_result"] = run_demo(
            question,
            online=online,
            model_config=model_config,
        )
    except Exception as exc:
        st.error(f"运行失败：{exc}")
        st.stop()

result = st.session_state["demo_result"]
st.caption(f"Runtime: {result['runtime_mode']}")
st.chat_message("user").write(result["question"])
st.chat_message("assistant").write(result["final_answer"])

with st.expander("Execution details", expanded=False):
    columns = st.columns(3)
    digest = result["execution_digest"]
    columns[0].metric("Events found", digest.get("events_found", 0))
    columns[1].metric("Valid events", (digest.get("event_summary") or {}).get("num_valid", "—"))
    columns[2].metric("Evaluation", result["evaluation"].get("status", "—"))
    st.write("Plan")
    st.dataframe(result["plan"], width="stretch", hide_index=True)
    st.write("Execution digest")
    st.json(digest)
