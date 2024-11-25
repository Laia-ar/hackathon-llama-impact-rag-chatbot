import streamlit as st
from get_random_answer import query_llm

def response():
    with st.spinner("Pensando..."):
        response = query_llm()
    st.session_state["chat_history"].append({"role": "ai", "content": response})
    st.rerun()