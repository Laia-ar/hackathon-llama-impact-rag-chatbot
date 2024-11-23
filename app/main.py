import streamlit as st
from get_random_answer import query_llm

st.set_page_config(page_title="Chatbot", layout="centered")
st.title("Chatbot with RAG")

user_input = st.text_input("Hacé una pregunta:", placeholder="Tipeá algo...")
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if st.button("Send"):
    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.spinner("Pensando..."):
            response = query_llm()
        st.session_state["chat_history"].append({"role": "bot", "content": response})

for chat in st.session_state["chat_history"]:
    if chat["role"] == "user":
        st.markdown(f"**Vos:** {chat['content']}")
    else:
        st.markdown(f"**Medic-bot:** {chat['content']}")
