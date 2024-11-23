import streamlit as st
from get_random_answer import query_llm
from audiorecorder import audiorecorder
from pydub import AudioSegment
import io

st.set_page_config(page_title="Chatbot", layout="centered")
st.title("Chatbot with RAG")

# Input para el chatbot
user_input = st.text_input("Hacé una pregunta:", placeholder="Tipeá algo...")

# Historial del chat
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Grabacion de audio para la transcripcion
audio = audiorecorder("Grabar audio", "Detener")
if len(audio) > 0:
    try:
        if isinstance(audio, AudioSegment):
            # AudioSegment a bytes
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            audio_bytes = buffer.getvalue()
            
            st.audio(audio_bytes, format="audio/wav")
            st.success("¡Audio grabado y reproducido!")

            # Placeholder para la transcripcrión
            transcribed_text = "[Transcripción de audio pendiente]"
            st.session_state["chat_history"].append({"role": "user", "content": transcribed_text})
        else:
            st.error("Formato de audio no compatible.")
    except Exception as e:
        st.error(f"Error al procesar el audio: {e}")

if st.button("Enviar"):
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