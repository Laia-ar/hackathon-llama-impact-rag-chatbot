import streamlit as st
from get_random_answer import query_llm
from audiorecorder import audiorecorder
from pydub import AudioSegment
import io
import whisper

st.set_page_config(page_title="Chatbot", layout="centered")
st.title("Chatbot with RAG")

# Input para el chatbot
user_input = st.text_input("Hacé una pregunta:", placeholder="Tipeá algo...")

# Tipo de modelo
whisper_model_type = st.radio("Please choose your model type", ('Tiny', 'Base', 'Small', 'Medium', 'Large'))

# Historial del chat
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def process_audio(filename, model_type):
    model = whisper.load_model(model_type)
    result = model.transcribe(filename)
    return result["text"]

# Grabacion de audio para la transcripcion
audio = audiorecorder("Grabar audio", "Detener")
if len(audio) > 0:
    try:
        if isinstance(audio, AudioSegment):
            # AudioSegment a bytes
            st.audio(audio.export().read())  
            audio.export("audio.wav", format="wav")
            
            st.success("¡Audio grabado y reproducido!")

            # Placeholder para la transcripción
            transcribed_text = process_audio("audio.wav", whisper_model_type.lower())
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

