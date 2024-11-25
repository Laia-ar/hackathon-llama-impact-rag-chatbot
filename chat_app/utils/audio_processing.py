import streamlit as st
from pydub import AudioSegment
import tempfile
import whisper
from .chat_history import response

whisper_model_type = 'Tiny'

def process_audio(filename, model_type):
    model = whisper.load_model(model_type)
    result = model.transcribe(filename)
    return result["text"]

def process_uploaded_audio(uploaded_file):
    """Procesa el archivo de audio subido"""
    if "file_processed" not in st.session_state:
        st.session_state["file_processed"] = False
        
    try:
        if uploaded_file is not None and not st.session_state["file_processed"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name 
                
            transcribed_text = process_audio(temp_file_path, whisper_model_type.lower())
            
            if transcribed_text:
                st.session_state["chat_history"].append({
                    "role": "user", 
                    "content": transcribed_text
                })
                st.session_state["file_processed"] = True
                
                # Limpiar el archivo temporal
                import os
                os.unlink(temp_file_path)
                
                # Cerrar el popup automáticamente después de procesar
                st.session_state["show_file_uploader"] = False
                
                # Forzar la actualización de la interfaz para reflejar el cambio
                response()
                
    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")



def check_audio(audio):
    if "audio_processed" not in st.session_state:
        st.session_state["audio_processed"] = False
            
    if (len(audio) > 0 and not st.session_state["audio_processed"]) or isinstance(audio, str):
        try:
            if isinstance(audio, AudioSegment) or isinstance(audio, str):
                if isinstance(audio, str):
                    audio_segment = AudioSegment.from_wav(audio)
                else:
                    audio_segment = audio
                
                audio_segment.export("audio.wav", format="wav")
                transcribed_text = process_audio("audio.wav", whisper_model_type.lower())
                        
                if transcribed_text:
                    st.session_state["chat_history"].append({
                        "role": "user", 
                        "content": transcribed_text
                    })
                    st.session_state["audio_processed"] = True
                    response()
            else:
                st.error("Formato de audio no compatible.")
                
        except Exception as e:
            st.error(f"Error al procesar el audio: {e}")
            
    if len(audio) == 0:
        st.session_state["audio_processed"] = False
