import streamlit as st
from get_random_answer import query_llm
import requests
import json

def response():
    with st.spinner("Pensando..."):
        response = query_llm()
    st.session_state["chat_history"].append({"role": "ai", "content": response})
    st.rerun()


def new_response():
    try:

        full_conversation = " ".join([
            chat["content"] for chat in st.session_state.chat_history 
            if chat["role"] == "user"
        ])


        API_URL = "http://localhost:8000/diagnose"  # Adjust based on your API deployment
        

        payload = {
            "text": full_conversation
        }


        api_response = requests.post(API_URL, json=payload)
        

        if api_response.status_code == 200:
            diagnosis = api_response.json()
            

            response_text = f"""Diagnóstico: {diagnosis['primary_diagnosis']}
Confianza: {diagnosis['confidence_score']:.2%}

Tratamiento Recomendado: {diagnosis['recommended_treatment']}

Notas Adicionales: {diagnosis['additional_notes']}"""
            

            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response_text
            })
        else:

            error_message = f"Error en diagnóstico. Código de estado: {api_response.status_code}"
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": error_message
            })
    
    except requests.exceptions.RequestException as e:

        error_message = f"No se pudo conectar con el servicio de diagnóstico: {str(e)}"
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": error_message
        })
    except Exception as e:

        error_message = f"Ocurrió un error inesperado: {str(e)}"
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": error_message
        })