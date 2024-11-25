
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from utils.session_manager import ChatSessionManager

st.set_page_config(
    page_title="Sistema de Pacientes",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inicializar el administrador de sesiones
ChatSessionManager.initialize_session_states()

# Variable para manejar el estado de "nuevo paciente"
if "show_input" not in st.session_state:
    st.session_state.show_input = False

def reset_input_state():
    st.session_state.show_input = False

# Contenedor principal
with st.container():
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col2:
        st.markdown(
            "<div style='text-align: center; font-size: 80px; color: #4CAF50;'>+</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        with cols[1]:
            if st.button("Nuevo Paciente", key="nuevo_btn"):
                st.session_state.show_input = True

    with col3:
        st.markdown(
            "<div style='text-align: center; font-size: 80px; color: #2196F3;'>🔍</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        with cols[1]:
            if st.button("Buscar Paciente", key="buscar_btn"):
                ChatSessionManager.save_current_session()
                switch_page("list")

if st.session_state.show_input:
    st.markdown("<br><br>", unsafe_allow_html=True)
    patient_name = st.text_input("Ingrese el nombre del nuevo paciente:", key="input_patient")
    if st.button("Continuar", key="continue_btn"):
        if patient_name.strip():
            ChatSessionManager.create_new_patient_session(patient_name)
            reset_input_state()  # Resetear el estado del input
            switch_page("chat")
        else:
            st.error("Por favor, ingrese un nombre válido para el paciente.")

