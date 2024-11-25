from datetime import datetime
import streamlit as st

class ChatSessionManager:
    @staticmethod
    def initialize_session_states():
        """Inicializa los estados de sesión necesarios si no existen"""
        if "patient_sessions" not in st.session_state:
            st.session_state.patient_sessions = {}
        if "current_patient" not in st.session_state:
            st.session_state.current_patient = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    @staticmethod
    def save_current_session():
        """Guarda la sesión actual del paciente"""
        if st.session_state.current_patient:
            st.session_state.patient_sessions[st.session_state.current_patient] = {
                'chat_history': st.session_state.chat_history,
                'last_access': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    @staticmethod
    def load_patient_session(patient_name):
        """Carga la sesión de un paciente específico"""
        if patient_name in st.session_state.patient_sessions:
            st.session_state.chat_history = st.session_state.patient_sessions[patient_name]['chat_history']
            st.session_state.patient_sessions[patient_name]['last_access'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            st.session_state.chat_history = []
        st.session_state.current_patient = patient_name

    @staticmethod
    def get_all_patients():
        """Retorna una lista de todos los pacientes con sus últimas fechas de acceso"""
        return [(name, data['last_access']) 
                for name, data in st.session_state.patient_sessions.items()]

    @staticmethod
    def create_new_patient_session(patient_name):
        """Crea una nueva sesión para un paciente"""
        if patient_name not in st.session_state.patient_sessions:
            st.session_state.patient_sessions[patient_name] = {
                'chat_history': [],
                'last_access': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        st.session_state.current_patient = patient_name
        st.session_state.chat_history = []

    @staticmethod
    def add_patient(patient_name):
        if "patients" not in st.session_state:
            st.session_state["patients"] = []
        st.session_state["patients"].append(patient_name)