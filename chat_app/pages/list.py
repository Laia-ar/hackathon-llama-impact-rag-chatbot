import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from utils.session_manager import ChatSessionManager

st.set_page_config(
    page_title="Lista de Pacientes",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inicializar el administrador de sesiones
ChatSessionManager.initialize_session_states()

# Estilos CSS para la lista de pacientes
st.markdown("""
    <style>
    .patient-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .patient-card:hover {
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .patient-name {
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .patient-description {
        color: #666;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Contenedor principal con márgenes
with st.container():
    # Barra de búsqueda
    search = st.text_input("🔍 Buscar paciente")
    
    # Obtener lista de pacientes del ChatSessionManager y ordenarla alfabéticamente
    patients = sorted(ChatSessionManager.get_all_patients(), key=lambda x: x[0].lower())
    
    # Filtrar pacientes si hay búsqueda
    if search:
        filtered_patients = [
            (name, last_access) for name, last_access in patients 
            if search.lower() in name.lower()
        ]
    else:
        filtered_patients = patients

    # Organizar pacientes por letra inicial
    current_letter = None
    
    # Mostrar la lista de pacientes agrupada por inicial
    for name, last_access in filtered_patients:
        # Obtener la primera letra del nombre (convertida a mayúscula)
        first_letter = name[0].upper()
        
        # Si es una nueva letra inicial, mostrar el separador
        if first_letter != current_letter:
            current_letter = first_letter
            st.markdown(f"### {current_letter}")
            st.markdown("<hr>", unsafe_allow_html=True)
        
        # Crear un contenedor clickeable para cada paciente
        patient_container = st.container()
        patient_container.markdown(f"""
            <div class="patient-card" onclick="alert('click')">
                <div class="patient-name">{name}</div>
                <div class="patient-description">Última consulta: {last_access}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Botón invisible que cubre toda la tarjeta
        if patient_container.button("Ver paciente", key=f"btn_{name}", use_container_width=True):
            ChatSessionManager.load_patient_session(name)
            switch_page("chat")

    # Mostrar mensaje si no hay resultados
    if not filtered_patients:
        st.info("No se encontraron pacientes que coincidan con la búsqueda.")