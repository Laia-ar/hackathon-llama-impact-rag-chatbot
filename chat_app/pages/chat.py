import streamlit as st
from streamlit_extras.bottom_container import bottom 
from components.st_fixed_container import st_fixed_container
from audiorecorder import audiorecorder
from utils.audio_processing import check_audio, process_uploaded_audio
from utils.chat_history import response
from utils.session_manager import ChatSessionManager

# Configuración de la página
st.set_page_config(
    page_title="Chatbot Médico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar el administrador de sesiones
ChatSessionManager.initialize_session_states()

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Si no hay paciente actual, redirigir a la página principal
if not st.session_state.current_patient:
    new_patient_name = st.text_input("No hay paciente actual. Ingresa el nombre del nuevo paciente:")
    if st.button("Agregar Paciente"):
        if new_patient_name:
            # Lógica para agregar el nuevo paciente
            ChatSessionManager.add_patient(new_patient_name)  # Asegúrate de tener este método implementado
            st.session_state.current_patient = new_patient_name  # Establecer el paciente actual
        else:
            st.warning("Por favor, ingresa un nombre para el paciente.")

# Mostrar el nombre del paciente actual solo si hay un paciente
if st.session_state.current_patient:
    st.title(f"Chat del Paciente: {st.session_state.current_patient}")
# No se muestra nada si no hay paciente actual

# Cargar el contenedor de entrada solo si hay un paciente actual
if st.session_state.current_patient:
    input_container = st.container()
    with input_container:
        with bottom():
            cols = st.columns([6, 0.27, 0.1])

            # Chat input
            with cols[0]:
                if prompt := st.chat_input("Escribe tu consulta..."):
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    ChatSessionManager.save_current_session()
                    response()
                
                # Audio recorder
            with cols[1]:
                audio = audiorecorder("🎤", "🔴", key="audio_recorder", 
                                        custom_style={'border': '1px solid rgba(49, 51, 63, 0.2)', 
                                                      'padding': '0.45rem 0.75rem', 
                                                      'border-radius': '0.5rem'})
                check_audio(audio)
                
                # File upload button and popup
            with cols[2]:
                    # Botón con ícono de clip
                if st.button("📎", help="Subir archivo WAV"):
                    st.session_state["show_file_uploader"] = not st.session_state.get("show_file_uploader", False)

                    # Generar el contenedor del popup flotante
                popup_placeholder = st.empty()

                if st.session_state.get("show_file_uploader", False):
                    with popup_placeholder.container():
                        st.markdown('<div class="upload-popup active">', unsafe_allow_html=True)
                            
                        if "file_processed" not in st.session_state:
                            st.session_state["file_processed"] = False

                            # Widget para subir archivos
                        uploaded_file = st.file_uploader(
                                "",  # Eliminamos el label ya que lo ocultamos con CSS
                                type=["wav"],
                                key="file_uploader"
                            )

                            # Procesar el archivo si es necesario
                        if uploaded_file is not None and not st.session_state["file_processed"]:
                            process_uploaded_audio(uploaded_file)

                        if uploaded_file is None and st.session_state["file_processed"]:
                            st.session_state["file_processed"] = False
                                
                            # Cerrar el contenedor del popup
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    popup_placeholder.empty()  # Elimina el contenedor del popup

# Botón de regreso
col1, col2, col3 = st.columns([1,6,0.001])
with col3:
    with st_fixed_container(mode="fixed", position="top"):
        if st.button("↩️"):
            ChatSessionManager.save_current_session()
            st.session_state.current_patient = None
            st.rerun()
            

# Área de mensajes
chat_container = st.container()
with chat_container:
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

# Sidebar con lista de pacientes
with st.sidebar:
    st.title("Pacientes")
    patients = ChatSessionManager.get_all_patients()
    
    for patient_name, last_access in sorted(patients):
        col1, col2 = st.columns([3, 2])
        with col1:
            if st.button(f"{patient_name}", key=f"patient_{patient_name}"):
                ChatSessionManager.save_current_session()  # Guardar sesión actual
                ChatSessionManager.load_patient_session(patient_name)  # Cargar nueva sesión
                st.rerun()
        with col2:
            st.caption(f"Último acceso:\n{last_access}")