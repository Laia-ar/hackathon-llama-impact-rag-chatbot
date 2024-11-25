# Rural Llama Health

### RAG + Llama to assist medical professionals in rural areas by providing quick access to curated medical knowledge.

## 🚀 Installation

### 🛠️ 1. Clone the Repository
<code>git clone git@github.com:Laia-ar/hackathon-llama-impact-rag-chatbot.git</code> </br>
<code>cd hackathon-llama-impact-rag-chatbot</code> </br>
<code>cd app</code>

### 🔑 2. Set Up API Keys: Create the .streamlit/secrets.toml and add your openai and pinecone API keys in the following format:
<code>[openai]</code> </br>
<code>api_key = "your_openai_api_key"</code> </br>
<code>base_url = "base_url"</code>

<code>[pinecone]</code>
<code>api_key = "your_pinecone_api_key"</code> </br>
<code>index_name = "your_index_name"</code>

### 📦 3.Install Dependencies
<code>pip install -r requirements.txt</code>

### 🚀 4.Run the Application:
<code>streamlit run rural-llama-health.py</code>

Ejemplos de prompts:

"Tengo un paciente de 36 años, sin una relacion estable (relaciones promiscuas) con los siguientes sintomas:
Sarpullido en la zona inginal, Dolor de Cabeza, seguida con Fiebre y durante la noche tiene escalofrios.
Creo que podria ser sifilis, pero me gustaria saber que otras patologias podrian asegurar el diagnostico, y que tratamiento podria seguir?
Dice que tiene estos sintomas, desde hace ya 6 meses."

///////////////////////////////////////////////////////////////////////////////////////////////////////////

"Eduardo de 22 años se hizo un estudio de sangre de rutina para un pre ocupacional y se le detecto HIV, 
como recomendas que se le comunique la patologia y los tratamientos a seguir?"

///////////////////////////////////////////////////////////////////////////////////////////////////////////

"Necesito tratamientos para la sifilis en etapa avanzada."

///////////////////////////////////////////////////////////////////////////////////////////////////////////

"Que medicamentos son contraindicados para pacientes con HIV? o que drogas habria que evitar?"

///////////////////////////////////////////////////////////////////////////////////////////////////////////

PREGUNTAS SIMPLES
¿Quién es el Ministro de Salud mencionado en el manual?
¿Qué porcentaje de personas con VIH en Argentina desconoce su diagnóstico?
¿Qué pruebas se mencionan en el manual para el diagnóstico rápido de VIH y sífilis?
¿Qué tipo de muestra se utiliza para realizar las pruebas rápidas?
¿Qué debe hacerse con una tira reactiva después de ser utilizada?

///////////////////////////////////////////////////////////////////////////////////////////////////////////

PREGUNTAS MEDIANAMENTE COMPLEJAS
¿Cuáles son los principios éticos mencionados para el diagnóstico de VIH?
¿Qué acciones se sugieren durante el asesoramiento pre y post-test?
¿Cuáles son las recomendaciones para el almacenamiento adecuado de las tiras reactivas?
¿Qué pasos se deben seguir para realizar una extracción de sangre mediante punción digital?
¿Qué estrategia se propone para realizar pruebas en campañas públicas, como en plazas o eventos?

///////////////////////////////////////////////////////////////////////////////////////////////////////////

PREGUNTAS MUY COMPLEJAS
¿Cómo se maneja el "período ventana" en el diagnóstico de VIH y qué implicaciones tiene?
Describe el procedimiento completo para realizar una prueba rápida de sífilis, desde la extracción hasta la interpretación de resultados.
¿Qué tratamiento se recomienda para personas gestantes con sífilis y cuáles son las indicaciones específicas para prevenir la transmisión congénita?
¿Qué consideraciones específicas se mencionan para pacientes con alergia a la penicilina y cómo debe manejarse en este caso?

///////////////////////////////////////////////////////////////////////////////////////////////////////////
