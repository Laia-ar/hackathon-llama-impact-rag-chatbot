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

### Some sample prompts to test:

> "I have a 36-year-old patient, without a stable relationship (promiscuous relationships) with the following symptoms:
Rash in the inguinal area, headache, followed by fever and chills at night.
I think it could be syphilis, but I'd like to know what other pathologies could confirm the diagnosis, and what treatment could be followed?
They've had these symptoms for 6 months now."

> "Eduardo, 22 years old, had a routine blood test for pre-employment screening and was detected with HIV.
How do you recommend communicating the pathology and treatments to follow?"

> "I need treatments for advanced stage syphilis."

> "What medications are contraindicated for HIV patients? Or what drugs should be avoided?"

### SIMPLE QUESTIONS
> Who is the Minister of Health mentioned in the manual?
> What percentage of people with HIV in Argentina are unaware of their diagnosis?
> What tests are mentioned in the manual for rapid diagnosis of HIV and syphilis?
> What type of sample is used for rapid tests?
> What should be done with a test strip after use?

### MODERATELY COMPLEX QUESTIONS
> What are the ethical principles mentioned for HIV diagnosis?
> What actions are suggested during pre and post-test counseling?
> What are the recommendations for proper storage of test strips?
> What steps should be followed for blood collection through finger prick?
> What strategy is proposed for conducting tests in public campaigns, such as in squares or events?

### VERY COMPLEX QUESTIONS
> How is the "window period" managed in HIV diagnosis and what are its implications?
> Describe the complete procedure for performing a rapid syphilis test, from collection to results interpretation.
> What treatment is recommended for pregnant people with syphilis and what are the specific guidelines for preventing congenital transmission?
> What specific considerations are mentioned for patients with penicillin allergy and how should it be managed in this case?



### RAG + Llama para asistir a profesionales médicos en áreas rurales proporcionando acceso rápido a conocimiento médico seleccionado.

## 🚀 Instalación

### 🛠️ 1. Clonar el repositorio
<code>git clone git@github.com:Laia-ar/hackathon-llama-impact-rag-chatbot.git</code> </br>
<code>cd hackathon-llama-impact-rag-chatbot</code> </br>
<code>cd app</code>

### 🔑 2. Configurar Claves API: Crea el archivo .streamlit/secrets.toml y agrega tus claves API de openai y pinecone en el siguiente formato
<code>[openai]</code> </br>
<code>api_key = "your_openai_api_key"</code> </br>
<code>base_url = "base_url"</code>

<code>[pinecone]</code>
<code>api_key = "your_pinecone_api_key"</code> </br>
<code>index_name = "your_index_name"</code>

### 📦 3.Instalar dependencias
<code>pip install -r requirements.txt</code>

### 🚀 4.Ejecutar la aplicación
<code>streamlit run rural-llama-health.py</code>





























///////////////////////////////////////////////////////////////////////////////////////////////////////////

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
