from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import json
import openai
from textsplitter import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Cargar variables de entorno
load_dotenv()
API_KEY = os.getenv("AIMLAPI_API_KEY")
TEMPERATURE = 0.1  # Bajo para mantener consistencia en respuestas médicas

# Inicializar FastAPI
app = FastAPI(title="API de Diagnóstico de ITS")

# Modelos Pydantic
class TranscriptionInput(BaseModel):
    text: str

class SymptomStructure(BaseModel):
    symptoms: List[str]
    duration: Optional[str]
    severity: Optional[str]
    risk_factors: List[str]
    previous_conditions: Optional[List[str]]
    medications: Optional[List[str]]

class DiagnosisResponse(BaseModel):
    primary_diagnosis: str
    confidence_score: float
    differential_diagnoses: List[Dict[str, float]]
    recommended_treatment: str
    additional_notes: str

# Cargar base de conocimientos
with open("knowledge_base.json", "r", encoding="utf-8") as f:
    KNOWLEDGE_BASE = json.load(f)

class STIDiagnosisSystem:
    def __init__(self):
        # Cargar diccionario de síntomas estandarizados
        self.standardized_symptoms = self._load_standardized_symptoms()

        self.client = openai.OpenAI(
            base_url="https://api.aimlapi.com/v1",
            api_key=API_KEY
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", ", ", " "]
        )
    
    def _load_standardized_symptoms(self) -> Dict[str, List[str]]:
        """
        Carga un diccionario de síntomas estandarizados desde la base de conocimientos.
        Estructura: {término_médico: [sinónimos_coloquiales]}
        """
        symptoms_dict = {}
        for condition in KNOWLEDGE_BASE:
            for symptom in condition["symptoms"]:
                if symptom not in symptoms_dict:
                    symptoms_dict[symptom] = []
        return symptoms_dict

    def translate_colloquial_symptom(self, colloquial_description: str) -> Dict[str, float]:
        """
        Traduce una descripción coloquial a términos médicos estandarizados usando LLM.
        Retorna un diccionario de posibles términos médicos con scores de confianza.
        """
        prompt = f"""Analiza la siguiente descripción coloquial de un síntoma y mapéala al término médico más apropiado.
        Ten en cuenta variaciones culturales y regionales en el lenguaje.
        
        Descripción coloquial: "{colloquial_description}"
        
        Términos médicos disponibles:
        {json.dumps(list(self.standardized_symptoms.keys()), indent=2, ensure_ascii=False)}
        
        Proporciona los 3 términos médicos más probables con sus scores de confianza (0-1).
        Considera:
        1. Precisión semántica
        2. Contexto médico
        3. Similitud sintomática
        
        Responde en formato JSON:
        {
            "matches": [
                {"term": "término_médico", "confidence": 0.95, "reasoning": "explicación"},
                ...
            ]
        }"""
        
        response = self.client.chat.completions.create(
            model="text",
            messages=[
                {"role": "system", "content": "Eres un médico experto en traducir descripciones coloquiales de síntomas a términos médicos precisos. Tienes amplia experiencia en comunicación médico-paciente y comprensión de variantes culturales."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE
        )
        
        return json.loads(response.choices[0].message.content)
    
    def process_colloquial_symptoms(self, symptoms_list: List[str]) -> List[Dict[str, any]]:
        """
        Procesa una lista de síntomas coloquiales y retorna sus equivalentes médicos.
        Incluye análisis de consistencia y posibles relaciones entre síntomas.
        """
        processed_symptoms = []
        
        # Primero traducimos cada síntoma individualmente
        for symptom in symptoms_list:
            translations = self.translate_colloquial_symptom(symptom)
            processed_symptoms.append(translations)
        
        # Analizamos la consistencia y relaciones entre síntomas traducidos
        consistency_prompt = f"""Analiza el siguiente conjunto de síntomas traducidos y evalúa su consistencia médica.
        
        Síntomas procesados:
        {json.dumps(processed_symptoms, indent=2, ensure_ascii=False)}
        
        Considera:
        1. Coherencia entre síntomas
        2. Posibles relaciones causales
        3. Patrones sintomáticos comunes
        4. Posibles contradicciones o redundancias
        
        Proporciona un análisis estructurado y ajusta los scores de confianza si es necesario."""
        
        consistency_response = self.client.chat.completions.create(
            model="text",
            messages=[
                {"role": "system", "content": "Eres un médico especialista realizando un análisis de consistencia de síntomas."},
                {"role": "user", "content": consistency_prompt}
            ],
            temperature=TEMPERATURE
        )
        
        consistency_analysis = json.loads(consistency_response.choices[0].message.content)
        
        # Ajustamos los síntomas basados en el análisis de consistencia
        return self._adjust_symptoms_based_on_consistency(processed_symptoms, consistency_analysis)
    
    def _adjust_symptoms_based_on_consistency(self, symptoms: List[Dict], consistency_analysis: Dict) -> List[Dict]:
        """
        Ajusta los síntomas procesados basándose en el análisis de consistencia.
        Puede modificar scores de confianza o agregar/eliminar matches basado en el contexto general.
        """
        adjusted_symptoms = []
        
        for symptom in symptoms:
            # Aplicar ajustes basados en el análisis de consistencia
            adjusted_symptom = symptom.copy()
            
            for match in adjusted_symptom["matches"]:
                # Ajustar score basado en coherencia con otros síntomas
                if match["term"] in consistency_analysis.get("reinforced_symptoms", []):
                    match["confidence"] = min(1.0, match["confidence"] * 1.2)
                elif match["term"] in consistency_analysis.get("contradictory_symptoms", []):
                    match["confidence"] *= 0.8
                
                # Agregar notas de contexto
                if match["term"] in consistency_analysis.get("contextual_notes", {}):
                    match["context_note"] = consistency_analysis["contextual_notes"][match["term"]]
            
            adjusted_symptoms.append(adjusted_symptom)
        
        return adjusted_symptoms

    def split_text(self, text: str) -> List[str]:
        """Divide el texto en chunks manejables manteniendo contexto médico"""
        return self.text_splitter.split_text(text)
    
    def extract_medical_info(self, chunk: str) -> Dict:
        """
        Primero identificamos expresiones coloquiales de síntomas y las procesamos
        antes de la extracción general de información médica.
        """
        # Identificar posibles descripciones coloquiales de síntomas
        prompt_symptoms = f"""Identifica todas las descripciones de síntomas en el siguiente texto,
        incluyendo expresiones coloquiales, modismos o descripciones no técnicas.
        
        Texto: {chunk}
        
        Extrae cada descripción de síntoma individualmente, manteniendo el lenguaje original del paciente.
        Ignora información que no sea sintomática."""
        
        symptoms_response = self.client.chat.completions.create(
            model="text",
            messages=[
                {"role": "system", "content": "Eres un experto en identificar descripciones de síntomas en lenguaje coloquial."},
                {"role": "user", "content": prompt_symptoms}
            ],
            temperature=TEMPERATURE
        )
        
        colloquial_symptoms = json.loads(symptoms_response.choices[0].message.content)
        
        # Procesar síntomas coloquiales
        processed_symptoms = self.process_colloquial_symptoms(colloquial_symptoms["symptoms"])
        
        # Continuar con la extracción general de información médica
        """Extrae información médica relevante de cada chunk"""
        prompt = f"""Analiza el siguiente texto médico y extrae información relevante sobre síntomas, factores de riesgo y otros datos clínicos importantes.
        Mantén un formato estructurado y preciso. Enfócate en detalles relacionados con ITS.
        
        Texto: {chunk}
        
        Extrae y estructura la siguiente información:
        1. Síntomas presentes
        2. Duración de los síntomas
        3. Factores de riesgo identificados
        4. Historial médico relevante
        5. Medicaciones actuales
        
        Responde en formato JSON."""
        
        response = self.client.chat.completions.create(
            model="text",
            messages=[{"role": "system", "content": "Eres un médico experto en ITS realizando un análisis estructurado."},
                     {"role": "user", "content": prompt}],
            temperature=TEMPERATURE
        )
        return json.loads(response.choices[0].message.content)

    def standardize_symptoms(self, extracted_info: List[Dict]) -> SymptomStructure:
        """Estandariza la información extraída al formato de la base de conocimientos"""
        prompt = f"""Analiza la siguiente información médica extraída y transfórmala al formato estándar de nuestra base de conocimientos.
        Mantén la precisión médica y asegura la compatibilidad con el sistema de diagnóstico.
        
        Información extraída: {json.dumps(extracted_info, indent=2)}
        
        Formato objetivo similar a:
        {json.dumps(KNOWLEDGE_BASE[0], indent=2)}
        
        Estandariza la información manteniendo relevancia clínica."""
        
        response = self.client.chat.completions.create(
            model="text",
            messages=[{"role": "system", "content": "Eres un especialista en estandarización de datos médicos."},
                     {"role": "user", "content": prompt}],
            temperature=TEMPERATURE
        )
        return SymptomStructure(**json.loads(response.choices[0].message.content))

    def calculate_diagnosis_similarity(self, patient_symptoms: SymptomStructure) -> List[Dict[str, float]]:
        """Calcula similitud entre síntomas del paciente y base de conocimientos usando embeddings"""
        patient_text = " ".join(patient_symptoms.symptoms + patient_symptoms.risk_factors)
        patient_embedding = self.get_embedding(patient_text)
        
        similarities = []
        for condition in KNOWLEDGE_BASE:
            condition_text = " ".join(condition["symptoms"] + condition.get("risk_factors", []))
            condition_embedding = self.get_embedding(condition_text)
            
            similarity = cosine_similarity(
                [patient_embedding],
                [condition_embedding]
            )[0][0]
            
            similarities.append({
                "condition": condition["name"],
                "similarity": float(similarity),
                "treatment": condition["treatment"]
            })
        
        return sorted(similarities, key=lambda x: x["similarity"], reverse=True)

    def get_embedding(self, text: str) -> List[float]:
        """Obtiene embedding para un texto dado"""
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def generate_diagnosis_report(self, similarities: List[Dict[str, float]], 
                                patient_symptoms: SymptomStructure) -> DiagnosisResponse:
        """Genera un reporte de diagnóstico detallado"""
        top_diagnosis = similarities[0]
        differential_diagnoses = similarities[1:4]
        
        prompt = f"""Genera un diagnóstico médico detallado basado en la siguiente información:
        
        Síntomas del paciente: {json.dumps(patient_symptoms.dict(), indent=2)}
        
        Diagnóstico principal: {top_diagnosis["condition"]} (Confianza: {top_diagnosis["similarity"]:.2f})
        Diagnósticos diferenciales: {json.dumps(differential_diagnoses, indent=2)}
        
        Genera un reporte que incluya:
        1. Evaluación del diagnóstico principal
        2. Justificación clínica
        3. Consideraciones sobre diagnósticos diferenciales
        4. Plan de tratamiento recomendado
        5. Advertencias o consideraciones adicionales
        
        Mantén un tono profesional y preciso."""
        
        response = self.client.chat.completions.create(
            model="text",
            messages=[{"role": "system", "content": "Eres un médico especialista generando un reporte de diagnóstico."},
                     {"role": "user", "content": prompt}],
            temperature=TEMPERATURE
        )
        
        report_content = json.loads(response.choices[0].message.content)
        
        return DiagnosisResponse(
            primary_diagnosis=top_diagnosis["condition"],
            confidence_score=float(top_diagnosis["similarity"]),
            differential_diagnoses=[{d["condition"]: d["similarity"]} for d in differential_diagnoses],
            recommended_treatment=top_diagnosis["treatment"],
            additional_notes=report_content["additional_notes"]
        )

# Inicializar sistema
diagnosis_system = STIDiagnosisSystem()

@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(input_data: TranscriptionInput):
    try:
        # 1. Dividir texto en chunks
        chunks = diagnosis_system.split_text(input_data.text)
        
        # 2. Extraer información médica de cada chunk
        extracted_info = []
        for chunk in chunks:
            chunk_info = diagnosis_system.extract_medical_info(chunk)
            extracted_info.append(chunk_info)
        
        # 3. Estandarizar información extraída
        standardized_symptoms = diagnosis_system.standardize_symptoms(extracted_info)
        
        # 4. Calcular similitudes y encontrar matches
        diagnosis_similarities = diagnosis_system.calculate_diagnosis_similarity(standardized_symptoms)
        
        # 5. Generar reporte de diagnóstico
        diagnosis_report = diagnosis_system.generate_diagnosis_report(
            diagnosis_similarities,
            standardized_symptoms
        )
        
        return diagnosis_report
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)