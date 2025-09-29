import os
import json
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()

# === Configuración ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4"

# === API ===
app = FastAPI(title="TutorAI IA Microservice")
# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir a ["https://tudominio.com"]
    allow_credentials=True,
    allow_methods=["*"],  # Permitir GET, POST, PUT, DELETE, OPTIONS...
    allow_headers=["*"],  # Permitir cualquier header
)
# === Schemas ===
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    conversation_id: str
    text_type: str = "general"
    language: str = "es-PE"
    messages: List[ChatMessage]

class TextRequest(BaseModel):
    student_text: str
    text_type: str | None = "general"
    language: str | None = "es-PE"

class SlideRequest(BaseModel):
    topic: str
    difficulty: str
    slides: int = 5
    description: str | None = None
    source_pdf: str | None = None

class TaskRequest(BaseModel):
    title: str
    description: str
    task_type: str
    curriculum_topic: str
    difficulty: str
    num_questions: int = 5
    time_limit: int = 30
    instructions: str | None = None
    include_open: bool = False
    include_essay: bool = False
    include_truefalse: bool = False

class ConnectorRequest(BaseModel):
    snippet: str
    need: str

class QuizRequest(BaseModel):
    student_level: str
    topic: str
    difficulty: str = "medium"
    language: str = "es-PE"

class CompareRequest(BaseModel):
    old_text: str
    new_text: str

@app.post("/feedback")
def generate_feedback(req: TextRequest):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "Eres un tutor de comunicación para secundaria. "
                    "Analiza el texto del estudiante y responde únicamente con JSON válido. "
                    "El JSON debe tener esta estructura exacta:\n\n"
                    "{\n"
                    '  "score": {\n'
                    '    "global": int,\n'
                    '    "categories": {\n'
                    '      "conectores": int,\n'
                    '      "gramática": int,\n'
                    '      "estructura": int,\n'
                    '      "vocabulario": int,\n'
                    '      "estilo": int\n'
                    "    }\n"
                    "  },\n"
                    '  "errors": [ { "type": str, "message": str } ],\n'
                    '  "suggestions": [ { "type": str, "message": str } ],\n'
                    '  "annotations": [ { "range": { "start": int, "end": int }, "severity": str, "message": str } ],\n'
                    '  "connectors": [ { "word": str, "type": str, "start": int, "end": int, "color": str } ],\n'
                    '  "highlighted_text": str\n'
                    "}\n\n"
                    "No devuelvas explicaciones, solo el JSON."
                )},
                {"role": "user", "content": f"Texto del estudiante:\n{req.student_text}"}
            ],
            max_completion_tokens=1600
        )

        raw_output = response.choices[0].message.content or ""
        return json.loads(raw_output)

    except json.JSONDecodeError:
        # Fallback si el modelo no devuelve JSON válido
        return {
            "error": "El modelo no devolvió JSON válido",
            "raw_output": raw_output,
            "expected_format": {
                "score": {
                    "global": 0,
                    "categories": {
                        "conectores": 0,
                        "gramática": 0,
                        "estructura": 0,
                        "vocabulario": 0,
                        "estilo": 0
                    }
                },
                "errors": [],
                "suggestions": [],
                "annotations": [],
                "connectors": [],
                "highlighted_text": ""
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Endpoint 2: Quiz con feedback inmediato ===
@app.post("/quiz/generate")
def generate_quiz(req: QuizRequest):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "Eres un generador de preguntas educativas para secundaria. "
                    "Crea una pregunta de práctica sobre conectores o gramática. "
                    "Devuelve SOLO un JSON en este formato:\n"
                    "{ 'questionId': str, 'statement': str, "
                    "'options': [ { 'text': str, 'correct': bool, 'feedback': str } ], "
                    "'metadata': { 'topic': str, 'difficulty': str } }"
                )},
                {"role": "user", "content": (
                    f"Nivel: {req.student_level}\n"
                    f"Tema: {req.topic}\n"
                    f"Dificultad: {req.difficulty}\n"
                    f"Idioma: {req.language}"
                )}
            ],
            max_completion_tokens=800,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint 3: Comparación de versiones de texto ===
@app.post("/text/compare")
def compare_texts(req: CompareRequest):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "Compara dos versiones de un texto. Devuelve JSON:\n"
                    "{ 'improvements': [str], 'regressions': [str], 'summary': str }"
                )},
                {"role": "user", "content": f"Versión anterior:\n{req.old_text}\n\nNueva versión:\n{req.new_text}"}
            ],
            max_completion_tokens=600,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint 4: Generar Slides ===
@app.post("/slides/generate")
def generate_slides(req: SlideRequest):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "Eres un generador de presentaciones interactivas para secundaria. "
                    "Devuelve SOLO JSON válido con este formato:\n"
                    "{ 'title': str, 'slides': [ { 'id': int, 'question': str, 'options': [ { 'label': str, 'text': str, 'correct': bool } ], 'explanation': str } ] }"
                )},
                {"role": "user", "content": (
                    f"Tema: {req.topic}\n"
                    f"Descripción: {req.description if req.description else 'N/A'}\n"
                    f"Dificultad: {req.difficulty}\n"
                    f"Número de diapositivas: {req.slides}\n"
                    f"Fuente PDF: {req.source_pdf if req.source_pdf else 'N/A'}"
                )}
            ],
            max_completion_tokens=1500,
        )
        return json.loads(response.choices[0].message.content.strip())
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="El modelo no devolvió un JSON válido.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint 5: Generar Tareas ===
@app.post("/tasks/generate")
def generate_task(req: TaskRequest):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "Eres un creador de tareas educativas siguiendo el currículo MINEDU. "
                    "Debes generar una lista de preguntas con distintos tipos (opción múltiple, abierto, redacción, V/F). "
                    "Devuelve SOLO JSON válido con este formato:\n"
                    "{ 'title': str, 'description': str, 'task_type': str, 'difficulty': str, "
                    "'curriculum_topic': str, 'time_limit': int, 'total_points': int, "
                    "'instructions': str, 'questions': [ { 'id': int, 'question': str, 'type': str, 'points': int, 'options': [ { 'label': str, 'text': str, 'correct': bool } ] | null } ] }"
                )},
                {"role": "user", "content": (
                    f"Título: {req.title}\n"
                    f"Descripción: {req.description}\n"
                    f"Tipo de tarea: {req.task_type}\n"
                    f"Tema del currículo: {req.curriculum_topic}\n"
                    f"Dificultad: {req.difficulty}\n"
                    f"Número de preguntas: {req.num_questions}\n"
                    f"Incluir abiertas: {req.include_open}\n"
                    f"Incluir redacción: {req.include_essay}\n"
                    f"Incluir V/F: {req.include_truefalse}\n"
                    f"Tiempo límite: {req.time_limit} minutos\n"
                    f"Instrucciones: {req.instructions if req.instructions else 'N/A'}"
                )}
            ],
            max_completion_tokens=1800,
        )
        return json.loads(response.choices[0].message.content.strip())
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="El modelo no devolvió un JSON válido.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint 6: Chat de ayuda en escritura ===
@app.post("/chat/write-helper")
def write_helper(req: ChatRequest):
    try:
        formatted_messages = [
            {"role": "system", "content": (
                f"Eres un tutor de comunicación en secundaria. "
                f"Ayuda al alumno a escribir un texto de tipo '{req.text_type}'. "
                f"No escribas todo el texto por él, guía paso a paso con preguntas, ejemplos y sugerencias. "
                f"Responde siempre en {req.language}."
            )}
        ]

        for m in req.messages:
            formatted_messages.append({"role": m.role, "content": m.content})

        response = client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            max_completion_tokens=600,
        )

        reply = response.choices[0].message.content.strip()
        return {"conversation_id": req.conversation_id, "reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
