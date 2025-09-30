import os
import json
import re
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
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
class Position(BaseModel):
    start: int = 0
    end: int = 0

class ErrorItem(BaseModel):
    text: str
    suggestion: Optional[str] = ""
    type: str  # e.g., spelling, grammar, style, punctuation
    severity: Literal["high", "medium", "low"] = "medium"
    position: Position = Position()
    explanation: str

class SuggestionItem(BaseModel):
    text: str
    suggestion: Optional[str] = ""
    type: str  # e.g., Style, Grammar, Vocabulary

class Statistics(BaseModel):
    wordCount: int = 0
    sentenceCount: int = 0
    paragraphCount: int = 0
    readabilityLevel: str = "Secundaria"

class Competencies(BaseModel):
    grammar: int = Field(0, ge=0, le=100)
    vocabulary: int = Field(0, ge=0, le=100)
    coherence: int = Field(0, ge=0, le=100)
    creativity: int = Field(0, ge=0, le=100)

class Annotation(BaseModel):
    start: int
    end: int
    type: Literal["highlight"] = "highlight"
    message: str

class FeedbackOut(BaseModel):
    score: int = Field(..., ge=0, le=100)
    errors: List[ErrorItem] = []
    suggestions: List[SuggestionItem] = []
    statistics: Statistics = Statistics()
    competencies: Competencies = Competencies()
    annotations: List[Annotation] = []

def _extract_json_block(text: str) -> str:
    """Intenta extraer el bloque JSON más grande desde la primera '{' hasta la última '}'."""
    if not text:
        raise ValueError("Respuesta vacía del modelo")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No se encontró JSON en la respuesta")
    return text[start:end+1]

def _basic_stats_from_text(s: str) -> Statistics:
    words = [w for w in re.split(r"\s+", s.strip()) if w]
    sentences = re.split(r"[.!?¡¿]+", s.strip())
    sentences = [t for t in sentences if t.strip()]
    paragraphs = [p for p in s.split("\n") if p.strip()]
    return Statistics(
        wordCount=len(words),
        sentenceCount=len(sentences),
        paragraphCount=len(paragraphs),
        readabilityLevel="Secundaria"
    )

def _fill_defaults(parsed: dict, student_text: str) -> dict:
    # statistics
    if "statistics" not in parsed or not isinstance(parsed.get("statistics"), dict):
        parsed["statistics"] = _basic_stats_from_text(student_text).model_dump()

    # competencies
    if "competencies" not in parsed or not isinstance(parsed.get("competencies"), dict):
        # Heurística simple basada en score si existe
        score = int(parsed.get("score", 0)) if str(parsed.get("score", "0")).isdigit() else 0
        base = max(0, min(100, score))
        parsed["competencies"] = {
            "grammar": base,
            "vocabulary": max(0, min(100, base)),
            "coherence": max(0, min(100, base)),
            "creativity": max(0, min(100, base))
        }

    # arrays
    for key in ["errors", "suggestions", "annotations"]:
        if key not in parsed or not isinstance(parsed.get(key), list):
            parsed[key] = []

    return parsed

@app.post("/feedback")
def generate_feedback(req: TextRequest):
    SYSTEM_PROMPT = (
        "Eres un tutor de comunicación para secundaria. "
        "Analiza el texto del estudiante y responde ÚNICAMENTE con JSON VÁLIDO, sin comentarios ni texto adicional. "
        "Debes usar exactamente este esquema:\n"
        "{\n"
        '  "score": int,\n'
        '  "errors": [\n'
        "    {\n"
        '      "text": str,\n'
        '      "suggestion": str,\n'
        '      "type": str,\n'
        '      "severity": "high|medium|low",\n'
        '      "position": { "start": int, "end": int },\n'
        '      "explanation": str\n'
        "    }\n"
        "  ],\n"
        '  "suggestions": [ { "text": str, "suggestion": str, "type": str } ],\n'
        '  "statistics": { "wordCount": int, "sentenceCount": int, "paragraphCount": int, "readabilityLevel": str },\n'
        '  "competencies": { "grammar": int, "vocabulary": int, "coherence": int, "creativity": int },\n'
        '  "annotations": [ { "start": int, "end": int, "type": "highlight", "message": str } ]\n'
        "}\n\n"
        "Instrucciones adicionales:\n"
        "- Calcula siempre las posiciones (start, end) como índices basados en caracteres en el texto del estudiante.\n"
        "- 'start' es el índice de la primera letra de la palabra/frase con error.\n"
        "- 'end' es EXCLUSIVO: apunta justo después del último carácter de la palabra/frase.\n"
        "- Si no encuentras el término, usa {\"start\": 0, \"end\": 0}.\n"
        "- No devuelvas explicaciones ni notas fuera del JSON."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Texto del estudiante:\n{req.student_text}\n\nIdioma: {req.language} | Tipo: {req.text_type}"}
            ],
            # Usa 'max_tokens' para Chat Completions
            max_tokens=1200,
            temperature=0.2,
        )

        raw_output = (response.choices[0].message.content or "").strip()

        # 1) Intento directo
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            # 2) Rescate: extraer el bloque JSON más probable
            parsed = json.loads(_extract_json_block(raw_output))

        # 3) Rellenar faltantes sensatos
        parsed = _fill_defaults(parsed, req.student_text)

        # 4) Validar con Pydantic y normalizar tipos/rangos
        try:
            feedback = FeedbackOut.model_validate(parsed)
        except ValidationError as ve:
            # Ajuste mínimo: si 'score' invalida rango, normaliza y reintenta
            if "score" in parsed:
                try:
                    parsed["score"] = max(0, min(100, int(parsed["score"])))
                except Exception:
                    parsed["score"] = 0
            feedback = FeedbackOut.model_validate(parsed)

        return feedback.model_dump()

    except Exception as e:
        # Fallback final: siempre entrega el formato pedido
        return FeedbackOut(
            score=0,
            errors=[],
            suggestions=[],
            statistics=_basic_stats_from_text(getattr(req, "student_text", "")),
            competencies=Competencies(grammar=0, vocabulary=0, coherence=0, creativity=0),
            annotations=[],
        ).model_dump()

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
