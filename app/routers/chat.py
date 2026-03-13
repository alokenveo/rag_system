from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.services.chat_service import answer_question

router = APIRouter(prefix="/chat", tags=["Chat"])


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          description="Pregunta sobre los documentos subidos")
    doc_id: Optional[str] = Field(None,
                                   description="Si se indica, filtra la búsqueda a ese documento concreto")


@router.post("/")
async def chat_with_documents(request: QuestionRequest):
    """
    Hace una pregunta al sistema RAG.
    Busca los fragmentos más relevantes en ChromaDB
    y genera una respuesta con Gemini usando ese contexto.
    """
    try:
        result = answer_question(
            question=request.question,
            doc_id=request.doc_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la pregunta: {str(e)}"
        )
    
    if result["chunks_used"] == 0:
        raise HTTPException(
            status_code=404,
            detail="No hay documentos en el sistema. Sube alguno primero."
        )
    
    return result