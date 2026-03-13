from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from app.services.vector_store import search_similar_chunks
from app.config import get_settings

settings = get_settings()

SYSTEM_PROMPT = """Eres un asistente experto en analizar documentos empresariales.
Tienes acceso a fragmentos de documentos que el usuario ha subido al sistema.

Responde SIEMPRE basándote en el contexto proporcionado.
Si la información no está en el contexto, dilo claramente: "No encontré esa información en los documentos disponibles."
Sé preciso, claro y útil. Si el contexto contiene precios, fechas o datos concretos, inclúyelos en tu respuesta."""

def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

def build_context_from_chunks(chunks: list[dict]) -> str:
    """Construye el bloque de contexto a partir de los chunks recuperados."""
    if not chunks:
        return "No se encontraron fragmentos relevantes en los documentos."
    
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Fragmento {i} — {chunk['filename']}]\n{chunk['content']}"
        )
    
    return "\n\n---\n\n".join(context_parts)

def answer_question(question: str, doc_id: str = None) -> dict:
    """
    Responde una pregunta usando RAG:
    1. Busca chunks relevantes en el vector store
    2. Construye el contexto
    3. Llama a Gemini con el contexto + pregunta
    4. Devuelve la respuesta y las fuentes usadas
    """
    # Paso 1: recuperar chunks relevantes
    chunks = search_similar_chunks(question, k=settings.max_results)
    
    # Si se especifica doc_id, filtrar solo chunks de ese documento
    if doc_id:
        chunks = [c for c in chunks if c["doc_id"] == doc_id]
    
    # Paso 2: construir contexto
    context = build_context_from_chunks(chunks)
    
    # Paso 3: construir prompt y llamar a Gemini
    llm = get_llm()
    
    user_message = f"""Contexto extraído de los documentos:

{context}

---

Pregunta del usuario: {question}"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    
    response = llm.invoke(messages)
    
    # Paso 4: devolver respuesta + fuentes
    sources = [
        {
            "filename": c["filename"],
            "chunk_index": c["chunk_index"],
            "relevance_score": round(1 - c["score"], 3),  # Chroma devuelve distancia, lo invertimos
        }
        for c in chunks
    ]
    
    return {
        "answer": response.content,
        "sources": sources,
        "chunks_used": len(chunks),
    }