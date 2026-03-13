from google import genai
from app.config import get_settings

settings = get_settings()

_client = None

def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Genera embeddings para una lista de textos usando la API de Google directamente."""
    client = get_client()
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=texts,
    )
    return [e.values for e in result.embeddings]

def embed_query(text: str) -> list[float]:
    """Genera el embedding de un único texto (para búsqueda)."""
    return embed_texts([text])[0]