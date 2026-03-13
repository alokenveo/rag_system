from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat, documents
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="RAG System API",
    description="Sistema RAG con Gemini, LangChain y ChromaDB",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir router

app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/")
async def root():
    return {
        "message": "RAG System API",
        "version": "1.0.0",
        "endpoints": {"docs": "/docs", "documents": "/documents", "chat": "/chat"},
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

