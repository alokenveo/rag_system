from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    google_api_key: str
    chroma_db_path: str = "./data/chroma_db"
    collection_name: str = "documents"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "models/gemini-embedding-001"
    llm_model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_tokens: int = 2048
    max_results: int = 3

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings():
    return Settings()
