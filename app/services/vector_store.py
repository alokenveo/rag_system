import chromadb
from app.services.embeddings_service import embed_texts, embed_query
from app.config import get_settings

settings = get_settings()

_client = None
_collection = None

def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_db_path)
        _collection = _client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection

def add_document_to_store(doc_id: str, filename: str, chunks: list[str]) -> None:
    """Añade los chunks de un documento al vector store con metadatos."""
    collection = get_collection()

    embeddings = embed_texts(chunks)
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"doc_id": doc_id, "filename": filename, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

def search_similar_chunks(query: str, k: int = None) -> list[dict]:
    """Busca los k chunks más relevantes para una query."""
    collection = get_collection()
    k = k or settings.max_results

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "content": results["documents"][0][i],
            "filename": results["metadatas"][0][i].get("filename", "desconocido"),
            "doc_id": results["metadatas"][0][i].get("doc_id", ""),
            "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
            "score": results["distances"][0][i],
        })

    return chunks

def delete_document_from_store(doc_id: str) -> int:
    """Elimina todos los chunks de un documento por su doc_id."""
    collection = get_collection()

    results = collection.get(where={"doc_id": doc_id})
    ids_to_delete = results["ids"]

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    return len(ids_to_delete)

def list_documents_in_store() -> list[dict]:
    """Devuelve la lista de documentos únicos almacenados."""
    collection = get_collection()
    results = collection.get()

    seen = {}
    for metadata in results["metadatas"]:
        doc_id = metadata.get("doc_id")
        if doc_id and doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "filename": metadata.get("filename", "desconocido"),
            }

    return list(seen.values())