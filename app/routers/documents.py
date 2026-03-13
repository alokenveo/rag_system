from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.document_processor import process_document
from app.services.vector_store import (
    add_document_to_store,
    delete_document_from_store,
    list_documents_in_store,
)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Sube y procesa un documento PDF o DOCX.
    Lo divide en chunks y los almacena en ChromaDB con embeddings.
    """
    allowed_types = ["application/pdf",
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    
    if file.content_type not in allowed_types and \
       not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(
            status_code=400,
            detail="Formato no soportado. Sube un PDF o DOCX."
        )
    
    file_bytes = await file.read()
    
    if len(file_bytes) > 10 * 1024 * 1024:  # 10MB límite
        raise HTTPException(
            status_code=400,
            detail="El archivo supera el límite de 10MB."
        )
    
    try:
        result = process_document(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    add_document_to_store(
        doc_id=result["doc_id"],
        filename=result["filename"],
        chunks=result["chunks"],
    )
    
    return {
        "message": "Documento procesado correctamente.",
        "doc_id": result["doc_id"],
        "filename": result["filename"],
        "total_chunks": result["total_chunks"],
        "total_chars": result["total_chars"],
    }


@router.get("/")
async def list_documents():
    """Lista todos los documentos almacenados en el sistema."""
    documents = list_documents_in_store()
    return {
        "total": len(documents),
        "documents": documents,
    }


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Elimina un documento y todos sus chunks del vector store."""
    deleted_chunks = delete_document_from_store(doc_id)
    
    if deleted_chunks == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró ningún documento con id '{doc_id}'."
        )
    
    return {
        "message": "Documento eliminado correctamente.",
        "doc_id": doc_id,
        "chunks_deleted": deleted_chunks,
    }