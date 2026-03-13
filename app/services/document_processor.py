import uuid
from pathlib import Path
from pypdf import PdfReader
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import get_settings

settings = get_settings()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    import io

    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    import io

    doc = DocxDocument(io.BytesIO(file_bytes))
    text = ""
    for paragraph in doc.pages:
        if paragraph.text.strip():
            text += paragraph.text + "\n"
    return text.strip()


def split_text_into_chunks(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def process_document(file_bytes: bytes, filename: str) -> dict:
    """
    Recibe los bytes de un archivo y devuelve:
    - doc_id: identificador único
    - filename: nombre del archivo
    - chunks: lista de fragmentos de texto
    - total_chars: total de caracteres extraídos
    """

    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    elif filename_lower.endswith(".docx") or filename_lower.endswith(".doc"):
        text = extract_text_from_docx(file_bytes)
    else:
        raise ValueError(f"Formato no soportado: {filename}. Usa PDF o DOCX.")

    if not text:
        raise ValueError("No se pudo extraer texto del documento. ¿Está escaneado?")

    chunks = split_text_into_chunks(text)

    return {
        "doc_id": str(uuid.uuid4()),
        "filename": filename,
        "chunks": chunks,
        "total_chars": len(text),
        "total_chunks": len(chunks),
    }
