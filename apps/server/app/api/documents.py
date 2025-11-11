"""
Document management API endpoints.
"""

import logging
from pathlib import Path
from typing import List
import aiofiles
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..rag.document_processor import get_document_processor
from ..rag.embeddings import get_embedding_generator
from ..rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Storage directory
UPLOAD_DIR = Path("data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Supported file types
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


class DocumentInfo(BaseModel):
    """Document information response."""

    document_id: str
    filename: str
    file_type: str
    file_size: int
    char_count: int
    word_count: int
    chunk_count: int
    uploaded_at: str


class DocumentListResponse(BaseModel):
    """List of documents response."""

    documents: List[DocumentInfo]
    total: int


async def process_document_background(file_path: Path, document_id: str):
    """Background task to process document."""
    try:
        logger.info(f"Background processing started for {file_path.name}")

        # Get processors
        doc_processor = get_document_processor()
        embedding_gen = get_embedding_generator()
        vector_store = get_vector_store()

        # Process document
        full_text, chunks, metadata = doc_processor.process_document(
            file_path, document_id
        )

        # Generate embeddings for chunks
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = embedding_gen.embed_texts(chunk_texts, show_progress=False)

        # Store in vector store
        vector_store.add_documents(
            texts=chunk_texts,
            embeddings=embeddings.tolist(),
            metadatas=[chunk.metadata for chunk in chunks],
            ids=[chunk.chunk_id for chunk in chunks],
        )

        logger.info(
            f"Document {document_id} processed successfully: "
            f"{len(chunks)} chunks indexed"
        )

    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        # Clean up on error
        if file_path.exists():
            file_path.unlink()


@router.post("/upload", response_model=DocumentInfo)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a document (PDF or TXT).

    The document will be processed in the background:
    - Parsed and chunked
    - Embeddings generated
    - Stored in vector database
    """
    # Validate file type
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # Read file
    content = await file.read()
    file_size = len(content)

    # Check file size
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB",
        )

    # Generate document ID
    doc_processor = get_document_processor()
    document_id = doc_processor._generate_document_id(Path(file.filename))

    # Save file
    file_path = UPLOAD_DIR / f"{document_id}_{file.filename}"
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    logger.info(f"File uploaded: {file.filename} -> {document_id}")

    # Schedule background processing
    background_tasks.add_task(process_document_background, file_path, document_id)

    # Return immediate response with basic info
    return DocumentInfo(
        document_id=document_id,
        filename=file.filename,
        file_type=file_extension,
        file_size=file_size,
        char_count=0,  # Will be updated after processing
        word_count=0,
        chunk_count=0,
        uploaded_at="",  # Will be set in background
    )


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """
    List all uploaded documents.
    """
    vector_store = get_vector_store()
    all_documents = vector_store.get_all_documents()

    documents = []
    for doc_metadata in all_documents:
        documents.append(
            DocumentInfo(
                document_id=doc_metadata.get("document_id", ""),
                filename=doc_metadata.get("filename", ""),
                file_type=doc_metadata.get("file_type", ""),
                file_size=int(doc_metadata.get("file_size", 0)),
                char_count=int(doc_metadata.get("char_count", 0)),
                word_count=int(doc_metadata.get("word_count", 0)),
                chunk_count=int(doc_metadata.get("total_chunks", 0)),
                uploaded_at=doc_metadata.get("uploaded_at", ""),
            )
        )

    return DocumentListResponse(documents=documents, total=len(documents))


@router.get("/{document_id}", response_model=DocumentInfo)
async def get_document_info(document_id: str):
    """
    Get information about a specific document.
    """
    vector_store = get_vector_store()
    all_documents = vector_store.get_all_documents()

    for doc_metadata in all_documents:
        if doc_metadata.get("document_id") == document_id:
            return DocumentInfo(
                document_id=document_id,
                filename=doc_metadata.get("filename", ""),
                file_type=doc_metadata.get("file_type", ""),
                file_size=int(doc_metadata.get("file_size", 0)),
                char_count=int(doc_metadata.get("char_count", 0)),
                word_count=int(doc_metadata.get("word_count", 0)),
                chunk_count=int(doc_metadata.get("total_chunks", 0)),
                uploaded_at=doc_metadata.get("uploaded_at", ""),
            )

    raise HTTPException(status_code=404, detail="Document not found")


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its chunks from the vector store.
    """
    vector_store = get_vector_store()

    # Delete from vector store
    deleted_count = vector_store.delete_by_document_id(document_id)

    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete physical file if exists
    for file_path in UPLOAD_DIR.glob(f"{document_id}_*"):
        try:
            file_path.unlink()
            logger.info(f"Deleted file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not delete file {file_path}: {e}")

    return JSONResponse(
        content={
            "message": f"Document deleted successfully",
            "document_id": document_id,
            "chunks_deleted": deleted_count,
        }
    )
