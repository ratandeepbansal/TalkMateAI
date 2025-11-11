"""
Document processing module for RAG system.
Handles parsing, chunking, and metadata extraction for PDF and TXT files.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

# PDF parsing
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

# Text chunking
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a chunk of text from a document."""

    def __init__(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_id: str,
        chunk_index: int,
    ):
        self.text = text
        self.metadata = metadata
        self.chunk_id = chunk_id
        self.chunk_index = chunk_index

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
        }


class DocumentProcessor:
    """Process documents for RAG system."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
    ):
        """
        Initialize document processor.

        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize text splitter
        if RecursiveCharacterTextSplitter:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        else:
            self.text_splitter = None
            logger.warning("RecursiveCharacterTextSplitter not available")

        logger.info(
            f"DocumentProcessor initialized (chunk_size={chunk_size}, overlap={chunk_overlap})"
        )

    def parse_txt(self, file_path: Path) -> str:
        """
        Parse TXT file.

        Args:
            file_path: Path to TXT file

        Returns:
            Extracted text content
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            logger.info(f"Parsed TXT file: {file_path.name} ({len(text)} chars)")
            return text
        except Exception as e:
            logger.error(f"Error parsing TXT file {file_path}: {e}")
            raise

    def parse_pdf(self, file_path: Path) -> str:
        """
        Parse PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text content
        """
        if not PdfReader:
            raise ImportError("PyPDF2 not installed. Install with: pip install pypdf2")

        try:
            reader = PdfReader(str(file_path))
            text_parts = []

            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num}: {e}")
                    continue

            text = "\n\n".join(text_parts)
            logger.info(
                f"Parsed PDF file: {file_path.name} ({len(reader.pages)} pages, {len(text)} chars)"
            )
            return text

        except Exception as e:
            logger.error(f"Error parsing PDF file {file_path}: {e}")
            raise

    def parse_document(self, file_path: Path) -> str:
        """
        Parse document based on file extension.

        Args:
            file_path: Path to document

        Returns:
            Extracted text content
        """
        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            return self.parse_txt(file_path)
        elif suffix == ".pdf":
            return self.parse_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def chunk_text(
        self, text: str, document_id: str, metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk
            document_id: Unique document identifier
            metadata: Document metadata

        Returns:
            List of DocumentChunk objects
        """
        if not self.text_splitter:
            # Fallback: simple chunking
            chunks_text = self._simple_chunk(text)
        else:
            chunks_text = self.text_splitter.split_text(text)

        chunks = []
        for idx, chunk_text in enumerate(chunks_text):
            # Create unique chunk ID
            chunk_id = self._generate_chunk_id(document_id, idx, chunk_text)

            # Add chunk-specific metadata
            chunk_metadata = {
                **metadata,
                "chunk_index": idx,
                "total_chunks": len(chunks_text),
                "document_id": document_id,
            }

            chunk = DocumentChunk(
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_id=chunk_id,
                chunk_index=idx,
            )
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks for document {document_id}")
        return chunks

    def _simple_chunk(self, text: str) -> List[str]:
        """Simple fallback chunking strategy."""
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space

            if current_size >= self.chunk_size:
                chunks.append(" ".join(current_chunk))
                # Keep overlap
                overlap_words = int(
                    len(current_chunk) * (self.chunk_overlap / self.chunk_size)
                )
                current_chunk = (
                    current_chunk[-overlap_words:] if overlap_words > 0 else []
                )
                current_size = sum(len(w) + 1 for w in current_chunk)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _generate_chunk_id(self, document_id: str, chunk_index: int, text: str) -> str:
        """Generate unique chunk ID."""
        content = f"{document_id}_{chunk_index}_{text[:50]}"
        return hashlib.md5(content.encode()).hexdigest()

    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from file.

        Args:
            file_path: Path to file

        Returns:
            Dictionary of metadata
        """
        stat = file_path.stat()

        metadata = {
            "filename": file_path.name,
            "file_type": file_path.suffix.lower(),
            "file_size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "uploaded_at": datetime.now().isoformat(),
        }

        return metadata

    def process_document(
        self, file_path: Path, document_id: Optional[str] = None
    ) -> tuple[str, List[DocumentChunk], Dict[str, Any]]:
        """
        Process a document: parse, extract metadata, and chunk.

        Args:
            file_path: Path to document
            document_id: Optional document ID (auto-generated if not provided)

        Returns:
            Tuple of (full_text, chunks, metadata)
        """
        if not document_id:
            document_id = self._generate_document_id(file_path)

        # Extract metadata
        metadata = self.extract_metadata(file_path)

        # Parse document
        text = self.parse_document(file_path)
        metadata["char_count"] = len(text)
        metadata["word_count"] = len(text.split())

        # Chunk text
        chunks = self.chunk_text(text, document_id, metadata)

        logger.info(f"Processed document: {file_path.name} -> {len(chunks)} chunks")

        return text, chunks, metadata

    def _generate_document_id(self, file_path: Path) -> str:
        """Generate unique document ID from file path and timestamp."""
        content = f"{file_path.name}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


# Singleton instance
_processor_instance = None


def get_document_processor() -> DocumentProcessor:
    """Get or create singleton DocumentProcessor instance."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = DocumentProcessor()
    return _processor_instance
