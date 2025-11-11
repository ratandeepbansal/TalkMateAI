"""
Vector store module for RAG system.
Handles storage and retrieval of document embeddings using ChromaDB.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-based vector store for document embeddings."""

    def __init__(
        self,
        persist_directory: str = "data/vector_store",
        collection_name: str = "documents",
    ):
        """
        Initialize vector store.

        Args:
            persist_directory: Directory to persist vector data
            collection_name: Name of the collection
        """
        if not chromadb:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb"
            )

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document embeddings for RAG"},
        )

        logger.info(
            f"VectorStore initialized (collection: {collection_name}, "
            f"documents: {self.collection.count()})"
        )

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        """
        Add documents to vector store.

        Args:
            texts: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: List of unique document IDs
        """
        if not texts:
            logger.warning("No documents to add")
            return

        # Ensure all lists have same length
        assert (
            len(texts) == len(embeddings) == len(metadatas) == len(ids)
        ), "All input lists must have the same length"

        # Convert metadata values to strings (ChromaDB requirement)
        processed_metadatas = []
        for metadata in metadatas:
            processed_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (dict, list)):
                    processed_metadata[key] = json.dumps(value)
                else:
                    processed_metadata[key] = str(value)
            processed_metadatas.append(processed_metadata)

        # Add to collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=processed_metadatas,
            ids=ids,
        )

        logger.info(f"Added {len(texts)} documents to vector store")

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float], List[str]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            Tuple of (texts, metadatas, distances, ids)
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict,
        )

        texts = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []
        ids = results["ids"][0] if results["ids"] else []

        logger.info(
            f"Search returned {len(texts)} results (top distance: {distances[0] if distances else 'N/A'})"
        )

        return texts, metadatas, distances, ids

    def delete_by_document_id(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a document.

        Args:
            document_id: Document ID to delete

        Returns:
            Number of chunks deleted
        """
        # Get all chunks for this document
        results = self.collection.get(
            where={"document_id": str(document_id)},
        )

        if not results["ids"]:
            logger.warning(f"No chunks found for document_id: {document_id}")
            return 0

        # Delete the chunks
        self.collection.delete(ids=results["ids"])

        count = len(results["ids"])
        logger.info(f"Deleted {count} chunks for document_id: {document_id}")
        return count

    def delete_by_ids(self, ids: List[str]) -> None:
        """
        Delete documents by IDs.

        Args:
            ids: List of document IDs to delete
        """
        if not ids:
            return

        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} chunks by ID")

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """
        Get documents by IDs.

        Args:
            ids: List of document IDs

        Returns:
            Dictionary with documents, metadatas, and embeddings
        """
        results = self.collection.get(ids=ids)
        return results

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all unique documents in the store.

        Returns:
            List of document metadata
        """
        results = self.collection.get()

        if not results["metadatas"]:
            return []

        # Group by document_id and get unique documents
        documents = {}
        for metadata in results["metadatas"]:
            doc_id = metadata.get("document_id")
            if doc_id and doc_id not in documents:
                documents[doc_id] = metadata

        return list(documents.values())

    def count(self) -> int:
        """Get total number of chunks in the store."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete all data from the collection (use with caution!)."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"description": "Document embeddings for RAG"},
        )
        logger.warning("Vector store has been reset")


# Singleton instance
_vector_store_instance = None


def get_vector_store(
    persist_directory: str = "data/vector_store",
    collection_name: str = "documents",
) -> VectorStore:
    """
    Get or create singleton VectorStore instance.

    Args:
        persist_directory: Directory to persist vector data
        collection_name: Name of the collection

    Returns:
        VectorStore instance
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    return _vector_store_instance
