"""
Retrieval module for RAG system.
Implements various retrieval strategies for document search.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .vector_store import VectorStore, get_vector_store
from .embeddings import EmbeddingGenerator, get_embedding_generator

logger = logging.getLogger(__name__)


class Retriever:
    """Document retrieval with various strategies."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        default_k: int = 5,
        similarity_threshold: float = 0.5,
    ):
        """
        Initialize retriever.

        Args:
            vector_store: VectorStore instance
            embedding_generator: EmbeddingGenerator instance
            default_k: Default number of results to retrieve
            similarity_threshold: Minimum similarity score (0-1, higher is more similar)
        """
        self.vector_store = vector_store or get_vector_store()
        self.embedding_generator = embedding_generator or get_embedding_generator()
        self.default_k = default_k
        self.similarity_threshold = similarity_threshold

        logger.info(
            f"Retriever initialized (k={default_k}, threshold={similarity_threshold})"
        )

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        use_mmr: bool = False,
        mmr_lambda: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query text
            k: Number of results (uses default_k if None)
            filter_dict: Optional metadata filter
            use_mmr: Whether to use MMR (Maximal Marginal Relevance) for diversity
            mmr_lambda: MMR lambda parameter (0=diversity, 1=relevance)

        Returns:
            List of retrieved document chunks with metadata and scores
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        k = k or self.default_k

        # Generate query embedding
        query_embedding = self.embedding_generator.embed_query(query)

        # Retrieve from vector store
        texts, metadatas, distances, ids = self.vector_store.search(
            query_embedding=query_embedding.tolist(),
            n_results=k * 2 if use_mmr else k,  # Get more for MMR filtering
            filter_dict=filter_dict,
        )

        if not texts:
            logger.info("No documents found matching query")
            return []

        # Convert distances to similarity scores (closer distance = higher similarity)
        # ChromaDB uses L2 distance, so we convert: similarity = 1 / (1 + distance)
        similarities = [1 / (1 + d) for d in distances]

        # Filter by threshold
        filtered_results = []
        for text, metadata, similarity, doc_id in zip(
            texts, metadatas, similarities, ids
        ):
            if similarity >= self.similarity_threshold:
                filtered_results.append(
                    {
                        "text": text,
                        "metadata": metadata,
                        "similarity": similarity,
                        "id": doc_id,
                    }
                )

        if not filtered_results:
            logger.info(
                f"No documents above similarity threshold {self.similarity_threshold}"
            )
            return []

        # Apply MMR if requested
        if use_mmr and len(filtered_results) > k:
            filtered_results = self._apply_mmr(
                filtered_results, query_embedding, k, mmr_lambda
            )
        else:
            # Just take top k
            filtered_results = filtered_results[:k]

        logger.info(
            f"Retrieved {len(filtered_results)} documents "
            f"(top similarity: {filtered_results[0]['similarity']:.3f})"
        )

        return filtered_results

    def _apply_mmr(
        self,
        results: List[Dict[str, Any]],
        query_embedding: np.ndarray,
        k: int,
        lambda_param: float,
    ) -> List[Dict[str, Any]]:
        """
        Apply Maximal Marginal Relevance to diversify results.

        Args:
            results: List of retrieved results
            query_embedding: Query embedding vector
            k: Number of final results
            lambda_param: Balance between relevance and diversity (0-1)

        Returns:
            Diversified list of results
        """
        # Get embeddings for all results (we'd need to store these or re-compute)
        # For now, use a simplified version based on text similarity

        selected = []
        remaining = results.copy()

        # Always select the most relevant first
        selected.append(remaining.pop(0))

        while len(selected) < k and remaining:
            best_score = -float("inf")
            best_idx = 0

            for idx, candidate in enumerate(remaining):
                # Relevance score
                relevance = candidate["similarity"]

                # Diversity score (1 - max similarity to selected)
                diversity = 1.0
                for selected_item in selected:
                    # Simple text-based diversity (could use embeddings for better results)
                    text_overlap = self._text_similarity(
                        candidate["text"], selected_item["text"]
                    )
                    diversity = min(diversity, 1 - text_overlap)

                # MMR score
                mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected.append(remaining.pop(best_idx))

        logger.debug(f"MMR selected {len(selected)} diverse results")
        return selected

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def retrieve_by_document_id(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks from a specific document.

        Args:
            document_id: Document ID

        Returns:
            List of chunks from the document
        """
        texts, metadatas, distances, ids = self.vector_store.search(
            query_embedding=[0.0] * self.embedding_generator.get_embedding_dimension(),
            n_results=1000,  # Get many results
            filter_dict={"document_id": str(document_id)},
        )

        results = []
        for text, metadata, doc_id in zip(texts, metadatas, ids):
            results.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "id": doc_id,
                }
            )

        return results


# Singleton instance
_retriever_instance = None


def get_retriever(
    default_k: int = 5,
    similarity_threshold: float = 0.5,
) -> Retriever:
    """
    Get or create singleton Retriever instance.

    Args:
        default_k: Default number of results
        similarity_threshold: Minimum similarity threshold

    Returns:
        Retriever instance
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever(
            default_k=default_k,
            similarity_threshold=similarity_threshold,
        )
    return _retriever_instance
