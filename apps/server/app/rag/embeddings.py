"""
Embeddings module for RAG system.
Handles text embedding generation using sentence-transformers.
"""

import logging
from typing import List, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: Name of sentence-transformers model
            device: Device to run model on ('cpu' or 'cuda')
        """
        if not SentenceTransformer:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.device = device

        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Embedding model loaded (dimension: {self.embedding_dim})")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding

    def embed_texts(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Array of embedding vectors
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.
        Alias for embed_text() for consistency with retrieval systems.

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        return self.embed_text(query)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.embedding_dim


# Singleton instance
_embedding_generator_instance = None


def get_embedding_generator(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
) -> EmbeddingGenerator:
    """
    Get or create singleton EmbeddingGenerator instance.

    Args:
        model_name: Name of sentence-transformers model
        device: Device to run model on

    Returns:
        EmbeddingGenerator instance
    """
    global _embedding_generator_instance
    if _embedding_generator_instance is None:
        _embedding_generator_instance = EmbeddingGenerator(
            model_name=model_name, device=device
        )
    return _embedding_generator_instance
