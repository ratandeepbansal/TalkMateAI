"""
RAG context builder for LLM prompts.
Retrieves relevant documents and builds context for generation.
"""

import logging
from typing import Optional, Dict, Any, List

from .retriever import get_retriever
from .persona_manager import get_persona_manager, Persona

logger = logging.getLogger(__name__)


class RAGContextBuilder:
    """Build context for LLM generation using RAG."""

    def __init__(self):
        """Initialize RAG context builder."""
        self.retriever = get_retriever()
        self.persona_manager = get_persona_manager()
        logger.info("RAGContextBuilder initialized")

    def build_context(
        self,
        query: str,
        persona_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build RAG context for a query.

        Args:
            query: User query text
            persona_id: Optional persona ID to use

        Returns:
            Dictionary with context information
        """
        context = {
            "has_persona": False,
            "has_knowledge": False,
            "persona": None,
            "system_prompt": None,
            "retrieved_docs": [],
            "context_text": "",
        }

        # Get persona if specified
        persona = None
        if persona_id:
            persona = self.persona_manager.get_persona(persona_id)
            if persona:
                context["has_persona"] = True
                context["persona"] = {
                    "name": persona.name,
                    "role": persona.role,
                    "description": persona.description,
                }
                context["system_prompt"] = persona.system_prompt

                # Retrieve relevant documents if persona has knowledge base
                if persona.knowledge_base_ids:
                    try:
                        retrieved_docs = self.retriever.retrieve(
                            query=query,
                            k=persona.settings.get("retrieval_k", 5),
                            use_mmr=persona.settings.get("use_mmr", True),
                            mmr_lambda=persona.settings.get("mmr_lambda", 0.7),
                            filter_dict=(
                                {"document_id": {"$in": persona.knowledge_base_ids}}
                                if persona.knowledge_base_ids
                                else None
                            ),
                        )

                        if retrieved_docs:
                            context["has_knowledge"] = True
                            context["retrieved_docs"] = retrieved_docs
                            context["context_text"] = self._format_retrieved_docs(
                                retrieved_docs
                            )

                            logger.info(
                                f"Retrieved {len(retrieved_docs)} docs for persona {persona.name}"
                            )

                    except Exception as e:
                        logger.error(f"Error retrieving documents: {e}")

        return context

    def _format_retrieved_docs(self, docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a readable context string.

        Args:
            docs: List of retrieved document dictionaries

        Returns:
            Formatted context string
        """
        if not docs:
            return ""

        context_parts = ["Relevant knowledge from the knowledge base:\n"]

        for idx, doc in enumerate(docs, 1):
            text = doc["text"].strip()
            metadata = doc.get("metadata", {})
            filename = metadata.get("filename", "Unknown")
            similarity = doc.get("similarity", 0)

            context_parts.append(
                f"\n[Document {idx} from {filename} - Relevance: {similarity:.2f}]\n{text}\n"
            )

        return "\n".join(context_parts)

    def build_prompt(
        self,
        user_query: str,
        rag_context: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Build complete prompt with RAG context and persona.

        Args:
            user_query: User's query
            rag_context: RAG context from build_context()
            conversation_history: Optional conversation history

        Returns:
            Complete prompt string
        """
        prompt_parts = []

        # Add system prompt if persona specified
        if rag_context.get("system_prompt"):
            prompt_parts.append(f"System: {rag_context['system_prompt']}\n")

        # Add knowledge base context if available
        if rag_context.get("context_text"):
            prompt_parts.append(rag_context["context_text"])
            prompt_parts.append("\n")

        # Add conversation history if available
        if conversation_history:
            prompt_parts.append("Previous conversation:\n")
            for msg in conversation_history[-3:]:  # Last 3 exchanges
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {content}\n")
            prompt_parts.append("\n")

        # Add current user query
        prompt_parts.append(f"User: {user_query}")

        return "".join(prompt_parts)


# Singleton instance
_context_builder_instance = None


def get_context_builder() -> RAGContextBuilder:
    """
    Get or create singleton RAGContextBuilder instance.

    Returns:
        RAGContextBuilder instance
    """
    global _context_builder_instance
    if _context_builder_instance is None:
        _context_builder_instance = RAGContextBuilder()
    return _context_builder_instance
