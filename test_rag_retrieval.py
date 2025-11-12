#!/usr/bin/env python3
"""
Test RAG retrieval system directly
This script demonstrates how the RAG context builder works
"""

import sys
import os

# Add the server directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "apps/server"))

from app.rag.retriever import get_retriever
from app.rag.persona_manager import get_persona_manager
from app.rag.context_builder import get_context_builder
from app.rag.vector_store import get_vector_store


def test_retrieval():
    """Test document retrieval"""
    print("🔍 Testing RAG Retrieval System")
    print("=" * 50)
    print()

    # Get instances
    retriever = get_retriever()
    vector_store = get_vector_store()
    context_builder = get_context_builder()

    # Check document count
    doc_count = vector_store.count()
    print(f"📊 Total chunks in vector store: {doc_count}")
    print()

    if doc_count == 0:
        print("❌ No documents found! Please upload documents first:")
        print("   Run: ./test_rag.sh")
        return

    # List all unique documents
    documents = vector_store.get_all_documents()
    print(f"📚 Found {len(documents)} unique documents:")
    for doc in documents:
        print(
            f"   - {doc.get('filename', 'Unknown')} (ID: {doc.get('document_id', 'N/A')})"
        )
    print()

    # Test queries
    test_queries = [
        "What is TalkMateAI?",
        "What models does it use?",
        "What are the key features?",
        "How does the technical stack work?",
    ]

    for query in test_queries:
        print(f"🔎 Query: '{query}'")
        print("-" * 50)

        try:
            results = retriever.retrieve(query, k=3)

            if not results:
                print("   ❌ No relevant documents found")
            else:
                print(f"   ✅ Found {len(results)} relevant chunks:")
                for idx, result in enumerate(results, 1):
                    similarity = result.get("similarity", 0)
                    text = result.get("text", "")
                    metadata = result.get("metadata", {})

                    print(f"\n   [{idx}] Similarity: {similarity:.3f}")
                    print(f"   Source: {metadata.get('filename', 'Unknown')}")
                    print(f"   Text: {text[:200]}...")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        print()

    # Test persona-based retrieval
    print("=" * 50)
    print("🎭 Testing Persona-Based Context Building")
    print("=" * 50)
    print()

    persona_manager = get_persona_manager()
    personas = persona_manager.list_personas()

    if personas:
        persona = personas[0]
        print(f"Using persona: {persona.name} (ID: {persona.id})")
        print(f"Knowledge base: {len(persona.knowledge_base_ids)} documents")
        print()

        query = "Explain the features of TalkMateAI"
        print(f"Query: '{query}'")
        print()

        try:
            rag_context = context_builder.build_context(
                query=query, persona_id=persona.id
            )

            print(f"Has persona: {rag_context['has_persona']}")
            print(f"Has knowledge: {rag_context['has_knowledge']}")
            print(f"Retrieved docs: {len(rag_context['retrieved_docs'])}")
            print()

            if rag_context["context_text"]:
                print("Generated context:")
                print("-" * 50)
                print(rag_context["context_text"][:500] + "...")
                print()

            # Build full prompt
            prompt = context_builder.build_prompt(
                user_query=query, rag_context=rag_context
            )

            print("Full prompt that would be sent to LLM:")
            print("-" * 50)
            print(prompt[:800] + "...")

        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("ℹ️  No personas found. Create one first:")
        print("   Run: ./test_rag.sh")

    print()
    print("=" * 50)
    print("✅ RAG Retrieval Test Complete!")
    print()


if __name__ == "__main__":
    test_retrieval()
