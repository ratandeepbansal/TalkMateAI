#!/bin/bash

# TalkMateAI RAG System Test Script
# This script tests the RAG (Retrieval Augmented Generation) endpoints

BASE_URL="http://127.0.0.1:8000"
echo "🧪 Testing TalkMateAI RAG System"
echo "================================"
echo ""

# Test 1: Check server health
echo "1️⃣ Testing server connection..."
curl -s "$BASE_URL/stats" > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Server is running!"
else
    echo "❌ Server is not responding. Make sure it's running with 'pnpm dev'"
    exit 1
fi
echo ""

# Test 2: Create a test document
echo "2️⃣ Creating a test document..."
cat > /tmp/test_doc.txt << 'EOF'
TalkMateAI System Information

TalkMateAI is a real-time voice-controlled 3D avatar with multimodal AI capabilities.

Key Features:
- Speech-to-Text using OpenAI Whisper
- Vision understanding with SmolVLM2
- Natural text-to-speech with Kokoro TTS
- 3D avatar animation with lip-sync
- Real-time camera integration

Technical Stack:
- Backend: FastAPI with Python 3.10
- Frontend: Next.js 15 with TypeScript
- Models: Whisper, SmolVLM2, Kokoro TTS
- Communication: WebSocket for real-time interaction

Use Cases:
- Virtual assistant for hands-free interaction
- Customer support with visual context
- Educational tutoring with avatar engagement
- Accessibility tool for vision-impaired users
EOF

echo "✅ Test document created: /tmp/test_doc.txt"
echo ""

# Test 3: Upload document
echo "3️⃣ Uploading document to RAG system..."
UPLOAD_RESPONSE=$(curl -s -X POST "$BASE_URL/api/documents/upload" \
  -F "file=@/tmp/test_doc.txt" \
  -H "accept: application/json")

DOCUMENT_ID=$(echo $UPLOAD_RESPONSE | grep -o '"document_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$DOCUMENT_ID" ]; then
    echo "❌ Document upload failed!"
    echo "Response: $UPLOAD_RESPONSE"
    exit 1
fi

echo "✅ Document uploaded successfully!"
echo "   Document ID: $DOCUMENT_ID"
echo ""

# Wait for background processing
echo "⏳ Waiting 10 seconds for background processing (embedding generation)..."
sleep 10
echo ""

# Test 4: List documents
echo "4️⃣ Listing all documents..."
curl -s "$BASE_URL/api/documents" | python3 -m json.tool | grep -E "(document_id|filename|chunk_count)" | head -10
echo ""

# Test 5: Get document info
echo "5️⃣ Getting document information..."
curl -s "$BASE_URL/api/documents/$DOCUMENT_ID" | python3 -m json.tool
echo ""

# Test 6: Create a persona
echo "6️⃣ Creating a test persona..."
PERSONA_RESPONSE=$(curl -s -X POST "$BASE_URL/api/personas" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "TalkMate Assistant",
    "role": "Technical Support Specialist",
    "description": "Helpful assistant for TalkMateAI questions",
    "system_prompt": "You are a friendly and knowledgeable assistant specializing in TalkMateAI. Use the knowledge base to answer questions accurately. Be concise and helpful.",
    "knowledge_base_ids": [],
    "settings": {
      "retrieval_k": 3,
      "use_mmr": true,
      "mmr_lambda": 0.7,
      "similarity_threshold": 0.5
    }
  }')

PERSONA_ID=$(echo $PERSONA_RESPONSE | grep -o '"id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$PERSONA_ID" ]; then
    echo "❌ Persona creation failed!"
    echo "Response: $PERSONA_RESPONSE"
    exit 1
fi

echo "✅ Persona created successfully!"
echo "   Persona ID: $PERSONA_ID"
echo ""

# Test 7: Link document to persona
echo "7️⃣ Linking document to persona..."
curl -s -X POST "$BASE_URL/api/personas/$PERSONA_ID/documents" \
  -H "Content-Type: application/json" \
  -d "{\"document_id\": \"$DOCUMENT_ID\"}" | python3 -m json.tool | head -20
echo ""

# Test 8: List personas
echo "8️⃣ Listing all personas..."
curl -s "$BASE_URL/api/personas" | python3 -m json.tool
echo ""

# Test 9: Test retrieval (manual API test)
echo "9️⃣ Testing RAG retrieval..."
echo "   To test retrieval programmatically, we'd need to call the retriever directly."
echo "   For now, you can test via Python:"
echo ""
echo "   python3 -c \""
echo "   from app.rag.retriever import get_retriever"
echo "   retriever = get_retriever()"
echo "   results = retriever.retrieve('What is TalkMateAI?', k=3)"
echo "   for r in results: print(f'Score: {r[\\\"similarity\\\"]:.3f} - {r[\\\"text\\\"][:100]}...')"
echo "   \""
echo ""

# Summary
echo "================================"
echo "✅ RAG System Test Complete!"
echo ""
echo "Summary:"
echo "  - Document uploaded: $DOCUMENT_ID"
echo "  - Persona created: $PERSONA_ID"
echo "  - Document linked to persona"
echo ""
echo "🎯 Next Steps:"
echo "  1. Visit http://127.0.0.1:8000/docs for interactive API testing"
echo "  2. Test retrieval with different queries"
echo "  3. Upload more documents (PDF or TXT)"
echo "  4. Create additional personas with different configurations"
echo ""
echo "📚 API Endpoints:"
echo "  - GET  /api/documents          - List all documents"
echo "  - POST /api/documents/upload   - Upload new document"
echo "  - GET  /api/personas           - List all personas"
echo "  - POST /api/personas           - Create new persona"
echo "================================"
