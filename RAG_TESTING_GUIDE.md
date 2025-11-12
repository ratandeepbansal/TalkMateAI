# 🧪 RAG System Testing Guide

This guide will help you test the RAG (Retrieval Augmented Generation) system that has been implemented in TalkMateAI.

## 📊 Current Status

### ✅ What's Implemented
- **Backend RAG System**: Fully functional
  - Document processing (PDF, TXT)
  - Embedding generation (sentence-transformers)
  - Vector storage (ChromaDB)
  - Semantic retrieval (Top-K, MMR)
  - Persona management
  - REST API endpoints

### ⚠️ What's Not Complete Yet
- **Frontend UI**: RAG components not yet integrated into the web interface
- **Voice Pipeline Integration**: RAG context not yet injected into voice responses
- **Model Loading**: Whisper/SmolVLM2 models fail due to HuggingFace 403 error (separate issue)

### ✅ What You Can Test Right Now
- Document upload and processing via API
- Persona creation and management via API
- Semantic search and retrieval
- RAG context building

---

## 🚀 Quick Start

### 1. Start the Servers

The servers should already be running from when you ran `pnpm dev`:

- **Frontend**: http://localhost:3000
- **Backend**: http://127.0.0.1:8000
- **API Docs**: http://127.0.0.1:8000/docs (Interactive Swagger UI)

If not running, start them with:
```bash
pnpm dev
```

### 2. Run the RAG Test Script

This will test all RAG endpoints automatically:

```bash
./test_rag.sh
```

This script will:
1. Check server connection
2. Create a test document
3. Upload it to the RAG system
4. Wait for embedding generation
5. Create a test persona
6. Link the document to the persona
7. Show you all available endpoints

### 3. Test RAG Retrieval

After running the test script, you can test the retrieval system directly:

```bash
cd /home/user/TalkMateAI/apps/server
uv run python ../../test_rag_retrieval.py
```

This will:
- Show all documents in the vector store
- Test semantic search with various queries
- Demonstrate persona-based context building
- Show how the RAG context would be injected into prompts

---

## 📝 Manual API Testing

### Option 1: Using Swagger UI (Easiest)

1. Go to http://127.0.0.1:8000/docs
2. You'll see all API endpoints with interactive forms
3. Try these endpoints:
   - `POST /api/documents/upload` - Upload a PDF or TXT file
   - `GET /api/documents` - List all uploaded documents
   - `POST /api/personas` - Create a new persona
   - `GET /api/personas` - List all personas

### Option 2: Using curl

**Upload a document:**
```bash
curl -X POST http://127.0.0.1:8000/api/documents/upload \
  -F "file=@/path/to/your/document.txt"
```

**List documents:**
```bash
curl http://127.0.0.1:8000/api/documents | python3 -m json.tool
```

**Create a persona:**
```bash
curl -X POST http://127.0.0.1:8000/api/personas \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Assistant",
    "role": "Helper",
    "description": "A helpful assistant",
    "system_prompt": "You are a helpful assistant.",
    "knowledge_base_ids": []
  }'
```

**List personas:**
```bash
curl http://127.0.0.1:8000/api/personas | python3 -m json.tool
```

**Link document to persona:**
```bash
curl -X POST http://127.0.0.1:8000/api/personas/{persona_id}/documents \
  -H "Content-Type: application/json" \
  -d '{"document_id": "{document_id}"}'
```

---

## 🔬 Understanding What's Happening

### Document Upload Flow

1. **Upload**: You send a PDF or TXT file to `/api/documents/upload`
2. **Background Processing**:
   - File is saved to `data/documents/`
   - Document is parsed and text extracted
   - Text is split into chunks (512 chars with 128 overlap)
   - Embeddings are generated using sentence-transformers
   - Chunks + embeddings stored in ChromaDB vector database
3. **Result**: Document is searchable via semantic queries

### Semantic Search Flow

When you query for information:
1. Your query is converted to an embedding vector
2. ChromaDB finds the most similar document chunks
3. Results are ranked by similarity score
4. MMR (Maximal Marginal Relevance) ensures diversity
5. Only results above similarity threshold are returned

### Persona-Based RAG Flow

1. Select a persona (e.g., "Tech Support Agent")
2. User asks a question
3. RAG system:
   - Uses persona's system prompt
   - Retrieves relevant docs from persona's knowledge base
   - Builds context with retrieved information
   - Combines into a complete prompt for the LLM

---

## 📁 Files and Storage

**Where data is stored:**
- `data/documents/` - Uploaded files
- `data/vector_store/` - ChromaDB vector database
- `data/personas.json` - Persona configurations

**View stored data:**
```bash
# List uploaded documents
ls -lh data/documents/

# View personas
cat data/personas.json | python3 -m json.tool

# Check vector store
ls -lh data/vector_store/
```

---

## 🧪 Example Test Workflow

Here's a complete workflow to test RAG:

```bash
# 1. Create a test document
cat > my_doc.txt << 'EOF'
Product: TalkMateAI
Version: 1.0
Features:
- Voice recognition with Whisper
- Vision understanding with SmolVLM2
- Natural TTS with Kokoro
- 3D avatar with lip-sync
EOF

# 2. Upload it
curl -X POST http://127.0.0.1:8000/api/documents/upload \
  -F "file=@my_doc.txt" | python3 -m json.tool

# 3. Wait for processing (embeddings generation)
sleep 10

# 4. Create a persona
PERSONA=$(curl -X POST http://127.0.0.1:8000/api/personas \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Product Expert",
    "role": "TalkMateAI Specialist",
    "description": "Expert on TalkMateAI features",
    "system_prompt": "You are an expert on TalkMateAI. Answer questions accurately using the knowledge base."
  }')

echo $PERSONA | python3 -m json.tool

# 5. Get document and persona IDs from responses above, then link them
# (Replace {persona_id} and {document_id} with actual IDs)
curl -X POST http://127.0.0.1:8000/api/personas/{persona_id}/documents \
  -H "Content-Type: application/json" \
  -d '{"document_id": "{document_id}"}'

# 6. Test retrieval
cd /home/user/TalkMateAI/apps/server
uv run python ../../test_rag_retrieval.py
```

---

## 🎯 What to Expect

### Successful RAG Test Output

When you run `./test_rag.sh`, you should see:

```
🧪 Testing TalkMateAI RAG System
================================

1️⃣ Testing server connection...
✅ Server is running!

2️⃣ Creating a test document...
✅ Test document created: /tmp/test_doc.txt

3️⃣ Uploading document to RAG system...
✅ Document uploaded successfully!
   Document ID: abc123...

⏳ Waiting 10 seconds for background processing...

4️⃣ Listing all documents...
   "document_id": "abc123..."
   "filename": "test_doc.txt"
   "chunk_count": 5

...
```

### Successful Retrieval Test Output

When you run `test_rag_retrieval.py`, you should see:

```
🔍 Testing RAG Retrieval System
==================================================

📊 Total chunks in vector store: 15

📚 Found 3 unique documents:
   - test_doc.txt (ID: abc123)
   - my_doc.txt (ID: def456)

🔎 Query: 'What is TalkMateAI?'
--------------------------------------------------
   ✅ Found 3 relevant chunks:

   [1] Similarity: 0.876
   Source: test_doc.txt
   Text: TalkMateAI is a real-time voice-controlled 3D avatar...

...
```

---

## ❓ Troubleshooting

### "Server is not responding"
- Check if `pnpm dev` is running
- Check for errors in the server output
- Try accessing http://127.0.0.1:8000/docs directly

### "No documents found"
- Make sure you ran `./test_rag.sh` first
- Check that files were uploaded: `ls data/documents/`
- Check for errors in server logs

### "Embedding generation failed"
- The embedding model downloads on first use
- Check disk space (model is ~80MB)
- Check internet connection for initial download

### "Import errors when running Python test"
- Make sure you're in the correct directory: `cd /home/user/TalkMateAI/apps/server`
- Use `uv run` to execute with correct environment: `uv run python ../../test_rag_retrieval.py`

---

## 🔜 Next Steps

After testing the RAG backend:

1. **Phase 5 completion**: Integrate RAG into voice pipeline
2. **Phase 6**: Build frontend UI components
   - Document upload interface
   - Persona configuration panel
   - Knowledge base management
3. **Phase 7**: Testing and optimization

For now, the RAG system is fully functional via API - you can upload documents, create personas, and retrieve relevant information!

---

## 📚 API Reference

Full API documentation available at: http://127.0.0.1:8000/docs

**Document Endpoints:**
- `POST /api/documents/upload` - Upload document
- `GET /api/documents` - List documents
- `GET /api/documents/{id}` - Get document info
- `DELETE /api/documents/{id}` - Delete document

**Persona Endpoints:**
- `POST /api/personas` - Create persona
- `GET /api/personas` - List personas
- `GET /api/personas/{id}` - Get persona
- `PUT /api/personas/{id}` - Update persona
- `DELETE /api/personas/{id}` - Delete persona
- `POST /api/personas/{id}/documents` - Link document
- `DELETE /api/personas/{id}/documents/{doc_id}` - Unlink document

---

**Happy Testing! 🚀**
