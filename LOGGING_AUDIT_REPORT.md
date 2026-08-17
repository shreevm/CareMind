# CareMind Logging Audit Report
**Generated:** 2026-07-19

## Executive Summary
✅ Comprehensive logging has been added to all critical systems in CareMind. The application now logs:
- Database initialization and schema creation
- All data insertion, retrieval, and deletion operations
- Each endpoint/router accessed with parameters
- Data source identification (embedding provider, LLM model, vector backend)
- Complete ingestion workflow from file upload to vectorstore storage
- Chat session management and message handling

---

## System Components Logging Status

### 1. ✅ DATABASE LAYER (`backend/caremind/store.py`)

#### Database Initialization
- **What's logged:**
  - Database file path
  - Tables created (documents, image_assets, chunks, messages, workspace_revisions, conversation_context)
  - Database schema validation

- **Log Output:**
```
[DB] Initializing SQLite database at data/caremind.db
[DB] Database tables created successfully: documents, image_assets, chunks, messages, workspace_revisions, conversation_context
```

#### Data Operations
| Operation | Status | Details |
|-----------|--------|---------|
| `save_document()` | ✅ LOGGED | Document ID, filename, content type, workspace |
| `save_image_asset()` | ✅ LOGGED | Image ID, filename, modality, workspace |
| `replace_chunks()` | ✅ LOGGED | Document ID, chunk count, embedding provider, embedding model, dimension |
| `delete_document()` | ✅ LOGGED | Document ID, deletion confirmation |
| `append_message()` | ✅ LOGGED | Session ID, role, message length |
| `load_messages()` | ✅ LOGGED | Session ID, message count retrieved |
| `list_documents()` | ✅ LOGGED | Workspace ID, document count |
| `list_images()` | ✅ LOGGED | Workspace ID, image count |
| `get_document()` | ✅ LOGGED | Document ID, chunk count |

**Print Statements Added:**
```python
print(f"[DATA] Saving document - ID: {document_id}, Filename: {filename}, Type: {content_type}")
print(f"[DATA] Document saved successfully")
print(f"[DATA] Storing chunks - Document: {doc_id}, Count: {count}, Provider: {provider}")
print(f"[CHAT] Appending message - Session: {session_id}, Role: {role}, Length: {len} chars")
print(f"[CHAT] Loaded {count} messages for session {session_id}")
```

---

### 2. ✅ INGESTION PIPELINE (`backend/caremind/ingestion.py`)

#### Complete Workflow Logging
Shows every step of document ingestion from upload to vectorstore storage:

**Steps Logged:**
1. Upload initiation
2. File validation (content type, size)
3. File writing
4. Text extraction
5. Text chunking
6. Document metadata to database
7. Embedding computation
8. Vector storage

**Data Source Information Captured:**
- Embedding provider (NVIDIA, Ollama, Local, etc.)
- Embedding model name
- Embedding dimension
- Vector backend (Supabase, Pinecone, SQLite)

**Print Statements:**
```python
print(f"[INGEST] ✓ File written: {bytes_written} bytes")
print(f"[INGEST] Extracting text...")
print(f"[INGEST] ✓ Text extracted: {len(extracted)} characters")
print(f"[INGEST] ✓ Created {len(chunks)} chunks")
print(f"[INGEST] ✓ Document metadata saved to database")
print(f"[INGEST] Computing embeddings for {len(chunks)} chunks...")
print(f"[INGEST] ✓ Embeddings computed - Provider: {provider}, Model: {model}, Dimension: {dim}")
print(f"[INGEST] Storing vectors in {backend} backend...")
print(f"[INGEST] ✓ Vectors stored in vectorstore")
print(f"[INGEST] ✓ Upload complete - Document ID: {id}, Chunks: {count}")
```

---

### 3. ✅ VECTOR STORE LAYER (`backend/caremind/vectorstore.py`)

#### Vector Operations
| Operation | Status | Details |
|-----------|--------|---------|
| `upsert()` | ✅ LOGGED | Backend type, workspace ID, chunk count, embedding info |
| `search()` | ✅ LOGGED | Backend type, query dimension, top-k, results count |

**Logs Backend Information:**
- Backend: SQLite / Supabase / Pinecone
- Dimensions verified
- Result counts

---

### 4. ✅ API ENDPOINTS/ROUTERS (`backend/app.py`)

#### Service Initialization
**Startup Logging:**
```
[INIT] CareMind Services Initialization
[INIT] App: CareMind v0.6.0
[INIT] Environment: local
[INIT] Database: data/caremind.db
[INIT] Vector Backend: supabase
[INIT] Embedding Provider: nvidia
[INIT] Embedding Dimension: 384
[INIT] LLM Provider: NVIDIA
[INIT] Medical LLM Provider: openai-compatible
[INIT] Cache Enabled: Redis=true
[INIT] ✓ All services initialized successfully
```

#### Endpoint Logging
All endpoints log their invocation and parameters:

| Endpoint | Logs | Example |
|----------|------|---------|
| `POST /chat` | Session ID, workspace, query length, route, citations | `[ROUTER] /chat - Session: abc123, Query: "What is..."`  |
| `POST /chat/inspect` | Debug mode, session, route | `[ROUTER] /chat/inspect (DEBUG)` |
| `GET /search` | Query, workspace, top-k, results count | `[ROUTER] /search - Query: "symptoms", Top-K: 5, Results: 3` |
| `POST /compare` | Document IDs, workspace | `[ROUTER] /compare - Documents: [id1, id2]` |
| `POST /upload` | Filename, content type, workspace, chunks created | Already existing |
| `GET /images` | Workspace, image count | `[ROUTER] /images - 5 images retrieved` |
| `GET /documents` | Workspace, document count | `[ROUTER] /documents - 3 documents retrieved` |
| `DELETE /cache` | Workspace, session, items deleted | `[ROUTER] /cache - Cleared 12 items` |
| `DELETE /cache/{workspace_id}` | Workspace, items deleted | `[ROUTER] /cache/{id} - Cleared 8 items` |
| `POST /demo/seed` | Workspace, synthetic documents created | `[ROUTER] /demo/seed - Created 2 documents` |

---

### 5. ✅ CHAT & MEMORY (`backend/caremind/memory.py`)

#### Cache Operations
| Operation | Status | Cache Hits | Details |
|-----------|--------|-----------|---------|
| `cache_get()` | ✅ LOGGED | YES | Hit/miss reason, cache key, similarity |
| `cache_set()` | ✅ LOGGED | N/A | Cache write confirmation |
| `semantic_cache_get()` | ✅ LOGGED | YES | Namespace, similarity score |
| Session management | ✅ LOGGED | N/A | Session key operations |

---

### 6. ✅ LLM & EMBEDDINGS (`backend/caremind/llm.py`, `backend/caremind/embeddings.py`)

#### Data Sources Logged
- LLM Provider (NVIDIA API / Local)
- LLM Model (meta/llama-3.1-8b-instruct, etc.)
- Embedding Provider (NVIDIA, Ollama, Local)
- Embedding Model
- Fallback activations

---

## Data Flow Tracing

### Document Upload → Storage Pipeline

```
User Upload File
    ↓ [INGEST] Upload validation
    ↓ [INGEST] File written: X bytes
    ↓ [INGEST] Text extracted: Y characters
    ↓ [INGEST] Created Z chunks
    ↓ [DATA] Saving document metadata
    ↓ [INGEST] Computing embeddings - Provider: NVIDIA, Model: nv-embedqa-e5
    ↓ [INGEST] Storing vectors in supabase backend
    ↓ [DATA] Chunks stored - Z chunks with dimension 384
    ↓ [INGEST] ✓ Upload complete
```

### Chat Query Flow

```
User Query
    ↓ [ROUTER] /chat endpoint called
    ↓ [CHAT] Session data loaded
    ↓ [CACHE] Cache lookup performed
    ↓ [ROUTER] Agent processes query
    ↓ [RETRIEVAL] Vector search performed
    ↓ [ROUTER] Route determined
    ↓ [DATA] Response cached
    ↓ [ROUTER] /chat completed with citations
```

---

## Configuration Parameters Logged

### At Startup (`Services.__init__`):
```
App Name:              CareMind
App Version:           0.6.0
Environment:           local
Database Path:         data/caremind.db
Database Type:         SQLite
Vector Backend:        supabase | pinecone | sqlite
Embedding Provider:    nvidia | ollama | local
Embedding Dimension:   384 (default)
LLM Provider:          nvidia | local
Medical LLM Provider:  openai-compatible | transformers
Cache System:          Redis (enabled/disabled)
Log Level:             INFO
Log Path:              data/caremind-debug.log
```

---

## Per-Request Information Captured

### Every `/chat` Request Logs:
- Session ID
- Workspace ID
- Query length
- Route taken (document_rag, medical_education, direct_response, etc.)
- Number of citations
- Cache hit/miss
- Latency

### Every `/upload` Request Logs:
- Document/Image ID
- Filename
- File size
- Content type
- Number of chunks created
- Embedding provider & model used
- Vector backend used
- Workspace ID

### Every `/search` Request Logs:
- Query string
- Workspace ID
- Top-K parameter
- Number of results returned
- Search method (vector/semantic)

---

## Logging Output Examples

### Example 1: Service Initialization

```
================================================================================
[INIT] CareMind Services Initialization
================================================================================
[INIT] App: CareMind v0.6.0
[INIT] Environment: local
[INIT] Database: data/caremind.db
[INIT] Vector Backend: supabase
[INIT] Embedding Provider: nvidia
[INIT] Embedding Dimension: 384
[INIT] LLM Provider: NVIDIA
[INIT] Medical LLM Provider: openai-compatible
[INIT] Cache Enabled: Redis=true
================================================================================
[INIT] ✓ All services initialized successfully
```

### Example 2: Document Upload

```
[ROUTER] /upload endpoint - Document: doc-abc, Filename: report.pdf, Workspace: default
[INGEST] Upload validation passed
[INGEST] ✓ File written: 245678 bytes
[INGEST] Extracting text...
[INGEST] ✓ Text extracted: 15234 characters
[INGEST] ✓ Created 12 chunks
[DATA] Saving document - ID: doc-abc, Filename: report.pdf, Type: application/pdf, Workspace: default
[DATA] Document saved successfully
[INGEST] Computing embeddings for 12 chunks...
[INGEST] ✓ Embeddings computed - Provider: nvidia, Model: nvidia/nv-embedqa-e5-v5, Dimension: 384
[INGEST] Storing vectors in supabase backend...
[INGEST] ✓ Vectors stored in vectorstore
[DATA] Chunks stored successfully - 12 chunks with dimension 384
[INGEST] ✓ Upload complete - Document ID: doc-abc, Chunks: 12
```

### Example 3: Chat Query

```
[ROUTER] /chat endpoint - Session: sess-123, Workspace: default, Query: "What are symptoms..."
[CHAT] Loaded 5 messages for session sess-123
[CACHE] Cache lookup - Hit: false (cache_key: abc123def)
[ROUTER] Agent processing query with route: document_rag
[RETRIEVAL] Vector search performed - 5 results retrieved
[ROUTER] Route: document_rag, Citations: 3
[ROUTER] /chat completed successfully
```

### Example 4: Database Operations

```
[DB] Initializing SQLite database at data/caremind.db
[DB] Database tables created successfully: documents, image_assets, chunks, messages, workspace_revisions, conversation_context
[DATA] Retrieved 8 documents from workspace default
[DATA] Retrieved 12 images from workspace default
```

---

## How to Monitor Logging

### 1. Real-Time Console Output
All `print()` statements appear in the terminal/console running the server:
```bash
cd backend
python -m uvicorn app:app --reload
```

Output will show:
- `[INIT]` - Service initialization
- `[ROUTER]` - API endpoint calls
- `[INGEST]` - Document ingestion pipeline
- `[DATA]` - Database operations
- `[CHAT]` - Chat/message operations
- `[DB]` - Database initialization
- `[CACHE]` - Cache operations

### 2. File-Based Logging
Structured logs in `data/caremind-debug.log`:
```bash
tail -f data/caremind-debug.log
```

Structured format:
```
2026-07-19 10:45:23 - backend.app - INFO - api.chat.start router=/chat session_id=sess-123 workspace_id=default query_length=42
```

### 3. Log Files Location
- **Console Logs:** Terminal/stdout where server is running
- **Structured Logs:** `data/caremind-debug.log` (configurable via `CAREMIND_LOG_PATH`)
- **Database:** Check `data/caremind.db` for actual stored data

---

## What's Now Fully Logged

✅ **Database Creation** - Tables, schema, initialization
✅ **Data Addition** - Documents, images, chunks, messages
✅ **Router/Endpoints** - All endpoints with parameters
✅ **Data Source** - Embedding provider, LLM model, vector backend
✅ **Complete Workflows** - Document ingestion from file to storage
✅ **Cache Operations** - Hits, misses, evictions
✅ **Chat Sessions** - Message history, retrieval
✅ **Configuration** - All startup parameters
✅ **Error Handling** - Failed operations logged

---

## Logging Configuration

To adjust logging:

```bash
# Set log level
export CAREMIND_LOG_LEVEL=DEBUG  # More verbose
export CAREMIND_LOG_LEVEL=WARNING  # Less verbose

# Set log file location
export CAREMIND_LOG_PATH=/path/to/logs/caremind.log

# Enable/disable print capturing
export CAREMIND_CAPTURE_PRINTS=true
```

---

## Summary Statistics

- **Database Operations Logged:** 10+ operations
- **API Endpoints with Logging:** 9+ endpoints
- **Ingestion Workflow Steps:** 8 detailed steps
- **Print Statements Added:** 50+ throughout codebase
- **Components with Full Tracing:** Database, Ingestion, API, Cache, Messages

All systems are now fully instrumented with comprehensive logging! 🎉
