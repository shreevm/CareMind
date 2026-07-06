# CareMind Data Model

## Entities

### User
- id
- username
- password_hash
- created_at

### Workspace
- id
- user_id
- name
- created_at

### Session
- id
- workspace_id
- started_at
- last_active_at

### Document
- id
- workspace_id
- filename
- file_type
- source_type
- storage_path
- created_at

### ImageAsset
- id
- workspace_id
- document_id
- modality
- storage_path
- created_at

### Chunk
- id
- document_id
- chunk_index
- text
- embedding_id
- metadata

### Citation
- id
- answer_id
- source_type
- source_id
- chunk_id
- quote
- offset_start
- offset_end

### Message
- id
- session_id
- role
- content
- modality
- created_at

### CacheEntry
- id
- workspace_id
- doc_set_hash
- query_text
- query_embedding
- answer
- citations
- route
- ttl_expires_at

### EvalRun
- id
- created_at
- dataset_name
- metrics_json
- status
- baseline_flag

### TraceRef
- id
- eval_run_id
- route
- langsmith_trace_url
- created_at