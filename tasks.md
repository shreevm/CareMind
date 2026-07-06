# CareMind Tasks

## Phase 1: Foundation
- [x] Create repo structure.
- [x] Add environment variable handling.
- [x] Set up FastAPI app.
- [x] Add health endpoint.
- [ ] Add logging and config modules.

## Phase 2: Documents
- [x] Implement PDF/text upload.
- [x] Extract text from uploaded files.
- [x] Chunk documents.
- [x] Generate embeddings.
- [x] Upsert vectors to Pinecone.
- [x] Add local fallback retrieval.

## Phase 3: Agent
- [x] Implement query router.
- [x] Add document QA route.
- [x] Add education QA route.
- [x] Add report comparison route.
- [x] Add clarification route.
- [x] Add emergency redirect route.
- [x] Add citation formatting.

## Phase 4: Memory and Cache
- [x] Add Redis session storage.
- [x] Add exact-match response cache.
- [x] Add semantic cache.
- [x] Add SQLite fallback.

## Phase 5: Safety
- [x] Add prompt injection screening.
- [x] Add medical safety screening.
- [x] Add disclaimer injection.
- [ ] Add output leak checks.

## Phase 6: Evaluation
- [x] Create internal eval set.
- [ ] Create adversarial guardrail set.
- [x] Implement evaluate.py.
- [x] Save eval runs to JSON.
- [x] Compare against baseline.
- [x] Emit PASS/WARN/FAIL results.

## Phase 7: UI
- [x] Build web chat UI.
- [x] Add document upload UI.
- [x] Add citation display.
- [x] Add comparison UI.
- [x] Add metrics page.

## Phase 8: VS Code
- [x] Build extension scaffold.
- [x] Add sidebar chat.
- [x] Connect to backend.
- [x] Sync session and workspace.

## Phase 9: Voice
- [x] Add microphone input.
- [x] Transcribe speech to text.
- [x] Send transcript to agent.
- [ ] Add optional TTS.

## Phase 10: Imaging
- [x] Add image upload.
- [x] Store image metadata.
- [x] Implement image grounding route.
- [ ] Add vision model integration.

## Phase 11: Observability
- [ ] Add LangSmith tracing.
- [x] Add Prometheus metrics.
- [ ] Add Grafana-ready panels.
