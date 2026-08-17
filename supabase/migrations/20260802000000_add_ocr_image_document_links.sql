alter table if exists public.caremind_image_assets
    add column if not exists ocr_document_id text;

alter table if exists public.caremind_document_chunks
    add column if not exists metadata jsonb not null default '{}'::jsonb;

create index if not exists caremind_document_chunks_metadata_source_idx
    on public.caremind_document_chunks ((metadata->>'source'));

create index if not exists caremind_document_chunks_metadata_image_idx
    on public.caremind_document_chunks ((metadata->>'image_id'));

-- SQLite stores document metadata locally. Supabase vector rows carry OCR provenance
-- in metadata JSON: {"source": "ocr", "image_id": "..."}.
