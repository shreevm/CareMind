create extension if not exists vector with schema extensions;

create table if not exists public.caremind_document_chunks_v2 (
    chunk_id text primary key,
    document_id text not null,
    workspace_id text not null,
    document_name text not null,
    page integer,
    position integer not null,
    text text not null,
    embedding extensions.vector(1024) not null,
    embedding_provider text not null default '',
    embedding_model text not null default '',
    embedding_dimension integer not null default 1024,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists caremind_document_chunks_v2_workspace_idx
    on public.caremind_document_chunks_v2 (workspace_id);

create index if not exists caremind_document_chunks_v2_document_idx
    on public.caremind_document_chunks_v2 (document_id);

create index if not exists caremind_document_chunks_v2_embedding_hnsw_idx
    on public.caremind_document_chunks_v2
    using hnsw (embedding extensions.vector_cosine_ops);

create or replace function public.match_caremind_document_chunks(
    query_embedding extensions.vector(1024),
    match_workspace_id text,
    match_count int default 5
)
returns table (
    chunk_id text,
    document_id text,
    document_name text,
    text text,
    page integer,
    score double precision
)
language sql
stable
as $$
    select
        c.chunk_id,
        c.document_id,
        c.document_name,
        c.text,
        c.page,
        1 - (c.embedding <=> query_embedding) as score
    from public.caremind_document_chunks_v2 c
    where c.workspace_id = match_workspace_id
    order by c.embedding <=> query_embedding
    limit match_count;
$$;

notify pgrst, 'reload schema';
