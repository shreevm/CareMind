create extension if not exists vector with schema extensions;

create table if not exists public.caremind_document_chunks_qwen3_1024 (
    chunk_id text primary key,
    document_id text not null,
    workspace_id text not null,
    document_name text not null,
    page integer,
    position integer not null,
    text text not null,
    embedding extensions.vector(1024) not null,
    embedding_provider text not null default 'qwen',
    embedding_model text not null default 'Qwen/Qwen3-Embedding-0.6B',
    embedding_dimension integer not null default 1024,
    embedding_index_version text not null default 'qwen3-0.6b-v1-1024',
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists caremind_qwen3_chunks_workspace_idx
    on public.caremind_document_chunks_qwen3_1024 (workspace_id);

create index if not exists caremind_qwen3_chunks_document_idx
    on public.caremind_document_chunks_qwen3_1024 (document_id);

create index if not exists caremind_qwen3_chunks_version_idx
    on public.caremind_document_chunks_qwen3_1024 (
        workspace_id,
        embedding_provider,
        embedding_model,
        embedding_dimension,
        embedding_index_version
    );

create index if not exists caremind_qwen3_chunks_embedding_hnsw_idx
    on public.caremind_document_chunks_qwen3_1024
    using hnsw (embedding extensions.vector_cosine_ops);

create or replace function public.match_caremind_document_chunks_qwen3_1024(
    query_embedding extensions.vector(1024),
    match_workspace_id text,
    match_count int default 5,
    match_embedding_provider text default 'qwen',
    match_embedding_model text default 'Qwen/Qwen3-Embedding-0.6B',
    match_embedding_dimension int default 1024,
    match_embedding_index_version text default 'qwen3-0.6b-v1-1024'
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
    from public.caremind_document_chunks_qwen3_1024 c
    where c.workspace_id = match_workspace_id
      and c.embedding_provider = match_embedding_provider
      and c.embedding_model = match_embedding_model
      and c.embedding_dimension = match_embedding_dimension
      and c.embedding_index_version = match_embedding_index_version
    order by c.embedding <=> query_embedding
    limit match_count;
$$;

notify pgrst, 'reload schema';
