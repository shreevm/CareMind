create extension if not exists vector with schema extensions;

drop index if exists public.caremind_document_chunks_embedding_hnsw_idx;

drop function if exists public.match_caremind_document_chunks(extensions.vector, text, int);

truncate table public.caremind_document_chunks;

alter table if exists public.caremind_document_chunks
    add column if not exists embedding_provider text not null default '',
    add column if not exists embedding_model text not null default '',
    add column if not exists embedding_dimension integer,
    alter column embedding type extensions.vector(1024);

create index if not exists caremind_document_chunks_embedding_hnsw_idx
    on public.caremind_document_chunks
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
    from public.caremind_document_chunks c
    where c.workspace_id = match_workspace_id
    order by c.embedding <=> query_embedding
    limit match_count;
$$;

notify pgrst, 'reload schema';
