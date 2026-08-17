create table if not exists public.caremind_image_assets (
    image_id text primary key,
    workspace_id text not null,
    filename text not null,
    content_type text,
    file_path text not null,
    modality text not null default 'clinical_image',
    report_text_summary text,
    metadata jsonb not null default '{}'::jsonb,
    uploaded_at timestamptz not null default now()
);

create index if not exists caremind_image_assets_workspace_idx
    on public.caremind_image_assets (workspace_id);
