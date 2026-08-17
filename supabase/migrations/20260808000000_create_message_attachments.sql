create table if not exists public.message_attachments (
  id text primary key,
  message_id bigint,
  conversation_id text not null,
  workspace_id text not null,
  user_id text,
  attachment_type text not null,
  filename text not null,
  mime_type text,
  file_size bigint not null default 0,
  storage_bucket text not null default 'caremind-attachments',
  storage_path text not null,
  image_asset_id text,
  document_id text,
  processing_status text not null default 'completed',
  processing_error text,
  created_at timestamptz not null default now(),
  expires_at timestamptz,
  persistence_mode text not null default 'saved_compat',
  constraint message_attachments_persistence_mode_check
    check (persistence_mode in ('temporary', 'saved', 'saved_compat'))
);

create index if not exists idx_message_attachments_message
  on public.message_attachments(message_id);

create index if not exists idx_message_attachments_conversation
  on public.message_attachments(workspace_id, conversation_id, created_at desc);

create index if not exists idx_message_attachments_document
  on public.message_attachments(document_id);

create index if not exists idx_message_attachments_image
  on public.message_attachments(image_asset_id);
