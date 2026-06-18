-- ============================================================
-- PharmaRegBot — Supabase users table
-- Run this ONCE in your Supabase project: SQL Editor → New query → Run.
-- ============================================================

create table if not exists public.users (
    id            bigint generated always as identity primary key,
    username      text        unique not null,
    full_name     text,
    email         text,
    password_hash text        not null,
    salt          text        not null,
    iterations    integer     not null,
    created_at    timestamptz not null default now()
);

-- Fast lookups by username (login / existence checks).
create index if not exists users_username_idx on public.users (username);

-- Enable Row Level Security and add NO policies. This blocks the public/anon
-- key entirely; only the service_role key (used by the PharmaRegBot backend,
-- which bypasses RLS) can read/write the table. Password hashes stay private.
alter table public.users enable row level security;
