import { createClient } from "@supabase/supabase-js";

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_KEY = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY;

let client = null;

export function authEnabled() {
  return Boolean(SUPABASE_URL && SUPABASE_KEY);
}

export function supabase() {
  if (!authEnabled()) return null;
  if (!client) client = createClient(SUPABASE_URL, SUPABASE_KEY);
  return client;
}

// Returns {} when Supabase auth is not configured so the browser's basic-auth flow keeps working.
export async function authHeaders() {
  const sb = supabase();
  if (!sb) return {};
  const { data } = await sb.auth.getSession();
  const token = data?.session?.access_token;
  return token ? { Authorization: `Bearer ${token}` } : {};
}
