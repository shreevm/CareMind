"use client";

import { useEffect, useState } from "react";
import { authEnabled, supabase } from "./auth";

export default function AuthGate({ children }) {
  const enabled = authEnabled();
  const [session, setSession] = useState(null);
  const [ready, setReady] = useState(!enabled);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState("signin");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!enabled) return undefined;
    const sb = supabase();
    sb.auth.getSession().then(({ data }) => {
      setSession(data.session);
      setReady(true);
    });
    const { data } = sb.auth.onAuthStateChange((_event, next) => setSession(next));
    return () => data.subscription.unsubscribe();
  }, [enabled]);

  if (!enabled) return children;
  if (!ready) return null;

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setMessage("");
    const sb = supabase();
    const { error, data } =
      mode === "signin"
        ? await sb.auth.signInWithPassword({ email, password })
        : await sb.auth.signUp({ email, password });
    setBusy(false);
    if (error) setMessage(error.message);
    else if (mode === "signup" && !data.session) setMessage("Check your email to confirm your account, then sign in.");
  }

  if (!session) {
    return (
      <main style={{ maxWidth: 360, margin: "12vh auto", padding: 16 }}>
        <h1>CareMind</h1>
        <p>Sign in to access your private workspace.</p>
        <form onSubmit={submit} style={{ display: "grid", gap: 10 }}>
          <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={8}
            autoComplete={mode === "signin" ? "current-password" : "new-password"}
          />
          <button type="submit" disabled={busy}>{mode === "signin" ? "Sign in" : "Create account"}</button>
          <button type="button" onClick={() => setMode(mode === "signin" ? "signup" : "signin")}>
            {mode === "signin" ? "Need an account? Sign up" : "Have an account? Sign in"}
          </button>
          {message && <p role="alert">{message}</p>}
        </form>
      </main>
    );
  }

  return (
    <>
      <button
        type="button"
        onClick={() => supabase().auth.signOut()}
        style={{ position: "fixed", top: 8, right: 8, zIndex: 50 }}
        title={session.user?.email || ""}
      >
        Sign out
      </button>
      {children}
    </>
  );
}
