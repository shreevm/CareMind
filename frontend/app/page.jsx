"use client";

import { useEffect, useMemo, useRef, useState } from "react";

const CHAT_HISTORY_KEY = "caremind-chat-history";
const CHAT_MODE_KEY = "caremind-chat-mode";
const WELCOME_MESSAGE =
  "Upload a synthetic medical report or seed the demo, then ask what changed, what the key findings are, or which evidence supports a statement.";

function apiBase() {
  if (process.env.NEXT_PUBLIC_API_BASE_URL) return process.env.NEXT_PUBLIC_API_BASE_URL.replace(/\/$/, "");
  if (typeof window !== "undefined" && ["3000", "3001"].includes(window.location.port)) return "http://127.0.0.1:8002";
  return "";
}

function apiErrorMessage(body, fallback) {
  if (!body) return fallback || "Request failed.";
  try {
    const payload = JSON.parse(body);
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail)) return payload.detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
    if (typeof payload.error === "string") return payload.error;
  } catch {
    return body;
  }
  return body;
}

async function apiRequest(path, options = {}) {
  const response = await fetch(`${apiBase()}${path}`, options);
  if (!response.ok) {
    const detail = apiErrorMessage(await response.text(), response.statusText);
    throw new Error(detail);
  }
  return response.json();
}

async function transcribeAudio(blob, { sessionId, startedAt, endedAt }) {
  const form = new FormData();
  form.append("file", blob, "voice-input.webm");
  form.append("workspace_id", "default");
  form.append("session_id", sessionId || "default");
  form.append("started_at", startedAt || "");
  form.append("ended_at", endedAt || "");
  return apiRequest("/speech/transcribe", {
    method: "POST",
    body: form,
  });
}

async function streamChat(payload, handlers = {}) {
  const response = await fetch(`${apiBase()}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok || !response.body) {
    const detail = apiErrorMessage(await response.text(), response.statusText);
    throw new Error(detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = "message";
  let finalData = null;

  function dispatch(event, dataText) {
    if (!dataText) return;
    const data = JSON.parse(dataText);
    if (event === "delta") handlers.onDelta?.(data.text || "");
    if (event === "metric") handlers.onMetric?.(data);
    if (event === "start") handlers.onStart?.(data);
    if (event === "status") handlers.onStatus?.(data);
    if (event === "route") handlers.onRoute?.(data);
    if (event === "replace") handlers.onReplace?.(data.answer || "");
    if (event === "final") finalData = data;
    if (event === "error") throw new Error(data.error || "Streaming failed");
  }

  function consumeBuffer(flush = false) {
    let separatorIndex;
    while ((separatorIndex = buffer.indexOf("\n\n")) >= 0) {
      const rawEvent = buffer.slice(0, separatorIndex);
      buffer = buffer.slice(separatorIndex + 2);
      let dataText = "";
      currentEvent = "message";
      for (const line of rawEvent.split("\n")) {
        if (line.startsWith("event:")) currentEvent = line.slice(6).trim();
        if (line.startsWith("data:")) dataText += line.slice(5).trim();
      }
      dispatch(currentEvent, dataText);
    }
    if (flush && buffer.trim()) {
      let dataText = "";
      currentEvent = "message";
      for (const line of buffer.split("\n")) {
        if (line.startsWith("event:")) currentEvent = line.slice(6).trim();
        if (line.startsWith("data:")) dataText += line.slice(5).trim();
      }
      dispatch(currentEvent, dataText);
      buffer = "";
    }
  }

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    consumeBuffer();
  }
  buffer += decoder.decode();
  consumeBuffer(true);
  return finalData;
}

function getStoredHistory() {
  if (typeof window === "undefined") return [];
  try {
    const rows = JSON.parse(localStorage.getItem(CHAT_HISTORY_KEY) || "[]");
    return Array.isArray(rows) ? rows : [];
  } catch {
    return [];
  }
}

function saveStoredHistory(rows) {
  localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(rows.slice(0, 30)));
}

function cleanMarkdownSyntax(text = "") {
  return String(text)
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1")
    .replace(/\*\*/g, "")
    .replace(/__/g, "");
}

function cleanAnswerForDisplay(answer = "") {
  const cleaned = cleanMarkdownSyntax(answer);
  const lines = cleaned.split("\n");
  const firstContentIndex = lines.findIndex((line) => line.trim());
  if (firstContentIndex >= 0 && lines[firstContentIndex].trim().toLowerCase() === "caremind") {
    lines.splice(firstContentIndex, 1);
  }
  return lines.join("\n").trim();
}

function titleCase(value) {
  return value
    .replace(/\.[^.]+$/, "")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
    .trim();
}

function friendlyName(filename = "") {
  const lower = filename.toLowerCase();
  if (lower.includes("baseline")) return "Baseline Lab Report";
  if (lower.includes("followup") || lower.includes("follow-up")) return "Follow-up Lab Report";
  if (lower.includes("cxr") || lower.includes("xray") || lower.includes("x-ray")) return "Chest X-ray Image";
  return titleCase(filename);
}

function uniqueByName(items, nameGetter) {
  const seen = new Set();
  return items.filter((item) => {
    const key = nameGetter(item).toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function assetTimestamp(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

function routeLabel(route = "") {
  const labels = {
    clinical_document_qa: "Document",
    medical_knowledge_qa: "Education",
    nursing_care_qa: "Nursing",
    imaging_qa: "Imaging",
    report_comparison: "Compare",
    emergency_redirect: "Safety",
    clarify: "Clarify",
    direct: "Help",
    prompt_injection_blocked: "Safety",
  };
  return labels[route] || route.replace(/_/g, " ") || "CareMind";
}

function isExternalCitation(citation) {
  return citation?.document_id?.startsWith("pubmed:");
}

function citationUrl(citation) {
  if (!isExternalCitation(citation)) return "";
  const pmid = citation.document_id.replace(/^pubmed:/, "").trim();
  return pmid ? `https://pubmed.ncbi.nlm.nih.gov/${encodeURIComponent(pmid)}/` : "";
}

function InlineCitations({ text, citations }) {
  return cleanMarkdownSyntax(text)
    .split(/(\[\d+\])/g)
    .map((part, index) => {
      const match = part.match(/^\[(\d+)\]$/);
      if (!match) return part ? <span key={index}>{part}</span> : null;
      const citationIndex = Number(match[1]);
      const citation = citations[citationIndex - 1];
      const href = citationUrl(citation);
      if (!href) return <span key={index}>{part}</span>;
      return (
        <a key={index} className="citation-link" href={href} target="_blank" rel="noopener noreferrer">
          {part}
        </a>
      );
    });
}

function AnswerText({ answer, citations }) {
  const blocks = cleanAnswerForDisplay(answer)
    .split(/\n{2,}/)
    .map((item) => item.trim())
    .filter(Boolean);

  return blocks.map((block, blockIndex) => {
    const lines = block
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    const nodes = [];
    let index = 0;
    while (index < lines.length) {
      const line = lines[index];
      const heading = /^[A-Za-z][A-Za-z0-9 /&()-]{1,60}:$/.test(line);
      if (heading) {
        nodes.push(
          <h3 key={`${blockIndex}-${index}`} className="response-heading">
            {line.replace(/:$/, "")}
          </h3>,
        );
        index += 1;
        continue;
      }
      if (!/^[-*]\s+/.test(line)) {
        nodes.push(
          <p key={`${blockIndex}-${index}`}>
            <InlineCitations text={line} citations={citations} />
          </p>,
        );
        index += 1;
        continue;
      }
      const items = [];
      while (index < lines.length && /^[-*]\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^[-*]\s+/, ""));
        index += 1;
      }
      nodes.push(
        <ul key={`${blockIndex}-${index}`} className="response-list">
          {items.map((item, itemIndex) => (
            <li key={itemIndex}>
              <InlineCitations text={item} citations={citations} />
            </li>
          ))}
        </ul>,
      );
    }
    return <div key={blockIndex}>{nodes}</div>;
  });
}

function References({ citations }) {
  const externalCitations = citations
    .map((citation, index) => ({ citation, index }))
    .filter(({ citation }) => isExternalCitation(citation));
  if (!externalCitations.length) return null;
  return (
    <div className="references">
      {externalCitations.map(({ citation, index }) => (
        <a
          key={citation.document_id || index}
          className="reference-card"
          href={citationUrl(citation)}
          target="_blank"
          rel="noopener noreferrer"
        >
          <span>
            [{index + 1}] {friendlyName(citation.document_name)}
          </span>
          <small>{citation.quote || "Source passage"}</small>
        </a>
      ))}
    </div>
  );
}

function TracePanel({ trace, data }) {
  if (!trace) return null;
  const cacheDebug = trace.cache_debug || trace.cache_metadata?.cache_debug || {};
  const rows = [
    ["Route", trace.route || data.route || "-"],
    ["Generation model", trace.generation_model || "-"],
    ["Embedding model", trace.embedding_model || "-"],
    ["Vector backend", trace.vector_backend || "-"],
    ["Tools", (data.tool_calls || []).join(", ") || "none"],
    ["Cache", trace.cache_hit ? "hit" : "miss"],
    ["Cache source", trace.cache_source || "-"],
    ["Redis", cacheDebug.redis_enabled ? "connected" : cacheDebug.redis_error ? `off (${cacheDebug.redis_error})` : "off"],
    ["Cache key", trace.cache_key ? trace.cache_key.slice(0, 12) : "-"],
    ["Cache namespace", trace.cache_namespace ? trace.cache_namespace.slice(0, 12) : "-"],
    ["Semantic cache", cacheDebug.semantic_cache_enabled ? "enabled" : "disabled"],
    ["TTFT", trace.time_to_first_token_ms == null ? "-" : `${Math.round(trace.time_to_first_token_ms)} ms`],
    ["Latency", `${trace.total_latency_ms ?? "-"} ms`],
    ["Stream latency", trace.stream_total_latency_ms == null ? "-" : `${trace.stream_total_latency_ms} ms`],
  ];
  return (
    <details className="trace-panel">
      <summary>Agent trace</summary>
      <div className="trace-grid">
        {rows.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      {!!trace.steps?.length && (
        <ol className="trace-steps">
          {trace.steps.map((step, index) => (
            <li key={index}>{`${step.node}: ${JSON.stringify(step)}`}</li>
          ))}
        </ol>
      )}
      {!!trace.retrieved_chunks?.length && (
        <div className="trace-contexts">
          {trace.retrieved_chunks.map((chunk, index) => (
            <details key={chunk.chunk_id || index}>
              <summary>
                {chunk.document_name || "source"}
                {chunk.score == null ? "" : ` | score ${Number(chunk.score).toFixed(3)}`}
              </summary>
              <p>{chunk.preview || chunk.quote || ""}</p>
            </details>
          ))}
        </div>
      )}
    </details>
  );
}

function Message({ message, chatMode }) {
  const role = message.role || "assistant";
  if (role === "assistant" && message.data) {
    const data = message.data;
    const citations = data.citations || [];
    const showAgentDetails = chatMode === "agent";
    return (
      <div className="message assistant">
        <div className="message-label">CareMind</div>
        <div className="message-body response-content">
          {showAgentDetails && (
            <div className="response-meta">
              <span className="route-badge">{routeLabel(data.route || data.trace?.route)}</span>
              <span className="route-badge secondary-badge">{data.trace?.cache_hit ? "Cache hit" : "Fresh"}</span>
            </div>
          )}
          <AnswerText answer={data.answer || ""} citations={citations} />
          <References citations={citations} />
          {showAgentDetails && <TracePanel trace={data.trace} data={data} />}
        </div>
      </div>
    );
  }
  return (
    <div className={`message ${role} ${message.extraClass || ""}`.trim()}>
      <div className="message-label">{role === "user" ? "You" : role === "assistant" ? "CareMind" : role}</div>
      <div className="message-body">{role === "assistant" ? cleanAnswerForDisplay(message.content) : message.content}</div>
    </div>
  );
}

function sanitizePatientData(data) {
  if (!data) return data;
  return {
    ...data,
    route: "caremind_answer",
    tool_calls: [],
    trace: {},
  };
}

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="M17 8l-5-5-5 5" />
      <path d="M12 3v12" />
    </svg>
  );
}

function MicIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v5a3 3 0 0 0 3 3Z" />
      <path d="M19 11a7 7 0 0 1-14 0" />
      <path d="M12 18v4" />
      <path d="M8 22h8" />
    </svg>
  );
}

export default function CareMindPage() {
  const [mounted, setMounted] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [history, setHistory] = useState([]);
  const [messages, setMessages] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [images, setImages] = useState([]);
  const [selectedDocuments, setSelectedDocuments] = useState([]);
  const [status, setStatus] = useState("Ready.");
  const [theme, setTheme] = useState("light");
  const [chatMode, setChatMode] = useState("patient");
  const [composerAttachments, setComposerAttachments] = useState([]);
  const [draft, setDraft] = useState("");
  const [pendingTranscript, setPendingTranscript] = useState(null);
  const [voiceSupported, setVoiceSupported] = useState(false);
  const [voiceRecording, setVoiceRecording] = useState(false);
  const [voiceBusy, setVoiceBusy] = useState(false);
  const fileInputRef = useRef(null);
  const messagesRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const voiceChunksRef = useRef([]);
  const voiceStartedAtRef = useRef("");

  const currentConversation = useMemo(
    () => history.find((item) => item.session_id === sessionId),
    [history, sessionId],
  );

  useEffect(() => {
    const savedSessionId = localStorage.getItem("caremind-session-id") || crypto.randomUUID();
    localStorage.setItem("caremind-session-id", savedSessionId);
    const savedTheme = localStorage.getItem("caremind-theme");
    const savedMode = localStorage.getItem(CHAT_MODE_KEY);
    const preferredTheme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    const nextTheme = savedTheme || preferredTheme;
    const nextMode = savedMode === "agent" ? "agent" : "patient";
    setSessionId(savedSessionId);
    setTheme(nextTheme);
    setChatMode(nextMode);
    document.documentElement.dataset.theme = nextTheme;
    const rows = getStoredHistory();
    const existing = rows.find((item) => item.session_id === savedSessionId);
    if (!existing) {
      rows.unshift({
        session_id: savedSessionId,
        title: "New chat",
        updated_at: new Date().toISOString(),
        messages: [],
      });
      saveStoredHistory(rows);
    }
    setHistory(rows);
    setMessages(existing?.messages?.length ? existing.messages : [{ role: "assistant", content: WELCOME_MESSAGE }]);
    setVoiceSupported(Boolean(navigator.mediaDevices?.getUserMedia && window.MediaRecorder));
    setMounted(true);
    loadAssets();
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    if (mounted) localStorage.setItem("caremind-theme", theme);
  }, [mounted, theme]);

  useEffect(() => {
    if (mounted) localStorage.setItem(CHAT_MODE_KEY, chatMode);
  }, [mounted, chatMode]);

  useEffect(() => {
    if (messagesRef.current) messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
  }, [messages]);

  function persistConversation(nextMessages, titleSource = "") {
    const rows = getStoredHistory();
    let conversation = rows.find((item) => item.session_id === sessionId);
    if (!conversation) {
      conversation = {
        session_id: sessionId,
        title: "New chat",
        updated_at: new Date().toISOString(),
        messages: [],
      };
      rows.unshift(conversation);
    }
    conversation.messages = nextMessages.filter((message) => message.content?.trim() || message.data).slice(-40);
    if (titleSource && (!conversation.title || conversation.title === "New chat")) {
      conversation.title = titleSource.length > 42 ? `${titleSource.slice(0, 42)}...` : titleSource;
    }
    conversation.updated_at = new Date().toISOString();
    const nextHistory = [conversation, ...rows.filter((item) => item.session_id !== sessionId)];
    saveStoredHistory(nextHistory);
    setHistory(nextHistory);
  }

  async function loadAssets() {
    try {
      const [docs, imageRows] = await Promise.all([apiRequest("/documents"), apiRequest("/images")]);
      setDocuments(docs);
      setImages(imageRows);
    } catch (error) {
      setMessages((rows) => [...rows, { role: "assistant", content: error.message, extraClass: "notice" }]);
    }
  }

  async function deleteUploadedAsset(kind, id) {
    if (!id) return;
    const path =
      kind === "image"
        ? `/images/${encodeURIComponent(id)}?workspace_id=default`
        : `/documents/${encodeURIComponent(id)}?workspace_id=default`;
    setStatus(`Removing ${kind}...`);
    try {
      await apiRequest(path, { method: "DELETE" });
      await loadAssets();
      setSelectedDocuments((items) => items.filter((item) => item !== id));
      setComposerAttachments((items) => items.filter((item) => item.assetId !== id));
      setStatus(`${kind === "image" ? "Image" : "Document"} removed.`);
    } catch (error) {
      setStatus(error.message);
    }
  }

  function switchSession(nextSessionId) {
    localStorage.setItem("caremind-session-id", nextSessionId);
    setSessionId(nextSessionId);
    const rows = getStoredHistory();
    const conversation = rows.find((item) => item.session_id === nextSessionId);
    setHistory(rows);
    setMessages(conversation?.messages?.length ? conversation.messages : [{ role: "assistant", content: WELCOME_MESSAGE }]);
  }

  async function deleteConversation(event, targetSessionId) {
    event.preventDefault();
    event.stopPropagation();
    const remaining = getStoredHistory().filter((item) => item.session_id !== targetSessionId);
    saveStoredHistory(remaining);
    setHistory(remaining);
    try {
      await apiRequest(`/chat/sessions/${encodeURIComponent(targetSessionId)}?workspace_id=default`, { method: "DELETE" });
      setStatus("Chat deleted.");
    } catch (error) {
      setStatus(`Local chat deleted. Server cleanup failed: ${error.message}`);
    }
    if (targetSessionId !== sessionId) return;
    const next = remaining[0];
    if (next) {
      localStorage.setItem("caremind-session-id", next.session_id);
      setSessionId(next.session_id);
      setMessages(next.messages?.length ? next.messages : [{ role: "assistant", content: WELCOME_MESSAGE }]);
      return;
    }
    startNewChat();
  }

  function startNewChat() {
    const nextSessionId = crypto.randomUUID();
    localStorage.setItem("caremind-session-id", nextSessionId);
    const conversation = {
      session_id: nextSessionId,
      title: "New chat",
      updated_at: new Date().toISOString(),
      messages: [],
    };
    const nextHistory = [conversation, ...getStoredHistory()];
    saveStoredHistory(nextHistory);
    setSessionId(nextSessionId);
    setHistory(nextHistory);
    setMessages([{ role: "assistant", content: WELCOME_MESSAGE }]);
  }

  async function uploadSelectedFile(file) {
    if (!file) return;
    const attachmentId = crypto.randomUUID();
    const attachmentType = file.type.startsWith("image/") ? "Image" : "Document";
    setComposerAttachments((items) => [
      ...items,
      {
        id: attachmentId,
        name: friendlyName(file.name),
        type: attachmentType,
        status: "Uploading",
      },
    ]);
    const form = new FormData();
    form.append("file", file);
    form.append("workspace_id", "default");
    form.append("session_id", sessionId || "default");
    form.append("modality", "clinical_image");
    form.append("report_text", "");
    setStatus(`Uploading ${file.name}...`);
    try {
      const uploaded = await apiRequest("/upload", { method: "POST", body: form });
      await loadAssets();
      setStatus(`${friendlyName(file.name)} is ready.`);
      setComposerAttachments((items) =>
        items.map((item) =>
          item.id === attachmentId
            ? {
                ...item,
                status: "Ready",
                assetId: uploaded.document_id || uploaded.image_id || "",
                assetKind: uploaded.image_id ? "image" : "document",
              }
            : item,
        ),
      );
    } catch (error) {
      const message = error.message || "Upload failed.";
      setStatus(`Upload failed: ${message}`);
      setComposerAttachments((items) =>
        items.map((item) => (item.id === attachmentId ? { ...item, status: "Failed", error: message } : item)),
      );
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function seedDemoReports() {
    setStatus("Seeding demo reports...");
    try {
      await apiRequest("/demo/seed", { method: "POST" });
      await loadAssets();
      setStatus("Demo reports are ready.");
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function sendMessage(event) {
    event.preventDefault();
    const message = draft.trim();
    if (!message) return;
    setDraft("");
    const transcript = pendingTranscript?.text === message ? pendingTranscript : null;
    setPendingTranscript(null);
    const pendingId = crypto.randomUUID();
    const nextMessages = [
      ...messages,
      { role: "user", content: message },
      { id: pendingId, role: "assistant", content: "Preparing response...", extraClass: "notice" },
    ];
    setMessages(nextMessages);
    persistConversation(nextMessages, message);
    try {
      let streamedAnswer = "";
      const data = await streamChat(
        {
          message,
          session_id: sessionId,
          workspace_id: "default",
          modality: transcript ? "voice" : "text",
          transcript,
          response_mode: chatMode,
          debug: chatMode === "agent",
          bypass_cache: chatMode === "agent",
        },
        {
          onDelta: (text) => {
            streamedAnswer += text;
            setMessages((rows) =>
              rows.map((item) =>
                item.id === pendingId ? { ...item, content: cleanAnswerForDisplay(streamedAnswer), extraClass: "" } : item,
              ),
            );
          },
          onReplace: (answer) => {
            streamedAnswer = answer;
            setMessages((rows) =>
              rows.map((item) => (item.id === pendingId ? { ...item, content: cleanAnswerForDisplay(answer), extraClass: "" } : item)),
            );
          },
          onStart: () => {
            setMessages((rows) =>
              rows.map((item) => (item.id === pendingId ? { ...item, content: "Preparing response..." } : item)),
            );
          },
          onStatus: (status) => {
            const stage = status.stage === "generating" ? "Generating response..." : "Preparing response...";
            setMessages((rows) =>
              rows.map((item) => (item.id === pendingId ? { ...item, content: stage } : item)),
            );
          },
          onMetric: (metric) => {
            if (metric.time_to_first_token_ms != null) setStatus(`First token: ${metric.time_to_first_token_ms} ms`);
          },
        },
      );
      if (!data) throw new Error("Streaming ended before a final response was received.");
      const visibleData = chatMode === "agent" ? data : sanitizePatientData(data);
      const answered = nextMessages.map((item) =>
        item.id === pendingId
          ? {
              role: "assistant",
              content: cleanAnswerForDisplay(visibleData.answer || streamedAnswer),
              data: { ...visibleData, question: message },
            }
          : item,
      );
      setMessages(answered);
      persistConversation(answered, message);
    } catch (error) {
      const failed = nextMessages.map((item) =>
        item.id === pendingId ? { role: "assistant", content: error.message, extraClass: "notice" } : item,
      );
      setMessages(failed);
      persistConversation(failed, message);
    }
  }

  async function compareSelected() {
    if (selectedDocuments.length < 2) {
      setMessages((rows) => [...rows, { role: "assistant", content: "Select two uploaded documents to compare.", extraClass: "notice" }]);
      return;
    }
    try {
      const data = await apiRequest("/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_ids: selectedDocuments.slice(0, 2), workspace_id: "default" }),
      });
      const nextMessages = [...messages, { role: "assistant", content: data.summary }];
      setMessages(nextMessages);
      persistConversation(nextMessages);
    } catch (error) {
      setMessages((rows) => [...rows, { role: "assistant", content: error.message, extraClass: "notice" }]);
    }
  }

  function toggleSelectedDocument(documentId) {
    setSelectedDocuments((current) =>
      current.includes(documentId) ? current.filter((item) => item !== documentId) : [...current, documentId],
    );
  }

  async function startVoiceInput() {
    if (mediaRecorderRef.current?.state === "recording") {
      setStatus("Stopping recording...");
      mediaRecorderRef.current.stop();
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setStatus("Voice input is not supported in this browser.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      voiceChunksRef.current = [];
      voiceStartedAtRef.current = new Date().toISOString();
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data?.size) voiceChunksRef.current.push(event.data);
      };
      recorder.onstop = async () => {
        const endedAt = new Date().toISOString();
        stream.getTracks().forEach((track) => track.stop());
        setVoiceRecording(false);
        setVoiceBusy(true);
        setStatus("Transcribing voice input...");
        try {
          const transcript = await transcribeAudio(
            new Blob(voiceChunksRef.current, { type: recorder.mimeType || "audio/webm" }),
            {
              sessionId,
              startedAt: voiceStartedAtRef.current,
              endedAt,
            },
          );
          setDraft(transcript.text || "");
          setPendingTranscript({
            ...transcript,
            modality: "voice",
            input_modality: "voice",
          });
          const language = transcript.detected_language || transcript.language || "auto";
          const confidence =
            transcript.language_confidence == null ? "" : ` (${Math.round(transcript.language_confidence * 100)}%)`;
          setStatus(`Voice transcript is ready. Language: ${language}${confidence}.`);
        } catch (error) {
          setStatus(`Voice transcription failed: ${error.message}`);
        } finally {
          setVoiceBusy(false);
          mediaRecorderRef.current = null;
          voiceChunksRef.current = [];
        }
      };
      recorder.start();
      setVoiceRecording(true);
      setStatus("Recording... click the mic again to stop.");
    } catch (error) {
      setStatus(`Voice input failed: ${error.message}`);
    }
  }

  if (!mounted) {
    return <main className="app-shell" />;
  }

  return (
    <main className="app-shell">
      <section className="sidebar">
        <div>
          <p className="eyebrow">CareMind</p>
          <h1>Medical AI Assistant</h1>
        </div>

        <div className="toolbar">
          <button type="button" className="secondary" onClick={startNewChat}>
            New chat
          </button>
          <button type="button" className="secondary" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
            {theme === "dark" ? "Light mode" : "Dark mode"}
          </button>
        </div>

        <div className="panel chat-history-panel">
          <div className="panel-title">Chat History</div>
          <div className="chat-history">
            {history.length ? (
              history.map((item) => (
                <button
                  key={item.session_id}
                  type="button"
                  className={`history-row${item.session_id === sessionId ? " is-active" : ""}`}
                  onClick={() => switchSession(item.session_id)}
                >
                  <span className="history-title">{item.title || "New chat"}</span>
                  <small>
                    {item.messages?.length || 0} message{(item.messages?.length || 0) === 1 ? "" : "s"}
                  </small>
                  <span
                    role="button"
                    tabIndex={0}
                    className="history-delete"
                    title="Delete chat"
                    aria-label={`Delete ${item.title || "chat"}`}
                    onClick={(event) => deleteConversation(event, item.session_id)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") deleteConversation(event, item.session_id);
                    }}
                  >
                    x
                  </span>
                </button>
              ))
            ) : (
              "No chats yet."
            )}
          </div>
        </div>

        <div className="status" role="status">
          {status}
        </div>

        <button className="secondary" type="button" onClick={seedDemoReports}>
          Seed demo reports
        </button>

        <div className="panel">
          <div className="panel-title">Documents</div>
          <div className="documents">
            {documents.length ? (
              documents.map((doc) => (
                <div key={doc.document_id} className="doc-row asset-row">
                  <input
                    type="checkbox"
                    value={doc.document_id}
                    checked={selectedDocuments.includes(doc.document_id)}
                    onChange={() => toggleSelectedDocument(doc.document_id)}
                  />
                  <span>
                    {friendlyName(doc.filename)}
                    <small>{assetTimestamp(doc.uploaded_at)}</small>
                  </span>
                  <button type="button" className="asset-delete" onClick={() => deleteUploadedAsset("document", doc.document_id)}>
                    Remove
                  </button>
                </div>
              ))
            ) : (
              "No documents uploaded yet."
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-title">Images</div>
          <div className="documents">
            {images.length ? (
              images.map((image) => (
                <div key={image.image_id} className="doc-row asset-row">
                  <span className="asset-marker">Image</span>
                  <span>
                    {friendlyName(image.filename)}
                    <small>{assetTimestamp(image.uploaded_at)}</small>
                  </span>
                  <button type="button" className="asset-delete" onClick={() => deleteUploadedAsset("image", image.image_id)}>
                    Remove
                  </button>
                </div>
              ))
            ) : (
              "No images uploaded yet."
            )}
          </div>
        </div>

        <button className="secondary" type="button" onClick={compareSelected}>
          Compare selected
        </button>
      </section>

      <section className="chat">
        <div className="chat-header">
          <div className="disclaimer-banner" role="note">
            CareMind is for medical education and document interpretation only. It is not a diagnosis or treatment plan.
          </div>
          <div className="mode-switch" aria-label="Chat mode">
            <button
              type="button"
              className={chatMode === "patient" ? "is-active" : ""}
              aria-pressed={chatMode === "patient"}
              onClick={() => setChatMode("patient")}
            >
              Patient
            </button>
            <button
              type="button"
              className={chatMode === "agent" ? "is-active" : ""}
              aria-pressed={chatMode === "agent"}
              onClick={() => setChatMode("agent")}
            >
              Agent
            </button>
          </div>
        </div>
        <div ref={messagesRef} className="messages" aria-live="polite">
          {messages.map((message, index) => (
            <Message key={message.id || `${message.role}-${index}`} message={message} chatMode={chatMode} />
          ))}
        </div>
        <form className="composer-shell" aria-label="Chat composer" onSubmit={sendMessage}>
          {!!composerAttachments.length && (
            <div className="attachment-tray" aria-label="Attached files">
              {composerAttachments.map((item) => (
                <div key={item.id} className={`attachment-chip ${item.status.toLowerCase()}`}>
                  <span className="attachment-kind">{item.type}</span>
                  <span className="attachment-name">{item.name}</span>
                  <small title={item.error || item.status}>{item.error ? `${item.status}: ${item.error}` : item.status}</small>
                  <button
                    type="button"
                    aria-label={`${item.status === "Ready" && item.assetId ? "Remove" : "Hide"} ${item.name}`}
                    title={item.status === "Ready" && item.assetId ? "Remove upload" : "Hide attachment"}
                    onClick={() =>
                      item.status === "Ready" && item.assetId
                        ? deleteUploadedAsset(item.assetKind, item.assetId)
                        : setComposerAttachments((items) => items.filter((attachment) => attachment.id !== item.id))
                    }
                  >
                    x
                  </button>
                </div>
              ))}
            </div>
          )}
          <div className="composer">
            <input
              id="message"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              autoComplete="off"
              placeholder="Ask about the uploaded reports..."
            />
          <input
            ref={fileInputRef}
            className="hidden-file-input"
            name="file"
            type="file"
            accept=".pdf,.txt,.md,.png,.jpg,.jpeg,.webp,text/plain,text/markdown,application/pdf,image/png,image/jpeg,image/webp"
            onChange={(event) => uploadSelectedFile(event.target.files?.[0])}
          />
          <button
            type="button"
            className="icon-button"
            title="Upload document"
            aria-label="Upload document"
            onClick={() => fileInputRef.current?.click()}
          >
            <UploadIcon />
          </button>
          <button
            type="button"
            className={`icon-button${voiceRecording ? " is-recording" : ""}`}
            title={
              voiceSupported
                ? voiceRecording
                  ? "Stop recording"
                  : "Dictate question"
                : "Voice input is not supported in this browser"
            }
            aria-label="Dictate question"
            disabled={!voiceSupported || voiceBusy}
            onClick={startVoiceInput}
          >
            <MicIcon />
          </button>
          <button type="submit">Send</button>
          </div>
        </form>
      </section>
    </main>
  );
}
