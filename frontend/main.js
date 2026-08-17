const savedSessionId = localStorage.getItem("caremind-session-id");
let sessionId = savedSessionId || crypto.randomUUID();
localStorage.setItem("caremind-session-id", sessionId);
const messages = document.querySelector("#messages");
const documentsEl = document.querySelector("#documents");
const imagesEl = document.querySelector("#images");
const uploadForm = document.querySelector("#upload-form");
const fileInput = document.querySelector("#file");
const chatForm = document.querySelector("#chat-form");
const messageInput = document.querySelector("#message");
const voiceButton = document.querySelector("#voice");
const composerUploadButton = document.querySelector("#composer-upload");
const seedDemo = document.querySelector("#seed-demo");
const compareButton = document.querySelector("#compare");
const statusEl = document.querySelector("#status");
const themeToggle = document.querySelector("#theme-toggle");
const newChatButton = document.querySelector("#new-chat");
const chatHistoryEl = document.querySelector("#chat-history");
const patientModeButton = document.querySelector("#patient-mode");
const agentModeButton = document.querySelector("#agent-mode");
const attachmentTray = document.querySelector("#attachment-tray");
let pendingTranscript = null;
let mediaRecorder = null;
let voiceChunks = [];
let voiceStartedAt = null;
let chatMode = localStorage.getItem("caremind-chat-mode") === "agent" ? "agent" : "patient";
let composerAttachments = [];
const CHAT_HISTORY_KEY = "caremind-chat-history";
const WELCOME_MESSAGE =
  "Upload a synthetic medical report or seed the demo, then ask what changed, what the key findings are, or which evidence supports a statement.";

function setStatus(message) {
  statusEl.textContent = message;
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  themeToggle.textContent = theme === "dark" ? "Light mode" : "Dark mode";
  localStorage.setItem("caremind-theme", theme);
}

const savedTheme = localStorage.getItem("caremind-theme");
const preferredTheme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
applyTheme(savedTheme || preferredTheme);

themeToggle.addEventListener("click", () => {
  const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(nextTheme);
});

function getChatHistory() {
  try {
    const rows = JSON.parse(localStorage.getItem(CHAT_HISTORY_KEY) || "[]");
    return Array.isArray(rows) ? rows : [];
  } catch {
    return [];
  }
}

function saveChatHistory(rows) {
  localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(rows.slice(0, 30)));
}

function setChatMode(nextMode) {
  chatMode = nextMode === "agent" ? "agent" : "patient";
  localStorage.setItem("caremind-chat-mode", chatMode);
  patientModeButton?.classList.toggle("is-active", chatMode === "patient");
  agentModeButton?.classList.toggle("is-active", chatMode === "agent");
  patientModeButton?.setAttribute("aria-pressed", String(chatMode === "patient"));
  agentModeButton?.setAttribute("aria-pressed", String(chatMode === "agent"));
  renderStoredMessages(currentConversation());
}

function currentConversation() {
  const history = getChatHistory();
  let conversation = history.find((item) => item.session_id === sessionId);
  if (!conversation) {
    conversation = {
      session_id: sessionId,
      title: "New chat",
      updated_at: new Date().toISOString(),
      messages: [],
    };
    history.unshift(conversation);
    saveChatHistory(history);
  }
  return conversation;
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

function sanitizePatientData(data) {
  if (!data) return data;
  return {
    ...data,
    route: "caremind_answer",
    tool_calls: [],
    trace: {},
  };
}

function rememberMessage(role, content, data = null, attachments = []) {
  if (!content?.trim() && !data) return;
  const history = getChatHistory();
  let conversation = history.find((item) => item.session_id === sessionId);
  if (!conversation) {
    conversation = {
      session_id: sessionId,
      title: "New chat",
      updated_at: new Date().toISOString(),
      messages: [],
    };
    history.unshift(conversation);
  }
  const savedMessage = {
    role,
    content,
    created_at: new Date().toISOString(),
  };
  if (data) savedMessage.data = data;
  if (attachments.length) {
    savedMessage.attachments = attachments.map(({ attachmentId, name, type, assetId, assetKind }) => ({
      attachmentId,
      name,
      type,
      assetId,
      assetKind,
    }));
  }
  conversation.messages = [...(conversation.messages || []), savedMessage].slice(-40);
  if (role === "user" && (!conversation.title || conversation.title === "New chat")) {
    conversation.title = content.length > 42 ? `${content.slice(0, 42)}...` : content;
  }
  conversation.updated_at = new Date().toISOString();
  saveChatHistory([conversation, ...history.filter((item) => item.session_id !== sessionId)]);
  renderChatHistory();
}

function renderChatHistory() {
  if (!chatHistoryEl) return;
  const history = getChatHistory();
  chatHistoryEl.innerHTML = "";
  if (!history.length) {
    chatHistoryEl.textContent = "No chats yet.";
    return;
  }
  for (const item of history) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `history-row${item.session_id === sessionId ? " is-active" : ""}`;
    const title = document.createElement("span");
    title.className = "history-title";
    title.textContent = item.title || "New chat";
    const meta = document.createElement("small");
    const count = item.messages?.length || 0;
    meta.textContent = `${count} message${count === 1 ? "" : "s"}`;
    const remove = document.createElement("span");
    remove.className = "history-delete";
    remove.role = "button";
    remove.tabIndex = 0;
    remove.title = "Delete chat";
    remove.setAttribute("aria-label", `Delete ${item.title || "chat"}`);
    remove.textContent = "x";
    remove.addEventListener("click", (event) => deleteConversation(event, item.session_id));
    remove.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") deleteConversation(event, item.session_id);
    });
    button.append(title, meta, remove);
    button.addEventListener("click", () => switchSession(item.session_id));
    chatHistoryEl.appendChild(button);
  }
}

function renderStoredMessages(conversation) {
  messages.innerHTML = "";
  if (!conversation?.messages?.length) {
    addMessage("assistant", WELCOME_MESSAGE, "", false);
    return;
  }
  for (const message of conversation.messages) {
    if (message.role === "assistant" && message.data) {
      const node = addMessage("assistant", "", "", false);
      renderAssistantResponse(node, message.data, message.data.question || "", false);
    } else {
      addMessage(message.role, message.content, "", false, message.attachments || []);
    }
  }
}

function switchSession(nextSessionId) {
  sessionId = nextSessionId;
  localStorage.setItem("caremind-session-id", sessionId);
  renderStoredMessages(currentConversation());
  renderChatHistory();
  messageInput.focus();
}

async function deleteConversation(event, targetSessionId) {
  event.preventDefault();
  event.stopPropagation();
  const remaining = getChatHistory().filter((item) => item.session_id !== targetSessionId);
  saveChatHistory(remaining);
  try {
    await request(`/chat/sessions/${encodeURIComponent(targetSessionId)}?workspace_id=default`, { method: "DELETE" });
    setStatus("Chat deleted.");
  } catch (error) {
    setStatus(`Local chat deleted. Server cleanup failed: ${error.message}`);
  }
  if (targetSessionId === sessionId) {
    if (remaining.length) {
      sessionId = remaining[0].session_id;
      localStorage.setItem("caremind-session-id", sessionId);
      renderStoredMessages(remaining[0]);
    } else {
      startNewChat();
    }
  }
  renderChatHistory();
}

function startNewChat() {
  sessionId = crypto.randomUUID();
  localStorage.setItem("caremind-session-id", sessionId);
  currentConversation();
  messages.innerHTML = "";
  addMessage("assistant", WELCOME_MESSAGE, "", false);
  renderChatHistory();
  messageInput.focus();
}

function addMessage(role, content, extraClass = "", persist = true, attachments = []) {
  const node = document.createElement("div");
  node.className = `message ${role} ${extraClass}`.trim();
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : role === "assistant" ? "CareMind" : role;
  const body = document.createElement("div");
  body.className = "message-body";
  if (attachments.length) {
    body.appendChild(renderMessageAttachments(attachments));
  }
  const text = document.createElement("div");
  text.className = "message-text";
  text.textContent = role === "assistant" ? cleanAnswerForDisplay(content) : content;
  body.appendChild(text);
  node.append(label, body);
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
  if (persist) rememberMessage(role, content, null, attachments);
  return node;
}

function renderMessageAttachments(attachments = []) {
  const wrap = document.createElement("div");
  wrap.className = "message-attachments";
  for (const attachment of attachments) {
    const item = document.createElement("div");
    item.className = "message-attachment";
    if (attachment.previewUrl && attachment.type === "Image") {
      const img = document.createElement("img");
      img.alt = attachment.name;
      img.src = attachment.previewUrl;
      item.appendChild(img);
    }
    const meta = document.createElement("span");
    meta.textContent = attachment.name;
    const kind = document.createElement("small");
    kind.textContent = attachment.type || "Attachment";
    item.append(meta, kind);
    wrap.appendChild(item);
  }
  return wrap;
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const detail = apiErrorMessage(await response.text(), response.statusText);
    throw new Error(detail);
  }
  return response.json();
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

async function streamChat(payload, handlers = {}) {
  const response = await fetch("/chat/stream", {
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
      let event = "message";
      let dataText = "";
      for (const line of rawEvent.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) dataText += line.slice(5).trim();
      }
      dispatch(event, dataText);
    }
    if (flush && buffer.trim()) {
      let event = "message";
      let dataText = "";
      for (const line of buffer.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) dataText += line.slice(5).trim();
      }
      dispatch(event, dataText);
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

function titleCase(value) {
  return value
    .replace(/\.[^.]+$/, "")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
    .trim();
}

function friendlyName(filename) {
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

function isExternalCitation(citation) {
  return citation?.document_id?.startsWith("pubmed:");
}

function citationUrl(citation) {
  if (!isExternalCitation(citation)) return "";
  const pmid = citation.document_id.replace(/^pubmed:/, "").trim();
  return pmid ? `https://pubmed.ncbi.nlm.nih.gov/${encodeURIComponent(pmid)}/` : "";
}

function makeCitationLink(index, citation) {
  const href = citationUrl(citation);
  if (!href) return null;
  const link = document.createElement("a");
  link.className = "citation-link";
  link.textContent = `[${index}]`;
  link.title = `Open source ${index}`;
  link.href = href;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

function appendInlineCitations(parent, text, citations) {
  for (const part of cleanMarkdownSyntax(text).split(/(\[\d+\])/g)) {
    const match = part.match(/^\[(\d+)\]$/);
    if (match) {
      const index = Number(match[1]);
      const citation = citations[index - 1];
      const link = makeCitationLink(index, citation);
      if (link) parent.appendChild(link);
    } else if (part) {
      parent.appendChild(document.createTextNode(part));
    }
  }
}

function renderReferences(container, citations) {
  const externalCitations = (citations || [])
    .map((citation, index) => ({ citation, index }))
    .filter(({ citation }) => isExternalCitation(citation));
  if (!externalCitations.length) return;
  const refs = document.createElement("div");
  refs.className = "references";
  externalCitations.forEach(({ citation, index }) => {
    const originalIndex = index + 1;
    const card = document.createElement("a");
    card.id = `reference-${index + 1}`;
    card.className = "reference-card";
    card.href = citationUrl(citation);
    card.target = "_blank";
    card.rel = "noopener noreferrer";
    const title = document.createElement("span");
    title.textContent = `[${originalIndex}] ${friendlyName(citation.document_name)}`;
    const quote = document.createElement("small");
    quote.textContent = citation.quote || "Source passage";
    card.append(title, quote);
    refs.appendChild(card);
  });
  container.appendChild(refs);
}

function renderTrace(container, trace, data) {
  if (!trace) return;
  const cacheDebug = trace.cache_debug || trace.cache_metadata?.cache_debug || {};
  const details = document.createElement("details");
  details.className = "trace-panel";
  const summary = document.createElement("summary");
  summary.textContent = "Agent trace";
  details.appendChild(summary);

  const grid = document.createElement("div");
  grid.className = "trace-grid";
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
  for (const [label, value] of rows) {
    const item = document.createElement("div");
    const key = document.createElement("span");
    key.textContent = label;
    const val = document.createElement("strong");
    val.textContent = value;
    item.append(key, val);
    grid.appendChild(item);
  }
  details.appendChild(grid);

  if (trace.steps?.length) {
    const steps = document.createElement("ol");
    steps.className = "trace-steps";
    for (const step of trace.steps) {
      const item = document.createElement("li");
      item.textContent = `${step.node}: ${JSON.stringify(step)}`;
      steps.appendChild(item);
    }
    details.appendChild(steps);
  }

  if (trace.retrieved_chunks?.length) {
    const list = document.createElement("div");
    list.className = "trace-contexts";
    for (const chunk of trace.retrieved_chunks) {
      const card = document.createElement("details");
      const title = document.createElement("summary");
      title.textContent = `${chunk.document_name || "source"}${chunk.score == null ? "" : ` | score ${Number(chunk.score).toFixed(3)}`}`;
      const preview = document.createElement("p");
      preview.textContent = chunk.preview || chunk.quote || "";
      card.append(title, preview);
      list.appendChild(card);
    }
    details.appendChild(list);
  }

  container.appendChild(details);
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

function renderAnswerText(container, answer, citations) {
  const blocks = cleanAnswerForDisplay(answer).split(/\n{2,}/).map((item) => item.trim()).filter(Boolean);
  for (const block of blocks) {
    const lines = block.split("\n").map((line) => line.trim()).filter(Boolean);
    let index = 0;
    while (index < lines.length) {
      const line = lines[index];
      const heading = /^[A-Za-z][A-Za-z0-9 /&()-]{1,60}:$/.test(line);
      if (heading) {
        const title = document.createElement("h3");
        title.className = "response-heading";
        title.textContent = line.replace(/:$/, "");
        container.appendChild(title);
        index += 1;
        continue;
      }

      if (!/^[-*]\s+/.test(line)) {
        const paragraph = document.createElement("p");
        appendInlineCitations(paragraph, line, citations);
        container.appendChild(paragraph);
        index += 1;
        continue;
      }

      const list = document.createElement("ul");
      list.className = "response-list";
      while (index < lines.length && /^[-*]\s+/.test(lines[index])) {
        const item = document.createElement("li");
        appendInlineCitations(item, lines[index].replace(/^[-*]\s+/, ""), citations);
        list.appendChild(item);
        index += 1;
      }
      container.appendChild(list);
    }
  }
}

function renderAssistantResponse(node, data, question, persist = true) {
  try {
    node.className = "message assistant";
    node.textContent = "";
    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = "CareMind";
    const content = document.createElement("div");
    content.className = "message-body response-content";
    const citations = data.citations || [];
    if (chatMode === "agent") {
      const meta = document.createElement("div");
      meta.className = "response-meta";
      const routeBadge = document.createElement("span");
      routeBadge.className = "route-badge";
      routeBadge.textContent = routeLabel(data.route || data.trace?.route);
      const cacheBadge = document.createElement("span");
      cacheBadge.className = "route-badge secondary-badge";
      cacheBadge.textContent = data.trace?.cache_hit ? "Cache hit" : "Fresh";
      meta.append(routeBadge, cacheBadge);
      content.appendChild(meta);
    }

    renderAnswerText(content, data.answer || "", citations);

    renderReferences(content, citations);
    if (chatMode === "agent") renderTrace(content, data.trace, data);
    node.append(label, content);
    if (persist) {
      rememberMessage("assistant", cleanAnswerForDisplay(data.answer || ""), { ...data, question });
    }
    messages.scrollTop = messages.scrollHeight;
  } catch (error) {
    node.className = "message assistant";
    node.textContent = "";
    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = "CareMind";
    const body = document.createElement("div");
    body.className = "message-body";
    body.textContent = cleanAnswerForDisplay(data?.answer || "I received a response, but could not render it.");
    node.append(label, body);
    if (persist) rememberMessage("assistant", body.textContent, data ? { ...data, question } : null);
  }
}

function renderAttachmentTray() {
  if (!attachmentTray) return;
  attachmentTray.innerHTML = "";
  attachmentTray.hidden = composerAttachments.length === 0;
  for (const attachment of composerAttachments) {
    const chip = document.createElement("div");
    chip.className = `attachment-chip ${attachment.status.toLowerCase()}`;
    const kind = document.createElement("span");
    kind.className = "attachment-kind";
    kind.textContent = attachment.type;
    if (attachment.previewUrl && attachment.type === "Image") {
      chip.classList.add("has-preview");
      const preview = document.createElement("img");
      preview.className = "attachment-preview";
      preview.alt = attachment.name;
      preview.src = attachment.previewUrl;
      chip.appendChild(preview);
    }
    const name = document.createElement("span");
    name.className = "attachment-name";
    name.textContent = attachment.name;
    const status = document.createElement("small");
    status.textContent = attachment.error
      ? `${attachment.status}: ${attachment.error}`
      : attachment.status === "Uploading"
        ? `${attachment.progress || 0}%`
        : attachment.status;
    status.title = attachment.error || attachment.status;
    const clear = document.createElement("button");
    clear.type = "button";
    clear.title = "Remove attachment";
    clear.setAttribute("aria-label", `Remove ${attachment.name}`);
    clear.textContent = "x";
    clear.addEventListener("click", () => removeComposerAttachment(attachment.id));
    if (attachment.status === "Failed" && attachment.file) {
      const retry = document.createElement("button");
      retry.type = "button";
      retry.title = "Retry upload";
      retry.setAttribute("aria-label", `Retry ${attachment.name}`);
      retry.textContent = "Retry";
      retry.addEventListener("click", async () => {
        removeComposerAttachment(attachment.id);
        await uploadSelectedFile(attachment.file);
      });
      chip.append(kind, name, status, retry, clear);
    } else {
      chip.append(kind, name, status, clear);
    }
    attachmentTray.appendChild(chip);
  }
}

function removeComposerAttachment(id) {
  composerAttachments = composerAttachments.filter((item) => item.id !== id);
  renderAttachmentTray();
}

async function loadDocuments() {
  const [docs, images] = await Promise.all([
    request("/documents"),
    imagesEl ? request("/images") : Promise.resolve([]),
  ]);

  documentsEl.innerHTML = "";
  if (!docs.length) {
    documentsEl.textContent = "No documents uploaded yet.";
  } else {
    for (const doc of docs) {
      const row = document.createElement("div");
      row.className = "doc-row asset-row";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = doc.document_id;
      const label = document.createElement("span");
      label.textContent = `📄 ${friendlyName(doc.filename)}`;
      const uploadedAt = document.createElement("small");
      uploadedAt.textContent = assetTimestamp(doc.uploaded_at);
      label.appendChild(uploadedAt);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "asset-delete";
      remove.textContent = "Remove";
      remove.addEventListener("click", () => deleteUploadedAsset("document", doc.document_id));
      row.append(checkbox, label, remove);
      documentsEl.appendChild(row);
    }
  }

  if (!imagesEl) return;
  imagesEl.innerHTML = "";
  if (!images.length) {
    imagesEl.textContent = "No images uploaded yet.";
    return;
  }
  for (const image of images) {
    const row = document.createElement("div");
    row.className = "doc-row asset-row";
    const marker = document.createElement("span");
    marker.className = "asset-marker";
    marker.textContent = "🩻";
    const label = document.createElement("span");
    label.textContent = friendlyName(image.filename);
    const uploadedAt = document.createElement("small");
    uploadedAt.textContent = assetTimestamp(image.uploaded_at);
    label.appendChild(uploadedAt);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "asset-delete";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => deleteUploadedAsset("image", image.image_id));
    row.append(marker, label, remove);
    imagesEl.appendChild(row);
  }
}

async function deleteUploadedAsset(kind, id) {
  if (!id) return;
  const path = kind === "image" ? `/images/${encodeURIComponent(id)}?workspace_id=default` : `/documents/${encodeURIComponent(id)}?workspace_id=default`;
  setStatus(`Removing ${kind}...`);
  try {
    await request(path, { method: "DELETE" });
    await loadDocuments();
    composerAttachments = composerAttachments.filter((item) => item.assetId !== id);
    renderAttachmentTray();
    setStatus(`${kind === "image" ? "Image" : "Document"} removed.`);
  } catch (error) {
    setStatus(error.message);
  }
}

async function uploadSelectedFile(file) {
  if (!file) return;
  const attachmentId = crypto.randomUUID();
  composerAttachments = [
    ...composerAttachments,
    {
      id: attachmentId,
      name: friendlyName(file.name),
      type: file.type.startsWith("image/") ? "Image" : "Document",
      status: "Uploading",
      progress: 0,
      file,
      previewUrl: file.type.startsWith("image/") ? URL.createObjectURL(file) : "",
    },
  ];
  renderAttachmentTray();
  const form = new FormData();
  form.append("file", file);
  form.append("workspace_id", "default");
  form.append("session_id", sessionId || "default");
  form.append("modality", document.querySelector("#modality")?.value || "clinical_image");
  form.append("report_text", document.querySelector("#image-note")?.value.trim() || "");
  setStatus(`Uploading ${file.name}...`);
  try {
    const uploaded = await uploadWithProgress(form, (progress) => {
      composerAttachments = composerAttachments.map((item) =>
        item.id === attachmentId ? { ...item, progress } : item,
      );
      renderAttachmentTray();
    });
    await loadDocuments();
    setStatus(`${friendlyName(file.name)} is ready.`);
    composerAttachments = composerAttachments.map((item) =>
      item.id === attachmentId
        ? {
            ...item,
            status: "Ready",
            progress: 100,
            attachmentId: uploaded.attachment_id || uploaded.attachment?.id || "",
            assetId: uploaded.document_id || uploaded.image_id || "",
            assetKind: uploaded.image_id ? "image" : "document",
          }
        : item,
    );
  } catch (error) {
    const message = error.message || "Upload failed.";
    setStatus(`Upload failed: ${message}`);
    composerAttachments = composerAttachments.map((item) =>
      item.id === attachmentId ? { ...item, status: "Failed", error: message } : item,
    );
  } finally {
    renderAttachmentTray();
    if (fileInput) fileInput.value = "";
  }
}

function uploadWithProgress(form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload");
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      onProgress(Math.max(1, Math.round((event.loaded / event.total) * 100)));
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new Error("Upload response was not valid JSON."));
        }
      } else {
        reject(new Error(apiErrorMessage(xhr.responseText, xhr.statusText)));
      }
    };
    xhr.onerror = () => reject(new Error("Upload failed."));
    xhr.send(form);
  });
}

async function transcribeAudio(blob, startedAt, endedAt) {
  const form = new FormData();
  form.append("file", blob, "voice-input.webm");
  form.append("workspace_id", "default");
  form.append("session_id", sessionId || "default");
  form.append("started_at", startedAt || "");
  form.append("ended_at", endedAt || "");
  return request("/speech/transcribe", {
    method: "POST",
    body: form,
  });
}

async function uploadSelectedFiles(fileList) {
  const files = [...(fileList || [])];
  for (const file of files) {
    await uploadSelectedFile(file);
  }
}

uploadForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  await uploadSelectedFile(fileInput?.files?.[0]);
});

composerUploadButton?.addEventListener("click", () => {
  setStatus("Choose a PDF or text file...");
});

fileInput?.addEventListener("change", async () => {
  if (!fileInput.files?.length) {
    setStatus("Ready.");
    return;
  }
  await uploadSelectedFiles(fileInput.files);
});

chatForm?.addEventListener("dragover", (event) => {
  event.preventDefault();
  chatForm.classList.add("is-dragging");
});

chatForm?.addEventListener("dragleave", () => {
  chatForm.classList.remove("is-dragging");
});

chatForm?.addEventListener("drop", async (event) => {
  event.preventDefault();
  chatForm.classList.remove("is-dragging");
  await uploadSelectedFiles(event.dataTransfer?.files);
});

seedDemo.addEventListener("click", async () => {
  setStatus("Seeding demo reports...");
  try {
    await request("/demo/seed", { method: "POST" });
    await loadDocuments();
    setStatus("Demo reports are ready.");
  } catch (error) {
    setStatus(error.message);
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) return;
  const readyAttachments = composerAttachments.filter((item) => item.status === "Ready" && item.attachmentId);
  const blocking = composerAttachments.find((item) => item.status === "Uploading");
  if (blocking) {
    setStatus(`Still uploading ${blocking.name}...`);
    return;
  }
  const failed = composerAttachments.find((item) => item.status === "Failed");
  if (failed) {
    setStatus(`Remove or retry failed attachment: ${failed.name}`);
    return;
  }
  messageInput.value = "";
  addMessage("user", message, "", true, readyAttachments);
  composerAttachments = [];
  renderAttachmentTray();
  const pending = addMessage("assistant", "Preparing response...", "notice", false);
  const transcript = pendingTranscript?.text === message ? pendingTranscript : null;
  pendingTranscript = null;
  try {
    let streamedAnswer = "";
    const data = await streamChat(
      {
        message,
        attachment_ids: readyAttachments.map((item) => item.attachmentId),
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
          pending.className = "message assistant";
          pending.querySelector(".message-body").textContent = cleanAnswerForDisplay(streamedAnswer);
          messages.scrollTop = messages.scrollHeight;
        },
        onStart: () => {
          pending.querySelector(".message-body").textContent = "Preparing response...";
        },
        onStatus: (status) => {
          const stage = status.stage === "generating" ? "Generating response..." : "Preparing response...";
          pending.querySelector(".message-body").textContent = stage;
        },
        onReplace: (answer) => {
          streamedAnswer = answer;
          pending.className = "message assistant";
          pending.querySelector(".message-body").textContent = cleanAnswerForDisplay(answer);
          messages.scrollTop = messages.scrollHeight;
        },
        onMetric: (metric) => {
          if (metric.time_to_first_token_ms != null) setStatus(`First token: ${metric.time_to_first_token_ms} ms`);
        },
      },
    );
    if (!data) throw new Error("Streaming ended before a final response was received.");
    renderAssistantResponse(pending, chatMode === "agent" ? data : sanitizePatientData(data), message);
  } catch (error) {
    pending.className = "message assistant notice";
    pending.querySelector(".message-body").textContent = error.message;
  }
});

compareButton.addEventListener("click", async () => {
  const ids = [...documentsEl.querySelectorAll("input:checked")].map((input) => input.value);
  if (ids.length < 2) {
    addMessage("assistant", "Select two uploaded documents to compare.", "notice");
    return;
  }
  const data = await request("/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_ids: ids.slice(0, 2), workspace_id: "default" }),
  });
  addMessage("assistant", data.summary);
});

async function startVoiceRecording() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Voice input is not supported in this browser.");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    voiceChunks = [];
    voiceStartedAt = new Date().toISOString();
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (event) => {
      if (event.data?.size) voiceChunks.push(event.data);
    };
    mediaRecorder.onstop = async () => {
      const endedAt = new Date().toISOString();
      stream.getTracks().forEach((track) => track.stop());
      voiceButton.classList.remove("is-recording");
      voiceButton.disabled = true;
      setStatus("Transcribing voice input...");
      try {
        const transcript = await transcribeAudio(new Blob(voiceChunks, { type: mediaRecorder.mimeType || "audio/webm" }), voiceStartedAt, endedAt);
        messageInput.value = transcript.text || "";
        pendingTranscript = {
          ...transcript,
          modality: "voice",
          input_modality: "voice",
        };
        const language = transcript.detected_language || transcript.language || "auto";
        const confidence = transcript.language_confidence == null ? "" : ` (${Math.round(transcript.language_confidence * 100)}%)`;
        setStatus(`Voice transcript is ready. Language: ${language}${confidence}.`);
        messageInput.focus();
      } catch (error) {
        setStatus(`Voice transcription failed: ${error.message}`);
      } finally {
        voiceButton.disabled = false;
        mediaRecorder = null;
        voiceChunks = [];
      }
    };
    mediaRecorder.start();
    voiceButton.classList.add("is-recording");
    setStatus("Recording... click the mic again to stop.");
  } catch (error) {
    setStatus(`Voice input failed: ${error.message}`);
  }
}

function stopVoiceRecording() {
  if (mediaRecorder?.state === "recording") {
    setStatus("Stopping recording...");
    mediaRecorder.stop();
  }
}

if (!voiceButton) {
  if (voiceButton) {
    voiceButton.disabled = true;
    voiceButton.title = "Voice input is not supported in this browser";
  }
} else {
  voiceButton.addEventListener("click", () => {
    if (mediaRecorder?.state === "recording") {
      stopVoiceRecording();
      return;
    }
    startVoiceRecording();
  });
}

newChatButton?.addEventListener("click", startNewChat);
patientModeButton?.addEventListener("click", () => setChatMode("patient"));
agentModeButton?.addEventListener("click", () => setChatMode("agent"));

currentConversation();
setChatMode(chatMode);
renderStoredMessages(currentConversation());
renderChatHistory();
renderAttachmentTray();
loadDocuments().catch((error) => addMessage("assistant", error.message));
