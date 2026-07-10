const sessionId = crypto.randomUUID();
const messages = document.querySelector("#messages");
const documentsEl = document.querySelector("#documents");
const imagesEl = document.querySelector("#images");
const uploadForm = document.querySelector("#upload-form");
const chatForm = document.querySelector("#chat-form");
const messageInput = document.querySelector("#message");
const voiceButton = document.querySelector("#voice");
const seedDemo = document.querySelector("#seed-demo");
const compareButton = document.querySelector("#compare");
const statusEl = document.querySelector("#status");
const themeToggle = document.querySelector("#theme-toggle");
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let pendingTranscript = null;

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

function addMessage(role, content, extraClass = "") {
  const node = document.createElement("div");
  node.className = `message ${role} ${extraClass}`.trim();
  node.textContent = content;
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
  return node;
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  return response.json();
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

function makeCitationLink(index) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "citation-link";
  button.textContent = `[${index}]`;
  button.title = `Show source ${index}`;
  button.addEventListener("click", () => {
    const card = document.querySelector(`#reference-${index}`);
    if (!card) return;
    card.classList.toggle("is-open");
    card.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });
  return button;
}

function appendInlineCitations(parent, text, citations) {
  for (const part of text.split(/(\[\d+\])/g)) {
    const match = part.match(/^\[(\d+)\]$/);
    if (match && citations[Number(match[1]) - 1]) {
      parent.appendChild(makeCitationLink(Number(match[1])));
    } else if (part) {
      parent.appendChild(document.createTextNode(part));
    }
  }
}

function renderReferences(container, citations) {
  if (!citations?.length) return;
  const refs = document.createElement("div");
  refs.className = "references";
  citations.forEach((citation, index) => {
    const card = document.createElement("button");
    card.type = "button";
    card.id = `reference-${index + 1}`;
    card.className = "reference-card";
    const title = document.createElement("span");
    title.textContent = `[${index + 1}] ${friendlyName(citation.document_name)}`;
    const quote = document.createElement("small");
    quote.textContent = citation.quote || "Source passage";
    card.append(title, quote);
    card.addEventListener("click", () => card.classList.toggle("is-open"));
    refs.appendChild(card);
  });
  container.appendChild(refs);
}

function renderTrace(container, trace, data) {
  if (!trace) return;
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
    ["Latency", `${trace.total_latency_ms ?? "-"} ms`],
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

function renderAnswerText(container, answer, citations) {
  const blocks = answer.split(/\n{2,}/).map((item) => item.trim()).filter(Boolean);
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

function renderAssistantResponse(node, data, question) {
  try {
    node.className = "message assistant";
    node.textContent = "";
    const content = document.createElement("div");
    content.className = "response-content";
    const citations = data.citations || [];

    renderAnswerText(content, data.answer || "", citations);

    renderReferences(content, citations);
    renderTrace(content, data.trace, data);
    node.appendChild(content);
    messages.scrollTop = messages.scrollHeight;
  } catch (error) {
    node.className = "message assistant";
    node.textContent = data?.answer || "I received a response, but could not render it.";
  }
}

async function loadDocuments() {
  const [docs, images] = await Promise.all([
    request("/documents"),
    imagesEl ? request("/images") : Promise.resolve([]),
  ]);

  documentsEl.innerHTML = "";
  const visibleDocs = uniqueByName(docs, (doc) => friendlyName(doc.filename));
  if (!visibleDocs.length) {
    documentsEl.textContent = "No documents uploaded yet.";
  } else {
    for (const doc of visibleDocs) {
      const row = document.createElement("label");
      row.className = "doc-row";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = doc.document_id;
      const label = document.createElement("span");
      label.textContent = `📄 ${friendlyName(doc.filename)}`;
      row.append(checkbox, label);
      documentsEl.appendChild(row);
    }
  }

  if (!imagesEl) return;
  imagesEl.innerHTML = "";
  const visibleImages = uniqueByName(images, (image) => friendlyName(image.filename));
  if (!visibleImages.length) {
    imagesEl.textContent = "No images uploaded yet.";
    return;
  }
  for (const image of visibleImages) {
    const row = document.createElement("div");
    row.className = "doc-row";
    const marker = document.createElement("span");
    marker.className = "asset-marker";
    marker.textContent = "🩻";
    const label = document.createElement("span");
    label.textContent = friendlyName(image.filename);
    row.append(marker, label);
    imagesEl.appendChild(row);
  }
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = document.querySelector("#file").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  form.append("workspace_id", "default");
  form.append("modality", document.querySelector("#modality")?.value || "clinical_image");
  form.append("report_text", document.querySelector("#image-note")?.value.trim() || "");
  setStatus(`Uploading ${file.name}...`);
  try {
    await request("/upload", { method: "POST", body: form });
    await loadDocuments();
    setStatus(`${friendlyName(file.name)} is ready.`);
  } catch (error) {
    setStatus(error.message);
  }
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
  messageInput.value = "";
  addMessage("user", message);
  const pending = addMessage("assistant", "Thinking...", "notice");
  const transcript = pendingTranscript?.text === message ? pendingTranscript : null;
  pendingTranscript = null;
  try {
    const data = await request("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        workspace_id: "default",
        modality: transcript ? "voice" : "text",
        transcript,
      }),
    });
    renderAssistantResponse(pending, data, message);
  } catch (error) {
    pending.textContent = error.message;
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

if (!SpeechRecognition || !voiceButton) {
  if (voiceButton) {
    voiceButton.disabled = true;
    voiceButton.title = "Voice input is not supported in this browser";
  }
} else {
  const recognition = new SpeechRecognition();
  recognition.lang = navigator.language || "en-US";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  voiceButton.addEventListener("click", () => {
    setStatus("Listening...");
    voiceButton.disabled = true;
    const startedAt = new Date().toISOString();
    recognition.onresult = (event) => {
      const result = event.results[0][0];
      const endedAt = new Date().toISOString();
      messageInput.value = result.transcript;
      pendingTranscript = {
        text: result.transcript,
        confidence: result.confidence,
        language: recognition.lang,
        started_at: startedAt,
        ended_at: endedAt,
        input_modality: "voice",
      };
      setStatus("Voice transcript is ready.");
      voiceButton.disabled = false;
      messageInput.focus();
    };
    recognition.onerror = (event) => {
      setStatus(`Voice input failed: ${event.error}`);
      voiceButton.disabled = false;
    };
    recognition.onend = () => {
      voiceButton.disabled = false;
    };
    recognition.start();
  });
}

loadDocuments().catch((error) => addMessage("assistant", error.message));
