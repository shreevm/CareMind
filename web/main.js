const sessionId = crypto.randomUUID();
const messages = document.querySelector("#messages");
const documentsEl = document.querySelector("#documents");
const uploadForm = document.querySelector("#upload-form");
const chatForm = document.querySelector("#chat-form");
const messageInput = document.querySelector("#message");
const seedDemo = document.querySelector("#seed-demo");
const compareButton = document.querySelector("#compare");
const statusEl = document.querySelector("#status");
const themeToggle = document.querySelector("#theme-toggle");

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

async function loadDocuments() {
  const docs = await request("/documents");
  documentsEl.innerHTML = "";
  if (!docs.length) {
    documentsEl.textContent = "No documents uploaded yet.";
    return;
  }
  for (const doc of docs) {
    const row = document.createElement("label");
    row.className = "doc-row";
    row.innerHTML = `
      <input type="checkbox" value="${doc.document_id}">
      <span>${doc.filename}<small>${doc.chunk_count} chunks</small></span>
    `;
    documentsEl.appendChild(row);
  }
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = document.querySelector("#file").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  form.append("workspace_id", "default");
  setStatus(`Uploading ${file.name}...`);
  try {
    await request("/upload", { method: "POST", body: form });
    await loadDocuments();
    setStatus(`${file.name} is indexed and ready for questions.`);
  } catch (error) {
    setStatus(error.message);
  }
});

seedDemo.addEventListener("click", async () => {
  setStatus("Seeding two synthetic reports...");
  try {
    await request("/demo/seed", { method: "POST" });
    await loadDocuments();
    setStatus("Demo reports are ready. Try asking: what changed between the reports?");
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
  try {
    const data = await request("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId, workspace_id: "default" }),
    });
    const citations = data.citations
      .map((citation, index) => `[${index + 1}] ${citation.document_name}: ${citation.quote}`)
      .join("\n");
    pending.className = "message assistant";
    pending.textContent = citations ? `${data.answer}\n\nEvidence\n${citations}` : data.answer;
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

loadDocuments().catch((error) => addMessage("assistant", error.message));
