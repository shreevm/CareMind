import * as vscode from "vscode";
import { randomUUID } from "crypto";

export function activate(context: vscode.ExtensionContext) {
  const provider = new CareMindViewProvider(context.extensionUri, context);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("caremind.chatView", provider)
  );
}

export function deactivate() {}

class CareMindViewProvider implements vscode.WebviewViewProvider {
  private sessionId: string;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly context: vscode.ExtensionContext
  ) {
    this.sessionId = this.resolveSessionId();
  }

  resolveWebviewView(webviewView: vscode.WebviewView) {
    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this.extensionUri],
    };
    webviewView.webview.html = this.html();

    webviewView.webview.onDidReceiveMessage(async (message) => {
      if (message.type !== "chat") return;
      const apiBaseUrl = vscode.workspace
        .getConfiguration("caremind")
        .get<string>("apiBaseUrl", "http://127.0.0.1:8000");
      this.sessionId = this.resolveSessionId();

      try {
        const response = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: message.text,
            session_id: this.sessionId,
            workspace_id: "default",
          }),
        });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        webviewView.webview.postMessage({ type: "answer", data });
      } catch (error) {
        webviewView.webview.postMessage({
          type: "error",
          message: error instanceof Error ? error.message : String(error),
        });
      }
    });
  }

  private html() {
    return /* html */ `
      <!doctype html>
      <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <style>
            body { font-family: var(--vscode-font-family); padding: 12px; }
            #messages { display: grid; gap: 10px; margin-bottom: 12px; }
            .msg { border: 1px solid var(--vscode-panel-border); border-radius: 6px; padding: 10px; white-space: pre-wrap; }
            .user { background: var(--vscode-input-background); }
            form { display: grid; grid-template-columns: 1fr auto; gap: 8px; }
            input { min-width: 0; }
          </style>
        </head>
        <body>
          <div id="messages">
            <div class="msg">Ask about documents uploaded to the shared CareMind backend.</div>
          </div>
          <form id="form">
            <input id="input" placeholder="Ask CareMind..." />
            <button>Send</button>
          </form>
          <script>
            const vscode = acquireVsCodeApi();
            const messages = document.querySelector("#messages");
            const form = document.querySelector("#form");
            const input = document.querySelector("#input");
            function add(text, cls = "") {
              const node = document.createElement("div");
              node.className = "msg " + cls;
              node.textContent = text;
              messages.appendChild(node);
            }
            form.addEventListener("submit", (event) => {
              event.preventDefault();
              const text = input.value.trim();
              if (!text) return;
              input.value = "";
              add(text, "user");
              vscode.postMessage({ type: "chat", text });
            });
            window.addEventListener("message", (event) => {
              const message = event.data;
              if (message.type === "answer") {
                const citations = (message.data.citations || []).map((citation, index) =>
                  "[" + (index + 1) + "] " + citation.document_name + ": " + citation.quote
                ).join("\\n");
                add(citations ? message.data.answer + "\\n\\nEvidence\\n" + citations : message.data.answer);
              }
              if (message.type === "error") add(message.message);
            });
          </script>
        </body>
      </html>
    `;
  }

  private resolveSessionId(): string {
    const configured = vscode.workspace
      .getConfiguration("caremind")
      .get<string>("sessionId", "")
      .trim();
    if (configured) return configured;
    const stored = this.context.globalState.get<string>("caremind.sessionId");
    if (stored) return stored;
    const generated = randomUUID();
    this.context.globalState.update("caremind.sessionId", generated);
    return generated;
  }
}
