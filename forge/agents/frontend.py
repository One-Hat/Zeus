"""Frontend Worker Agent: generates frontend UI assets and client logic."""

from __future__ import annotations

import json
from typing import Optional
from forge.llm import LLMClient
from forge.sandbox.runner import DockerSandboxRunner
from forge.schemas import Spec, WorkerOutput

FRONTEND_SYSTEM_PROMPT = """You are the Forge Frontend Worker Agent.
You receive the application `Spec` and ONLY the `frontend` tasks.
Produce a `WorkerOutput` with `role="frontend"` and `files` mapping relative paths to complete HTML/CSS/JS files.
Ensure the frontend connects to the OpenAPI endpoints defined in `api_contract`.
"""

DEFAULT_FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Forge App</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; }
    form { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; }
    input[type="text"] { flex: 1; padding: 0.5rem; }
    button { padding: 0.5rem 1rem; cursor: pointer; }
    ul { list-style: none; padding: 0; }
    li { padding: 0.5rem; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; }
  </style>
</head>
<body>
  <h1>Forge Application</h1>
  <form id="item-form">
    <input type="text" id="item-input" placeholder="Add new item..." required />
    <button type="submit">Add</button>
  </form>
  <ul id="item-list"></ul>
  <script src="app.js"></script>
</body>
</html>
"""

DEFAULT_FRONTEND_JS = """const API_BASE = '/api/items';

async function fetchItems() {
  const res = await fetch(API_BASE);
  if (!res.ok) return;
  const items = await res.json();
  const list = document.getElementById('item-list');
  list.innerHTML = '';
  for (const item of items) {
    const li = document.createElement('li');
    li.textContent = item.title;
    list.appendChild(li);
  }
}

document.getElementById('item-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = document.getElementById('item-input');
  if (!input.value.trim()) return;
  await fetch(API_BASE, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: input.value.trim(), completed: false })
  });
  input.value = '';
  await fetchItems();
});

fetchItems();
"""


class FrontendWorker:
    """Worker that implements frontend tasks and writes UI files to the sandbox."""

    def __init__(self, llm_client: LLMClient, sandbox: Optional[DockerSandboxRunner] = None):
        self.llm = llm_client
        self.sandbox = sandbox

    @staticmethod
    def build_default_output() -> WorkerOutput:
        return WorkerOutput(
            role="frontend",
            files={
                "frontend/index.html": DEFAULT_FRONTEND_HTML,
                "frontend/app.js": DEFAULT_FRONTEND_JS,
            },
            summary="Created responsive HTML5/JS single-page interface integrated with /api/items.",
        )

    def run(self, spec: Spec, override_model: Optional[str] = None) -> WorkerOutput:
        frontend_tasks = [t.model_dump() for t in spec.tasks if t.role == "frontend"]
        self.llm.mock_responses.setdefault("frontend", self.build_default_output())

        payload = {
            "app_name": spec.name,
            "api_contract": spec.api_contract,
            "assigned_tasks": frontend_tasks,
        }

        messages = [
            {"role": "system", "content": FRONTEND_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ]
        output = self.llm.call_structured(
            role="frontend",
            messages=messages,
            response_model=WorkerOutput,
            override_model=override_model,
        )

        if self.sandbox is not None and output.files:
            self.sandbox.write_files(output.files)

        return output
