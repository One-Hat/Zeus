"""Backend Worker Agent: generates backend API, persistence, and unit tests."""

from __future__ import annotations

import json
from typing import Optional
from forge.llm import LLMClient
from forge.sandbox.runner import DockerSandboxRunner
from forge.schemas import Spec, WorkerOutput

BACKEND_SYSTEM_PROMPT = """You are the Forge Backend Worker Agent.
You receive the application `Spec` and ONLY the `backend` tasks.
Produce a `WorkerOutput` with `role="backend"` and `files` mapping relative paths to complete, working Python code.
Use FastAPI, Pydantic, and sqlite3 so it runs cleanly inside the sandbox.
Always include:
- `backend/__init__.py`
- `backend/app.py` (with `app = FastAPI()`, `/health`, and all endpoints from `api_contract`)
- `backend/test_app.py` (pytest unit tests using FastAPI `TestClient`)
- `requirements.txt`
"""

DEFAULT_BACKEND_APP_PY = '''"""Generated FastAPI application."""

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Forge Generated App", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ItemCreate(BaseModel):
    title: str
    completed: bool = False


class Item(ItemCreate):
    id: int


_ITEMS_DB: List[Item] = []


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/items", response_model=List[Item])
def list_items():
    return _ITEMS_DB


@app.post("/api/items", response_model=Item, status_code=status.HTTP_201_CREATED)
def create_item(payload: ItemCreate):
    new_item = Item(id=len(_ITEMS_DB) + 1, title=payload.title, completed=payload.completed)
    _ITEMS_DB.append(new_item)
    return new_item


@app.delete("/api/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int):
    global _ITEMS_DB
    _ITEMS_DB = [i for i in _ITEMS_DB if i.id != item_id]
    return Response(status_code=status.HTTP_204_NO_CONTENT)
'''

DEFAULT_BACKEND_TEST_PY = '''"""Unit tests for the generated FastAPI backend."""

from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_create_and_list_items():
    res = client.post("/api/items", json={"title": "Test Task", "completed": False})
    assert res.status_code == 201
    data = res.json()
    assert data["title"] == "Test Task"

    list_res = client.get("/api/items")
    assert list_res.status_code == 200
    assert any(i["title"] == "Test Task" for i in list_res.json())
'''


class BackendWorker:
    """Worker that implements backend tasks and writes files to the sandbox."""

    def __init__(self, llm_client: LLMClient, sandbox: Optional[DockerSandboxRunner] = None):
        self.llm = llm_client
        self.sandbox = sandbox

    @staticmethod
    def build_default_output() -> WorkerOutput:
        return WorkerOutput(
            role="backend",
            files={
                "backend/__init__.py": "",
                "backend/app.py": DEFAULT_BACKEND_APP_PY,
                "backend/test_app.py": DEFAULT_BACKEND_TEST_PY,
                "requirements.txt": "fastapi>=0.110.0\nuvicorn>=0.29.0\nhttpx>=0.27.0\npytest>=8.0.0\n",
            },
            summary="Implemented FastAPI CRUD backend, in-memory/SQLite persistence, and unit tests.",
        )

    def run(self, spec: Spec, override_model: Optional[str] = None) -> WorkerOutput:
        backend_tasks = [t.model_dump() for t in spec.tasks if t.role == "backend"]
        self.llm.mock_responses.setdefault("backend", self.build_default_output())

        payload = {
            "app_name": spec.name,
            "api_contract": spec.api_contract,
            "db_schema": spec.db_schema,
            "acceptance_tests": [t.model_dump() for t in spec.acceptance_tests],
            "assigned_tasks": backend_tasks,
        }

        messages = [
            {"role": "system", "content": BACKEND_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ]
        output = self.llm.call_structured(
            role="backend",
            messages=messages,
            response_model=WorkerOutput,
            override_model=override_model,
        )

        if self.sandbox is not None and output.files:
            self.sandbox.write_files(output.files)

        return output
