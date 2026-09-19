import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vault.agent import AgentTools, VaultAgent

app = FastAPI(title="Vault Enterprise RAG API", version="1.0.0")

# Request / Response Schemas
class AskRequest(BaseModel):
    question: str
    role: str = "public"

class AskResponse(BaseModel):
    question: str
    role: str
    answer: str
    citations: list[str]
    refused: bool
    refusal_reason: str | None
    latency_ms: float

# Initialize Agent
tools = AgentTools()
agent = VaultAgent(tools=tools)

# Static file directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/health")
def health_check() -> dict[str, str]:
    """Healthcheck endpoint."""
    return {"status": "ok", "service": "vault-api"}

@app.get("/")
def read_root() -> FileResponse:
    """Serve static web frontend."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_file):
        raise HTTPException(status_code=404, detail="Frontend HTML not found.")
    return FileResponse(index_file)

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """
    Executes VaultAgent pipeline and returns structured answer with latency.
    """
    start_time = time.perf_counter()
    response = agent.run(question=request.question, role=request.role)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return AskResponse(
        question=response.question,
        role=response.role,
        answer=response.answer,
        citations=response.citations,
        refused=response.refused,
        refusal_reason=response.refusal_reason,
        latency_ms=round(elapsed_ms, 2)
    )
