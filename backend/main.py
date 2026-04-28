"""
FastAPI application — INFERENCE multi-agent debate backend.

Improvements over v1:
- Pydantic request/response schemas replace raw dict parsing.
- Proper HTTPException on bad input rather than bare dict errors.
- Dataset store encapsulated in a typed class with LRU eviction.
- SSE endpoint validates request before opening the stream.
- Uvicorn log level wired to Python logging.
"""
from __future__ import annotations

import io
import json
import logging
import uuid
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from backend.config import AVAILABLE_MODELS, DEFAULT_MODEL
from backend.engine.debate_graph import run_debate_stream
from backend.rl.trajectory import get_trajectory, get_stats, get_trajectory_list

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="INFERENCE — Multi-Agent Debate Model",
    version="2.0.0",
    description="Bias auditing via structured AI agent debate.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dataset store — simple in-process LRU, max 50 sessions
# ---------------------------------------------------------------------------

class _DatasetStore:
    """Thread-safe (single-process) in-memory CSV store with LRU eviction."""

    MAX = 50

    def __init__(self) -> None:
        self._store: OrderedDict[str, str] = OrderedDict()

    def put(self, csv_text: str) -> str:
        key = str(uuid.uuid4())
        if len(self._store) >= self.MAX:
            self._store.popitem(last=False)  # evict oldest
        self._store[key] = csv_text
        return key

    def get(self, key: str) -> str:
        """Return CSV text, or empty string if not found / expired."""
        return self._store.get(key, "")


_datasets = _DatasetStore()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    rows: int
    columns: list[str]
    dtypes: dict[str, str]


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question or audit instruction.")
    session_id: str = Field("", description="Dataset session ID from /api/upload.")
    use_case: str = Field("general", description="Domain: job_hiring, loan_approval, healthcare, education, general.")
    protected_attributes: list[str] = Field(default_factory=list)
    target_column: str = Field("", description="Outcome/label column name.")
    model: str = Field(DEFAULT_MODEL, description="Model identifier.")
    max_rounds: int = Field(2, ge=1, le=3, description="Number of debate rounds (1-3).")

    @field_validator("model")
    @classmethod
    def model_must_be_valid(cls, v: str) -> str:
        all_models = AVAILABLE_MODELS
        if v not in all_models:
            raise ValueError(f"Unknown model '{v}'. Valid options: {all_models}")
        return v


class ModelListResponse(BaseModel):
    models: list[str]
    default: str


class TrajectoryMeta(BaseModel):
    debate_id: str
    timestamp: str
    use_case: str
    model_backend: str
    max_rounds: int
    protected_attributes: list[str]
    target_column: str
    bias_score: int | None
    severity: str | None
    total_reward: float | None


class TrajectoryStatsResponse(BaseModel):
    total_debates: int
    avg_bias_score: float | None
    avg_total_reward: float | None
    avg_per_agent_rewards: dict
    use_case_breakdown: dict
    model_breakdown: dict


# ---------------------------------------------------------------------------
# Static / frontend
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_index() -> HTMLResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not built.")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok", "service": "INFERENCE"}


@app.get("/api/models", response_model=ModelListResponse, tags=["meta"])
async def list_models() -> ModelListResponse:
    return ModelListResponse(models=AVAILABLE_MODELS, default=DEFAULT_MODEL)


@app.post("/api/upload", response_model=UploadResponse, tags=["data"])
async def upload_dataset(file: UploadFile = File(...)) -> UploadResponse:
    """Accept a CSV file and return its session_id + schema info."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    content = await file.read()
    try:
        csv_text = content.decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}") from exc

    session_id = _datasets.put(csv_text)
    log.info("Dataset uploaded: file=%s rows=%d cols=%d session=%s",
             file.filename, len(df), len(df.columns), session_id)

    return UploadResponse(
        session_id=session_id,
        filename=file.filename or "",
        rows=len(df),
        columns=list(df.columns),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
    )


@app.post("/api/chat/stream", tags=["debate"])
async def chat_stream(req: ChatRequest) -> EventSourceResponse:
    """
    Run the full multi-agent debate and stream events via Server-Sent Events.

    Event types:
    - `round_start`   → {round: int}
    - `agent_message` → {agent, role, content, round}
    - `final_report`  → {bias_score, severity, summary, flagged_issues, ...}
    - `error`         → {message: str}
    - `done`          → {}
    """
    csv_text = _datasets.get(req.session_id)
    log.info("Debate started: model=%s use_case=%s rounds=%d session=%s",
             req.model, req.use_case, req.max_rounds, req.session_id or "<none>")

    async def _generate():
        async for event in run_debate_stream(
            user_query=req.query,
            dataset_csv=csv_text,
            use_case=req.use_case,
            protected_attributes=req.protected_attributes,
            target_column=req.target_column,
            model_backend=req.model,
            max_rounds=req.max_rounds,
        ):
            yield {
                "event": event.get("event", "message"),
                "data": json.dumps(event.get("data", {})),
            }

    return EventSourceResponse(_generate())


# ---------------------------------------------------------------------------
# Trajectory / History endpoints
# ---------------------------------------------------------------------------

@app.get("/api/trajectories", tags=["history"], response_model=list[TrajectoryMeta])
async def list_trajectories(limit: int = 50):
    """
    Return a lightweight summary list of past debates, newest-first.
    Each item includes debate_id, timestamp, bias_score, severity, total_reward.
    """
    return get_trajectory_list(limit=limit)


@app.get("/api/trajectories/stats", tags=["history"], response_model=TrajectoryStatsResponse)
async def trajectory_stats():
    """
    Aggregate statistics across all logged debates:
    total count, average bias score, average reward, per-agent reward means,
    use-case breakdown, and model breakdown.
    """
    return get_stats()


@app.get("/api/trajectories/{debate_id}", tags=["history"])
async def get_debate(debate_id: str) -> dict:
    """
    Return the full record for a single debate: transcript, fairness metrics,
    final report, global reward, and per-agent rewards.
    """
    record = get_trajectory(debate_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Debate '{debate_id}' not found.")
    return record


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    from backend.config import HOST, PORT
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True, log_level="info")
