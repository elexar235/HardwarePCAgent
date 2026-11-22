"""FastAPI web server and browser chat UI for the PC Builder Agent."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.agent.graph import app as agent_graph
from src.agent.nodes import _determine_workload_profile

STATIC_DIR = Path(__file__).resolve().parent / "static"

web_app = FastAPI(
    title="HardwarePCAgent Web UI",
    description="Browser chat interface for the Autonomous PC Builder Agent",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class ChatMeta(BaseModel):
    parser_source: str
    profile: str
    profile_label: str
    budget_usd: Optional[int] = None
    resolution: Optional[str] = None
    target_games: List[str] = Field(default_factory=list)
    work_tasks: List[str] = Field(default_factory=list)
    priorities: List[str] = Field(default_factory=list)
    budget_weights: Optional[Dict[str, float]] = None
    silence_focus: bool = False
    prefer_liquid_cooler: bool = False
    prefer_aesthetic_case: bool = False


class ChatResponse(BaseModel):
    response: str
    meta: Optional[ChatMeta] = None


PROFILE_LABELS = {
    "gaming": "🎮 High-Performance Gaming",
    "productivity": "💼 Workstation & Development",
    "ai_3d": "🧠 AI / ML & 3D Rendering",
    "office": "📄 General Office & Web Surfing",
}


def _run_agent_sync(query: str) -> Dict[str, Any]:
    initial_state = {
        "user_query": query,
        "requirements": None,
        "budget_allocation": None,
        "current_build": None,
        "validation_result": None,
        "iteration_count": 0,
        "feedback_history": [],
        "final_response": None,
        "parser_source": None,
    }
    return agent_graph.invoke(initial_state)


@web_app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@web_app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=500, detail="Frontend template not found.")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@web_app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    final_state = await run_in_threadpool(_run_agent_sync, query)
    req = final_state.get("requirements")
    source = final_state.get("parser_source") or "Deterministic Heuristic Engine"

    meta = None
    if req:
        profile = _determine_workload_profile(req)
        meta = ChatMeta(
            parser_source=source,
            profile=profile,
            profile_label=PROFILE_LABELS.get(profile, profile),
            budget_usd=req.budget_usd,
            resolution=req.resolution.upper() if req.resolution else "Auto",
            target_games=req.target_games,
            work_tasks=req.work_tasks,
            priorities=req.priorities,
            budget_weights=req.budget_weights,
            silence_focus=req.silence_focus,
            prefer_liquid_cooler=req.prefer_liquid_cooler,
            prefer_aesthetic_case=req.prefer_aesthetic_case,
        )

    response_text = final_state.get("final_response") or "Failed to generate PC configuration."
    return ChatResponse(response=response_text, meta=meta)
