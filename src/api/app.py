import json
import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Optional
import time
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.workflow.service import AgentOrchestrator

logger = logging.getLogger(__name__)

# ------------------ #
# --- RATE LIMIT --- #
# ------------------ #

_RATE_LIMIT_REQUESTS = 20   # max requests per window
_RATE_LIMIT_WINDOW  = 60.0  # seconds

_request_log: dict[str, deque] = defaultdict(deque)


def _get_client_ip(request: Request) -> str:
    """Extract the real client IP, respecting ``X-Forwarded-For`` proxy headers."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(client_ip: str) -> None:
    """Enforce a sliding-window rate limit for ``client_ip``.

    Raises:
        HTTPException: 429 Too Many Requests when the limit is exceeded.
    """
    now = time.monotonic()
    window = _request_log[client_ip]
    # Evict timestamps outside the sliding window
    while window and now - window[0] > _RATE_LIMIT_WINDOW:
        window.popleft()
    if len(window) >= _RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {_RATE_LIMIT_REQUESTS} requests per {int(_RATE_LIMIT_WINDOW)}s.",
        )
    window.append(now)


def _build_error_response(message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR) -> HTTPException:
    """Wrap a message in an ``HTTPException`` with the given status code."""
    return HTTPException(status_code=status_code, detail=message)


# --------------- #
# --- SCHEMAS --- #
# --------------- #

class ResumeRequest(BaseModel):
    thread_id: str = Field(..., description="Unique ID of the suspended conversation")
    approve: bool = Field(..., description="True to approve the exception, False to deny it")
    feedback: Optional[str] = Field(None, description="Optional manager comment or reason for the decision.")

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=5000, description="The user's question or prompt")
    thread_id: str = Field(..., description="Unique ID to maintain conversation memory")

class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    trace_id: Optional[str] = None

# ---------------- #
# --- LIFESPAN --- #
# ---------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the engine on server startup and stores it in global state."""
    logger.info("🚀 Starting Production API...")
    app.state.orchestrator = await AgentOrchestrator.create_async()
    yield
    logger.info("🛑 Shutting down server gracefully...")

app = FastAPI(title="Enterprise Agent API", lifespan=lifespan)

@app.get("/", include_in_schema=False)
def root():
    """Redirects to the API documentation."""
    return RedirectResponse(url="/docs")


# ----------------- #
# --- ENDPOINTS --- #
# ----------------- #

@app.post("/v1/chat", summary="Process a chat query", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, fast_request: Request):
    """Invoke the agent graph synchronously and return the final response."""
    _check_rate_limit(_get_client_ip(fast_request))
    orchestrator: AgentOrchestrator = fast_request.app.state.orchestrator

    try:
        result = await orchestrator.aprocess_query(request.query, request.thread_id)
        return ChatResponse(
            answer=result.get("respuesta", "No response generated."),
            thread_id=request.thread_id,
            trace_id=result.get("trace_id"),
        )
    except Exception as e:
        logger.error("Endpoint error: %s", e)
        raise _build_error_response("Error processing the request.")


@app.post("/v1/chat/stream", summary="Stream a chat query in real-time", response_model=ChatResponse)
async def chat_stream_endpoint(request: ChatRequest, fast_request: Request):
    """Server-Sent Events endpoint — streams graph node updates and final tokens."""
    _check_rate_limit(_get_client_ip(fast_request))
    orchestrator: AgentOrchestrator = fast_request.app.state.orchestrator

    async def event_generator():
        try:
            async for event in orchestrator.astream_query(request.query, request.thread_id):
                yield json.dumps(event) + "\n"
        except Exception as e:
            logger.error("Streaming error: %s", e)
            yield json.dumps({"type": "error", "content": "An error occurred during streaming."}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/v1/chat/resume", summary="Resume a suspended workflow after manager approval", response_model=ChatResponse)
async def resume_chat_endpoint(request: ResumeRequest, fast_request: Request):
    """
    Endpoint to resume a graph paused by a human interruption (HITL).
    Updates the state by injecting the 'manager_override' and executes the remaining node.
    """
    _check_rate_limit(_get_client_ip(fast_request))
    orchestrator: AgentOrchestrator = fast_request.app.state.orchestrator

    try:
        result = await orchestrator.aresume_suspended_query(
            request.thread_id,
            request.approve,
            request.feedback,
        )
        return ChatResponse(
            answer=result.get("respuesta", "No response generated."),
            thread_id=request.thread_id,
            trace_id=result.get("trace_id"),
        )
    except Exception as e:
        logger.error("Error resuming workflow: %s", e)
        raise _build_error_response("Error resuming the approval workflow.")


@app.get("/health", summary="Health check endpoint")
def health_check():
    return {"status": "ok"}