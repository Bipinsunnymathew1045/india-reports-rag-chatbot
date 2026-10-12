import os
import time
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

load_dotenv(".chatenv")

# ── Logging setup ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Import RAG pipeline ────────────────────────────────────────────
logger.info("Loading RAG pipeline...")
from query import ask, detect_domain
logger.info("RAG pipeline loaded successfully")

# ── FastAPI app ────────────────────────────────────────────────────
app = FastAPI(
    title="India Reports RAG API",
    description="""
    Ask questions about 10 official Government of India annual reports
    spanning education and health domains from 2020 to 2025.
    Answers are grounded in source documents with page-level citations.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS middleware ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Prometheus metrics ─────────────────────────────────────────────
# Automatically tracks: request count, latency, status codes
Instrumentator().instrument(app).expose(app)

# ── Request / Response models ──────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Question to ask about Indian government reports",
        example="How many AIIMS hospitals exist in India?"
    )

class AnswerResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    domain: str
    response_time_ms: float

class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    total_documents: str

class ErrorResponse(BaseModel):
    error: str
    detail: str

# ── Request counter for basic stats ───────────────────────────────
request_stats = {
    "total": 0,
    "health": 0,
    "education": 0,
    "errors": 0
}

# ── Routes ─────────────────────────────────────────────────────────

@app.get(
    "/",
    response_model=HealthResponse,
    summary="Root health check"
)
def root():
    """Basic health check — confirms API is running."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model": "gpt-4o-mini",
        "total_documents": "3,492 pages across 10 reports"
    }

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check for monitoring"
)
def health_check():
    """
    Health check endpoint for monitoring systems.
    Returns healthy when RAG pipeline is loaded and ready.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model": "gpt-4o-mini",
        "total_documents": "3,492 pages across 10 reports"
    }

@app.get(
    "/stats",
    summary="Request statistics"
)
def get_stats():
    """Returns basic request statistics since server started."""
    return {
        "total_requests": request_stats["total"],
        "health_domain_requests": request_stats["health"],
        "education_domain_requests": request_stats["education"],
        "total_errors": request_stats["errors"],
    }

@app.post(
    "/ask",
    response_model=AnswerResponse,
    summary="Ask a question about Indian government reports",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid question"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
def ask_question(request: QuestionRequest):
    """
    Ask any question about Indian government education
    and health reports from 2020 to 2025.

    Returns:
    - **answer**: grounded response from documents
    - **citations**: source report filenames and page numbers
    - **domain**: detected domain (health/education/both)
    - **response_time_ms**: time taken to generate answer
    """
    logger.info(f"Question received: {request.question[:60]}...")

    # Update stats
    request_stats["total"] += 1

    try:
        # Detect domain
        domain = detect_domain(request.question)
        if domain == "health":
            request_stats["health"] += 1
        else:
            request_stats["education"] += 1

        # Time the RAG call
        start_time = time.time()
        result = ask(request.question)
        response_time = (time.time() - start_time) * 1000

        logger.info(
            f"Answered in {response_time:.0f}ms | "
            f"domain={domain} | "
            f"citations={len(result['citations'])}"
        )

        return {
            "question": request.question,
            "answer": result["answer"],
            "citations": result["citations"],
            "domain": domain,
            "response_time_ms": round(response_time, 2)
        }

    except Exception as e:
        request_stats["errors"] += 1
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}"
        )

# ── Run the server ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )