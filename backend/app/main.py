"""FastAPI app — local server that drives the pipeline and serves the frontend (§10.2).

Run: `uvicorn app.main:app --port 8756` from the backend/ directory.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .batch import BatchRegistry
from .cache import Cache
from .jobs import JobRegistry
from .routes.api import router as api_router

# Repo paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent
FRONTEND_DIR = REPO_DIR / "frontend"
UPLOAD_DIR = BACKEND_DIR / "tmp" / "uploads"
CACHE_DIR = BACKEND_DIR / "tmp" / "cache"


def create_app() -> FastAPI:
    app = FastAPI(title="WhisperMerger")

    # Local single-user app; allow the browser frontend on localhost.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Shared state, reached in routes via request.app.state.
    app.state.jobs = JobRegistry()
    app.state.batches = BatchRegistry()
    app.state.cache = Cache(CACHE_DIR)
    app.state.uploads = {}          # file_id -> {"path": str, "source": InputSource}
    app.state.combined = {}         # combo key -> combined file_id (Label Track sequencing)
    app.state.upload_dir = UPLOAD_DIR

    app.include_router(api_router, prefix="/api")

    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    return app


app = create_app()
