"""
SCREEN FastAPI backend.

Run locally:
    uvicorn main:app --reload --port 8000

ENV variables:
    INDEX_DIR   path to folder with screen_hnsw.index / pca_model.joblib / metadata.csv
                (default: ../index)
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import SCREENInference


SUPPORTED_VIDEO = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
SUPPORTED_IMAGE = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    index_dir = os.getenv("INDEX_DIR", "../index")
    state["screen"] = SCREENInference(index_dir=index_dir)
    yield
    state.clear()


app = FastAPI(
    title="SCREEN API",
    description="Cinema Recognition Engine — identify a movie from an image or a video clip.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class MovieMatch(BaseModel):
    movie_name: str
    year:       str
    director:   str
    genre:      str
    confidence: float


class QueryResponse(BaseModel):
    results:      list[MovieMatch]
    frames_used:  int | None = None   # only present for video queries


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["meta"])
def health():
    screen: SCREENInference = state["screen"]
    return {
        "status":          "ok",
        "vectors_indexed": screen.index.ntotal,
        "device":          screen.device,
    }


@app.post("/query/image", response_model=QueryResponse, tags=["search"])
async def query_image(
    file:  UploadFile = File(..., description="JPEG / PNG / WEBP screenshot."),
    top_k: int = 5,
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_IMAGE and not (file.content_type or "").startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported image type: {file.content_type}. Use JPG, PNG, or WEBP.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        results = state["screen"].query_image(data, top_k=top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(results=[MovieMatch(**r) for r in results])


@app.post("/query/video", response_model=QueryResponse, tags=["search"])
async def query_video(
    file:  UploadFile = File(..., description="MP4 / MKV / AVI / MOV / WEBM clip."),
    fps:   int = 1,
    top_k: int = 5,
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_VIDEO:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported video type '{ext}'. Supported: {', '.join(SUPPORTED_VIDEO)}.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        results, n_frames = state["screen"].query_video(
            data, video_ext=ext, fps=fps, top_k=top_k
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        results=[MovieMatch(**r) for r in results],
        frames_used=n_frames,
    )


# ── Legacy route — keeps old clients working ──────────────────────────────────
@app.post("/query", response_model=QueryResponse, tags=["search"], deprecated=True)
async def query_legacy(
    file:  UploadFile = File(...),
    top_k: int = 5,
):
    """Deprecated: use /query/image or /query/video instead."""
    ext = Path(file.filename or "").suffix.lower()
    if ext in SUPPORTED_VIDEO:
        return await query_video(file=file, fps=1, top_k=top_k)
    return await query_image(file=file, top_k=top_k)