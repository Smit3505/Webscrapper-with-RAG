from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from app.config import get_settings
from app.models import (
    AnalyzeRequest, AnalyzeResponse, CrawlError, EmptyContentError,
    GenerationError, HealthResponse, LLMParseError,
)
from app.retrieval import get_embedding_model
from app.service import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Website Marketing Analyzer", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    model_loaded = False
    try:
        get_embedding_model()
        model_loaded = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embedding model not loaded: %s", exc)

    status = "ok" if settings.gemini_api_key and model_loaded else "degraded"
    return HealthResponse(
        status=status,
        gemini_key_configured=bool(settings.gemini_api_key),
        embedding_model_loaded=model_loaded,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    settings = get_settings()
    try:
        analysis, _, _ = await run_pipeline(str(payload.url), settings)
        return analysis
    except CrawlError as exc:
        raise HTTPException(status_code=502, detail=f"crawl_failure: {exc}") from exc
    except EmptyContentError as exc:
        raise HTTPException(status_code=422, detail=f"empty_content: {exc}") from exc
    except LLMParseError as exc:
        raise HTTPException(status_code=502, detail=f"llm_parse_failure: {exc}") from exc
    except GenerationError as exc:
        raise HTTPException(status_code=500, detail=f"llm_generation_failure: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"unexpected_error: {exc}") from exc
