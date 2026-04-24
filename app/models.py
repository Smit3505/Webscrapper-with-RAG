"""Unified models, constants, and error types."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

# ── Exceptions ───────────────────────────────────────────────────────────────

class AppError(Exception):
    """Base application error."""

class CrawlError(AppError): ...
class EmptyContentError(AppError): ...
class LLMParseError(AppError): ...
class GenerationError(AppError): ...

# ── Constants ────────────────────────────────────────────────────────────────

PRIORITY_LINK_KEYWORDS = [
    "about", "services", "service", "product", "products",
    "pricing", "features", "contact",
]

CTA_KEYWORDS = [
    "contact", "book", "schedule", "demo", "start",
    "signup", "sign up", "talk to sales", "get started", "call",
]

VALUE_PROP_KEYWORDS = [
    "we help", "our mission", "value", "solution",
    "platform", "for businesses",
]

QUESTION_SET = {
    "business_value": "What does this business do and what is its value proposition?",
    "audience": "Who is the likely target audience and what pain points are addressed?",
    "cta": "What are the primary calls-to-action on the site?",
    "trust_gaps": "What messaging or conversion gaps are visible and how can they be improved?",
}

# ── Pydantic Models ──────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    url: HttpUrl

class Issue(BaseModel):
    problem: str
    fix: str

class AnalyzeResponse(BaseModel):
    summary: str
    issues: list[Issue] = Field(min_length=2)

class ChunkMetadata(BaseModel):
    url: str
    page_type: str
    title: str
    heading_path: str
    chunk_index: int

class ContentChunk(BaseModel):
    text: str
    metadata: ChunkMetadata

class CrawledPage(BaseModel):
    url: str
    title: str
    html: str
    page_type: str
    success: bool = True
    failure_reason: str | None = None

class CleanedPage(BaseModel):
    url: str
    title: str
    page_type: str
    clean_markdown: str
    char_count: int
    paragraph_count: int

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    gemini_key_configured: bool
    embedding_model_loaded: bool
