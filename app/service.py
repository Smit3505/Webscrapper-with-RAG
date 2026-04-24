"""Orchestration pipeline — crawl → clean → chunk → embed → retrieve → LLM."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.cleaner import clean_pages
from app.chunker import chunk_pages
from app.config import Settings
from app.crawler import crawl_site
from app.llm import answer_questions, generate_analysis
from app.models import CrawlError, EmptyContentError, AnalyzeResponse
from app.retrieval import build_vector_store, retrieve_contexts

logger = logging.getLogger(__name__)


@dataclass
class AnalysisArtifacts:
    pages_crawled: int
    chunks_total: int
    context_preview: str


async def run_pipeline(
    url: str, settings: Settings,
) -> tuple[AnalyzeResponse, dict[str, str], AnalysisArtifacts]:

    # ── Crawl ────────────────────────────────────────────────────────────
    logger.info("crawl:start url=%s", url)
    try:
        pages = await crawl_site(
            url=url, max_pages=settings.crawl_max_pages,
            max_depth=settings.crawl_max_depth, timeout_ms=settings.crawl_timeout_ms,
        )
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)
        if "Timeout" in detail or "timed out" in detail.lower():
            raise CrawlError(
                f"Timed out loading {url} after {settings.crawl_timeout_ms}ms. "
                "Try increasing CRAWL_TIMEOUT_MS and/or retry with the site's canonical URL."
            ) from exc
        raise CrawlError(f"Crawler failed for {url}: {detail}") from exc

    successes = [p for p in pages if p.success]
    failures = [p for p in pages if not p.success]
    logger.info("crawl:done successes=%d failures=%d", len(successes), len(failures))
    if not successes:
        reasons = "; ".join(p.failure_reason or "unknown" for p in failures[:2])
        raise CrawlError(f"Unable to crawl any pages successfully. Sample failures: {reasons}")

    # ── Clean ────────────────────────────────────────────────────────────
    cleaned = clean_pages(successes, settings.min_clean_chars, settings.min_paragraphs)
    logger.info("clean:done kept=%d", len(cleaned))
    if not cleaned:
        raise EmptyContentError("No pages had enough clean content after extraction")

    # ── Chunk ────────────────────────────────────────────────────────────
    chunks = chunk_pages(cleaned, settings.chunk_min_section_chars,
                         settings.chunk_max_tokens, settings.chunk_overlap_tokens)
    logger.info("chunk:done chunks=%d", len(chunks))
    if not chunks:
        raise EmptyContentError("No chunks were produced from extracted content")

    # ── Embed + Retrieve ─────────────────────────────────────────────────
    store = build_vector_store(chunks)
    contexts = retrieve_contexts(store, settings.retrieval_top_k, settings.retrieval_final_context_chunks)
    logger.info("retrieve:done contexts=%d", len(contexts))

    # ── Assemble context text ────────────────────────────────────────────
    lines: list[str] = []
    for ctx in contexts:
        lines.append(f"\n## Question: {ctx.question}")
        for chunk in ctx.chunks:
            md = chunk.metadata
            lines.append(
                f"- Source: {md.url} | title={md.title} | type={md.page_type} | heading={md.heading_path}\n"
                f"{chunk.text}"
            )
    context_text = "\n".join(lines).strip()

    # ── LLM ──────────────────────────────────────────────────────────────
    logger.info("llm:start")
    analysis = generate_analysis(settings.gemini_api_key, settings.gemini_model, url, context_text)
    qa_answers = answer_questions(settings.gemini_api_key, settings.gemini_model, url, context_text)
    logger.info("llm:done")

    artifacts = AnalysisArtifacts(
        pages_crawled=len(successes), chunks_total=len(chunks),
        context_preview=context_text[:2000],
    )
    return analysis, qa_answers, artifacts


def run_pipeline_sync(
    url: str, settings: Settings,
) -> tuple[AnalyzeResponse, dict[str, str], AnalysisArtifacts]:
    return asyncio.run(run_pipeline(url, settings))
