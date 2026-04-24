from __future__ import annotations

import re

import trafilatura

from app.models import CleanedPage, CrawledPage


def _normalize_noise(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def clean_pages(
    pages: list[CrawledPage],
    min_clean_chars: int,
    min_paragraphs: int,
) -> list[CleanedPage]:
    cleaned: list[CleanedPage] = []
    for page in pages:
        if not page.success or not page.html:
            continue

        extracted = trafilatura.extract(
            page.html,
            output_format="markdown",
            include_comments=False,
            include_tables=False,
            favor_recall=True,
        )
        if not extracted:
            continue

        normalized = _normalize_noise(extracted)
        paragraphs = [p for p in normalized.split("\n\n") if p.strip()]

        if len(normalized) < min_clean_chars:
            continue
        if len(paragraphs) < min_paragraphs:
            continue

        cleaned.append(
            CleanedPage(
                url=page.url,
                title=page.title,
                page_type=page.page_type,
                clean_markdown=normalized,
                char_count=len(normalized),
                paragraph_count=len(paragraphs),
            )
        )

    return cleaned
