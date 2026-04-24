"""Playwright crawler with marketing-intent link prioritization."""
from __future__ import annotations

import logging
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import Browser, Page, async_playwright

from app.models import PRIORITY_LINK_KEYWORDS, CrawledPage

logger = logging.getLogger(__name__)


def _infer_page_type(url: str) -> str:
    low = url.lower()
    if low.endswith("/") or low.count("/") <= 3:
        return "homepage"
    for key in PRIORITY_LINK_KEYWORDS:
        if key in low:
            return key
    return "other"


def _extract_priority_links(base_url: str, html: str) -> list[str]:
    base_host = urlparse(base_url).netloc.lower()
    scored: dict[str, int] = {}

    for anchor in BeautifulSoup(html, "html.parser").find_all("a"):
        href = anchor.get("href")
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue

        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        if parsed.netloc.lower() != base_host:
            continue

        clean_url = parsed._replace(fragment="", query="").geturl().rstrip("/")
        text_and_path = f"{anchor.get_text(' ', strip=True)} {parsed.path}".lower()
        score = sum(2 for kw in PRIORITY_LINK_KEYWORDS if kw in text_and_path)
        scored[clean_url] = max(score, scored.get(clean_url, 0))

    return [url for url, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)]


async def _load_page(page: Page, url: str, timeout_ms: int) -> tuple[str, str]:
    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    return (await page.title()) or "Untitled", await page.content()


async def crawl_site(url: str, max_pages: int, max_depth: int, timeout_ms: int) -> list[CrawledPage]:
    if max_depth != 1:
        logger.warning("Only depth=1 is currently supported; received %s", max_depth)

    pages: list[CrawledPage] = []

    async with async_playwright() as pw:
        browser: Browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        try:
            home_title, home_html = await _load_page(page, url, timeout_ms)
            home_url = page.url.rstrip("/")
            pages.append(CrawledPage(
                url=home_url, title=home_title, html=home_html,
                page_type="homepage", success=True,
            ))

            for candidate in _extract_priority_links(home_url, home_html):
                if candidate == home_url or len(pages) >= max_pages:
                    continue
                try:
                    title, html = await _load_page(page, candidate, timeout_ms)
                    pages.append(CrawledPage(
                        url=page.url.rstrip("/"), title=title, html=html,
                        page_type=_infer_page_type(candidate), success=True,
                    ))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to crawl %s: %s", candidate, exc)
                    pages.append(CrawledPage(
                        url=candidate, title="", html="",
                        page_type=_infer_page_type(candidate),
                        success=False, failure_reason=str(exc),
                    ))

            return pages[:max_pages]
        finally:
            await context.close()
            await browser.close()
