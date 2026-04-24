"""Heading-aware markdown chunking with merge and overlap."""
from __future__ import annotations

from app.models import CleanedPage, ContentChunk, ChunkMetadata


def _split_sections(markdown: str) -> list[tuple[list[str], str]]:
    """Split markdown into (heading_path, body) pairs."""
    sections: list[tuple[list[str], str]] = []
    heading_stack: list[str] = ["Overview"]
    buffer: list[str] = []

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            body = "\n".join(buffer).strip()
            if body:
                sections.append((heading_stack.copy(), body))
            buffer = []
            level = len(stripped) - len(stripped.lstrip("#"))
            heading_stack = heading_stack[: max(1, level - 1)]
            heading_stack.append(stripped[level:].strip() or "Section")
        else:
            buffer.append(line)

    body = "\n".join(buffer).strip()
    if body:
        sections.append((heading_stack.copy(), body))
    return sections


def _window_split(text: str, max_tokens: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_tokens:
        return [text]
    windows, start = [], 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        windows.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(0, end - overlap)
    return windows


def chunk_pages(
    pages: list[CleanedPage], min_section_chars: int,
    max_tokens: int, overlap_tokens: int,
) -> list[ContentChunk]:
    chunks: list[ContentChunk] = []

    for page in pages:
        sections = _split_sections(page.clean_markdown)

        # Merge tiny sections into their predecessor
        merged: list[tuple[list[str], str]] = []
        for heading_path, body in sections:
            if merged and len(body) < min_section_chars:
                prev_h, prev_b = merged[-1]
                merged[-1] = (prev_h, f"{prev_b}\n\n{body}".strip())
            else:
                merged.append((heading_path, body))

        idx = 0
        for heading_path, body in merged:
            hp = " > ".join(heading_path)
            for part in _window_split(body, max_tokens, overlap_tokens):
                chunks.append(ContentChunk(
                    text=f"Heading Path: {hp}\n\n{part}".strip(),
                    metadata=ChunkMetadata(
                        url=page.url, page_type=page.page_type,
                        title=page.title, heading_path=hp, chunk_index=idx,
                    ),
                ))
                idx += 1

    return chunks
