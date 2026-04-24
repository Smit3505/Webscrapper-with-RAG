# Website Marketing Analyzer MVP

A FastAPI + CLI app that analyzes one website URL end-to-end:

- Crawl homepage + up to 2 internal pages (same-domain only, depth 1) with Playwright.
- Extract main content with Trafilatura and skip low-content pages.
- Build heading-aware chunks with metadata.
- Embed with all-MiniLM-L6-v2 and index with FAISS inner-product (cosine via normalization).
- Retrieve per marketing intent with lightweight reranking and dedupe.
- Generate strict JSON via Gemini 2.5 Flash.

## Output Schema

API returns strict JSON only:

```json
{
  "summary": "string",
  "issues": [
    {"problem": "string", "fix": "string"},
    {"problem": "string", "fix": "string"}
  ]
}
```

## Project Structure

- `app/main.py`: FastAPI app (`/health`, `/analyze`)
- `app/cli.py`: CLI entrypoint
- `app/service.py`: shared orchestration pipeline
- `app/crawler.py`: Playwright crawler + link prioritization
- `app/cleaner.py`: Trafilatura extraction + quality thresholds
- `app/chunker.py`: heading-aware chunking
- `app/embeddings.py`: MiniLM embeddings + FAISS index
- `app/retrieval.py`: top-k retrieval + boosts + dedupe
- `app/llm.py`: Gemini prompting + JSON validation/repair
- `app/schemas.py`: request/response and chunk metadata models

## Setup

1. Create and activate a virtual environment in python 3.10.

```powershell
python3.10 -m venv venv
```
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Install Playwright browser binaries:

```powershell
playwright install chromium
```

4. Configure environment variables:

Create .env file.

Set `GEMINI_API_KEY` in .env

## Run API

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

Analyze:

```powershell
curl -X POST http://127.0.0.1:8000/analyze -H "Content-Type: application/json" -d '{"url":"https://example.com"}'
```

## Run CLI

```powershell
python -m app.cli https://example.com
```

CLI prints strict JSON first, then separate answers to four marketing questions.

## Architecture Notes

- Crawl limits are hard bounded by `CRAWL_MAX_PAGES=3` and `CRAWL_MAX_DEPTH=1`.
- Internal links are ranked by marketing intent terms: about/services/product/pricing/features/contact.
- Section chunk metadata fields are: `url`, `page_type`, `title`, `heading_path`, `chunk_index`.
- Retrieval applies intent boosts (homepage/value proposition, CTA terms, pricing/contact/service terms).
- LLM output is parsed as strict JSON and has one repair attempt on malformed JSON.

## Troubleshooting

- `crawl_failure`: target site blocked/slow or browser render timeout. Increase `CRAWL_TIMEOUT_MS`.
- `empty_content`: extraction produced too little content; lower `MIN_CLEAN_CHARS` or `MIN_PARAGRAPHS`.
- `llm_generation_failure`: missing or invalid `GEMINI_API_KEY`.
- `llm_parse_failure`: model returned invalid JSON even after one repair pass.
