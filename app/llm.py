"""Gemini LLM integration — prompts, JSON parsing, and repair."""
from __future__ import annotations

import json
import re

import google.generativeai as genai

from app.models import QUESTION_SET, AnalyzeResponse, GenerationError, LLMParseError


def _get_model(api_key: str, model_name: str) -> genai.GenerativeModel:
    if not api_key:
        raise GenerationError("GEMINI_API_KEY is missing")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def _extract_text(response: object) -> str:
    text = getattr(response, "text", None)
    if text:
        return text.strip()
    candidates = getattr(response, "candidates", None)
    if candidates:
        try:
            return candidates[0].content.parts[0].text.strip()
        except Exception:  # noqa: BLE001
            pass
    return ""


def _extract_json(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _call_with_repair(model: genai.GenerativeModel, prompt: str,
                      repair_prompt: str, parser):
    """Generate → parse → on failure, repair once and re-parse."""
    raw = _extract_text(model.generate_content(prompt))
    try:
        return parser(raw)
    except LLMParseError:
        repaired = _extract_text(model.generate_content(
            f"Fix this to valid strict JSON only. Keep the same meaning and schema.\n\n"
            f"{repair_prompt}\n\n{raw}"
        ))
        return parser(repaired)

# ── Prompts ──────────────────────────────────────────────────────────────────

def _analysis_prompt(url: str, context: str) -> str:
    return f"""
You are a senior marketing strategist. Analyze website content for URL: {url}

Context:
{context}

Return strict JSON only with this exact schema:
{{
  "summary": "string",
  "issues": [
    {{"problem": "string", "fix": "string"}},
    {{"problem": "string", "fix": "string"}}
  ]
}}

Rules:
- Return only JSON, no markdown.
- issues must contain at least 2 items.
- Keep summary concise and factual.
""".strip()


def _questions_prompt(url: str, context: str) -> str:
    questions = "\n".join(f"- {k}: {v}" for k, v in QUESTION_SET.items())
    return f"""
You are analyzing this website: {url}

Using only the context below, answer each question in 1-3 sentences.

Questions:
{questions}

Return strict JSON only as:
{{
  "business_value": "...",
  "audience": "...",
  "cta": "...",
  "trust_gaps": "..."
}}

Context:
{context}
""".strip()

# ── Parsers ──────────────────────────────────────────────────────────────────

def _parse_analysis(payload: str) -> AnalyzeResponse:
    candidate = _extract_json(payload)
    if not candidate:
        raise LLMParseError("LLM returned empty response text for analysis")
    try:
        parsed = AnalyzeResponse.model_validate(json.loads(candidate))
    except Exception as exc:  # noqa: BLE001
        raise LLMParseError(f"Invalid analysis JSON: {exc}. Preview: {payload[:240]}") from exc
    if len(parsed.issues) < 2:
        raise LLMParseError("Model returned fewer than 2 issues")
    return parsed


def _parse_questions(payload: str) -> dict[str, str]:
    candidate = _extract_json(payload)
    if not candidate:
        raise LLMParseError("LLM returned empty response text for Q&A")
    try:
        raw = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMParseError(f"Invalid Q&A JSON: {exc}. Preview: {payload[:240]}") from exc
    return {k: str(raw.get(k, "")).strip() for k in QUESTION_SET}

# ── Public API ───────────────────────────────────────────────────────────────

_ANALYSIS_REPAIR_HINT = ""
_QUESTIONS_REPAIR_HINT = (
    'Use schema: {"business_value":"...","audience":"...","cta":"...","trust_gaps":"..."}'
)


def generate_analysis(api_key: str, model_name: str, url: str, context: str) -> AnalyzeResponse:
    model = _get_model(api_key, model_name)
    return _call_with_repair(model, _analysis_prompt(url, context), _ANALYSIS_REPAIR_HINT, _parse_analysis)


def answer_questions(api_key: str, model_name: str, url: str, context: str) -> dict[str, str]:
    model = _get_model(api_key, model_name)
    return _call_with_repair(model, _questions_prompt(url, context), _QUESTIONS_REPAIR_HINT, _parse_questions)
