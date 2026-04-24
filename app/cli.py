from __future__ import annotations

import argparse
import json
import sys

from app.config import get_settings
from app.models import QUESTION_SET, AppError
from app.service import run_pipeline_sync


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Website marketing analyzer CLI")
    parser.add_argument("url", type=str, help="Website URL to analyze")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    try:
        analysis, answers, artifacts = run_pipeline_sync(args.url, settings)
    except AppError as exc:
        print(
            json.dumps({
                "error": str(exc),
                "hint": "Check logs above. If crawl timed out, increase CRAWL_TIMEOUT_MS in .env.",
            }),
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps({
                "error": f"unexpected: {exc}",
                "hint": "Re-run with the same URL and share the logs for diagnosis.",
            }),
            file=sys.stderr,
        )
        return 1

    print(analysis.model_dump_json(indent=2))
    print("\n--- Marketing Question Answers ---")
    for key, question in QUESTION_SET.items():
        print(f"\nQ: {question}")
        print(f"A: {answers.get(key, '')}")

    print("\n--- Diagnostics ---")
    print(f"Pages crawled: {artifacts.pages_crawled}")
    print(f"Chunks indexed: {artifacts.chunks_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
