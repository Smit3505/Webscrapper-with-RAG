from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")

    app_host: str = Field(default="127.0.0.1", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    crawl_max_pages: int = Field(default=3, alias="CRAWL_MAX_PAGES")
    crawl_max_depth: int = Field(default=1, alias="CRAWL_MAX_DEPTH")
    crawl_timeout_ms: int = Field(default=20000, alias="CRAWL_TIMEOUT_MS")

    retrieval_top_k: int = Field(default=8, alias="RETRIEVAL_TOP_K")
    retrieval_final_context_chunks: int = Field(default=8, alias="RETRIEVAL_FINAL_CONTEXT_CHUNKS")

    chunk_min_section_chars: int = Field(default=220, alias="CHUNK_MIN_SECTION_CHARS")
    chunk_max_tokens: int = Field(default=220, alias="CHUNK_MAX_TOKENS")
    chunk_overlap_tokens: int = Field(default=30, alias="CHUNK_OVERLAP_TOKENS")

    min_clean_chars: int = Field(default=350, alias="MIN_CLEAN_CHARS")
    min_paragraphs: int = Field(default=2, alias="MIN_PARAGRAPHS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
