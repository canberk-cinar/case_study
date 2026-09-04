import logging
import logging.config
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]  # src/config.py -> case_study/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    LOG_LEVEL: str = "INFO"
    DATABASE_URL: str = "sqlite:///./case_study.db"

    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    REPORTS_DIR: str = "data/reports"

    INGEST_CHUNK_SIZE: int = 100_000
    DTYPE_SCAN_CHUNK_SIZE: int = 50_000
    COLUMN_BATCH_SIZE: int = 25
    PARQUET_COMPRESSION: str = "zstd"

    # Feeds Case 8's RAGContainer default config (its `openrouter_*` keys) — the LLM provider
    # Case 9's policy_explanation agent ultimately calls through. The brief's original ask for
    # Case 8/9 was local-only (Ollama); the case study team approved OpenRouter's free tier
    # instead once this machine's limited RAM ruled out running a local model well. Real
    # model name and API key belong in .env, never hardcoded here.
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = ""

    @property
    def raw_data_path(self) -> Path:
        return REPO_ROOT / self.RAW_DATA_DIR

    @property
    def processed_data_path(self) -> Path:
        return REPO_ROOT / self.PROCESSED_DATA_DIR

    @property
    def reports_path(self) -> Path:
        return REPO_ROOT / self.REPORTS_DIR


def configure_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "formatter": "default"},
        },
        "root": {"level": log_level, "handlers": ["console"]},
    })


settings = Settings()