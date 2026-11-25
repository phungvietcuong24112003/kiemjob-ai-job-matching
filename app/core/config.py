from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "ai-cv-job-matching"
    API_V1_STR: str = "/api/v1"
    ALGORITHM_VERSION: str = "v1-nlp-tfidf"
    ENVIRONMENT: str = "development"

    INTERNAL_SECRET_KEY: str = "kiemjob_secret_key_change_in_prod"
    CORS_ORIGINS: List[str] = ["*"]

    MAX_CV_CHARS: int = 20000
    MAX_JOB_CHARS: int = 10000

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
