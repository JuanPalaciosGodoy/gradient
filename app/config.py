from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Storage
    database_path: str = "gradient.db"
    max_csv_rows: int = 100_000
    max_file_size_mb: int = 50

    # Evaluation
    evaluator_mode: str = "heuristic"  # heuristic | exact_match | task_router | llm_judge

    # Real provider mode — set REAL_PROVIDER_MODE=true + at least one API key to enable
    real_provider_mode: bool = False
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""

    # LLM judge configuration (used when evaluator_mode=llm_judge)
    llm_judge_model: str = "gpt-4o-mini"
    llm_judge_provider: str = "openai"


settings = Settings()
