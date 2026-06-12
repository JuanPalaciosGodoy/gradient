from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_path: str = "gradient.db"
    max_csv_rows: int = 100_000
    max_file_size_mb: int = 50
    evaluator_mode: str = "heuristic"  # heuristic | prometheus | llm_judge


settings = Settings()
