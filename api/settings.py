from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    data_dir: Path
    history_dir: Path
    documents_dir: Path
    model_path: Path
    stats_workbook: Path
    database_path: Path
    providers_state_path: Path
    frontend_dist: Path
    jwt_secret: str
    jwt_access_expire_min: int
    jwt_refresh_expire_days: int
    admin_username: str
    admin_password: str
    frontend_origin: str
    similarity_threshold: float
    llm_timeout: int
    max_upload_mb: int
    app_port: int

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(os.getenv("AIMPACT_ROOT", Path.cwd())).resolve()
        data_dir = _path(root, os.getenv("DATA_DIR", "data"))
        return cls(
            root_dir=root,
            data_dir=data_dir,
            history_dir=_path(root, os.getenv("HISTORY_DIR", "history")),
            documents_dir=_path(root, os.getenv("DOCUMENTS_DIR", "documents")),
            model_path=_path(
                root,
                os.getenv("MODEL_PATH", "models/multilingual-e5-small-onnx"),
            ),
            stats_workbook=_path(
                root,
                os.getenv("STATS_WORKBOOK", "project_agent/Phu luc 1.xlsx"),
            ),
            database_path=_path(root, os.getenv("DATABASE_PATH", "data/app.db")),
            providers_state_path=_path(
                root,
                os.getenv("PROVIDERS_STATE_PATH", "data/providers.state.json"),
            ),
            frontend_dist=_path(root, os.getenv("FRONTEND_DIST", "web/dist")),
            jwt_secret=os.getenv("JWT_SECRET", ""),
            jwt_access_expire_min=int(os.getenv("JWT_ACCESS_EXPIRE_MIN", "15")),
            jwt_refresh_expire_days=int(os.getenv("JWT_REFRESH_EXPIRE_DAYS", "7")),
            admin_username=os.getenv("ADMIN_USERNAME", ""),
            admin_password=os.getenv("ADMIN_PASSWORD", ""),
            frontend_origin=os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.84")),
            llm_timeout=int(os.getenv("LLM_TIMEOUT", "60")),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "50")),
            app_port=int(os.getenv("APP_PORT", "8000")),
        )

    def ensure_directories(self) -> None:
        for directory in (self.data_dir, self.history_dir, self.documents_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        errors = []
        if len(self.jwt_secret) < 32:
            errors.append("JWT_SECRET must contain at least 32 characters")
        if not self.model_path.exists():
            errors.append(f"MODEL_PATH does not exist: {self.model_path}")
        if not self.stats_workbook.is_file():
            errors.append(f"STATS_WORKBOOK does not exist: {self.stats_workbook}")
        if not 0 <= self.similarity_threshold <= 1:
            errors.append("SIMILARITY_THRESHOLD must be between 0 and 1")
        if self.jwt_access_expire_min < 1 or self.jwt_refresh_expire_days < 1:
            errors.append("JWT token lifetimes must be positive")
        if errors:
            raise RuntimeError("Invalid API configuration: " + "; ".join(errors))


def get_settings() -> Settings:
    return Settings.from_env()
