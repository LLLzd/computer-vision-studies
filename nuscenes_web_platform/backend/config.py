"""Runtime configuration and startup validation for the NuScenes web platform."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    nuscenes_dataroot: Path
    nuscenes_version: str = "v1.0-mini"
    nuscenes_web_cache_dir: Path | None = None
    nuscenes_web_host: str = "127.0.0.1"
    nuscenes_web_port: int = 8765

    @field_validator("nuscenes_dataroot", mode="before")
    @classmethod
    def _coerce_path(cls, v):
        return Path(v).expanduser().resolve() if v else v

    @field_validator("nuscenes_web_cache_dir", mode="before")
    @classmethod
    def _coerce_cache(cls, v):
        if v is None or v == "":
            return None
        return Path(v).expanduser().resolve()

    def validate_dataset_layout(self) -> None:
        root = self.nuscenes_dataroot
        if not root.is_dir():
            raise FileNotFoundError(f"NUSCENES_DATAROOT is not a directory: {root}")
        meta = root / self.nuscenes_version
        if not meta.is_dir():
            raise FileNotFoundError(
                f"Missing meta folder {self.nuscenes_version!r} under {root}. "
                "Check NUSCENES_VERSION matches your download."
            )
        samples = root / "samples"
        if not samples.is_dir():
            raise FileNotFoundError(f"Missing samples/ under {root}")


def get_settings() -> Settings:
    return Settings()
