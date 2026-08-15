"""
Centralized Configuration for Local AI Moments Generator.
Loads configuration from environment variables with MOMENTS_ prefix or .env file.
"""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MOMENTS_",
        extra="ignore",
    )

    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = False

    # Model Configuration
    MODEL_NAME: str = "google/siglip2-base-patch16-224"
    MODEL_BACKEND: Literal["auto", "mlx", "pytorch_mps"] = "auto"
    MODEL_PRECISION: Literal["fp16", "8bit"] = "fp16"
    EMBEDDING_RESOLUTION: int = 224  # Model input resolution (NOT output video resolution)
    EMBED_BATCH_SIZE: int = 64  # Benchmark-proven saturation sweet spot

    # Ingestion Pipeline
    EXTRACT_WORKERS: int = 12  # Benchmark-proven optimal for Apple ImageIO decode
    INDEX_BATCH_SIZE: int = 100
    FILE_QUEUE_SIZE: int = 64
    FRAME_QUEUE_SIZE: int = 256
    VECTOR_QUEUE_SIZE: int = 512

    # Curation Parameters
    MIN_SIMILARITY_THRESHOLD: float = 0.05  # Lower threshold for high recall; Qdrant Top-K ranking handles precision
    MAX_OUTPUT_DURATION: int = 300  # 5 minutes
    DEFAULT_ASPECT_RATIO: str = "1:1"  # "1:1" | "16:9" | "9:16"
    IMAGE_DISPLAY_DURATION: float = 3.0  # seconds per photo
    VIDEO_SEGMENT_DURATION: float = 3.0  # seconds per video clip
    DEDUP_COSINE_THRESHOLD: float = 0.05
    DEDUP_PHASH_THRESHOLD: int = 5

    # Qdrant Vector DB
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "media_embeddings"

    # Working Directories
    DATA_DIR: str = "./data"
    EXPORTS_DIR: str = "./exports"
    MODELS_DIR: str = "./models"

    # Rendering (Apple Silicon VideoToolbox Hardware Accelerated)
    VIDEO_CODEC: str = "h264_videotoolbox"
    VIDEO_BITRATE: str = "6000k"
    VIDEO_FPS: int = 30


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
