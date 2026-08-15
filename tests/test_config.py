"""
Unit tests for configuration module.
"""

import os
from unittest.mock import patch
from app.config import Settings, get_settings


def test_default_settings():
    """Verify benchmark-proven defaults are set correctly."""
    settings = Settings()
    assert settings.MODEL_NAME == "google/siglip2-base-patch16-224"
    assert settings.MODEL_BACKEND == "auto"
    assert settings.MODEL_PRECISION == "fp16"
    assert settings.EMBED_BATCH_SIZE == 64
    assert settings.EXTRACT_WORKERS == 12
    assert settings.DEFAULT_ASPECT_RATIO == "1:1"
    assert settings.VIDEO_CODEC == "h264_videotoolbox"


def test_env_override():
    """Verify MOMENTS_ prefix environment overrides."""
    with patch.dict(os.environ, {
        "MOMENTS_EXTRACT_WORKERS": "8",
        "MOMENTS_MODEL_PRECISION": "8bit",
        "MOMENTS_EMBED_BATCH_SIZE": "32",
    }):
        settings = Settings()
        assert settings.EXTRACT_WORKERS == 8
        assert settings.MODEL_PRECISION == "8bit"
        assert settings.EMBED_BATCH_SIZE == 32


def test_get_settings_singleton():
    """Verify get_settings returns cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
