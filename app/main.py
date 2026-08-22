import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import router as api_router


def setup_logging(data_dir: Path):
    """Configure structured logging to console and rotating log file."""
    data_dir.mkdir(parents=True, exist_ok=True)
    log_file = data_dir / "moments_server.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if re-initialized
    if not any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)


def create_app() -> FastAPI:
    """Application factory for FastAPI."""
    settings = get_settings()
    data_dir = Path(settings.DATA_DIR).resolve()
    setup_logging(data_dir)

    app = FastAPI(
        title="Local AI Moments Generator",
        description="Offline, hardware-accelerated moments video generation on Apple Silicon using SigLIP 2.",
        version="0.1.0",
        debug=settings.DEBUG,
    )

    # Enable CORS for local UI and development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routes
    app.include_router(api_router, prefix="/api/v1")

    # Static UI mounting
    ui_dir = Path(__file__).parent / "ui"
    if ui_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root_redirect():
        """Redirect root to UI dashboard."""
        return RedirectResponse(url="/ui/")

    return app


app = create_app()
