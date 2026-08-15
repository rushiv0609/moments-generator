"""
Main FastAPI Application for Local AI Moments Generator.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import router as api_router


def create_app() -> FastAPI:
    """Application factory for FastAPI."""
    settings = get_settings()

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
