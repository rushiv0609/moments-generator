"""
Ollama Service Auto-Manager for Local AI Moments Generator.

Automatically ensures that the local Ollama daemon is running without requiring
the user to manually start it in a separate terminal.
"""

import os
import time
import shutil
import logging
import urllib.request
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_OLLAMA_PROCESS: Optional[subprocess.Popen] = None
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def find_ollama_binary() -> Optional[str]:
    """Find the Ollama executable on the host system."""
    # 1. Check standard PATH
    path = shutil.which("ollama")
    if path and Path(path).exists():
        return path

    # 2. Check common macOS and Linux install locations
    candidate_paths = [
        "/opt/homebrew/bin/ollama",
        "/usr/local/bin/ollama",
        os.path.expanduser("~/.local/bin/ollama"),
        "/usr/bin/ollama",
        "/Applications/Ollama.app/Contents/Resources/ollama",
    ]
    for cand in candidate_paths:
        if Path(cand).exists():
            return cand

    return None


def is_ollama_healthy(base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 1.0) -> bool:
    """Check if the Ollama API is responding with HTTP 200."""
    try:
        url = f"{base_url.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "MomentsGenerator/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def ensure_ollama_running(
    base_url: str = DEFAULT_OLLAMA_URL,
    log_dir: Optional[Path] = None,
    max_wait_seconds: float = 6.0,
) -> bool:
    """
    Ensure the Ollama server is running.
    If it is not running, launches the `ollama serve` subprocess and waits for readiness.
    Returns True if healthy/started, False if binary is not installed or failed to start.
    """
    global _OLLAMA_PROCESS

    if is_ollama_healthy(base_url):
        return True

    binary_path = find_ollama_binary()
    if not binary_path:
        logger.warning(
            "Ollama binary not found in PATH or standard locations. "
            "Install Ollama via 'brew install ollama' or visit https://ollama.com"
        )
        return False

    logger.info("Ollama is not responding. Auto-starting Ollama daemon via %s...", binary_path)

    # Determine log destination
    log_file = None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / "ollama_service.log", "a", encoding="utf-8")
    else:
        log_file = subprocess.DEVNULL

    try:
        env = os.environ.copy()
        env["OLLAMA_ORIGINS"] = "*"

        _OLLAMA_PROCESS = subprocess.Popen(
            [binary_path, "serve"],
            stdout=log_file,
            stderr=log_file,
            env=env,
            start_new_session=True,  # Detach process group
        )
        logger.info("Spawned Ollama daemon (PID: %d)", _OLLAMA_PROCESS.pid)

        # Poll for readiness
        start_time = time.time()
        while time.time() - start_time < max_wait_seconds:
            time.sleep(0.3)
            if is_ollama_healthy(base_url):
                logger.info(
                    "Ollama daemon started and healthy in %.2fs (PID: %d)",
                    time.time() - start_time,
                    _OLLAMA_PROCESS.pid,
                )
                return True
            if _OLLAMA_PROCESS.poll() is not None:
                logger.error("Ollama process exited prematurely with code %d", _OLLAMA_PROCESS.returncode)
                return False

        logger.warning("Ollama daemon did not respond within %.1fs", max_wait_seconds)
        return is_ollama_healthy(base_url)

    except Exception as e:
        logger.error("Failed to auto-start Ollama daemon: %s", e)
        return False


def stop_ollama() -> None:
    """Gracefully terminate any Ollama process spawned by this application."""
    global _OLLAMA_PROCESS
    if _OLLAMA_PROCESS is not None:
        try:
            logger.info("Stopping managed Ollama daemon (PID: %d)...", _OLLAMA_PROCESS.pid)
            _OLLAMA_PROCESS.terminate()
            try:
                _OLLAMA_PROCESS.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                _OLLAMA_PROCESS.kill()
            _OLLAMA_PROCESS = None
        except Exception as e:
            logger.warning("Error stopping Ollama process: %s", e)
