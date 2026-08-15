"""Shared test fixtures."""
import pytest
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests and clean up afterwards."""
    d = tempfile.mkdtemp(prefix="moments_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)
