"""
Project Workspace Manager for Local AI Moments Generator.

Implements the Desktop Project Bundle Architecture:
All manifest databases, embedded Qdrant vector storages, intermediate frame caches,
and exported videos are contained 100% inside the user-selected workspace directory.
"""

import json
import time
import shutil
import logging
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Union

from app.db.manifest import ManifestDB
from app.db.qdrant import QdrantVectorDB
from app.db.models import FileStatus
from app.config import get_settings

logger = logging.getLogger(__name__)


def open_native_finder_picker(
    prompt: str = "Select a Folder",
    default_path: Optional[str] = None,
) -> Optional[str]:
    """
    Open native macOS Finder folder selection dialog using AppleScript.
    Allows user to browse, select, or create folders in Finder.
    Returns POSIX path string or None if cancelled.
    """
    try:
        default_clause = ""
        if default_path and Path(default_path).exists():
            default_clause = f'default location POSIX file "{default_path}"'

        script = f'''
        tell application "System Events"
            activate
            set chosenFolder to choose folder with prompt "{prompt}" {default_clause}
            return POSIX path of chosenFolder
        end tell
        '''
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            selected = result.stdout.strip()
            # Remove trailing slash if present
            return str(Path(selected).resolve())
        return None
    except Exception as e:
        logger.warning("Native Finder picker failed or cancelled: %s", e)
        return None


@dataclass
class WorkspaceInfo:
    """
    Metadata and statistics for the active Project Workspace.
    """
    workspace_dir: str
    corpus_dir: Optional[str]
    manifest_db_path: str
    qdrant_storage_path: str
    exports_dir: str
    cache_dir: str
    created_at: float
    updated_at: float
    total_files: int = 0
    indexed_files: int = 0
    total_vectors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkspaceManager:
    """
    Manages active project workspace lifecycle, storage isolation,
    and access to project-scoped databases.
    """

    def __init__(self, workspace_path: Optional[Union[str, Path]] = None):
        self._workspace_path: Optional[Path] = None
        self._corpus_path: Optional[Path] = None
        self._manifest_db: Optional[ManifestDB] = None
        self._qdrant_db: Optional[QdrantVectorDB] = None

        if workspace_path:
            self.set_workspace(workspace_path)

    @property
    def is_active(self) -> bool:
        """Check if a project workspace is currently set."""
        return self._workspace_path is not None and self._workspace_path.exists()

    @property
    def workspace_path(self) -> Optional[Path]:
        return self._workspace_path

    @property
    def corpus_path(self) -> Optional[Path]:
        return self._corpus_path

    def set_workspace(
        self,
        workspace_path: Union[str, Path],
        corpus_path: Optional[Union[str, Path]] = None,
    ) -> WorkspaceInfo:
        """
        Activate or create a project workspace directory.
        Creates visible manifest.db, qdrant_storage/, cache/, and exports/ at workspace root.
        """
        w_path = Path(workspace_path).expanduser().resolve()
        w_path.mkdir(parents=True, exist_ok=True)

        exports_dir = w_path / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)

        cache_dir = w_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        qdrant_storage = w_path / "qdrant_storage"
        qdrant_storage.mkdir(parents=True, exist_ok=True)

        is_same_workspace = self._workspace_path == w_path and self._qdrant_db is not None

        if not is_same_workspace:
            # If switching to a new workspace, close existing Qdrant client to release directory lock
            if self._qdrant_db is not None:
                self._qdrant_db.close()
                self._qdrant_db = None

            self._workspace_path = w_path
            # Initialize project-scoped ManifestDB
            self._manifest_db = ManifestDB.for_workspace(w_path)
            # Initialize project-scoped embedded Qdrant DB
            self._qdrant_db = QdrantVectorDB(storage_path=qdrant_storage)
            self._qdrant_db.ensure_collection("media_embeddings", vector_size=768)

        if corpus_path:
            self._corpus_path = Path(corpus_path).expanduser().resolve()

        # Save / update project workspace metadata
        meta_file = w_path / "workspace_meta.json"
        now = time.time()
        meta: Dict[str, Any] = {
            "workspace_dir": str(w_path),
            "corpus_dir": str(self._corpus_path) if self._corpus_path else None,
            "created_at": now,
            "updated_at": now,
        }
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    existing = json.load(f)
                meta["created_at"] = existing.get("created_at", now)
            except Exception:
                pass

        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            "Activated Project Workspace at %s (Corpus: %s)",
            w_path,
            self._corpus_path,
        )
        return self.get_workspace_info()

    def get_manifest_db(self) -> ManifestDB:
        """Get the active project's ManifestDB instance."""
        if not self._manifest_db or not self._workspace_path:
            raise RuntimeError("No active project workspace set. Call set_workspace() first.")
        return self._manifest_db

    def get_qdrant_db(self) -> QdrantVectorDB:
        """Get the active project's QdrantVectorDB instance."""
        if not self._qdrant_db or not self._workspace_path:
            raise RuntimeError("No active project workspace set. Call set_workspace() first.")
        return self._qdrant_db

    def get_exports_dir(self) -> Path:
        """Get the active project's exports directory."""
        if not self._workspace_path:
            raise RuntimeError("No active project workspace set.")
        exports = self._workspace_path / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        return exports

    def get_cache_dir(self) -> Path:
        """Get the active project's intermediate cache directory."""
        if not self._workspace_path:
            raise RuntimeError("No active project workspace set.")
        cache = self._workspace_path / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        return cache

    def get_workspace_info(self) -> WorkspaceInfo:
        """Retrieve full details, path mapping, and live stats for active workspace."""
        if not self._workspace_path or not self._workspace_path.exists():
            raise RuntimeError("No active project workspace.")

        db_path = self._workspace_path / "manifest.db"
        qdrant_path = self._workspace_path / "qdrant_storage"
        exports_path = self._workspace_path / "exports"
        cache_path = self._workspace_path / "cache"

        created_at = time.time()
        updated_at = created_at
        meta_file = self._workspace_path / "workspace_meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                created_at = meta.get("created_at", created_at)
                updated_at = meta.get("updated_at", updated_at)
                if not self._corpus_path and meta.get("corpus_dir"):
                    self._corpus_path = Path(meta["corpus_dir"])
            except Exception:
                pass

        total_files = 0
        indexed_files = 0
        total_vectors = 0

        if self._manifest_db:
            try:
                all_records = self._manifest_db.get_all_files()
                total_files = len(all_records)
                indexed_files = sum(1 for r in all_records if r.status == FileStatus.INDEXED)
            except Exception as e:
                logger.debug("Could not query manifest stats: %s", e)

        if self._qdrant_db:
            try:
                total_vectors = self._qdrant_db.count("media_embeddings")
            except Exception as e:
                logger.debug("Could not query vector stats: %s", e)

        return WorkspaceInfo(
            workspace_dir=str(self._workspace_path),
            corpus_dir=str(self._corpus_path) if self._corpus_path else None,
            manifest_db_path=str(db_path),
            qdrant_storage_path=str(qdrant_path),
            exports_dir=str(exports_path),
            cache_dir=str(cache_path),
            created_at=created_at,
            updated_at=updated_at,
            total_files=total_files,
            indexed_files=indexed_files,
            total_vectors=total_vectors,
        )


# Global singleton instance
_workspace_manager: Optional[WorkspaceManager] = None


def get_workspace_manager() -> WorkspaceManager:
    """Retrieve or initialize the global WorkspaceManager instance."""
    global _workspace_manager
    if _workspace_manager is None:
        _workspace_manager = WorkspaceManager()
        # Default workspace under ~/Moments_Projects/Default (outside repository source tree)
        default_dir = Path.home() / "Moments_Projects" / "Default"
        _workspace_manager.set_workspace(default_dir)
    return _workspace_manager
