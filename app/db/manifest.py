"""
SQLite Manifest System for Local AI Moments Generator.
Provides ACID transactions, WAL mode concurrency, two-tier content hashing,
and crash-safe checkpointing for media ingestion and curation.
"""

import json
import time
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
import xxhash

from app.db.models import FileRecord, TimelineSegmentRecord, FileStatus, VALID_TRANSITIONS


def compute_fast_hash(file_size: int, mtime: float) -> str:
    """Fast change-detection hash: xxhash64(file_size + mtime)."""
    payload = f"{file_size}_{mtime}".encode("utf-8")
    return xxhash.xxh64_hexdigest(payload)


def compute_content_hash(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    """Content deduplication hash: xxhash64 of entire file stream (~10 GB/s)."""
    hasher = xxhash.xxh64()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


class ManifestDB:
    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path).resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection configured with WAL mode and row factory."""
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        
        # Ensure schema tables exist even if manifest.db was deleted/cleared on disk
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='files'")
        if not cursor.fetchone():
            self._init_schema_on_conn(conn)

        return conn

    def _init_schema(self):
        """Initialize manifest database tables and indices."""
        with sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False) as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            self._init_schema_on_conn(conn)

    def _init_schema_on_conn(self, conn: sqlite3.Connection):
        """Execute table creation DDL on a specific connection."""
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS manifest_meta (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS files (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path           TEXT    UNIQUE NOT NULL,
                file_hash           TEXT    NOT NULL,
                content_hash        TEXT,
                file_size           INTEGER NOT NULL,
                file_type           TEXT    NOT NULL,
                mime_type           TEXT,
                creation_timestamp  REAL,
                timestamp_source    TEXT,
                duration_seconds    REAL,
                status              TEXT    NOT NULL DEFAULT 'PENDING',
                error_message       TEXT,
                frame_count         INTEGER,
                model_name          TEXT,
                model_version       TEXT,
                qdrant_point_ids    TEXT,
                embedded_at         REAL,
                scanned_at          REAL,
                updated_at          REAL
            );

            CREATE INDEX IF NOT EXISTS idx_files_status       ON files(status);
            CREATE INDEX IF NOT EXISTS idx_files_hash         ON files(file_hash);
            CREATE INDEX IF NOT EXISTS idx_files_content_hash ON files(content_hash);
            CREATE INDEX IF NOT EXISTS idx_files_type         ON files(file_type);
            CREATE INDEX IF NOT EXISTS idx_files_ts           ON files(creation_timestamp);

            CREATE TABLE IF NOT EXISTS timeline_segments (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id           TEXT    NOT NULL,
                position         INTEGER NOT NULL,
                file_id          INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                segment_type     TEXT    NOT NULL,
                start_offset     REAL,
                duration         REAL    NOT NULL,
                similarity_score REAL,
                time_bucket      INTEGER,
                UNIQUE(job_id, position)
            );

            CREATE INDEX IF NOT EXISTS idx_timeline_job ON timeline_segments(job_id);
        """)


    @classmethod
    def for_workspace(cls, workspace_dir: Union[str, Path]) -> "ManifestDB":
        """
        Open or initialize a ManifestDB instance directly inside a Project Workspace.
        Stored at <workspace_dir>/manifest.db.
        """
        w_path = Path(workspace_dir).resolve()
        w_path.mkdir(parents=True, exist_ok=True)
        db_file = w_path / "manifest.db"
        return cls(str(db_file))

    @classmethod
    def open_or_create(cls, corpus_path: str, data_dir: str = "./data") -> "ManifestDB":
        """
        Open or initialize a ManifestDB instance keyed to a canonical corpus path.
        Database is stored under <data_dir>/manifests/<corpus_hash>.db.
        """
        canonical_path = str(Path(corpus_path).resolve())
        corpus_hash = xxhash.xxh64_hexdigest(canonical_path.encode("utf-8"))
        manifests_dir = Path(data_dir) / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        db_file = manifests_dir / f"manifest_{corpus_hash}.db"
        
        manifest = cls(str(db_file))
        manifest.set_meta("corpus_path", canonical_path)
        return manifest

    # =========================================================================
    # Metadata CRUD
    # =========================================================================

    def get_meta(self, key: str) -> Optional[str]:
        """Fetch metadata value by key."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT value FROM manifest_meta WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row["value"] if row else None

    def set_meta(self, key: str, value: str):
        """Set metadata key-value pair."""
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO manifest_meta (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at;
                """,
                (key, value, now),
            )

    # =========================================================================
    # File Record CRUD
    # =========================================================================

    def _row_to_record(self, row: sqlite3.Row) -> FileRecord:
        """Convert SQLite Row to FileRecord dataclass."""
        point_ids = []
        if row["qdrant_point_ids"]:
            try:
                point_ids = json.loads(row["qdrant_point_ids"])
            except Exception:
                point_ids = []

        return FileRecord(
            id=row["id"],
            file_path=row["file_path"],
            file_hash=row["file_hash"],
            content_hash=row["content_hash"],
            file_size=row["file_size"],
            file_type=row["file_type"],
            mime_type=row["mime_type"],
            creation_timestamp=row["creation_timestamp"],
            timestamp_source=row["timestamp_source"],
            duration_seconds=row["duration_seconds"],
            status=FileStatus(row["status"]),
            error_message=row["error_message"],
            frame_count=row["frame_count"],
            model_name=row["model_name"],
            model_version=row["model_version"],
            qdrant_point_ids=point_ids,
            embedded_at=row["embedded_at"],
            scanned_at=row["scanned_at"],
            updated_at=row["updated_at"],
        )

    def lookup(self, file_path: str) -> Optional[FileRecord]:
        """Lookup a file by its canonical file path."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM files WHERE file_path = ?", (file_path,))
            row = cursor.fetchone()
            return self._row_to_record(row) if row else None

    def lookup_by_id(self, file_id: int) -> Optional[FileRecord]:
        """Lookup a file by its database ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,))
            row = cursor.fetchone()
            return self._row_to_record(row) if row else None

    def lookup_by_content_hash(self, content_hash: str) -> Optional[FileRecord]:
        """Lookup an existing file record with an identical content hash (for dedup/rename)."""
        if not content_hash:
            return None
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM files WHERE content_hash = ? ORDER BY id ASC LIMIT 1", (content_hash,))
            row = cursor.fetchone()
            return self._row_to_record(row) if row else None

    def upsert_file(
        self,
        file_path: str,
        file_hash: str,
        file_size: int,
        file_type: str,
        content_hash: Optional[str] = None,
        mime_type: Optional[str] = None,
        creation_timestamp: Optional[float] = None,
        timestamp_source: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        status: FileStatus = FileStatus.PENDING,
        frame_count: Optional[int] = None,
    ) -> FileRecord:
        """Insert or update a media file record in the manifest."""
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO files (
                    file_path, file_hash, content_hash, file_size, file_type,
                    mime_type, creation_timestamp, timestamp_source, duration_seconds,
                    status, frame_count, scanned_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    file_hash = excluded.file_hash,
                    content_hash = COALESCE(excluded.content_hash, files.content_hash),
                    file_size = excluded.file_size,
                    file_type = excluded.file_type,
                    mime_type = COALESCE(excluded.mime_type, files.mime_type),
                    creation_timestamp = COALESCE(excluded.creation_timestamp, files.creation_timestamp),
                    timestamp_source = COALESCE(excluded.timestamp_source, files.timestamp_source),
                    duration_seconds = COALESCE(excluded.duration_seconds, files.duration_seconds),
                    status = excluded.status,
                    frame_count = COALESCE(excluded.frame_count, files.frame_count),
                    updated_at = excluded.updated_at
                RETURNING *;
                """,
                (
                    file_path, file_hash, content_hash, file_size, file_type,
                    mime_type, creation_timestamp, timestamp_source, duration_seconds,
                    status.value, frame_count, now, now,
                ),
            )
            row = cursor.fetchone()
            return self._row_to_record(row)

    def update_status(self, file_id: int, status: FileStatus, error_message: Optional[str] = None):
        """Transition file to a new status with error message handling."""
        now = time.time()
        with self._get_connection() as conn:
            # Check current status
            curr = conn.execute("SELECT status FROM files WHERE id = ?", (file_id,)).fetchone()
            if not curr:
                raise ValueError(f"File ID {file_id} not found")
            
            conn.execute(
                """
                UPDATE files
                SET status = ?, error_message = ?, updated_at = ?
                WHERE id = ?;
                """,
                (status.value, error_message, now, file_id),
            )

    def set_error(self, file_id: int, message: str):
        """Mark a file as ERROR with a detailed reason."""
        self.update_status(file_id, FileStatus.ERROR, error_message=message)

    def update_embedding_info(
        self,
        file_id: int,
        qdrant_point_ids: List[str],
        model_name: str,
        model_version: str = "1.0",
    ):
        """Mark file as INDEXED and store Qdrant point IDs."""
        now = time.time()
        points_json = json.dumps(qdrant_point_ids)
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE files
                SET status = 'INDEXED',
                    model_name = ?,
                    model_version = ?,
                    qdrant_point_ids = ?,
                    embedded_at = ?,
                    updated_at = ?
                WHERE id = ?;
                """,
                (model_name, model_version, points_json, now, now, file_id),
            )

    def get_files_by_status(self, status: FileStatus) -> List[FileRecord]:
        """Fetch all files currently in a given status."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM files WHERE status = ? ORDER BY id ASC", (status.value,))
            return [self._row_to_record(r) for r in cursor.fetchall()]

    def get_all_files(self) -> List[FileRecord]:
        """Fetch all files in the manifest."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM files ORDER BY id ASC")
            return [self._row_to_record(r) for r in cursor.fetchall()]

    def reset_embeddings(self) -> int:
        """
        Reset all INDEXED and EMBEDDED files back to SCANNED (used when model version changes).
        Preserves all file metadata, timestamps, and durations.
        Returns the number of affected rows.
        """
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE files
                SET status = 'SCANNED',
                    model_name = NULL,
                    model_version = NULL,
                    qdrant_point_ids = NULL,
                    embedded_at = NULL,
                    updated_at = ?
                WHERE status IN ('INDEXED', 'EMBEDDED');
                """,
                (now,),
            )
            return cursor.rowcount

    def remove_deleted_files(self, active_file_paths: List[str]) -> List[Tuple[int, List[str]]]:
        """
        Identify and remove files from the manifest that no longer exist on disk.
        Returns a list of (file_id, qdrant_point_ids) for cleaning up external vector indices.
        """
        deleted_records = []
        with self._get_connection() as conn:
            # Fetch all known files
            rows = conn.execute("SELECT id, file_path, qdrant_point_ids FROM files").fetchall()
            active_set = set(active_file_paths)
            ids_to_delete = []

            for r in rows:
                if r["file_path"] not in active_set:
                    ids_to_delete.append(r["id"])
                    point_ids = []
                    if r["qdrant_point_ids"]:
                        try:
                            point_ids = json.loads(r["qdrant_point_ids"])
                        except Exception:
                            pass
                    deleted_records.append((r["id"], point_ids))

            if ids_to_delete:
                placeholders = ",".join("?" * len(ids_to_delete))
                conn.execute(f"DELETE FROM files WHERE id IN ({placeholders})", ids_to_delete)

        return deleted_records

    def get_stats(self) -> Dict[str, Any]:
        """Calculate aggregate manifest statistics."""
        with self._get_connection() as conn:
            total_files = conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"]
            total_size = conn.execute("SELECT COALESCE(SUM(file_size), 0) as s FROM files").fetchone()["s"]
            
            status_counts = {}
            for row in conn.execute("SELECT status, COUNT(*) as c FROM files GROUP BY status"):
                status_counts[row["status"]] = row["c"]

            type_counts = {}
            for row in conn.execute("SELECT file_type, COUNT(*) as c FROM files GROUP BY file_type"):
                type_counts[row["file_type"]] = row["c"]

            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "status_counts": status_counts,
                "type_counts": type_counts,
            }

    # =========================================================================
    # Timeline Segments Persistence
    # =========================================================================

    def save_timeline(self, job_id: str, segments: List[TimelineSegmentRecord]):
        """Save or replace a curated timeline for a generation job."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM timeline_segments WHERE job_id = ?", (job_id,))
            conn.executemany(
                """
                INSERT INTO timeline_segments (
                    job_id, position, file_id, segment_type,
                    start_offset, duration, similarity_score, time_bucket
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    (
                        s.job_id, s.position, s.file_id, s.segment_type,
                        s.start_offset, s.duration, s.similarity_score, s.time_bucket,
                    )
                    for s in segments
                ],
            )

    def load_timeline(self, job_id: str) -> List[TimelineSegmentRecord]:
        """Load curated timeline segments for a job ordered by position."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM timeline_segments WHERE job_id = ? ORDER BY position ASC",
                (job_id,),
            )
            return [
                TimelineSegmentRecord(
                    id=row["id"],
                    job_id=row["job_id"],
                    position=row["position"],
                    file_id=row["file_id"],
                    segment_type=row["segment_type"],
                    start_offset=row["start_offset"],
                    duration=row["duration"],
                    similarity_score=row["similarity_score"],
                    time_bucket=row["time_bucket"],
                )
                for row in cursor.fetchall()
            ]
