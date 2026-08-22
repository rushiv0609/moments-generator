"""
Qdrant Vector Database Integration Layer for Local AI Moments Generator.

Provides high-performance indexing, storage, temporal range filtering,
and Top-K cosine similarity retrieval of 768-dim multimodal embeddings.
Supports hybrid deployment: Remote Docker, Embedded Local Disk, or In-Memory.
"""

import uuid
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType

from app.config import get_settings, Settings

logger = logging.getLogger(__name__)


@dataclass
class VectorPoint:
    """
    A single embedding point with structured media metadata payload.
    Supports dual-granularity indexing: individual frames ('frame') and scene summaries ('scene').
    """
    vector: Union[np.ndarray, List[float]]
    file_path: str
    file_id: Optional[int] = None
    file_type: str = "image"  # 'image' | 'video'
    frame_index: int = 0      # 0 for photos, 0..N for video clips
    source_offset: float = 0.0  # Seconds into video (0.0 for photos)
    creation_timestamp: Optional[float] = None
    duration_seconds: Optional[float] = None
    id: Optional[str] = None  # UUID string; generated automatically if None
    granularity: str = "frame"  # 'frame' | 'scene'
    scene_id: Optional[int] = None        # Scene index within video
    scene_start: Optional[float] = None   # Scene start timestamp in seconds
    scene_end: Optional[float] = None     # Scene end timestamp in seconds
    scene_frame_count: Optional[int] = None  # Number of frames in this scene
    is_scene_representative: bool = False  # True for the mean summary vector of a scene
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if isinstance(self.vector, np.ndarray):
            self.vector = self.vector.astype(np.float32).tolist()

    def to_payload(self) -> Dict[str, Any]:
        """Convert metadata fields into Qdrant JSON payload dictionary."""
        payload = {
            "file_path": self.file_path,
            "file_id": self.file_id,
            "file_type": self.file_type,
            "frame_index": self.frame_index,
            "source_offset": self.source_offset,
            "creation_timestamp": self.creation_timestamp,
            "duration_seconds": self.duration_seconds,
            "granularity": self.granularity,
            "scene_id": self.scene_id,
            "scene_start": self.scene_start,
            "scene_end": self.scene_end,
            "scene_frame_count": self.scene_frame_count,
            "is_scene_representative": self.is_scene_representative,
        }
        if self.extra:
            payload.update(self.extra)
        return payload

    def to_point_struct(self) -> models.PointStruct:
        """Convert to Qdrant PointStruct for upsert."""
        return models.PointStruct(
            id=self.id,
            vector=self.vector,
            payload=self.to_payload(),
        )


@dataclass
class SearchResult:
    """
    A single ranked result returned from semantic vector retrieval.
    """
    point_id: str
    score: float  # Cosine similarity score (typically -0.15 to +0.25 in SigLIP 2)
    file_path: str
    file_id: Optional[int] = None
    file_type: str = "image"
    frame_index: int = 0
    source_offset: float = 0.0
    creation_timestamp: Optional[float] = None
    duration_seconds: Optional[float] = None
    granularity: str = "frame"  # 'frame' | 'scene'
    scene_id: Optional[int] = None
    scene_start: Optional[float] = None
    scene_end: Optional[float] = None
    is_scene_representative: bool = False
    payload: Dict[str, Any] = field(default_factory=dict)


class QdrantVectorDB:
    """
    Manages vector collections, batch upserting, point deletion,
    and temporal range-filtered cosine similarity search.
    """

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        storage_path: Optional[Union[str, Path]] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        in_memory: bool = False,
    ):
        if client is not None:
            self.client = client
            self.mode = "custom"
        elif in_memory:
            self.client = QdrantClient(":memory:")
            self.mode = "in_memory"
        elif host and port:
            self.client = QdrantClient(host=host, port=port, timeout=3.0)
            self.mode = "remote"
        elif storage_path:
            storage_dir = Path(storage_path)
            storage_dir.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(storage_dir.resolve()))
            self.mode = "embedded"
        else:
            self.client = QdrantClient(":memory:")
            self.mode = "in_memory"

    def close(self) -> None:
        """Close the underlying client and release storage file lock."""
        if hasattr(self, "client") and self.client is not None:
            if hasattr(self.client, "close"):
                try:
                    self.client.close()
                except Exception as e:
                    logger.debug("Notice closing Qdrant client: %s", e)


    @classmethod
    def create(
        cls,
        settings: Optional[Settings] = None,
        in_memory: bool = False,
        prefer_remote: bool = True,
    ) -> "QdrantVectorDB":
        """
        Factory method to initialize Qdrant client based on system environment.
        Attempts remote Docker container first; if unavailable, falls back
        cleanly to local embedded storage without failing.
        """
        if in_memory:
            return cls(in_memory=True)

        if settings is None:
            settings = get_settings()

        if prefer_remote and settings.QDRANT_HOST and settings.QDRANT_PORT:
            try:
                # Test connectivity to remote Docker container
                client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                    timeout=1.5,
                )
                client.get_collections()
                logger.info(
                    "Connected to remote Qdrant server at %s:%d",
                    settings.QDRANT_HOST,
                    settings.QDRANT_PORT,
                )
                return cls(client=client)
            except Exception as e:
                logger.warning(
                    "Remote Qdrant unavailable (%s). Falling back to embedded local storage at %s/qdrant_storage",
                    e,
                    settings.DATA_DIR,
                )

        # Embedded local disk storage
        storage_dir = Path(settings.DATA_DIR) / "qdrant_storage"
        return cls(storage_path=storage_dir)

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists in the database."""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            logger.error("Error checking collection existence: %s", e)
            return False

    def ensure_collection(
        self,
        collection_name: str,
        vector_size: int = 768,
        distance: Distance = Distance.COSINE,
    ) -> bool:
        """
        Idempotently ensure a vector collection exists with the specified dimension.
        Creates payload indexes on key metadata fields.
        """
        try:
            if not self.collection_exists(collection_name):
                logger.info(
                    "Creating Qdrant collection '%s' (dim=%d, distance=%s)",
                    collection_name,
                    vector_size,
                    distance.name,
                )
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=distance),
                )
                self.create_payload_indices(collection_name)
            return True
        except Exception as e:
            logger.error("Failed to ensure collection '%s': %s", collection_name, e)
            raise

    def create_payload_indices(self, collection_name: str) -> None:
        """Create indexes on payload fields for high-performance filtering."""
        fields_to_index = [
            ("file_path", PayloadSchemaType.KEYWORD),
            ("file_type", PayloadSchemaType.KEYWORD),
            ("creation_timestamp", PayloadSchemaType.FLOAT),
            ("file_id", PayloadSchemaType.INTEGER),
            ("granularity", PayloadSchemaType.KEYWORD),
            ("scene_id", PayloadSchemaType.INTEGER),
            ("is_scene_representative", PayloadSchemaType.KEYWORD),
        ]
        for field_name, schema_type in fields_to_index:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception as e:
                # Local in-memory/embedded Qdrant may emit a warning that indexes are ignored
                logger.debug("Notice when creating payload index for %s: %s", field_name, e)

    def delete_collection(self, collection_name: str) -> bool:
        """Delete an entire collection (e.g. during model upgrade re-indexing)."""
        try:
            if self.collection_exists(collection_name):
                self.client.delete_collection(collection_name=collection_name)
                logger.info("Deleted collection '%s'", collection_name)
            return True
        except Exception as e:
            logger.error("Failed to delete collection '%s': %s", collection_name, e)
            raise

    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve collection configuration and point counts."""
        try:
            if not self.collection_exists(collection_name):
                return None
            info = self.client.get_collection(collection_name=collection_name)
            return {
                "name": collection_name,
                "points_count": info.points_count,
                "vectors_count": getattr(info, "vectors_count", info.points_count),
                "status": info.status.value if hasattr(info.status, "value") else str(info.status),
            }
        except Exception as e:
            logger.error("Failed to get collection info for '%s': %s", collection_name, e)
            return None

    def count(self, collection_name: str) -> int:
        """Return the total number of points in a collection."""
        try:
            if not self.collection_exists(collection_name):
                return 0
            res = self.client.count(collection_name=collection_name, exact=True)
            return res.count
        except Exception as e:
            logger.error("Failed to count points in collection '%s': %s", collection_name, e)
            return 0

    def upsert_points(
        self,
        collection_name: str,
        points: List[VectorPoint],
        batch_size: int = 100,
    ) -> List[str]:
        """
        Batch upsert vector points into Qdrant.
        Returns the list of point UUIDs indexed.
        """
        if not points:
            return []

        vec_dim = len(points[0].vector)
        self.ensure_collection(collection_name, vector_size=vec_dim)
        point_ids: List[str] = []

        # Process in batches to manage memory and network payload size
        for i in range(0, len(points), batch_size):
            chunk = points[i : i + batch_size]
            struct_chunk = [p.to_point_struct() for p in chunk]
            self.client.upsert(
                collection_name=collection_name,
                points=struct_chunk,
                wait=True,
            )
            point_ids.extend([p.id for p in chunk])

        logger.info(
            "Upserted %d points into collection '%s'",
            len(points),
            collection_name,
        )
        return point_ids

    def search(
        self,
        collection_name: str,
        query_vector: Union[np.ndarray, List[float]],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None,
        file_type: Optional[str] = None,
        file_path: Optional[str] = None,
        granularity: Optional[str] = None,
        scene_id: Optional[int] = None,
        is_scene_representative: Optional[bool] = None,
        must_not_file_paths: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Execute cosine similarity search with optional metadata filters (e.g. time range, granularity).
        Returns structured SearchResult objects ordered by score descending.
        """
        if not self.collection_exists(collection_name):
            return []

        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.astype(np.float32).tolist()

        # Build Filter conditions
        must_conditions = []
        must_not_conditions = []

        # Temporal range filter (e.g. for timeline bucketing)
        if start_timestamp is not None or end_timestamp is not None:
            range_kwargs: Dict[str, float] = {}
            if start_timestamp is not None:
                range_kwargs["gte"] = float(start_timestamp)
            if end_timestamp is not None:
                range_kwargs["lte"] = float(end_timestamp)

            must_conditions.append(
                models.FieldCondition(
                    key="creation_timestamp",
                    range=models.Range(**range_kwargs),
                )
            )

        # File type filter ('image' vs 'video')
        if file_type is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value=file_type),
                )
            )

        # Specific file path filter
        if file_path is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="file_path",
                    match=models.MatchValue(value=file_path),
                )
            )

        # Granularity filter ('frame' vs 'scene')
        if granularity is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="granularity",
                    match=models.MatchValue(value=granularity),
                )
            )

        # Specific scene filter
        if scene_id is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="scene_id",
                    match=models.MatchValue(value=scene_id),
                )
            )

        # Scene representative flag filter
        if is_scene_representative is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="is_scene_representative",
                    match=models.MatchValue(value=is_scene_representative),
                )
            )

        # Exclusion of specific file paths (e.g. for diverse time bucket curation)
        if must_not_file_paths:
            for p in must_not_file_paths:
                must_not_conditions.append(
                    models.FieldCondition(
                        key="file_path",
                        match=models.MatchValue(value=p),
                    )
                )

        query_filter = None
        if must_conditions or must_not_conditions:
            query_filter = models.Filter(
                must=must_conditions if must_conditions else None,
                must_not=must_not_conditions if must_not_conditions else None,
            )

        # Perform search using query_points API
        res = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
        )

        results: List[SearchResult] = []
        for p in res.points:
            payload = p.payload or {}
            results.append(
                SearchResult(
                    point_id=str(p.id),
                    score=float(p.score),
                    file_path=payload.get("file_path", ""),
                    file_id=payload.get("file_id"),
                    file_type=payload.get("file_type", "image"),
                    frame_index=payload.get("frame_index", 0),
                    source_offset=payload.get("source_offset", 0.0),
                    creation_timestamp=payload.get("creation_timestamp"),
                    duration_seconds=payload.get("duration_seconds"),
                    granularity=payload.get("granularity", "frame"),
                    scene_id=payload.get("scene_id"),
                    scene_start=payload.get("scene_start"),
                    scene_end=payload.get("scene_end"),
                    is_scene_representative=payload.get("is_scene_representative", False),
                    payload=payload,
                )
            )

        return results

    def get_point(self, collection_name: str, point_id: str) -> Optional[Tuple[SearchResult, List[float]]]:
        """
        Retrieve a single point with its raw vector and metadata payload.
        """
        if not self.collection_exists(collection_name):
            return None

        try:
            records = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_vectors=True,
                with_payload=True,
            )
            if not records:
                return None
            rec = records[0]
            payload = rec.payload or {}
            res = SearchResult(
                point_id=str(rec.id),
                score=1.0,
                file_path=payload.get("file_path", ""),
                file_id=payload.get("file_id"),
                file_type=payload.get("file_type", "image"),
                frame_index=payload.get("frame_index", 0),
                source_offset=payload.get("source_offset", 0.0),
                creation_timestamp=payload.get("creation_timestamp"),
                duration_seconds=payload.get("duration_seconds"),
                granularity=payload.get("granularity", "frame"),
                scene_id=payload.get("scene_id"),
                scene_start=payload.get("scene_start"),
                scene_end=payload.get("scene_end"),
                is_scene_representative=payload.get("is_scene_representative", False),
                payload=payload,
            )
            vec = rec.vector if isinstance(rec.vector, list) else list(rec.vector)
            return res, vec
        except Exception as e:
            logger.error("Failed to retrieve point %s: %s", point_id, e)
            return None

    def get_points_by_file(self, collection_name: str, file_path: str) -> List[SearchResult]:
        """
        Retrieve all points (frames and scenes) indexed for a specific file path.
        """
        if not self.collection_exists(collection_name):
            return []

        try:
            records, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchValue(value=file_path),
                        )
                    ]
                ),
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            results: List[SearchResult] = []
            for rec in records:
                payload = rec.payload or {}
                results.append(
                    SearchResult(
                        point_id=str(rec.id),
                        score=1.0,
                        file_path=payload.get("file_path", ""),
                        file_id=payload.get("file_id"),
                        file_type=payload.get("file_type", "image"),
                        frame_index=payload.get("frame_index", 0),
                        source_offset=payload.get("source_offset", 0.0),
                        creation_timestamp=payload.get("creation_timestamp"),
                        duration_seconds=payload.get("duration_seconds"),
                        granularity=payload.get("granularity", "frame"),
                        scene_id=payload.get("scene_id"),
                        scene_start=payload.get("scene_start"),
                        scene_end=payload.get("scene_end"),
                        is_scene_representative=payload.get("is_scene_representative", False),
                        payload=payload,
                    )
                )
            # Sort chronologically by source_offset, then by granularity (scenes first)
            results.sort(key=lambda x: (x.source_offset, 0 if x.granularity == "scene" else 1, x.frame_index))
            return results
        except Exception as e:
            logger.error("Failed to retrieve points for %s: %s", file_path, e)
            return []

    def delete_points(self, collection_name: str, point_ids: List[str]) -> bool:
        """Delete specific vector points by ID."""
        if not point_ids or not self.collection_exists(collection_name):
            return True

        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=point_ids),
                wait=True,
            )
            return True
        except Exception as e:
            logger.error("Failed to delete points from '%s': %s", collection_name, e)
            raise

    def delete_by_file_path(self, collection_name: str, file_path: str) -> bool:
        """Delete all points associated with a given file path."""
        if not self.collection_exists(collection_name):
            return True

        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_path",
                                match=models.MatchValue(value=file_path),
                            )
                        ]
                    )
                ),
                wait=True,
            )
            return True
        except Exception as e:
            logger.error("Failed to delete points by file_path '%s': %s", file_path, e)
            raise

