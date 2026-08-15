"""
High-Throughput Multithreaded Ingestion Pipeline for Local AI Moments Generator.

Orchestrates:
1. Manifest Discovery / Unindexed File Selection
2. Multithreaded Apple ImageIO / VideoToolbox Media Decoding (12 workers)
3. GPU Batch SigLIP 2 Embedding (MLX / MPS)
4. Atomic Qdrant Vector DB Upsert & SQLite Manifest Checkpointing
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple, Union

import numpy as np

from app.db.manifest import ManifestDB
from app.db.qdrant import QdrantVectorDB, VectorPoint
from app.db.models import FileRecord, FileStatus
from app.core.embedder import EmbedderInterface, create_embedder
from app.core.extractor import decode_image, extract_video_frames, FrameData
from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class PipelineProgress:
    """
    Live progress telemetry for the active ingestion pipeline.
    """
    stage: str  # 'SCANNING' | 'EXTRACTING' | 'EMBEDDING' | 'INDEXING' | 'COMPLETED' | 'FAILED'
    processed_count: int
    total_count: int
    percentage: float
    throughput_fps: float
    current_file: Optional[str] = None
    message: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "processed_count": self.processed_count,
            "total_count": self.total_count,
            "percentage": round(self.percentage, 1),
            "throughput_fps": round(self.throughput_fps, 1),
            "current_file": self.current_file,
            "message": self.message,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


@dataclass
class PipelineSummary:
    """
    Final results summary of an ingestion pipeline run.
    """
    total_files: int
    indexed_files: int
    error_files: int
    total_vectors: int
    elapsed_seconds: float
    average_throughput: float
    details: Dict[str, Any] = field(default_factory=dict)


class IngestionPipeline:
    """
    Orchestrates end-to-end media decoding, batch GPU embedding,
    and vector DB indexing with progress streaming.
    """

    def __init__(
        self,
        manifest: ManifestDB,
        qdrant: QdrantVectorDB,
        embedder: Optional[EmbedderInterface] = None,
        max_decode_workers: int = 12,
        batch_size: int = 32,
    ):
        self.manifest = manifest
        self.qdrant = qdrant
        self.embedder = embedder or create_embedder()
        self.max_decode_workers = max_decode_workers
        self.batch_size = batch_size
        self._cancelled = False

    def cancel(self):
        """Signal pipeline cancellation."""
        self._cancelled = True

    def _decode_file(self, record: FileRecord) -> Tuple[FileRecord, List[FrameData], Optional[str]]:
        """
        Worker function: decodes a single photo or video file into 224x224 FrameData chunks.
        Runs inside the ThreadPoolExecutor.
        """
        try:
            p = Path(record.file_path)
            if not p.exists():
                return record, [], f"File not found on disk: {record.file_path}"

            frames: List[FrameData] = []
            if record.file_type == "video":
                # Extract at 1 FPS with hardware acceleration
                for f in extract_video_frames(str(p), target_size=224, sampling_fps=1.0):
                    frames.append(f)
            else:
                # Photo decode via Apple ImageIO / Pillow
                frame = decode_image(str(p), target_size=224)
                if frame is not None:
                    frames.append(frame)
                else:
                    return record, [], "Failed to decode image file format."

            return record, frames, None
        except Exception as e:
            return record, [], str(e)

    def run(
        self,
        corpus_path: Optional[str] = None,
        force_reindex: bool = False,
        progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
    ) -> PipelineSummary:
        """
        Execute the full ingestion and embedding pipeline.
        """
        self._cancelled = False
        start_time = time.time()

        def emit_progress(stage: str, processed: int, total: int, current_file: Optional[str] = None, message: str = ""):
            elapsed = time.time() - start_time
            pct = (processed / total * 100.0) if total > 0 else 100.0
            fps = (processed / elapsed) if elapsed > 0 else 0.0
            prog = PipelineProgress(
                stage=stage,
                processed_count=processed,
                total_count=total,
                percentage=pct,
                throughput_fps=fps,
                current_file=current_file,
                message=message,
                elapsed_seconds=elapsed,
            )
            if progress_callback:
                try:
                    progress_callback(prog)
                except Exception as e:
                    logger.debug("Progress callback error: %s", e)

        # Stage 1: Checkpoint / File Discovery
        emit_progress("SCANNING", 0, 0, message="Inspecting manifest and corpus files...")

        all_records = self.manifest.get_all_files()
        if force_reindex:
            files_to_process = all_records
            logger.info("Force reindex enabled: reprocessing all %d files", len(files_to_process))
        else:
            files_to_process = [r for r in all_records if r.status != FileStatus.INDEXED]

        total_files = len(files_to_process)
        if total_files == 0:
            logger.info("All %d files already indexed in manifest. Nothing to process.", len(all_records))
            emit_progress("COMPLETED", len(all_records), len(all_records), message="All files already indexed.")
            return PipelineSummary(
                total_files=len(all_records),
                indexed_files=len(all_records),
                error_files=0,
                total_vectors=self.qdrant.count("media_embeddings"),
                elapsed_seconds=time.time() - start_time,
                average_throughput=0.0,
            )

        logger.info("Starting ingestion pipeline for %d files (Workers=%d, Batch=%d)", total_files, self.max_decode_workers, self.batch_size)
        emit_progress("EXTRACTING", 0, total_files, message=f"Decoding {total_files} media files across {self.max_decode_workers} threads...")

        processed_count = 0
        indexed_count = 0
        error_count = 0
        total_vectors_added = 0

        # Accumulators for batch GPU embedding
        pending_points_metadata: List[Tuple[FileRecord, FrameData]] = []
        pending_images_to_embed: List[np.ndarray] = []

        def flush_embedding_batch():
            nonlocal pending_points_metadata, pending_images_to_embed, total_vectors_added, indexed_count
            if not pending_images_to_embed:
                return

            batch_count = len(pending_images_to_embed)
            emit_progress(
                "EMBEDDING",
                processed_count,
                total_files,
                message=f"Computing SigLIP 2 GPU embeddings for batch of {batch_count} frames...",
            )

            # 1. GPU Forward pass (MLX / MPS)
            vectors = self.embedder.embed_images(pending_images_to_embed)

            # 2. Build Qdrant VectorPoints
            points_to_upsert: List[VectorPoint] = []
            file_to_point_ids: Dict[int, List[str]] = {}
            model_name = getattr(self.embedder, "model_name", "siglip2")

            for (record, frame), vec in zip(pending_points_metadata, vectors):
                pt = VectorPoint(
                    vector=vec,
                    file_path=record.file_path,
                    file_id=record.id,
                    file_type=record.file_type,
                    frame_index=frame.frame_index,
                    source_offset=frame.source_offset,
                    creation_timestamp=record.creation_timestamp,
                    duration_seconds=record.duration_seconds,
                )
                points_to_upsert.append(pt)
                if record.id not in file_to_point_ids:
                    file_to_point_ids[record.id] = []
                file_to_point_ids[record.id].append(pt.id)

            # 3. Batch upsert into Qdrant Vector DB
            emit_progress(
                "INDEXING",
                processed_count,
                total_files,
                message=f"Upserting {len(points_to_upsert)} vector points into Qdrant...",
            )
            self.qdrant.upsert_points("media_embeddings", points_to_upsert)
            total_vectors_added += len(points_to_upsert)

            # 4. Checkpoint SQLite Manifest state to INDEXED
            for file_id, point_ids in file_to_point_ids.items():
                self.manifest.update_embedding_info(
                    file_id=file_id,
                    qdrant_point_ids=point_ids,
                    model_name=model_name,
                )
                indexed_count += 1

            pending_points_metadata.clear()
            pending_images_to_embed.clear()

        # Execute media decode in parallel thread pool
        with ThreadPoolExecutor(max_workers=self.max_decode_workers) as executor:
            future_to_file = {executor.submit(self._decode_file, record): record for record in files_to_process}

            for future in as_completed(future_to_file):
                if self._cancelled:
                    logger.warning("Ingestion pipeline cancelled by user.")
                    emit_progress("FAILED", processed_count, total_files, message="Pipeline cancelled.")
                    break

                record, frames, err = future.result()
                processed_count += 1

                if err or not frames:
                    error_count += 1
                    logger.warning("Error processing file %s: %s", record.file_path, err)
                    self.manifest.set_error(record.id, err or "No frames extracted")
                    emit_progress(
                        "EXTRACTING",
                        processed_count,
                        total_files,
                        current_file=Path(record.file_path).name,
                        message=f"Error on {Path(record.file_path).name}: {err}",
                    )
                    continue

                # Add frames to batch accumulator
                for frame in frames:
                    pending_points_metadata.append((record, frame))
                    pending_images_to_embed.append(frame.pixels)

                emit_progress(
                    "EXTRACTING",
                    processed_count,
                    total_files,
                    current_file=Path(record.file_path).name,
                    message=f"Decoded {Path(record.file_path).name} ({len(frames)} frame(s))",
                )

                # Flush GPU batch when threshold reached
                if len(pending_images_to_embed) >= self.batch_size:
                    flush_embedding_batch()

        # Flush any remaining frames
        if not self._cancelled and pending_images_to_embed:
            flush_embedding_batch()

        total_elapsed = time.time() - start_time
        avg_fps = (total_files / total_elapsed) if total_elapsed > 0 else 0.0

        status_msg = f"Pipeline complete: {indexed_count} indexed, {error_count} errors in {total_elapsed:.1f}s ({avg_fps:.1f} img/s)"
        emit_progress("COMPLETED", total_files, total_files, message=status_msg)

        logger.info(status_msg)
        return PipelineSummary(
            total_files=total_files,
            indexed_files=indexed_count,
            error_files=error_count,
            total_vectors=total_vectors_added,
            elapsed_seconds=total_elapsed,
            average_throughput=avg_fps,
        )
