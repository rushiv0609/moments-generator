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
import gc
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple, Union

import numpy as np

from app.db.manifest import ManifestDB
from app.db.qdrant import QdrantVectorDB, VectorPoint
from app.db.models import FileRecord, FileStatus
from app.core.embedder import EmbedderInterface, create_embedder
from app.core.extractor import decode_image, extract_video_frames, FrameData
from app.core.scene_detector import detect_video_scenes, SceneBoundary
from app.core.scanner import scan_corpus
from app.core.telemetry import TelemetryMonitor
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

    def _decode_file(self, record: FileRecord, decode_queue: queue.Queue) -> Tuple[FileRecord, Optional[str]]:
        """
        Worker function: decodes photo or video file using C-accelerated PyAV / ImageIO.
        Streams frames directly into bounded decode_queue without IPC serialization overhead.
        """
        try:
            p = Path(record.file_path)
            if not p.exists():
                return record, f"File not found on disk: {record.file_path}"

            if record.file_type == "video":
                scenes: List[SceneBoundary] = detect_video_scenes(str(p))
                for f in extract_video_frames(str(p), target_size=224, sampling_fps=1.0):
                    if self._cancelled:
                        break
                    # Tag frame with its matching scene
                    for sc in scenes:
                        if sc.start_sec <= f.source_offset <= sc.end_sec:
                            f.scene_id = sc.scene_id
                            f.scene_start = sc.start_sec
                            f.scene_end = sc.end_sec
                            break
                    if f.scene_id is None and scenes:
                        f.scene_id = scenes[0].scene_id
                        f.scene_start = scenes[0].start_sec
                        f.scene_end = scenes[0].end_sec

                    # Push to bounded queue with cancellation check
                    while not self._cancelled:
                        try:
                            decode_queue.put((record, f), timeout=0.5)
                            break
                        except queue.Full:
                            continue
            else:
                frame = decode_image(str(p), target_size=224)
                if frame is not None:
                    while not self._cancelled:
                        try:
                            decode_queue.put((record, frame), timeout=0.5)
                            break
                        except queue.Full:
                            continue
                else:
                    return record, "Failed to decode image file format."

            return record, None
        except Exception as e:
            return record, str(e)

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

    def _embed_loop(self, decode_queue: queue.Queue, index_queue: queue.Queue):
        """
        Consumer of decode_queue and producer for index_queue.
        Reads frames, batches them, embeds on GPU, and puts points to index_queue.
        Timeout of 10s ensures partial batches are flushed.
        """
        pending_points_metadata = []
        pending_images_to_embed = []
        model_name = getattr(self.embedder, "model_name", "siglip2")

        def flush():
            if not pending_images_to_embed:
                return

            try:
                vectors = self.embedder.embed_images(pending_images_to_embed)
                points_to_upsert: List[VectorPoint] = []
                file_to_point_ids: Dict[int, List[str]] = {}
                scene_vectors_map: Dict[Tuple[int, int], List[Tuple[np.ndarray, Optional[float], Optional[float], FileRecord]]] = {}

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
                        granularity="frame",
                        scene_id=frame.scene_id,
                        scene_start=frame.scene_start,
                        scene_end=frame.scene_end,
                        is_scene_representative=False,
                    )
                    points_to_upsert.append(pt)
                    if record.id not in file_to_point_ids:
                        file_to_point_ids[record.id] = []
                    file_to_point_ids[record.id].append(pt.id)

                    if record.file_type == "video" and frame.scene_id is not None and record.id is not None:
                        key = (record.id, frame.scene_id)
                        if key not in scene_vectors_map:
                            scene_vectors_map[key] = []
                        scene_vectors_map[key].append((vec, frame.scene_start, frame.scene_end, record))

                for (file_id, scene_id), items in scene_vectors_map.items():
                    if not items:
                        continue
                    vecs = [it[0] for it in items]
                    first_rec = items[0][3]
                    sc_start = items[0][1]
                    sc_end = items[0][2]

                    mean_vec = np.mean(vecs, axis=0).astype(np.float32)
                    norm = np.linalg.norm(mean_vec)
                    if norm > 0:
                        mean_vec = mean_vec / norm

                    scene_pt = VectorPoint(
                        vector=mean_vec,
                        file_path=first_rec.file_path,
                        file_id=first_rec.id,
                        file_type="video",
                        frame_index=-1,
                        source_offset=sc_start if sc_start is not None else 0.0,
                        creation_timestamp=first_rec.creation_timestamp,
                        duration_seconds=first_rec.duration_seconds,
                        granularity="scene",
                        scene_id=scene_id,
                        scene_start=sc_start,
                        scene_end=sc_end,
                        scene_frame_count=len(vecs),
                        is_scene_representative=True,
                    )
                    points_to_upsert.append(scene_pt)
                    if first_rec.id in file_to_point_ids:
                        file_to_point_ids[first_rec.id].append(scene_pt.id)

                index_queue.put((points_to_upsert, file_to_point_ids, len(pending_images_to_embed)))

            except Exception as e:
                logger.error(f"Embed loop error: {e}", exc_info=True)
            finally:
                pending_points_metadata.clear()
                pending_images_to_embed.clear()
                self.embedder.empty_cache()

        while not self._cancelled:
            try:
                item = decode_queue.get(timeout=10.0)
                if item is None:
                    break
                record, frame = item
                pending_points_metadata.append((record, frame))
                pending_images_to_embed.append(frame.pixels)
                
                if len(pending_images_to_embed) >= self.batch_size:
                    flush()
            except queue.Empty:
                if pending_images_to_embed:
                    flush()

        if pending_images_to_embed:
            flush()
        index_queue.put(None)  # Sentinel for index worker

    def _index_loop(self, index_queue: queue.Queue):
        """
        Consumer of index_queue. Upserts vectors to Qdrant and updates SQLite.
        """
        model_name = getattr(self.embedder, "model_name", "siglip2")
        while not self._cancelled:
            item = index_queue.get()
            if item is None:
                break
            
            points_to_upsert, file_to_point_ids, frames_embedded = item
            
            try:
                if points_to_upsert:
                    self.qdrant.upsert_points("media_embeddings", points_to_upsert)
                    
                for file_id, point_ids in file_to_point_ids.items():
                    self.manifest.update_embedding_info(
                        file_id=file_id,
                        qdrant_point_ids=point_ids,
                        model_name=model_name,
                    )
                # We can't update self.total_vectors_added here directly without thread safety, 
                # but we can rely on main thread counting it if we want.
                # Actually we can just keep a counter and attach it to self.
                self._indexed_frames_count += frames_embedded
                self._upserted_vectors_count += len(points_to_upsert)
            except Exception as e:
                logger.error(f"Index loop error: {e}", exc_info=True)

    def run(
        self,
        corpus_path: Optional[str] = None,
        force_reindex: bool = False,
        progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
    ) -> PipelineSummary:
        """
        Execute the full ingestion and embedding pipeline using decoupled producer-consumer queues.
        """
        self._cancelled = False
        self._indexed_frames_count = 0
        self._upserted_vectors_count = 0
        start_time = time.time()
        
        decode_queue = queue.Queue(maxsize=512)
        index_queue = queue.Queue(maxsize=512)
        
        telemetry = TelemetryMonitor(decode_queue, index_queue, interval_seconds=2.0)
        telemetry.start()

        def emit_progress(stage: str, processed: int, total: int, current_file: Optional[str] = None, message: str = ""):
            elapsed = time.time() - start_time
            pct = (processed / total * 100.0) if total > 0 else 100.0
            fps = (processed / elapsed) if elapsed > 0 else 0.0
            
            # Inject queue sizes into message
            q_msg = f"[Q: dec={decode_queue.qsize()}/512 idx={index_queue.qsize()}/512] "
            msg_with_q = q_msg + message if message else q_msg
            
            prog = PipelineProgress(
                stage=stage,
                processed_count=processed,
                total_count=total,
                percentage=pct,
                throughput_fps=fps,
                current_file=current_file,
                message=msg_with_q,
                elapsed_seconds=elapsed,
            )
            if progress_callback:
                try:
                    progress_callback(prog)
                except Exception as e:
                    logger.debug("Progress callback error: %s", e)

        # Stage 1: Checkpoint / File Discovery
        emit_progress("SCANNING", 0, 0, message="Inspecting manifest and corpus files...")

        if corpus_path and Path(corpus_path).exists():
            try:
                emit_progress("SCANNING", 0, 0, message=f"Scanning corpus directory: {Path(corpus_path).name}...")
                scan_corpus(corpus_path, self.manifest)
            except Exception as e:
                logger.warning("Auto-scan error on %s: %s", corpus_path, e)

        all_records = self.manifest.get_all_files()
        if force_reindex:
            files_to_process = all_records
            logger.info("Force reindex enabled: reprocessing all %d files", len(files_to_process))
        else:
            files_to_process = [r for r in all_records if r.status != FileStatus.INDEXED]

        total_files = len(files_to_process)
        if total_files == 0:
            logger.info("All files already indexed.")
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
        error_count = 0

        # Start consumer threads
        embed_thread = threading.Thread(target=self._embed_loop, args=(decode_queue, index_queue))
        index_thread = threading.Thread(target=self._index_loop, args=(index_queue,))
        embed_thread.start()
        index_thread.start()

        # Execute media decode in parallel thread pool (PyAV & ImageIO release GIL in C layer)
        with ThreadPoolExecutor(max_workers=self.max_decode_workers) as executor:
            future_to_file = {executor.submit(self._decode_file, record, decode_queue): record for record in files_to_process}

            for future in as_completed(future_to_file):
                if self._cancelled:
                    logger.warning("Ingestion pipeline cancelled by user.")
                    emit_progress("FAILED", processed_count, total_files, message="Pipeline cancelled.")
                    for f in future_to_file:
                        f.cancel()
                    break

                record, err = future.result()
                processed_count += 1

                if err:
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
                else:
                    emit_progress(
                        "EXTRACTING",
                        processed_count,
                        total_files,
                        current_file=Path(record.file_path).name,
                        message=f"Decoded {Path(record.file_path).name} (Indexed {self._indexed_frames_count} frames)",
                    )

        # Signal embedder to stop
        decode_queue.put(None)
        
        # Wait for threads to finish
        emit_progress("INDEXING", processed_count, total_files, message="Waiting for indexer to flush remaining frames...")
        embed_thread.join()
        index_thread.join()

        total_elapsed = time.time() - start_time
        avg_fps = (total_files / total_elapsed) if total_elapsed > 0 else 0.0

        status_msg = f"Pipeline complete: {processed_count - error_count} indexed, {error_count} errors in {total_elapsed:.1f}s ({avg_fps:.1f} img/s)"
        emit_progress("COMPLETED", total_files, total_files, message=status_msg)

        logger.info(status_msg)

        # Final cleanup pass
        telemetry.stop()
        self.embedder.empty_cache()
        gc.collect()

        return PipelineSummary(
            total_files=total_files,
            indexed_files=processed_count - error_count,
            error_files=error_count,
            total_vectors=self._upserted_vectors_count,
            elapsed_seconds=total_elapsed,
            average_throughput=avg_fps,
        )
