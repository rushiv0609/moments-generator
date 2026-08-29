"""
Background Job Manager and Server-Sent Events (SSE) Dispatcher for Local AI Moments Generator.
"""

import uuid
import time
import json
import asyncio
import logging
import threading
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, AsyncGenerator, Callable

from app.core.workspace import get_workspace_manager
from app.core.pipeline import IngestionPipeline, PipelineProgress, PipelineSummary

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class JobEvent:
    """
    Real-time progress event broadcast via Server-Sent Events (SSE).
    """
    job_id: str
    event_type: str  # 'progress' | 'log' | 'completed' | 'error'
    progress_pct: float
    stage: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "event_type": self.event_type,
            "progress_pct": round(self.progress_pct, 1),
            "stage": self.stage,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    def to_sse_string(self) -> str:
        """Format as standard text/event-stream message."""
        payload = json.dumps(self.to_dict())
        return f"data: {payload}\n\n"


@dataclass
class Job:
    """
    Background Task Execution State.
    """
    id: str
    job_type: str  # 'indexing' | 'generation'
    status: JobStatus
    workspace_dir: str
    corpus_dir: Optional[str] = None
    progress: float = 0.0
    stage: str = "QUEUED"
    message: str = "Job queued..."
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_type": self.job_type,
            "status": self.status.value,
            "workspace_dir": self.workspace_dir,
            "corpus_dir": self.corpus_dir,
            "progress": round(self.progress, 1),
            "stage": self.stage,
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "summary": self.summary,
        }


class JobManager:
    """
    Manages background tasks, execution threads, and SSE subscriber queues.
    """

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._active_pipelines: Dict[str, IngestionPipeline] = {}
        self._lock = threading.Lock()

    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve job record by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[Job]:
        """List all tracked jobs."""
        with self._lock:
            return list(self._jobs.values())

    def broadcast_event(self, job_id: str, event: JobEvent):
        """Send an event to all connected SSE clients."""
        queues = self._subscribers.get(job_id, [])
        for q in list(queues):
            try:
                q.put_nowait(event)
            except Exception as e:
                logger.debug("Error broadcasting to subscriber: %s", e)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel an active background job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return False

            job.status = JobStatus.CANCELLED
            job.message = "Job cancelled by user."
            job.completed_at = time.time()

            pipeline = self._active_pipelines.get(job_id)
            if pipeline:
                pipeline.cancel()

        ev = JobEvent(
            job_id=job_id,
            event_type="error",
            progress_pct=job.progress,
            stage="CANCELLED",
            message="Job cancelled by user.",
        )
        self.broadcast_event(job_id, ev)
        return True

    def submit_indexing_job(
        self,
        workspace_dir: str,
        corpus_dir: Optional[str] = None,
        force_reindex: bool = False,
    ) -> Job:
        """
        Submit a new media indexing background job.
        """
        job_id = f"job_idx_{uuid.uuid4().hex[:8]}"
        job = Job(
            id=job_id,
            job_type="indexing",
            status=JobStatus.QUEUED,
            workspace_dir=workspace_dir,
            corpus_dir=corpus_dir,
        )

        with self._lock:
            self._jobs[job_id] = job

        # Launch background thread
        thread = threading.Thread(
            target=self._run_indexing_worker,
            args=(job_id, workspace_dir, corpus_dir, force_reindex),
            daemon=True,
            name=f"Worker-{job_id}",
        )
        thread.start()
        return job

    def _run_indexing_worker(
        self,
        job_id: str,
        workspace_dir: str,
        corpus_dir: Optional[str],
        force_reindex: bool,
    ):
        """Worker thread executing the ingestion pipeline."""
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            job.stage = "SCANNING"
            job.message = "Initializing ingestion pipeline..."

        ev_start = JobEvent(
            job_id=job_id,
            event_type="progress",
            progress_pct=0.0,
            stage="SCANNING",
            message=job.message,
        )
        self.broadcast_event(job_id, ev_start)

        try:
            workspace_mgr = get_workspace_manager()
            workspace_mgr.set_workspace(workspace_dir, corpus_path=corpus_dir)
            manifest = workspace_mgr.get_manifest_db()
            qdrant = workspace_mgr.get_qdrant_db()

            pipeline = IngestionPipeline(
                manifest=manifest,
                qdrant=qdrant,
                max_decode_workers=12,
                batch_size=32,
            )

            with self._lock:
                self._active_pipelines[job_id] = pipeline

            def on_progress(p: PipelineProgress):
                with self._lock:
                    job.progress = p.percentage
                    job.stage = p.stage
                    job.message = p.message

                ev = JobEvent(
                    job_id=job_id,
                    event_type="progress",
                    progress_pct=p.percentage,
                    stage=p.stage,
                    message=p.message,
                    data=p.to_dict(),
                )
                self.broadcast_event(job_id, ev)

            summary = pipeline.run(
                corpus_path=corpus_dir,
                force_reindex=force_reindex,
                progress_callback=on_progress,
            )

            with self._lock:
                job.status = JobStatus.COMPLETED
                job.progress = 100.0
                job.stage = "COMPLETED"
                job.completed_at = time.time()
                job.message = f"Indexed {summary.indexed_files} files into Qdrant in {summary.elapsed_seconds:.1f}s."
                job.summary = asdict(summary)

            ev_done = JobEvent(
                job_id=job_id,
                event_type="completed",
                progress_pct=100.0,
                stage="COMPLETED",
                message=job.message,
                data=job.summary,
            )
            self.broadcast_event(job_id, ev_done)

        except Exception as e:
            logger.error("Job %s failed: %s", job_id, e, exc_info=True)
            with self._lock:
                job.status = JobStatus.FAILED
                job.completed_at = time.time()
                job.error = str(e)
                job.message = f"Ingestion failed: {str(e)}"

            ev_err = JobEvent(
                job_id=job_id,
                event_type="failed",
                progress_pct=job.progress,
                stage="FAILED",
                message=job.error,
            )
            self.broadcast_event(job_id, ev_err)

        finally:
            with self._lock:
                if job_id in self._active_pipelines:
                    del self._active_pipelines[job_id]
            import gc
            if 'pipeline' in locals():
                del pipeline
            gc.collect()

    def start_generation_job(
        self,
        workspace_dir: str,
        prompt: str,
        corpus_dir: Optional[str] = None,
        target_duration: int = 30,
        model_name: str = "qwen2.5:7b",
        retrieval_mode: str = "dual",
        generate_alternatives: bool = True,
    ) -> Job:
        """Enqueue and launch a background Director Agent video generation job."""
        job_id = f"job_gen_{uuid.uuid4().hex[:8]}"
        job = Job(
            id=job_id,
            job_type="generation",
            status=JobStatus.QUEUED,
            workspace_dir=workspace_dir,
            corpus_dir=corpus_dir,
            message="Director Agent job queued...",
        )

        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_generation_worker,
            args=(
                job_id,
                workspace_dir,
                prompt,
                corpus_dir,
                target_duration,
                model_name,
                retrieval_mode,
                generate_alternatives,
            ),
            daemon=True,
            name=f"Director-{job_id}",
        )
        thread.start()
        return job

    def _run_generation_worker(
        self,
        job_id: str,
        workspace_dir: str,
        prompt: str,
        corpus_dir: Optional[str],
        target_duration: int,
        model_name: str,
        retrieval_mode: str,
        generate_alternatives: bool,
    ):
        """Worker thread executing the LangGraph Director state machine."""
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            job.stage = "PLANNING"
            job.message = f"Director planning storyline for '{prompt}'..."
            job.progress = 15.0

        ev_plan = JobEvent(
            job_id=job_id,
            event_type="progress",
            progress_pct=15.0,
            stage="PLANNING",
            message=job.message,
            data={"prompt": prompt, "model": model_name},
        )
        self.broadcast_event(job_id, ev_plan)

        try:
            from app.core.embedder import MLXEmbedder, PyTorchMPSEmbedder, HAS_MLX
            from app.core.director import DirectorAgent, get_director_llm, MockDirectorLLM

            workspace_mgr = get_workspace_manager()
            workspace_mgr.set_workspace(workspace_dir, corpus_path=corpus_dir)
            manifest = workspace_mgr.get_manifest_db()
            qdrant = workspace_mgr.get_qdrant_db()
            collection_name = getattr(workspace_mgr, "collection_name", "media_embeddings")

            # Initialize embedder and LLM (fail-fast without silent mock fallbacks)
            embedder = MLXEmbedder() if HAS_MLX else PyTorchMPSEmbedder()
            llm = get_director_llm(model_name=model_name, fallback_to_mock=False)

            agent = DirectorAgent(
                embedder=embedder,
                qdrant=qdrant,
                collection_name=collection_name,
                llm=llm,
                manifest=manifest,
                model_name=model_name,
            )

            # Stage: RETRIEVAL
            with self._lock:
                job.stage = "RETRIEVAL"
                job.progress = 40.0
                job.message = "Searching vector database for visual moments..."
            self.broadcast_event(
                job_id,
                JobEvent(job_id=job_id, event_type="progress", progress_pct=40.0, stage="RETRIEVAL", message=job.message),
            )

            # Stage: DRAFTING & EDITING
            with self._lock:
                job.stage = "DRAFTING"
                job.progress = 70.0
            def on_step_telemetry(step_data: Dict[str, Any]):
                stage_name = step_data.get("stage", "DRAFTING")
                node_name = step_data.get("node", "AGENT")
                summary = step_data.get("summary", "")
                with self._lock:
                    job.stage = stage_name
                    job.message = f"[{node_name}] {summary}"
                self.broadcast_event(
                    job_id,
                    JobEvent(
                        job_id=job_id,
                        event_type="telemetry",
                        progress_pct=job.progress,
                        stage=stage_name,
                        message=job.message,
                        data=step_data,
                    ),
                )

            if generate_alternatives:
                alternatives = agent.generate_alternatives(
                    prompt=prompt,
                    target_duration=target_duration,
                    job_id_prefix=job_id,
                    step_callback=on_step_telemetry,
                )
                primary_state = alternatives.get("alt_c_dual") or alternatives.get("alt_a_scene") or list(alternatives.values())[0]
                summary_data = {
                    "prompt": prompt,
                    "target_duration": target_duration,
                    "model": model_name,
                    "alternatives": {k: v for k, v in alternatives.items()},
                    "storyboard": primary_state.get("storyboard", []),
                    "search_queries": primary_state.get("search_queries", []),
                    "narrative_arc": primary_state.get("narrative_arc", ""),
                    "approved": primary_state.get("approved", True),
                    "editor_feedback": primary_state.get("editor_feedback", []),
                    "agent_telemetry": primary_state.get("agent_telemetry", []),
                }
            else:
                final_state = agent.run(
                    prompt=prompt,
                    target_duration=target_duration,
                    retrieval_mode=retrieval_mode,
                    job_id=job_id,
                    step_callback=on_step_telemetry,
                )
                summary_data = {
                    "prompt": prompt,
                    "target_duration": target_duration,
                    "model": model_name,
                    "storyboard": final_state.get("storyboard", []),
                    "search_queries": final_state.get("search_queries", []),
                    "narrative_arc": final_state.get("narrative_arc", ""),
                    "approved": final_state.get("approved", True),
                    "editor_feedback": final_state.get("editor_feedback", []),
                    "agent_telemetry": final_state.get("agent_telemetry", []),
                }

            with self._lock:
                job.status = JobStatus.COMPLETED
                job.progress = 100.0
                job.stage = "COMPLETED"
                job.completed_at = time.time()
                tot_dur = sum(s.get("duration", 0) for s in summary_data.get("storyboard", []))
                job.message = f"Curated {len(summary_data.get('storyboard', []))} moments ({tot_dur:.1f}s) successfully!"
                job.summary = summary_data

            ev_done = JobEvent(
                job_id=job_id,
                event_type="completed",
                progress_pct=100.0,
                stage="COMPLETED",
                message=job.message,
                data=job.summary,
            )
            self.broadcast_event(job_id, ev_done)

        except Exception as e:
            logger.error("Generation Job %s failed: %s", job_id, e, exc_info=True)
            with self._lock:
                job.status = JobStatus.FAILED
                job.completed_at = time.time()
                job.error = str(e)
                job.message = f"Curation failed: {str(e)}"

            ev_err = JobEvent(
                job_id=job_id,
                event_type="failed",
                progress_pct=job.progress,
                stage="FAILED",
                message=job.error,
            )
            self.broadcast_event(job_id, ev_err)

        finally:
            if 'llm' in locals() and llm and hasattr(llm, "unload"):
                try:
                    llm.unload()
                except Exception as e:
                    logger.debug("Notice unloading LLM in generation worker: %s", e)

    async def subscribe(self, job_id: str) -> AsyncGenerator[str, None]:
        """
        Subscribe to live Server-Sent Events (SSE) for a given job.
        Yields standard text/event-stream chunks.
        """
        q = asyncio.Queue()
        if job_id not in self._subscribers:
            self._subscribers[job_id] = []
        self._subscribers[job_id].append(q)

        # Emit initial state
        job = self.get_job(job_id)
        if job:
            init_ev = JobEvent(
                job_id=job_id,
                event_type="progress",
                progress_pct=job.progress,
                stage=job.stage,
                message=job.message,
            )
            yield init_ev.to_sse_string()

        try:
            while True:
                # Wait for next event or timeout check
                try:
                    event: JobEvent = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield event.to_sse_string()
                    if event.event_type in ("completed", "error"):
                        break
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
                    # Check if job terminated
                    curr_job = self.get_job(job_id)
                    if curr_job and curr_job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                        break
        finally:
            if job_id in self._subscribers and q in self._subscribers[job_id]:
                self._subscribers[job_id].remove(q)


# Global singleton instance
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Retrieve the global JobManager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
