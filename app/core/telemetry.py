import json
import time
import threading
import logging
from pathlib import Path
from queue import Queue
from typing import Optional
import os

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)

class TelemetryMonitor:
    """
    Asynchronous telemetry tracker for the ingestion pipeline.
    Samples CPU, RAM, and Queue sizes periodically without blocking.
    Writes JSON Lines to data/telemetry.jsonl.
    """
    def __init__(self, decode_queue: Queue, index_queue: Queue, interval_seconds: float = 2.0):
        self.decode_queue = decode_queue
        self.index_queue = index_queue
        self.interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._log_path = Path("data/telemetry.jsonl")
        
        # Ensure data directory exists
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # We need our own process ID for process-specific memory and cpu metrics
        if HAS_PSUTIL:
            self._process = psutil.Process(os.getpid())
            # Initialize cpu_percent
            self._process.cpu_percent()
        else:
            self._process = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        
        self._stop_event.clear()
        
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps({"event": "pipeline_start", "timestamp": time.time()}) + "\n")
        except Exception as e:
            logger.warning(f"Could not write to telemetry log: {e}")

        self._thread = threading.Thread(target=self._loop, name="TelemetryMonitor", daemon=True)
        self._thread.start()
        logger.info("Telemetry monitor started.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps({"event": "pipeline_stop", "timestamp": time.time()}) + "\n")
        except Exception as e:
            pass
        logger.info("Telemetry monitor stopped.")

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                metrics = {
                    "timestamp": time.time(),
                    "decode_qsize": self.decode_queue.qsize(),
                    "index_qsize": self.index_queue.qsize(),
                }
                
                if HAS_PSUTIL and self._process:
                    total_rss = 0
                    total_cpu = 0.0
                    
                    try:
                        total_rss += self._process.memory_info().rss
                        total_cpu += self._process.cpu_percent()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                    try:
                        children = self._process.children(recursive=True)
                        for child in children:
                            try:
                                total_rss += child.memory_info().rss
                                total_cpu += child.cpu_percent()
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                    metrics["cpu_percent"] = round(total_cpu, 1)
                    metrics["memory_rss_mb"] = round(total_rss / (1024 * 1024), 2)
                    
                    # Also get overall system stats just in case
                    metrics["system_cpu_percent"] = psutil.cpu_percent()
                    metrics["system_ram_percent"] = psutil.virtual_memory().percent
                else:
                    metrics["warning"] = "psutil not installed"

                with open(self._log_path, "a") as f:
                    f.write(json.dumps(metrics) + "\n")
                
            except Exception as e:
                logger.debug(f"Telemetry loop error: {e}")
            
            # Wait with interrupt check
            self._stop_event.wait(self.interval)
