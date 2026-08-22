"""
Video Scene Boundary Detection Engine using PySceneDetect.
Optimized with AdaptiveDetector for handheld trip videos (robust against camera motion).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import cv2

try:
    import scenedetect
    from scenedetect import detect, AdaptiveDetector, ContentDetector
    HAS_SCENEDETECT = True
except ImportError:
    HAS_SCENEDETECT = False

logger = logging.getLogger(__name__)


@dataclass
class SceneBoundary:
    """Represents a coherent temporal visual scene within a video."""
    scene_id: int
    start_sec: float
    end_sec: float
    duration_sec: float
    start_frame: int = 0
    end_frame: int = 0

    @property
    def midpoint_sec(self) -> float:
        return (self.start_sec + self.end_sec) / 2.0


def get_video_duration_cv2(file_path: str) -> float:
    """Helper to get exact video duration in seconds via OpenCV."""
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        return 0.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps and fps > 0 and frame_count and frame_count > 0:
            return float(frame_count / fps)
        return 0.0
    finally:
        cap.release()


def detect_video_scenes(
    file_path: str,
    adaptive_threshold: float = 3.0,
    min_scene_length_sec: float = 2.0,
) -> List[SceneBoundary]:
    """
    Detect scene cut and transition boundaries in a video file using AdaptiveDetector.
    
    Args:
        file_path: Path to local video file.
        adaptive_threshold: Rolling average sensitivity threshold (default: 3.0 for handheld).
        min_scene_length_sec: Minimum scene duration in seconds (default: 2.0s).
        
    Returns:
        List of SceneBoundary objects covering [0.0, video_duration].
    """
    path_obj = Path(file_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    total_duration = get_video_duration_cv2(str(path_obj))

    if not HAS_SCENEDETECT:
        logger.warning("PySceneDetect not installed. Treating entire video as a single scene.")
        return [
            SceneBoundary(
                scene_id=0,
                start_sec=0.0,
                end_sec=max(0.0, total_duration),
                duration_sec=max(0.0, total_duration),
            )
        ]

    try:
        # Determine fps to compute min_scene_len in frames
        cap = cv2.VideoCapture(str(path_obj))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        effective_fps = fps if (fps and fps > 0) else 30.0
        min_scene_frames = max(15, int(round(min_scene_length_sec * effective_fps)))

        detector = AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
            min_scene_len=min_scene_frames,
        )

        scene_list = detect(str(path_obj), detector)

        if not scene_list:
            # Entire video is one continuous shot without cuts
            return [
                SceneBoundary(
                    scene_id=0,
                    start_sec=0.0,
                    end_sec=max(0.0, total_duration),
                    duration_sec=max(0.0, total_duration),
                )
            ]

        results: List[SceneBoundary] = []
        for i, (start_time, end_time) in enumerate(scene_list):
            s_sec = getattr(start_time, "seconds", None)
            if s_sec is None:
                s_sec = start_time.get_seconds()
            e_sec = getattr(end_time, "seconds", None)
            if e_sec is None:
                e_sec = end_time.get_seconds()

            s_frame = getattr(start_time, "frame_num", None)
            if s_frame is None:
                s_frame = start_time.get_frames()
            e_frame = getattr(end_time, "frame_num", None)
            if e_frame is None:
                e_frame = end_time.get_frames()

            results.append(
                SceneBoundary(
                    scene_id=i,
                    start_sec=round(float(s_sec), 3),
                    end_sec=round(float(e_sec), 3),
                    duration_sec=round(float(e_sec) - float(s_sec), 3),
                    start_frame=int(s_frame),
                    end_frame=int(e_frame),
                )
            )

        return results

    except Exception as e:
        logger.warning("Scene detection failed for %s: %s. Falling back to 1 scene.", file_path, e)
        return [
            SceneBoundary(
                scene_id=0,
                start_sec=0.0,
                end_sec=max(0.0, total_duration),
                duration_sec=max(0.0, total_duration),
            )
        ]
