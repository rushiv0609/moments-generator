"""
Visual effects, motion filters (Ken Burns), and transitions for the Video Compiler.
Includes saliency-aware subject tracking, smoothstep easing, and content-aware motion selection.
"""

import enum
import logging
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class KenBurnsMotion(str, enum.Enum):
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"


def get_ken_burns_pattern(index: int) -> KenBurnsMotion:
    """Cycle through motion patterns to create cinematic visual rhythm."""
    patterns = [
        KenBurnsMotion.ZOOM_IN,
        KenBurnsMotion.PAN_RIGHT,
        KenBurnsMotion.ZOOM_OUT,
        KenBurnsMotion.PAN_LEFT,
    ]
    return patterns[index % len(patterns)]


def detect_subject_position(image_path: str) -> Tuple[float, float]:
    """
    Detect main subject center coordinates (cx, cy) normalized in [0.0, 1.0].
    Uses OpenCV Spectral Residual saliency map with center fallback.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return (0.5, 0.5)

        h, w = img.shape[:2]
        small = cv2.resize(img, (240, int(240 * h / w)))

        # Saliency detection
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(small)

        if success and saliency_map is not None:
            # Find center of mass of saliency
            thresh = cv2.threshold(saliency_map, 0.7 * saliency_map.max(), 255, cv2.THRESH_BINARY)[1]
            moments = cv2.moments(thresh)
            if moments["m00"] > 0:
                cx = (moments["m10"] / moments["m00"]) / small.shape[1]
                cy = (moments["m01"] / moments["m00"]) / small.shape[0]
                return (float(np.clip(cx, 0.1, 0.9)), float(np.clip(cy, 0.1, 0.9)))
    except Exception as e:
        logger.debug("Saliency detection notice on %s: %s", image_path, e)

    return (0.5, 0.5)


def choose_smart_ken_burns_pattern(image_path: str, index: int = 0) -> KenBurnsMotion:
    """
    Content-aware Ken Burns motion selection based on aspect ratio and subject center.
    """
    try:
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            aspect = w / h if h > 0 else 1.0
            cx, cy = detect_subject_position(image_path)

            # 1. Wide landscape (>= 16:9 or 3:2)
            if aspect >= 1.4:
                if cx < 0.42:
                    return KenBurnsMotion.PAN_LEFT  # Pan toward left subject
                elif cx > 0.58:
                    return KenBurnsMotion.PAN_RIGHT # Pan toward right subject
                else:
                    return KenBurnsMotion.ZOOM_OUT  # Reveal wide scenery

            # 2. Portrait / Square (< 1.2)
            elif aspect < 1.2:
                return KenBurnsMotion.ZOOM_IN if (0.4 <= cx <= 0.6) else KenBurnsMotion.ZOOM_OUT
    except Exception as e:
        logger.debug("Smart Ken Burns selection exception: %s", e)

    return get_ken_burns_pattern(index)


def build_ken_burns_filter(
    motion: KenBurnsMotion,
    duration: float,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
    zoom_delta: float = 0.08,
) -> str:
    """
    Build a smooth FFmpeg zoompan filter for static photos with Hermite smoothstep easing.
    
    Args:
        motion: Ken Burns pattern (zoom in/out, pan left/right).
        duration: Duration in seconds.
        fps: Frames per second.
        width: Canvas width.
        height: Canvas height.
        zoom_delta: Maximum zoom change (0.08 = 1.0 to 1.08x for smooth motion).
    
    Returns:
        FFmpeg filter string.
    """
    clamped_duration = min(5.0, max(1.0, float(duration)))
    total_frames = max(1, int(round(clamped_duration * fps)))
    
    # Hermite smoothstep normalized progress: t = on/total_frames; ease = 3*t^2 - 2*t^3
    t_expr = f"(on/{total_frames})"
    ease_expr = f"(3*{t_expr}*{t_expr}-2*{t_expr}*{t_expr}*{t_expr})"

    if motion == KenBurnsMotion.ZOOM_IN:
        # Smooth ease-in/ease-out zoom from 1.0x to (1.0 + zoom_delta)x centered
        z_expr = f"min(1.0+{zoom_delta:.3f},1.0+{zoom_delta:.3f}*{ease_expr})"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif motion == KenBurnsMotion.ZOOM_OUT:
        # Smooth ease-in/ease-out zoom from (1.0 + zoom_delta)x down to 1.0x centered
        z_expr = f"max(1.0,1.0+{zoom_delta:.3f}-{zoom_delta:.3f}*{ease_expr})"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif motion == KenBurnsMotion.PAN_RIGHT:
        # Smooth horizontal pan left-to-right with subtle zoom
        z_expr = f"{1.0 + zoom_delta:.3f}"
        x_expr = f"(iw-iw/zoom)*{ease_expr}"
        y_expr = "ih/2-(ih/zoom/2)"
    else:  # PAN_LEFT
        # Smooth horizontal pan right-to-left with subtle zoom
        z_expr = f"{1.0 + zoom_delta:.3f}"
        x_expr = f"(iw-iw/zoom)*(1.0-{ease_expr})"
        y_expr = "ih/2-(ih/zoom/2)"

    # Scale up proportionally for zoom headroom, then apply zoompan
    upscale_w = width * 2
    upscale_h = height * 2

    return (
        f"scale={upscale_w}:{upscale_h}:force_original_aspect_ratio=increase,"
        f"crop={upscale_w}:{upscale_h},"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={total_frames}:s={width}x{height}:fps={fps},"
        f"format=yuv420p"
    )


def build_video_normalization_filter(
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    mode: str = "letterbox",  # 'letterbox' or 'crop_fill'
) -> str:
    """
    Normalize video clips to target resolution and framerate without distortion.
    """
    if mode == "letterbox":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"fps={fps},format=yuv420p"
        )
    else:
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            f"fps={fps},format=yuv420p"
        )
