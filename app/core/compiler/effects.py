"""
Visual effects, motion filters (Ken Burns), and transitions for the Video Compiler.
"""

import enum
from typing import Tuple, Dict, Any, Optional


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


def build_ken_burns_filter(
    motion: KenBurnsMotion,
    duration: float,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
    zoom_delta: float = 0.15,
) -> str:
    """
    Build a smooth FFmpeg zoompan filter for static photos.
    
    Args:
        motion: Ken Burns pattern (zoom in/out, pan left/right).
        duration: Duration in seconds.
        fps: Frames per second.
        width: Canvas width.
        height: Canvas height.
        zoom_delta: Maximum zoom change (e.g. 0.15 = 1.0 to 1.15x).
    
    Returns:
        FFmpeg filter string.
    """
    # Strict upper bound: No image displayed for more than 3.0s
    clamped_duration = min(3.0, max(1.0, float(duration)))
    total_frames = max(1, int(round(clamped_duration * fps)))
    
    if motion == KenBurnsMotion.ZOOM_IN:
        # Smooth zoom from 1.0x to 1.15x centered
        z_expr = f"min(1.0+{zoom_delta:.2f},1.0+{zoom_delta:.2f}*on/{total_frames})"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif motion == KenBurnsMotion.ZOOM_OUT:
        # Smooth zoom from 1.15x down to 1.0x centered
        z_expr = f"max(1.0,1.0+{zoom_delta:.2f}-{zoom_delta:.2f}*on/{total_frames})"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif motion == KenBurnsMotion.PAN_RIGHT:
        # Subtle 1.08x zoom with horizontal pan left-to-right
        z_expr = "1.08"
        x_expr = f"(iw-iw/zoom)*(on/{total_frames})"
        y_expr = "ih/2-(ih/zoom/2)"
    else:  # PAN_LEFT
        # Subtle 1.08x zoom with horizontal pan right-to-left
        z_expr = "1.08"
        x_expr = f"(iw-iw/zoom)*(1-on/{total_frames})"
        y_expr = "ih/2-(ih/zoom/2)"

    # Scale up proportionally to 2x target resolution for zoom headroom, then crop to target aspect ratio
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
    mode: str = "crop_fill",  # 'crop_fill' or 'letterbox'
) -> str:
    """
    Normalize video clips to target resolution, framerate, and aspect ratio.
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
