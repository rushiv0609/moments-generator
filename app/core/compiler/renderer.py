"""
FFmpeg-based Video Compiler and Montage Renderer.
Stitches photos (with content-aware Ken Burns motion) and trimmed video clips into a single .mp4 export.
Supports source resolution preservation, smart speed clamping, and hardware acceleration.
"""

import os
import shutil
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple

import cv2
from app.core.compiler.effects import (
    KenBurnsMotion,
    get_ken_burns_pattern,
    choose_smart_ken_burns_pattern,
    build_ken_burns_filter,
    build_video_normalization_filter,
)

logger = logging.getLogger(__name__)


def _detect_media_resolution(file_path: str) -> Tuple[int, int]:
    """Detect native width and height of image or video file."""
    try:
        ext = Path(file_path).suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".webp", ".tiff", ".heic"):
            from PIL import Image, ImageOps
            import pillow_heif
            pillow_heif.register_heif_opener()
            with Image.open(file_path) as img:
                img = ImageOps.exif_transpose(img)
                return img.size # (width, height)
        else:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if w > 0 and h > 0:
                    return (w, h)
    except Exception as e:
        logger.debug("Resolution detection notice for %s: %s", file_path, e)
    return (1920, 1080)


class VideoCompiler:
    """
    Renders storyboard timelines into high-definition, color-standardized MP4 videos.
    """

    def __init__(
        self,
        output_width: int = 1920,
        output_height: int = 1080,
        fps: int = 30,
        transition_duration: float = 0.5,
        max_speed_factor: float = 1.25,
        use_hardware_accel: bool = True,
    ):
        self.output_width = output_width
        self.output_height = output_height
        self.fps = fps
        self.transition_duration = transition_duration
        self.max_speed_factor = max_speed_factor
        self.use_hardware_accel = use_hardware_accel
        self.ffmpeg_path = shutil.which("ffmpeg") or "ffmpeg"

    def _get_encoder_args(self, is_4k: bool = False) -> List[str]:
        """Check and return video encoder arguments (VideoToolbox on macOS or libx264)."""
        bitrate = "15000k" if is_4k else "8000k"
        if self.use_hardware_accel:
            try:
                out = subprocess.check_output(
                    [self.ffmpeg_path, "-encoders"],
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                if "h264_videotoolbox" in out:
                    return ["-c:v", "h264_videotoolbox", "-b:v", bitrate, "-pix_fmt", "yuv420p"]
            except Exception as e:
                logger.debug("VideoToolbox check failed, falling back to libx264: %s", e)

        return ["-c:v", "libx264", "-preset", "fast", "-crf", "20" if is_4k else "21", "-pix_fmt", "yuv420p"]

    def _render_image_segment(
        self,
        image_path: str,
        duration: float,
        out_path: str,
        pattern_index: int = 0,
        temp_dir: Optional[str] = None,
    ) -> bool:
        """Render a single image with content-aware Ken Burns animation into a temporary MP4 chunk."""
        # Clamped duration range: 1.0s to 5.0s
        duration = min(2.5, max(1.0, float(duration)))
        total_frames = max(1, int(round(duration * self.fps)))

        src_path = image_path
        cleanup_temp = False
        try:
            from PIL import Image, ImageOps
            import pillow_heif
            pillow_heif.register_heif_opener()

            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                temp_dir_path = temp_dir or tempfile.gettempdir()
                temp_jpg = os.path.join(temp_dir_path, f"norm_img_{pattern_index:03d}_{os.path.basename(image_path)}.jpg")
                img.save(temp_jpg, "JPEG", quality=95)
                src_path = temp_jpg
                cleanup_temp = True
        except Exception as e:
            logger.debug("Pillow normalization fallback for %s: %s", image_path, e)
            src_path = image_path

        # Content-aware smart Ken Burns pattern selection
        pattern = choose_smart_ken_burns_pattern(src_path, index=pattern_index)
        kb_filter = build_ken_burns_filter(
            motion=pattern,
            duration=duration,
            fps=self.fps,
            width=self.output_width,
            height=self.output_height,
        )

        is_4k = (self.output_width >= 3840 or self.output_height >= 2160)
        encoder_args = self._get_encoder_args(is_4k=is_4k)

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", src_path,
            "-vf", kb_filter,
            "-frames:v", str(total_frames),
            *encoder_args,
            "-an",
            out_path,
        ]

        logger.info("Rendering photo [%s] -> %s (Ken Burns: %s, %.1fs, %d frames)", image_path, out_path, pattern.value, duration, total_frames)
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if cleanup_temp and os.path.exists(src_path) and src_path != image_path:
            try:
                os.remove(src_path)
            except OSError:
                pass

        if res.returncode != 0:
            logger.error("FFmpeg error rendering photo segment %s:\n%s", image_path, res.stderr)
            return False
        return True

    def _render_video_segment(
        self,
        video_path: str,
        start_offset: float,
        duration: float,
        out_path: str,
    ) -> bool:
        """Trim and normalize a video clip into a temporary MP4 chunk with speed clamping (Max 10.0s)."""
        # Duration clamped to 1.5s - 10.0s
        duration = min(10.0, max(1.5, float(duration)))
        actual_dur = duration
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_sec = total_frames / src_fps if src_fps > 0 else 10.0
                cap.release()

                if start_offset >= total_sec:
                    start_offset = 0.0
                avail_sec = total_sec - start_offset
                if avail_sec < duration:
                    # Check implied speed factor: clamp at max 1.25x
                    implied_speed = duration / max(avail_sec, 0.5)
                    if implied_speed > self.max_speed_factor:
                        actual_dur = max(1.5, avail_sec)
                    else:
                        actual_dur = max(1.0, avail_sec)
        except Exception as e:
            logger.debug("Video probe notice: %s", e)

        norm_filter = build_video_normalization_filter(
            width=self.output_width,
            height=self.output_height,
            fps=self.fps,
        )

        is_4k = (self.output_width >= 3840 or self.output_height >= 2160)
        encoder_args = self._get_encoder_args(is_4k=is_4k)

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-ss", f"{start_offset:.3f}",
            "-t", f"{actual_dur:.3f}",
            "-i", video_path,
            "-vf", norm_filter,
            *encoder_args,
            "-an",  # Strip audio for silent visual montage
            out_path,
        ]

        logger.info("Rendering video clip [%s] (offset: %.1fs, dur: %.1fs) -> %s", video_path, start_offset, actual_dur, out_path)
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            logger.error("FFmpeg error rendering video segment %s:\n%s", video_path, res.stderr)
            return False
        return True

    def _get_media_duration(self, file_path: str) -> float:
        """Get exact duration of rendered media chunk in seconds."""
        try:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                fc = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                cap.release()
                if fps > 0 and fc > 0:
                    return float(fc / fps)
        except Exception:
            pass
        return 3.0

    def render(
        self,
        storyboard: List[Dict[str, Any]],
        output_file: str,
        progress_callback: Optional[Callable[[float, str, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Compile and render the full storyboard sequence into output_file.
        """
        if not storyboard:
            raise ValueError("Cannot render an empty storyboard.")

        out_path = Path(output_file).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg is not installed or not available in system PATH.")

        # Determine dominant resolution across storyboard candidates (4K vs 1080p)
        resolutions = [_detect_media_resolution(s.get("file_path", "")) for s in storyboard if s.get("file_path")]
        fourk_count = sum(1 for (w, h) in resolutions if w >= 3840 or h >= 2160)
        if fourk_count > len(resolutions) / 2 and len(resolutions) > 0:
            self.output_width = 3840
            self.output_height = 2160
            logger.info("Auto-detected 4K source media dominant in storyboard; rendering at 3840x2160")
        else:
            self.output_width = 1920
            self.output_height = 1080

        temp_dir = tempfile.mkdtemp(prefix="moments_render_")
        chunk_files: List[str] = []
        chunk_durations: List[float] = []

        try:
            total_segments = len(storyboard)
            logger.info("Starting VideoCompiler for %d segments (%dx%d) -> %s", total_segments, self.output_width, self.output_height, output_file)

            # Stage 1: Pre-render individual normalized chunks
            for idx, seg in enumerate(storyboard):
                file_path = seg.get("file_path", "")
                raw_seg_type = str(seg.get("segment_type", "image")).lower()
                ext = Path(file_path).suffix.lower()
                is_video = raw_seg_type in ("video", "video_clip") or ext in (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")

                if is_video:
                    duration = min(10.0, max(1.5, float(seg.get("duration", 3.5))))
                else:
                    duration = min(5.0, max(1.0, float(seg.get("duration", 2.5))))

                start_offset = float(seg.get("start_offset", 0.0))
                chunk_name = os.path.join(temp_dir, f"chunk_{idx:03d}.mp4")
                pct = 10.0 + (float(idx) / float(total_segments)) * 60.0

                if progress_callback:
                    file_base = os.path.basename(file_path)
                    progress_callback(
                        pct,
                        "RENDERING_CHUNKS",
                        f"Rendering moment {idx + 1}/{total_segments}: {file_base} ({duration:.1f}s)",
                    )

                if not os.path.exists(file_path):
                    logger.warning("Media file not found: %s, skipping", file_path)
                    continue

                success = False
                if is_video:
                    success = self._render_video_segment(file_path, start_offset, duration, chunk_name)
                else:
                    success = self._render_image_segment(file_path, duration, chunk_name, pattern_index=idx, temp_dir=temp_dir)

                if success and os.path.exists(chunk_name) and os.path.getsize(chunk_name) > 0:
                    chunk_files.append(chunk_name)
                    dur = self._get_media_duration(chunk_name)
                    chunk_durations.append(dur)
                else:
                    logger.warning("Failed rendering chunk %d for %s", idx, file_path)

            if not chunk_files:
                raise RuntimeError("Failed to render any individual media segments.")

            # Stage 2: Join chunks with crossfades or concatenation
            if progress_callback:
                progress_callback(75.0, "COMPOSITING", "Applying transitions and compositing timeline...")

            final_dur = sum(chunk_durations)

            if len(chunk_files) == 1:
                shutil.copyfile(chunk_files[0], str(out_path))
            else:
                inputs = []
                for cf in chunk_files:
                    inputs.extend(["-i", cf])

                filter_complex_parts = []
                cur_offset = 0.0
                last_stream = "[0:v]"
                t_dur = min(0.5, self.transition_duration)

                for i in range(1, len(chunk_files)):
                    prev_dur = chunk_durations[i - 1]
                    actual_t_dur = min(t_dur, prev_dur * 0.4)
                    cur_offset += max(0.1, prev_dur - actual_t_dur)
                    next_stream = f"[{i}:v]"
                    out_stream = f"[v{i}]" if i < len(chunk_files) - 1 else "[outv]"

                    filter_complex_parts.append(
                        f"{last_stream}{next_stream}xfade=transition=fade:duration={actual_t_dur:.2f}:offset={cur_offset:.2f}{out_stream}"
                    )
                    last_stream = out_stream

                filter_complex_str = ";".join(filter_complex_parts)
                is_4k = (self.output_width >= 3840 or self.output_height >= 2160)
                encoder_args = self._get_encoder_args(is_4k=is_4k)

                cmd_join = [
                    self.ffmpeg_path,
                    "-y",
                    *inputs,
                    "-filter_complex", filter_complex_str,
                    "-map", "[outv]",
                    *encoder_args,
                    "-movflags", "+faststart",
                    str(out_path),
                ]

                logger.info("Executing FFmpeg transition composite...")
                res_join = subprocess.run(cmd_join, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if res_join.returncode != 0:
                    logger.warning("xfade filter failed, falling back to concat demuxer: %s", res_join.stderr)
                    concat_txt = os.path.join(temp_dir, "concat.txt")
                    with open(concat_txt, "w") as f:
                        for cf in chunk_files:
                            f.write(f"file '{cf}'\n")

                    cmd_fallback = [
                        self.ffmpeg_path,
                        "-y",
                        "-f", "concat",
                        "-safe", "0",
                        "-i", concat_txt,
                        "-c", "copy",
                        "-movflags", "+faststart",
                        str(out_path),
                    ]
                    subprocess.run(cmd_fallback, check=True)

            if progress_callback:
                progress_callback(100.0, "COMPLETED", "Final video rendering complete!")

            file_size_bytes = os.path.getsize(str(out_path)) if out_path.exists() else 0
            logger.info("Rendering completed successfully: %s (%d bytes)", out_path, file_size_bytes)

            return {
                "file_path": str(out_path),
                "file_name": out_path.name,
                "file_size_bytes": file_size_bytes,
                "duration_seconds": round(final_dur, 2),
                "resolution": f"{self.output_width}x{self.output_height}",
                "fps": self.fps,
                "total_segments": len(chunk_files),
            }

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
