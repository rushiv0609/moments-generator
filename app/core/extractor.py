"""
Hardware-Accelerated Frame Extraction & Photo Decoding Engine for Local AI Moments Generator.
Uses Apple Native ImageIO (PyObjC) for photos and OpenCV for videos to produce 224x224 RGB
arrays for SigLIP 2 vision embeddings, while preserving original high-resolution file paths.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Generator, Optional, Tuple
import numpy as np
from PIL import Image
import pillow_heif
import cv2

# Apple Native ImageIO via PyObjC (macOS hardware-accelerated decoder)
try:
    import Quartz
    from Foundation import NSURL
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False

pillow_heif.register_heif_opener()


@dataclass
class FrameData:
    """
    A decoded frame ready for SigLIP 2 vision embedding.
    NOTE: `pixels` is strictly for semantic AI embedding (224x224).
    The original high-resolution file is preserved at `file_path` for rendering.
    """
    file_path: str          # Original file path (preserved for final video rendering)
    frame_index: int        # 0 for photos, 0..N for video frames
    pixels: np.ndarray      # Shape: (224, 224, 3), dtype: uint8, RGB
    source_offset: float    # Seconds into source video (0.0 for photos)
    file_type: str = "image"  # 'image' | 'video'
    scene_id: Optional[int] = None        # Scene ID if extracted from a video
    scene_start: Optional[float] = None   # Timestamp start of the scene
    scene_end: Optional[float] = None     # Timestamp end of the scene


def pad_to_square(arr: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Pad an image array to a square (target_size x target_size x 3) with zero-padding (black bars)
    to preserve 100% of the original photo/frame composition without any cropping.
    """
    h, w, c = arr.shape
    if h == target_size and w == target_size:
        return arr

    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)

    # Pad symmetrically or top-left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = np.pad(
        arr,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return padded[:target_size, :target_size, :3]


def decode_image_imageio(file_path: str, target_size: int = 224) -> Optional[np.ndarray]:
    """
    Hardware-accelerated thumbnail generation using Apple ImageIO (CGImageSource).
    Decodes directly to target thumbnail size on the Apple Silicon media engine.
    """
    if not HAS_IMAGEIO:
        return None

    try:
        url = NSURL.fileURLWithPath_(str(file_path))
        src = Quartz.CGImageSourceCreateWithURL(url, None)
        if not src:
            return None

        options = {
            Quartz.kCGImageSourceCreateThumbnailWithTransform: True,
            Quartz.kCGImageSourceCreateThumbnailFromImageAlways: True,
            Quartz.kCGImageSourceThumbnailMaxPixelSize: target_size,
            Quartz.kCGImageSourceShouldCache: False,
        }

        cg_img = Quartz.CGImageSourceCreateThumbnailAtIndex(src, 0, options)
        if not cg_img:
            return None

        w = Quartz.CGImageGetWidth(cg_img)
        h = Quartz.CGImageGetHeight(cg_img)
        dp = Quartz.CGImageGetDataProvider(cg_img)
        data = Quartz.CGDataProviderCopyData(dp)
        
        # Buffer is RGBA or RGB
        arr = np.frombuffer(bytes(data), dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
        return pad_to_square(arr, target_size)
    except Exception:
        return None


def decode_image_pillow(file_path: str, target_size: int = 224) -> np.ndarray:
    """
    Fallback image decoder using Pillow and pillow-heif.
    Scales aspect-ratio preserved and applies zero-padding to target_size.
    """
    with Image.open(file_path) as img:
        img = img.convert("RGB")
        img.thumbnail((target_size, target_size), Image.Resampling.BICUBIC)
        arr = np.array(img, dtype=np.uint8)
        return pad_to_square(arr, target_size)


def decode_image(file_path: str, target_size: int = 224) -> FrameData:
    """
    Decode an image (HEIC, JPEG, PNG, WEBP, TIFF) to a 224x224 RGB numpy array.
    Uses Apple ImageIO hardware acceleration where available, with Pillow fallback.
    """
    path_str = str(Path(file_path).resolve())
    pixels = decode_image_imageio(path_str, target_size)
    if pixels is None:
        pixels = decode_image_pillow(path_str, target_size)

    return FrameData(
        file_path=path_str,
        frame_index=0,
        pixels=pixels,
        source_offset=0.0,
        file_type="image",
    )


def extract_video_frames_av(
    file_path: str,
    target_size: int = 224,
    sampling_fps: float = 1.0,
) -> Generator[FrameData, None, None]:
    """
    High-performance, C-level multi-threaded video frame decoding using PyAV (FFmpeg).
    Releases the Python GIL during decoding and scaling in libavcodec / libswscale.
    """
    path_str = str(Path(file_path).resolve())
    with av.open(path_str) as container:
        video_streams = [s for s in container.streams if s.type == "video"]
        if not video_streams:
            raise ValueError(f"No video streams found in '{file_path}'")

        stream = video_streams[0]
        stream.thread_type = "AUTO"  # C-level multi-threaded codec decoding

        rate = stream.guessed_rate or stream.average_rate or stream.base_rate or 30
        fps = float(rate) if rate else 30.0

        frame_interval = max(1, int(round(fps / sampling_fps))) if (fps and fps > 0) else 30
        frame_idx = 0
        extracted_count = 0

        for frame in container.decode(stream):
            if frame.is_corrupt:
                frame_idx += 1
                continue

            if frame_idx % frame_interval == 0:
                if frame.time is not None:
                    sec = float(frame.time)
                elif frame.pts is not None and stream.time_base:
                    sec = float(frame.pts * stream.time_base)
                else:
                    sec = float(frame_idx / fps) if fps > 0 else float(extracted_count)

                # Scale & convert to RGB24 inside libswscale (C layer)
                rgb_frame = frame.reformat(width=target_size, height=target_size, format="rgb24")
                arr = rgb_frame.to_ndarray()

                yield FrameData(
                    file_path=path_str,
                    frame_index=extracted_count,
                    pixels=arr,
                    source_offset=round(sec, 3),
                    file_type="video",
                )
                extracted_count += 1

            frame_idx += 1


def extract_video_frames_cv2(
    file_path: str,
    target_size: int = 224,
    sampling_fps: float = 1.0,
) -> Generator[FrameData, None, None]:
    """
    Fallback video frame extractor using OpenCV with AVFoundation acceleration.
    """
    path_str = str(Path(file_path).resolve())
    cap = cv2.VideoCapture(path_str, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        raise ValueError(f"Unable to open video file at '{file_path}'")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(fps / sampling_fps)) if (fps and fps > 0) else 30
        frame_idx = 0
        extracted_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                resized = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                source_offset = frame_idx / fps if (fps and fps > 0) else float(extracted_count)

                yield FrameData(
                    file_path=path_str,
                    frame_index=extracted_count,
                    pixels=rgb,
                    source_offset=round(source_offset, 3),
                    file_type="video",
                )
                extracted_count += 1

            frame_idx += 1
    finally:
        cap.release()


def extract_video_frames(
    file_path: str,
    target_size: int = 224,
    sampling_fps: float = 1.0,
) -> Generator[FrameData, None, None]:
    """
    Extract video frames sampled at sampling_fps (default 1.0 FPS), scaled to target_size (224x224 RGB).
    Prefers macOS native AVFoundation hardware decoding (OpenCV); falls back to PyAV C-level decoder.
    """
    cv2_err = None
    try:
        yield from extract_video_frames_cv2(file_path, target_size, sampling_fps)
        return
    except Exception as e:
        cv2_err = e

    if HAS_AV:
        try:
            yield from extract_video_frames_av(file_path, target_size, sampling_fps)
            return
        except Exception as av_err:
            raise ValueError(f"Unable to open video file at '{file_path}': {av_err}") from av_err

    raise ValueError(f"Unable to open video file at '{file_path}': {cv2_err}")
