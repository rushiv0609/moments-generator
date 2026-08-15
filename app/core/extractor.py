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


def extract_video_frames(
    file_path: str,
    target_size: int = 224,
    sampling_fps: float = 1.0,
) -> Generator[FrameData, None, None]:
    """
    Extract video frames sampled at 1.0 FPS, scaled to target_size (224x224 RGB)
    for semantic embedding. Streams frames lazily as a generator.
    """
    path_str = str(Path(file_path).resolve())
    cap = cv2.VideoCapture(path_str)

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
                # Resize frame to target_size (squish to 224x224 or aspect scale)
                resized = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
                # Convert BGR (OpenCV) to RGB
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
