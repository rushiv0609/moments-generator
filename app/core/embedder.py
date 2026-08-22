"""
High-Performance Vision-Language Embedding Engine for SigLIP 2 on Apple Silicon.
Provides strategy pattern supporting Apple MLX JIT-kernel-fused inference (primary)
and PyTorch MPS inference (fallback).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from app.config import get_settings, Settings

# Try importing Apple MLX
try:
    import mlx.core as mx
    import mlx.nn as nn
    from app.core.mlx_siglip2 import load_mlx_siglip_vision_model
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class EmbedderInterface(ABC):
    """Abstract interface for multi-modal embedding engines."""

    @abstractmethod
    def embed_images(self, batch_pixels: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Embed a batch of (224, 224, 3) uint8 RGB image arrays.
        Returns: (N, 768) float32 numpy array, L2-normalized.
        """
        pass

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a natural language text query.
        Returns: (768,) float32 numpy array, L2-normalized.
        """
        pass

    @abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """Return engine metadata and model details."""
        pass

    @abstractmethod
    def empty_cache(self) -> None:
        """Clear GPU/accelerator memory caches."""
        pass


class MLXEmbedder(EmbedderInterface):
    """
    Apple MLX implementation with @mx.compile JIT Kernel Fusion for SigLIP 2.
    Achieves maximum throughput and near-zero memory footprint on Apple Silicon.
    """

    def __init__(self, model_name: str = "google/siglip2-base-patch16-224", precision: str = "fp16"):
        if not HAS_MLX:
            raise RuntimeError("Apple MLX is not installed or supported on this system.")

        self.model_name = model_name
        self.precision = precision.lower()
        self.embedding_dim = 768

        # 1. Load MLX vision model and tokenizer
        self.raw_vision_model, self.processor = load_mlx_siglip_vision_model(model_name)

        # 2. Apply 8-bit quantization if requested
        if self.precision == "8bit":
            nn.quantize(self.raw_vision_model, group_size=64, bits=8, class_predicate=lambda _, m: isinstance(m, nn.Linear))

        # 3. JIT-compile the forward pass Metal compute graph
        @mx.compile
        def _compiled_forward(x: mx.array) -> mx.array:
            return self.raw_vision_model(x)

        self._forward_fn = _compiled_forward

        # 4. Load PyTorch model on CPU for text encoder
        self.pt_model = AutoModel.from_pretrained(model_name)
        self.pt_model.eval()

        # Warmup Metal JIT compilation with dummy tensor
        dummy = mx.zeros((1, 224, 224, 3), dtype=mx.float16)
        _ = self._forward_fn(dummy)
        mx.eval(_)

    def embed_images(self, batch_pixels: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        if isinstance(batch_pixels, list):
            if len(batch_pixels) == 0:
                return np.empty((0, self.embedding_dim), dtype=np.float32)
            batch_arr = np.stack(batch_pixels, axis=0)
        else:
            batch_arr = batch_pixels

        if batch_arr.ndim == 3:
            batch_arr = batch_arr[np.newaxis, ...]

        # Input normalisation: (x / 255.0 - 0.5) / 0.5
        norm_pixels = (batch_arr.astype(np.float32) / 255.0 - 0.5) / 0.5
        mlx_input = mx.array(norm_pixels, dtype=mx.float16)

        # Fused Metal GPU execution
        out = self._forward_fn(mlx_input)
        mx.eval(out)
        res = np.array(out, dtype=np.float32)
        # Ensure exact unit-length float32 normalisation
        norms = np.linalg.norm(res, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return res / norms

    def embed_text(self, text: str) -> np.ndarray:
        # Prompt normalization for vision-language alignment
        prompt = text.strip()
        if not (prompt.startswith("a photo of") or prompt.startswith("a picture of") or prompt.startswith("a video of")):
            prompt = f"a photo of {prompt}"

        inputs = self.processor(text=[prompt], padding="max_length", max_length=64, return_tensors="pt")
        with torch.inference_mode():
            emb = self.pt_model.get_text_features(**inputs)
            vec = emb.pooler_output if hasattr(emb, "pooler_output") else (emb[0] if isinstance(emb, (tuple, list)) else emb)
            vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.cpu().numpy().reshape(-1).astype(np.float32)

    def model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model_name,
            "backend": "mlx",
            "precision": self.precision,
            "embedding_dim": self.embedding_dim,
            "fused_kernel": True,
        }

    def empty_cache(self) -> None:
        if HAS_MLX:
            mx.metal.clear_cache()


class PyTorchMPSEmbedder(EmbedderInterface):
    """
    HuggingFace transformers + PyTorch MPS fallback backend.
    """

    def __init__(self, model_name: str = "google/siglip2-base-patch16-224"):
        self.model_name = model_name
        self.embedding_dim = 768
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.model.eval()

    def embed_images(self, batch_pixels: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        if isinstance(batch_pixels, list):
            if len(batch_pixels) == 0:
                return np.empty((0, self.embedding_dim), dtype=np.float32)
            batch_arr = np.stack(batch_pixels, axis=0)
        else:
            batch_arr = batch_pixels

        if batch_arr.ndim == 3:
            batch_arr = batch_arr[np.newaxis, ...]

        # Convert (B, H, W, C) to (B, C, H, W) for PyTorch
        batch_arr = batch_arr.transpose(0, 3, 1, 2)
        norm_pixels = ((torch.from_numpy(batch_arr).to(device=self.device, dtype=torch.float16) / 255.0) - 0.5) / 0.5

        with torch.inference_mode():
            feats = self.model.get_image_features(pixel_values=norm_pixels)
            vec = feats.pooler_output if hasattr(feats, "pooler_output") else (feats[0] if isinstance(feats, (tuple, list)) else feats)
            vec = vec / vec.norm(dim=-1, keepdim=True)
            if self.device == "mps":
                torch.mps.synchronize()

        return vec.cpu().numpy().astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        prompt = text.strip()
        if not (prompt.startswith("a photo of") or prompt.startswith("a picture of") or prompt.startswith("a video of")):
            prompt = f"a photo of {prompt}"

        inputs = self.processor(text=[prompt], padding="max_length", max_length=64, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            emb = self.model.get_text_features(**inputs)
            vec = emb.pooler_output if hasattr(emb, "pooler_output") else (emb[0] if isinstance(emb, (tuple, list)) else emb)
            vec = vec / vec.norm(dim=-1, keepdim=True)
            if self.device == "mps":
                torch.mps.synchronize()
        return vec.cpu().numpy().reshape(-1).astype(np.float32)

    def model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model_name,
            "backend": "pytorch_mps" if self.device == "mps" else "pytorch_cpu",
            "precision": "fp16",
            "embedding_dim": self.embedding_dim,
            "fused_kernel": False,
        }


def create_embedder(settings: Optional[Settings] = None) -> EmbedderInterface:
    """
    Factory creating the optimal embedding engine based on system hardware and config.
    Defaults to Apple MLX on macOS Apple Silicon, falling back to PyTorch MPS.
    """
    s = settings or get_settings()
    backend = s.MODEL_BACKEND

    if backend == "mlx":
        return MLXEmbedder(model_name=s.MODEL_NAME, precision=s.MODEL_PRECISION)
    elif backend == "pytorch_mps":
        return PyTorchMPSEmbedder(model_name=s.MODEL_NAME)
    else:  # "auto"
        if HAS_MLX:
            try:
                return MLXEmbedder(model_name=s.MODEL_NAME, precision=s.MODEL_PRECISION)
            except Exception:
                pass
        return PyTorchMPSEmbedder(model_name=s.MODEL_NAME)
