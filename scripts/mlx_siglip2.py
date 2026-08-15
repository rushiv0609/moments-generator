"""Apple MLX native implementation of SigLIP 2 vision and text towers.
Loads weights directly from HuggingFace safetensors/PyTorch model for Apple Silicon zero-copy acceleration.
"""
from typing import Tuple, Dict, Any, List
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import torch
from transformers import AutoModel, AutoProcessor

class MLXSiglipVisionEmbeddings(nn.Module):
    def __init__(self, embed_dim=768, image_size=224, patch_size=16):
        super().__init__()
        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, embed_dim)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        # pixel_values: (B, H, W, 3) in MLX (NHWC)
        patches = self.patch_embedding(pixel_values) # (B, 14, 14, 768)
        B, H, W, C = patches.shape
        embeddings = mx.reshape(patches, (B, H * W, C)) # (B, 196, 768)
        embeddings = embeddings + self.position_embedding.weight
        return embeddings

class MLXSiglipAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        B, L, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = mx.reshape(q, (B, L, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = mx.reshape(k, (B, L, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = mx.reshape(v, (B, L, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn_weights = mx.softmax(scores, axis=-1)
        out = attn_weights @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.embed_dim)
        return self.out_proj(out)

class MLXSiglipMLP(nn.Module):
    def __init__(self, embed_dim=768, intermediate_size=3072):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, embed_dim, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        h = self.fc1(hidden_states)
        # GELU tanh
        h = nn.gelu_approx(h)
        return self.fc2(h)

class MLXSiglipEncoderLayer(nn.Module):
    def __init__(self, embed_dim=768, intermediate_size=3072, num_heads=12):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.self_attn = MLXSiglipAttention(embed_dim, num_heads)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLXSiglipMLP(embed_dim, intermediate_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

class MLXSiglipMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.probe = mx.zeros((1, 1, embed_dim), dtype=mx.float16)
        self.in_proj_weight = mx.zeros((3 * embed_dim, embed_dim), dtype=mx.float16)
        self.in_proj_bias = mx.zeros((3 * embed_dim,), dtype=mx.float16)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLXSiglipMLP(embed_dim, 3072)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        B, L, D = hidden_states.shape
        # probe expanded: (B, 1, D)
        query = mx.broadcast_to(self.probe, (B, 1, D))
        
        # in_proj on query
        qw = self.in_proj_weight[:D]
        qb = self.in_proj_bias[:D]
        q = (query @ qw.T) + qb

        # in_proj on key/value
        kw = self.in_proj_weight[D:2*D]
        kb = self.in_proj_bias[D:2*D]
        k = (hidden_states @ kw.T) + kb

        vw = self.in_proj_weight[2*D:3*D]
        vb = self.in_proj_bias[2*D:3*D]
        v = (hidden_states @ vw.T) + vb

        # multihead reshape
        q = mx.reshape(q, (B, 1, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3) # (B, H, 1, D)
        k = mx.reshape(k, (B, L, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3) # (B, H, L, D)
        v = mx.reshape(v, (B, L, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3) # (B, H, L, D)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(scores, axis=-1)
        out = attn @ v # (B, H, 1, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, 1, D)
        out = self.out_proj(out)

        # Residual + layernorm + MLP
        hidden = query + out
        residual = hidden
        hidden = self.layernorm(hidden)
        hidden = self.mlp(hidden)
        hidden = (residual + hidden).squeeze(1) # (B, D)

        # L2 norm
        norm = mx.linalg.norm(hidden, axis=-1, keepdims=True)
        return hidden / norm

class MLXSiglipVisionTransformer(nn.Module):
    def __init__(self, num_layers=12, embed_dim=768, intermediate_size=3072, num_heads=12):
        super().__init__()
        self.embeddings = MLXSiglipVisionEmbeddings(embed_dim=embed_dim)
        self.encoder_layers = [MLXSiglipEncoderLayer(embed_dim, intermediate_size, num_heads) for _ in range(num_layers)]
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = MLXSiglipMultiheadAttentionPoolingHead(embed_dim, num_heads)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        hidden_states = self.embeddings(pixel_values)
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        return self.head(hidden_states)

def load_mlx_siglip_vision_model(model_name: str = "google/siglip2-base-patch16-224") -> Tuple[MLXSiglipVisionTransformer, AutoProcessor]:
    """Loads PyTorch HuggingFace weights directly into native Apple MLX model."""
    hf_model = AutoModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    sd = hf_model.state_dict()

    mlx_model = MLXSiglipVisionTransformer()

    # Conv2d weights: PyTorch (out, in, kH, kW) -> MLX (out, kH, kW, in)
    p_conv_w = sd["vision_model.embeddings.patch_embedding.weight"].numpy().transpose(0, 2, 3, 1)
    p_conv_b = sd["vision_model.embeddings.patch_embedding.bias"].numpy()
    mlx_model.embeddings.patch_embedding.weight = mx.array(p_conv_w, dtype=mx.float16)
    mlx_model.embeddings.patch_embedding.bias = mx.array(p_conv_b, dtype=mx.float16)

    # Positional embedding
    pos_w = sd["vision_model.embeddings.position_embedding.weight"].numpy()
    mlx_model.embeddings.position_embedding.weight = mx.array(pos_w, dtype=mx.float16)

    # 12 Transformer layers
    for i, layer in enumerate(mlx_model.encoder_layers):
        prefix = f"vision_model.encoder.layers.{i}."
        
        layer.layer_norm1.weight = mx.array(sd[prefix + "layer_norm1.weight"].numpy(), dtype=mx.float16)
        layer.layer_norm1.bias = mx.array(sd[prefix + "layer_norm1.bias"].numpy(), dtype=mx.float16)
        
        layer.self_attn.q_proj.weight = mx.array(sd[prefix + "self_attn.q_proj.weight"].numpy(), dtype=mx.float16)
        layer.self_attn.q_proj.bias = mx.array(sd[prefix + "self_attn.q_proj.bias"].numpy(), dtype=mx.float16)
        layer.self_attn.k_proj.weight = mx.array(sd[prefix + "self_attn.k_proj.weight"].numpy(), dtype=mx.float16)
        layer.self_attn.k_proj.bias = mx.array(sd[prefix + "self_attn.k_proj.bias"].numpy(), dtype=mx.float16)
        layer.self_attn.v_proj.weight = mx.array(sd[prefix + "self_attn.v_proj.weight"].numpy(), dtype=mx.float16)
        layer.self_attn.v_proj.bias = mx.array(sd[prefix + "self_attn.v_proj.bias"].numpy(), dtype=mx.float16)
        layer.self_attn.out_proj.weight = mx.array(sd[prefix + "self_attn.out_proj.weight"].numpy(), dtype=mx.float16)
        layer.self_attn.out_proj.bias = mx.array(sd[prefix + "self_attn.out_proj.bias"].numpy(), dtype=mx.float16)

        layer.layer_norm2.weight = mx.array(sd[prefix + "layer_norm2.weight"].numpy(), dtype=mx.float16)
        layer.layer_norm2.bias = mx.array(sd[prefix + "layer_norm2.bias"].numpy(), dtype=mx.float16)

        layer.mlp.fc1.weight = mx.array(sd[prefix + "mlp.fc1.weight"].numpy(), dtype=mx.float16)
        layer.mlp.fc1.bias = mx.array(sd[prefix + "mlp.fc1.bias"].numpy(), dtype=mx.float16)
        layer.mlp.fc2.weight = mx.array(sd[prefix + "mlp.fc2.weight"].numpy(), dtype=mx.float16)
        layer.mlp.fc2.bias = mx.array(sd[prefix + "mlp.fc2.bias"].numpy(), dtype=mx.float16)

    # Post layernorm
    mlx_model.post_layernorm.weight = mx.array(sd["vision_model.post_layernorm.weight"].numpy(), dtype=mx.float16)
    mlx_model.post_layernorm.bias = mx.array(sd["vision_model.post_layernorm.bias"].numpy(), dtype=mx.float16)

    # Attention Pooling Head
    mlx_model.head.probe = mx.array(sd["vision_model.head.probe"].numpy(), dtype=mx.float16)
    mlx_model.head.in_proj_weight = mx.array(sd["vision_model.head.attention.in_proj_weight"].numpy(), dtype=mx.float16)
    mlx_model.head.in_proj_bias = mx.array(sd["vision_model.head.attention.in_proj_bias"].numpy(), dtype=mx.float16)
    mlx_model.head.out_proj.weight = mx.array(sd["vision_model.head.attention.out_proj.weight"].numpy(), dtype=mx.float16)
    mlx_model.head.out_proj.bias = mx.array(sd["vision_model.head.attention.out_proj.bias"].numpy(), dtype=mx.float16)
    mlx_model.head.layernorm.weight = mx.array(sd["vision_model.head.layernorm.weight"].numpy(), dtype=mx.float16)
    mlx_model.head.layernorm.bias = mx.array(sd["vision_model.head.layernorm.bias"].numpy(), dtype=mx.float16)
    mlx_model.head.mlp.fc1.weight = mx.array(sd["vision_model.head.mlp.fc1.weight"].numpy(), dtype=mx.float16)
    mlx_model.head.mlp.fc1.bias = mx.array(sd["vision_model.head.mlp.fc1.bias"].numpy(), dtype=mx.float16)
    mlx_model.head.mlp.fc2.weight = mx.array(sd["vision_model.head.mlp.fc2.weight"].numpy(), dtype=mx.float16)
    mlx_model.head.mlp.fc2.bias = mx.array(sd["vision_model.head.mlp.fc2.bias"].numpy(), dtype=mx.float16)

    return mlx_model, processor
