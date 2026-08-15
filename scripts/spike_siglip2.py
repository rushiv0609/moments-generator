"""Spike: Verify SigLIP 2 loads and produces embeddings on this machine.
Run this right after Milestone 1 to confirm model viability.
"""
import time
import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

print("=== SigLIP 2 Spike Test ===")

# 1. Try MLX
mlx_ok = False
try:
    import mlx.core as mx
    print(f"✓ MLX imported (version: {mx.__version__})")
    mlx_ok = True
except ImportError:
    print("✗ MLX not available — will use PyTorch MPS")

# 2. Test PyTorch MPS
print(f"PyTorch version: {torch.__version__}")
mps_available = torch.backends.mps.is_available()
print(f"MPS available: {mps_available}")
assert mps_available, "MPS not available — Apple Silicon GPU acceleration required"

# 3. Load SigLIP 2 via transformers
model_name = "google/siglip2-base-patch16-224"
print(f"\nLoading {model_name} via Transformers + PyTorch MPS...")
t0 = time.time()

model = AutoModel.from_pretrained(model_name, dtype=torch.float16)
model = model.to("mps")
model.eval()
print(f"✓ Model loaded in {time.time() - t0:.2f}s")

processor = AutoProcessor.from_pretrained(model_name)

# 4. Test text embedding
text_prompts = ["a beautiful mountain landscape with snowy peaks", "people dancing at a fun party"]
text_inputs = processor(text=text_prompts, padding="max_length", return_tensors="pt").to("mps")

with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    # If returned as model output object with pooler_output or tensor:
    if hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
        text_emb = text_features.pooler_output
    elif hasattr(text_features, "last_hidden_state"):
        text_emb = text_features.last_hidden_state[:, 0]
    elif torch.is_tensor(text_features):
        text_emb = text_features
    else:
        text_emb = text_features[0]
    
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

print(f"✓ Text embeddings (2 prompts) shape: {text_emb.shape}")

# 5. Test image embedding
dummy_img_1 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
dummy_img_2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
image_inputs = processor(images=[dummy_img_1, dummy_img_2], return_tensors="pt").to("mps")

with torch.no_grad():
    image_features = model.get_image_features(**image_inputs)
    if hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
        img_emb = image_features.pooler_output
    elif hasattr(image_features, "last_hidden_state"):
        img_emb = image_features.last_hidden_state[:, 0]
    elif torch.is_tensor(image_features):
        img_emb = image_features
    else:
        img_emb = image_features[0]

    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

print(f"✓ Image embeddings (2 images) shape: {img_emb.shape}")

# 6. Test cosine similarity matrix
sim_matrix = (text_emb @ img_emb.T).cpu().numpy()
print(f"✓ Text-Image similarity matrix shape: {sim_matrix.shape}")
print(f"  Similarity values:\n  {sim_matrix}")

# 7. Memory usage
rss_mb = os.popen(f'ps -o rss= -p {os.getpid()}').read().strip()
if rss_mb:
    print(f"\nProcess RSS Memory: {int(rss_mb) // 1024} MB")
print(f"Embedding dimension: {text_emb.shape[-1]}")
print("\n=== Spike PASSED ✓ ===")
