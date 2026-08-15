"""
Cross-Modal Semantic Alignment & Image-Text Retrieval Test Suite for SigLIP 2.
Evaluates cosine similarity matrix between diverse reference images and corresponding prompts.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from app.core.embedder import create_embedder
from app.core.extractor import decode_image
from app.config import Settings

TEST_ASSETS_DIR = Path(__file__).parent / "assets" / "semantic_test"
CONFIG_FILE = TEST_ASSETS_DIR / "test_cases.json"


def load_test_cases():
    if not CONFIG_FILE.exists():
        return []
    with open(CONFIG_FILE, "r") as f:
        data = json.load(f)
    return data.get("test_cases", [])


def find_image_file(stem: str) -> Path | None:
    """Find image file matching the stem with any supported extension."""
    for ext in [".jpg", ".jpeg", ".png", ".heic", ".HEIC", ".webp", ".tif", ".tiff"]:
        candidate = TEST_ASSETS_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


@pytest.fixture(scope="module")
def embedder():
    settings = Settings(MODEL_NAME="google/siglip2-base-patch16-224", MODEL_BACKEND="auto")
    return create_embedder(settings)


def test_semantic_cross_modal_alignment_matrix(embedder):
    """
    Evaluates Top-1 cross-modal retrieval precision and positive-vs-negative margins
    for all available test images in tests/assets/semantic_test/.
    """
    test_cases = load_test_cases()
    if not test_cases:
        pytest.skip("No test cases defined in test_cases.json")

    valid_pairs = []
    for tc in test_cases:
        img_path = find_image_file(tc["filename_stem"])
        if img_path is not None:
            valid_pairs.append({
                "id": tc["id"],
                "stem": tc["filename_stem"],
                "category": tc["category"],
                "img_path": str(img_path),
                "positive_prompt": tc["positive_prompt"],
                "negative_prompt": tc["negative_prompt"],
            })

    if len(valid_pairs) == 0:
        pytest.skip(f"No test images found in {TEST_ASSETS_DIR}. Place images matching test_cases.json to run.")

    print(f"\n--- Testing Semantic Alignment on {len(valid_pairs)} Loaded Images ---")

    # 1. Decode and embed images
    decoded_images = [decode_image(item["img_path"]).pixels for item in valid_pairs]
    img_embeddings = embedder.embed_images(decoded_images)  # (N, 768)

    # 2. Embed all positive text prompts
    text_prompts = [item["positive_prompt"] for item in valid_pairs]
    text_embeddings = np.stack([embedder.embed_text(p) for p in text_prompts], axis=0)  # (N, 768)

    # 3. Compute NxN Cross-Modal Cosine Similarity Matrix
    sim_matrix = img_embeddings @ text_embeddings.T  # (N, N)

    top1_correct = 0
    margins = []

    print(f"\n{'#':<3} | {'Category':<22} | {'Pos Sim':<9} | {'Max Neg Sim':<11} | {'Margin':<8} | {'Top-1 Match'}")
    print("-" * 75)

    for i, item in enumerate(valid_pairs):
        pos_sim = float(sim_matrix[i, i])
        
        # Negative similarities (all other prompts)
        other_sims = [float(sim_matrix[i, j]) for j in range(len(valid_pairs)) if j != i]
        max_neg = max(other_sims) if other_sims else 0.0
        margin = pos_sim - max_neg
        margins.append(margin)

        # Check if the correct positive prompt is ranked #1 for this image
        predicted_idx = int(np.argmax(sim_matrix[i]))
        is_top1 = (predicted_idx == i)
        if is_top1:
            top1_correct += 1

        status_str = "✅ YES" if is_top1 else f"❌ NO (Ranked #{sorted(sim_matrix[i], reverse=True).index(pos_sim)+1})"
        print(f"{item['id']:<3} | {item['category']:<22} | {pos_sim:7.4f}  | {max_neg:9.4f}   | {margin:+6.4f} | {status_str}")

    top1_acc = (top1_correct / len(valid_pairs)) * 100
    avg_margin = float(np.mean(margins))

    print("-" * 75)
    print(f"Top-1 Retrieval Accuracy : {top1_correct}/{len(valid_pairs)} ({top1_acc:.1f}%)")
    print(f"Average Positive Margin  : {avg_margin:+.4f}")
    print("-" * 75)

    # Mathematical assertions
    assert top1_acc >= 80.0, f"Top-1 accuracy {top1_acc:.1f}% is below 80% threshold"
    assert avg_margin > 0.05, f"Average margin {avg_margin:.4f} is too low"
