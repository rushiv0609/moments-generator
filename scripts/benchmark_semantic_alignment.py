"""
Interactive Benchmark & Diagnostic Runner for Semantic Image-Text Alignment.
Run with: python scripts/benchmark_semantic_alignment.py
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.embedder import create_embedder
from app.core.extractor import decode_image
from app.config import get_settings

TEST_ASSETS_DIR = Path(__file__).parent.parent / "tests" / "assets" / "semantic_test"
CONFIG_FILE = TEST_ASSETS_DIR / "test_cases.json"


def find_image_file(stem: str) -> Path | None:
    for ext in [".jpg", ".jpeg", ".png", ".heic", ".HEIC", ".webp", ".tif", ".tiff"]:
        candidate = TEST_ASSETS_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    print("=" * 90)
    print(" 🎯 SIGLIP 2 CROSS-MODAL SEMANTIC ALIGNMENT & RETRIEVAL BENCHMARK")
    print("=" * 90)
    print(f"Target Assets Directory: {TEST_ASSETS_DIR}\n")

    if not CONFIG_FILE.exists():
        print(f"Error: {CONFIG_FILE} not found.")
        sys.exit(1)

    with open(CONFIG_FILE, "r") as f:
        data = json.load(f)
    test_cases = data.get("test_cases", [])

    valid_pairs = []
    missing_stems = []
    for tc in test_cases:
        img_path = find_image_file(tc["filename_stem"])
        if img_path is not None:
            valid_pairs.append({
                "id": tc["id"],
                "stem": tc["filename_stem"],
                "category": tc["category"],
                "img_path": img_path,
                "positive_prompt": tc["positive_prompt"],
                "negative_prompt": tc["negative_prompt"],
            })
        else:
            missing_stems.append(tc["filename_stem"])

    print(f"Found {len(valid_pairs)} / {len(test_cases)} test images in directory.")
    if missing_stems:
        print(f"Missing {len(missing_stems)} images: {', '.join(missing_stems[:5])}{'...' if len(missing_stems) > 5 else ''}\n")

    if len(valid_pairs) == 0:
        print(f"⚠️  Please drop your images into '{TEST_ASSETS_DIR}' with names like:")
        for tc in test_cases[:5]:
            print(f"    • {tc['filename_stem']}.jpg (or .png / .heic)")
        return

    print("Initializing SigLIP 2 Embedding Engine (Apple Silicon)...")
    t0 = time.perf_counter()
    embedder = create_embedder(get_settings())
    print(f"✓ Embedder loaded in {time.perf_counter() - t0:.2f}s | Backend: {embedder.model_info()['backend']} ({embedder.model_info().get('precision', 'fp16')})\n")

    print("Encoding images and text prompts...")
    t0 = time.perf_counter()
    decoded = [decode_image(str(item["img_path"])).pixels for item in valid_pairs]
    img_embs = embedder.embed_images(decoded)  # (N, 768)

    prompts = [item["positive_prompt"] for item in valid_pairs]
    text_embs = np.stack([embedder.embed_text(p) for p in prompts], axis=0)  # (N, 768)
    encode_time = time.perf_counter() - t0
    print(f"✓ Encoded {len(valid_pairs)} images and {len(valid_pairs)} prompts in {encode_time:.2f}s\n")

    # Compute NxN Similarity Matrix
    sim_matrix = img_embs @ text_embs.T

    print("=" * 90)
    print(f"{'#':<3} | {'Category':<22} | {'Pos Sim':<9} | {'Max Neg Sim':<11} | {'Margin':<8} | {'Top-1 Match'}")
    print("-" * 90)

    top1_count = 0
    top3_count = 0
    margins = []

    for i, item in enumerate(valid_pairs):
        pos_sim = float(sim_matrix[i, i])
        other_sims = [float(sim_matrix[i, j]) for j in range(len(valid_pairs)) if j != i]
        max_neg = max(other_sims) if other_sims else 0.0
        margin = pos_sim - max_neg
        margins.append(margin)

        # Ranking
        ranked_indices = np.argsort(-sim_matrix[i])
        rank = int(np.where(ranked_indices == i)[0][0]) + 1
        if rank == 1:
            top1_count += 1
        if rank <= 3:
            top3_count += 1

        status_str = "✅ YES (Rank #1)" if rank == 1 else f"❌ Rank #{rank} (Expected #1)"
        print(f"{item['id']:<3} | {item['category']:<22} | {pos_sim:7.4f}  | {max_neg:9.4f}   | {margin:+6.4f} | {status_str}")

    top1_acc = (top1_count / len(valid_pairs)) * 100
    top3_acc = (top3_count / len(valid_pairs)) * 100
    avg_margin = float(np.mean(margins))

    print("=" * 90)
    print(f"📊 SUMMARY RESULTS:")
    print(f"  • Top-1 Retrieval Accuracy : {top1_count}/{len(valid_pairs)} ({top1_acc:.1f}%)")
    print(f"  • Top-3 Retrieval Accuracy : {top3_count}/{len(valid_pairs)} ({top3_acc:.1f}%)")
    print(f"  • Average Positive Margin  : {avg_margin:+.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
