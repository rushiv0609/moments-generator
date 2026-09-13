"""
Unit tests for Aesthetic and Technical Quality Scoring Module.
"""

import numpy as np
import pytest
from app.core.aesthetic import (
    compute_sharpness,
    compute_colorfulness,
    compute_contrast,
    compute_technical_quality,
    NimaScorer,
    evaluate_frame_quality,
)


def test_opencv_metrics_range_and_stability():
    # 1. Solid black image
    black_img = np.zeros((224, 224, 3), dtype=np.uint8)
    black_res = compute_technical_quality(black_img)
    assert 0.0 <= black_res["sharpness"] <= 1.0
    assert 0.0 <= black_res["colorfulness"] <= 1.0
    assert 0.0 <= black_res["contrast"] <= 1.0
    assert 0.0 <= black_res["technical_composite"] <= 1.0

    # 2. High contrast & colorful synthetic pattern
    vibrant_img = np.zeros((224, 224, 3), dtype=np.uint8)
    vibrant_img[:112, :112] = [255, 0, 0]    # Red
    vibrant_img[:112, 112:] = [0, 255, 0]    # Green
    vibrant_img[112:, :112] = [0, 0, 255]    # Blue
    vibrant_img[112:, 112:] = [255, 255, 0]  # Yellow

    vib_res = compute_technical_quality(vibrant_img)
    assert vib_res["colorfulness"] > 0.3
    assert vib_res["contrast"] > 0.5


def test_nima_batch_scoring():
    scorer = NimaScorer()
    
    # Create batch of 3 test frames
    img1 = np.full((224, 224, 3), 128, dtype=np.uint8)
    img2 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img3 = np.zeros((224, 224, 3), dtype=np.uint8)

    scores = scorer.score_batch([img1, img2, img3])
    assert len(scores) == 3
    for s in scores:
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    # Test empty batch
    assert scorer.score_batch([]) == []


def test_evaluate_frame_quality():
    img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    res = evaluate_frame_quality(img)
    
    expected_keys = [
        "sharpness",
        "colorfulness",
        "contrast",
        "technical_composite",
        "nima_aesthetic",
        "epic_composite",
    ]
    for k in expected_keys:
        assert k in res
        assert 0.0 <= res[k] <= 1.0
