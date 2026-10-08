"""Unit tests for thresholding, morphological cleanup, and connected components extraction."""

from __future__ import annotations

import numpy as np
import pytest
from app.tools.segmentation.clustering import extract_connected_components
from app.tools.segmentation.postprocess import (
    cleanup_binary_mask,
    logits_to_binary_mask,
    sigmoid,
)


def test_sigmoid():
    logits = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    probs = sigmoid(logits)
    assert probs[0] < 0.001
    assert pytest.approx(probs[1], 0.01) == 0.5
    assert probs[2] > 0.999


def test_logits_to_binary_mask():
    logits = np.array([
        [-5.0, -2.0, 0.0],
        [1.0, 5.0, 10.0],
    ], dtype=np.float32)

    # threshold 0.5 means logits >= 0.0 become 1
    binary = logits_to_binary_mask(logits, threshold=0.5, apply_sigmoid=True)
    assert binary.shape == (2, 3)
    assert binary.dtype == np.uint8
    assert np.array_equal(binary, [
        [0, 0, 1],
        [1, 1, 1],
    ])


def test_cleanup_binary_mask_holes_and_noise():
    # 20x20 canvas with a 10x10 object with a 2x2 hole in the middle, and an isolated 2-pixel noise blob
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:12, 2:12] = 1
    mask[5:7, 5:7] = 0  # Interior hole
    mask[18:19, 18:20] = 1  # 2-pixel isolated noise

    cleaned = cleanup_binary_mask(mask, min_area_pixels=10, fill_holes=True)
    # Hole should be filled
    assert np.all(cleaned[5:7, 5:7] == 1)
    # Noise should be removed
    assert np.all(cleaned[18:20, 18:20] == 0)
    # Main component intact
    assert np.sum(cleaned) == 100


def test_extract_connected_components():
    mask = np.zeros((50, 50), dtype=np.uint8)
    # Component 1: 10x10 = 100 pixels
    mask[5:15, 5:15] = 1
    # Component 2: 5x5 = 25 pixels
    mask[30:35, 30:35] = 1
    # Noise: 2x2 = 4 pixels (should be ignored with min_area=10)
    mask[40:42, 40:42] = 1

    components = extract_connected_components(mask, min_area_pixels=10)
    assert len(components) == 2

    comp1, comp2 = components[0], components[1]
    # Sorted by area descending
    assert comp1.area_pixels == 100
    assert comp2.area_pixels == 25

    # Check bounding box for comp1
    assert comp1.bbox_px.x_min == 5.0
    assert comp1.bbox_px.y_min == 5.0
    assert comp1.bbox_px.x_max == 15.0
    assert comp1.bbox_px.y_max == 15.0

    # Check centroid for comp1
    cx, cy = comp1.centroid
    assert pytest.approx(cx, abs=0.5) == 9.5
    assert pytest.approx(cy, abs=0.5) == 9.5
