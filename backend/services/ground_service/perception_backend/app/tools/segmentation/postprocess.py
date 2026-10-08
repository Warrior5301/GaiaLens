"""Logit processing, thresholding, and morphological cleanup for segmentation masks."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Compute element-wise sigmoid on logits."""
    # Prevent overflow in exp
    clipped = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def logits_to_binary_mask(
    logits_or_probs: np.ndarray,
    threshold: float = 0.5,
    apply_sigmoid: bool = True,
) -> np.ndarray:
    """Convert raw model logits or probabilities to a uint8 binary mask (0 or 1).

    Args:
        logits_or_probs: 2D numpy array of float logits or probabilities.
        threshold: Decision threshold for positive mask assignment.
        apply_sigmoid: Whether to apply sigmoid first (set False if already probabilities).

    Returns:
        2D binary numpy array of dtype uint8 containing 0s and 1s.
    """
    if logits_or_probs.ndim != 2:
        raise ValueError(f"Expected 2D mask array, got shape {logits_or_probs.shape}")

    probs = sigmoid(logits_or_probs) if apply_sigmoid else logits_or_probs
    binary = (probs >= threshold).astype(np.uint8)
    return binary


def cleanup_binary_mask(
    binary_mask: np.ndarray,
    min_area_pixels: int = 10,
    fill_holes: bool = True,
    opening_size: int = 0,
) -> np.ndarray:
    """Apply morphological cleanup to binary mask: fill holes, remove isolated noise.

    Args:
        binary_mask: 2D uint8 or boolean array.
        min_area_pixels: Connected components smaller than this area are eliminated as noise.
        fill_holes: If True, fill interior holes inside solid object components.
        opening_size: Structuring element size for morphological opening (0 = disabled).

    Returns:
        Cleaned 2D uint8 binary mask.
    """
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {binary_mask.shape}")

    mask = binary_mask.astype(bool)

    # 1. Morphological opening to sever thin bridges if requested
    if opening_size > 0:
        struct = ndimage.generate_binary_structure(2, 1)
        if opening_size > 1:
            struct = ndimage.iterate_structure(struct, opening_size)
        mask = ndimage.binary_opening(mask, structure=struct)

    # 2. Fill interior holes if requested
    if fill_holes and np.any(mask):
        mask = ndimage.binary_fill_holes(mask)

    # 3. Filter out components smaller than min_area_pixels
    if min_area_pixels > 1 and np.any(mask):
        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            # Keep labels whose area >= min_area_pixels
            keep_labels = set(
                idx + 1 for idx, size in enumerate(component_sizes) if size >= min_area_pixels
            )
            filtered = np.isin(labeled, list(keep_labels))
            mask = filtered

    return mask.astype(np.uint8)
