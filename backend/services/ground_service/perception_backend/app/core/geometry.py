"""Geometrical transformation and spatial evaluation utilities.

Implements:
- OBB to 4 corner points (canonical CCW from image X-axis)
- OBB to Axis-Aligned Bounding Box (AABB) envelope for VRS-Bench adapter
- AABB IoU computation
- Coordinate boundary clipping and normalization
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np
from app.schemas.common import BBoxPx, OBB, PointPx


def obb_to_corners(obb: OBB) -> list[tuple[float, float]]:
    """Convert an Oriented Bounding Box to 4 ordered corner coordinates.

    Canonical convention:
    - angle_deg is counter-clockwise (CCW) rotation in degrees from positive image X-axis.
    - Image coordinates: origin (0, 0) is top-left, X is rightwards, Y is downwards.
    In image coordinates where Y points down:
      A CCW rotation in standard Cartesian space maps to:
      dx' = dx * cos(theta) + dy * sin(theta)  (accounting for Y-down)
      Specifically:
      rotation matrix R(theta) for vector [dx, dy]^T:
      [ x_rot ] = [  cos(theta)  -sin(theta) ] [ dx ]
      [ y_rot ]   [  sin(theta)   cos(theta) ] [ dy ]
      with theta in radians.

    Returns:
        List of 4 (x, y) tuples: [top-left, top-right, bottom-right, bottom-left] before rotation.
    """
    rad = math.radians(obb.angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    hw = obb.width / 2.0
    hh = obb.height / 2.0

    # Local corners relative to center (unrotated)
    local_corners = [
        (-hw, -hh),  # corner 0
        (hw, -hh),   # corner 1
        (hw, hh),    # corner 2
        (-hw, hh),   # corner 3
    ]

    rotated_corners: list[tuple[float, float]] = []
    for dx, dy in local_corners:
        rx = obb.cx + (dx * cos_a - dy * sin_a)
        ry = obb.cy + (dx * sin_a + dy * cos_a)
        rotated_corners.append((rx, ry))

    return rotated_corners


def obb_to_aabb(obb: OBB, clip_bounds: tuple[int, int] | None = None) -> BBoxPx:
    """Convert an Oriented Bounding Box into its minimal enclosing Axis-Aligned Bounding Box.

    Used by the VRS-Bench adapter to feed rotated production detections into
    axis-aligned benchmark evaluators.

    Args:
        obb: The oriented bounding box.
        clip_bounds: Optional (max_width, max_height) to clamp coordinates inside image bounds.

    Returns:
        BBoxPx envelope.
    """
    corners = obb_to_corners(obb)
    xs = [pt[0] for pt in corners]
    ys = [pt[1] for pt in corners]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    if clip_bounds is not None:
        max_w, max_h = clip_bounds
        x_min = max(0.0, min(float(max_w), x_min))
        y_min = max(0.0, min(float(max_h), y_min))
        x_max = max(0.0, min(float(max_w), x_max))
        y_max = max(0.0, min(float(max_h), y_max))

    return BBoxPx(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def compute_aabb_iou(box1: BBoxPx | Sequence[float], box2: BBoxPx | Sequence[float]) -> float:
    """Compute Intersection over Union (IoU) between two Axis-Aligned Bounding Boxes.

    Accepts BBoxPx instances or [x_min, y_min, x_max, y_max] sequences.
    """
    if isinstance(box1, BBoxPx):
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.x_min, box1.y_min, box1.x_max, box1.y_max
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])

    if isinstance(box2, BBoxPx):
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.x_min, box2.y_min, box2.x_max, box2.y_max
    else:
        b2_x1, b2_y1, b2_x2, b2_y2 = float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area1 = max(0.0, b1_x2 - b1_x1) * max(0.0, b1_y2 - b1_y1)
    area2 = max(0.0, b2_x2 - b2_x1) * max(0.0, b2_y2 - b2_y1)

    union = area1 + area2 - intersection
    if union <= 0.0:
        return 0.0

    return float(intersection / union)


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute pixel-level IoU between two boolean or binary 2D mask arrays."""
    if mask1.shape != mask2.shape:
        raise ValueError(f"Mask shapes mismatch: {mask1.shape} vs {mask2.shape}")

    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)
