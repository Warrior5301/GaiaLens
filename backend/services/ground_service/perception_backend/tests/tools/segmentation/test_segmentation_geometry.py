"""Unit tests for geometric conversions: AABB, OBB, polygon, and mask IoU."""

from __future__ import annotations

import numpy as np
import pytest
from app.tools.segmentation.geometry import (
    compute_mask_iou,
    mask_to_aabb,
    mask_to_obb,
    mask_to_polygon,
)


def test_compute_mask_iou():
    m1 = np.zeros((20, 20), dtype=np.uint8)
    m2 = np.zeros((20, 20), dtype=np.uint8)

    m1[5:15, 5:15] = 1  # 10x10 = 100
    m2[5:15, 10:20] = 1  # 10x10 = 100, overlaps on 5:15, 10:15 = 50

    # intersection = 50, union = 150 -> IoU = 50/150 = 1/3
    iou = compute_mask_iou(m1, m2)
    assert pytest.approx(iou, 0.001) == 1.0 / 3.0

    # Identical masks
    assert compute_mask_iou(m1, m1) == 1.0

    # Disjoint masks
    m3 = np.zeros((20, 20), dtype=np.uint8)
    m3[0:2, 0:2] = 1
    assert compute_mask_iou(m1, m3) == 0.0


def test_mask_to_aabb():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:30, 15:40] = 1

    bbox = mask_to_aabb(mask)
    assert bbox.x_min == 15.0
    assert bbox.y_min == 10.0
    assert bbox.x_max == 40.0
    assert bbox.y_max == 30.0

    # Empty mask
    empty = np.zeros((20, 20), dtype=np.uint8)
    empty_bbox = mask_to_aabb(empty)
    assert empty_bbox.to_xyxy() == [0.0, 0.0, 0.0, 0.0]


def test_mask_to_obb():
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Axis-aligned rectangle: y: 20-60 (h=40), x: 30-70 (w=40)
    mask[20:60, 30:70] = 1

    obb = mask_to_obb(mask)
    assert obb is not None
    assert obb.angle_convention == "ccw_from_image_x"
    assert pytest.approx(obb.cx, abs=1.0) == 50.0
    assert pytest.approx(obb.cy, abs=1.0) == 40.0
    assert pytest.approx(obb.width, abs=2.0) == 40.0
    assert pytest.approx(obb.height, abs=2.0) == 40.0


def test_mask_to_polygon():
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[10:40, 10:40] = 1

    poly = mask_to_polygon(mask, tolerance=1.0, max_points=20)
    assert len(poly) >= 4
    # Ensure closed loop: first and last point match
    assert poly[0] == poly[-1]
    # Points should lie roughly on boundaries
    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]
    assert min(xs) >= 10.0 and max(xs) <= 40.0
    assert min(ys) >= 10.0 and max(ys) <= 40.0
