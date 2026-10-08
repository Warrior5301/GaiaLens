"""Unit tests for geometry calculations and OBB/AABB conversions."""

from __future__ import annotations

import math
import numpy as np
import pytest
from app.core.geometry import (
    compute_aabb_iou,
    compute_mask_iou,
    obb_to_aabb,
    obb_to_corners,
)
from app.schemas.common import BBoxPx, OBB


def test_obb_to_corners_unrotated():
    obb = OBB(cx=100.0, cy=100.0, width=40.0, height=20.0, angle_deg=0.0)
    corners = obb_to_corners(obb)

    # 4 corners of unrotated box around (100, 100)
    # hw=20, hh=10
    assert len(corners) == 4
    assert pytest.approx(corners[0][0]) == 80.0
    assert pytest.approx(corners[0][1]) == 90.0
    assert pytest.approx(corners[1][0]) == 120.0
    assert pytest.approx(corners[1][1]) == 90.0
    assert pytest.approx(corners[2][0]) == 120.0
    assert pytest.approx(corners[2][1]) == 110.0
    assert pytest.approx(corners[3][0]) == 80.0
    assert pytest.approx(corners[3][1]) == 110.0


def test_obb_to_aabb_envelope():
    # 90-degree rotated box width=40, height=20 becomes AABB width=20, height=40
    obb = OBB(cx=100.0, cy=100.0, width=40.0, height=20.0, angle_deg=90.0)
    aabb = obb_to_aabb(obb)

    assert pytest.approx(aabb.width, abs=1e-3) == 20.0
    assert pytest.approx(aabb.height, abs=1e-3) == 40.0
    assert pytest.approx(aabb.x_min, abs=1e-3) == 90.0
    assert pytest.approx(aabb.y_min, abs=1e-3) == 80.0


def test_aabb_iou_computation():
    box1 = BBoxPx(x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0)  # Area 100
    box2 = BBoxPx(x_min=5.0, y_min=0.0, x_max=15.0, y_max=10.0)  # Area 100
    # Intersection = 5 * 10 = 50. Union = 100 + 100 - 50 = 150. IoU = 50/150 = 1/3
    iou = compute_aabb_iou(box1, box2)
    assert pytest.approx(iou, rel=1e-4) == 1.0 / 3.0

    # Disjoint boxes
    box3 = BBoxPx(x_min=20.0, y_min=20.0, x_max=30.0, y_max=30.0)
    assert compute_aabb_iou(box1, box3) == 0.0

    # Identical boxes
    assert compute_aabb_iou(box1, box1) == 1.0


def test_mask_iou_computation():
    m1 = np.zeros((10, 10), dtype=bool)
    m2 = np.zeros((10, 10), dtype=bool)

    m1[0:5, 0:5] = True  # 25 pixels
    m2[0:5, 0:5] = True  # 25 pixels
    assert compute_mask_iou(m1, m2) == 1.0

    m2[5:10, 5:10] = True  # 25 + 25 pixels
    # intersection = 25, union = 50 -> 0.5
    assert compute_mask_iou(m1, m2) == 0.5
