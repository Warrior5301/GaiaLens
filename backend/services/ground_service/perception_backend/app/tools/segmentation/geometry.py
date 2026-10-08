"""Geometric conversions for segmentation masks: AABB, OBB, and simplified polygons."""

from __future__ import annotations

import math
from typing import Optional, Sequence
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from sac_shared_contracts.common import BBoxPx, OBB


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute pixel-level Intersection over Union between two binary masks."""
    if mask1.shape != mask2.shape:
        raise ValueError(f"Mask shapes mismatch: {mask1.shape} vs {mask2.shape}")

    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


def mask_to_aabb(mask: np.ndarray) -> BBoxPx:
    """Convert a 2D binary mask to its enclosing Axis-Aligned Bounding Box (AABB)."""
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return BBoxPx(x_min=0.0, y_min=0.0, x_max=0.0, y_max=0.0)

    x_min = float(np.min(x_indices))
    y_min = float(np.min(y_indices))
    x_max = float(np.max(x_indices) + 1.0)
    y_max = float(np.max(y_indices) + 1.0)

    return BBoxPx(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def mask_to_obb(mask: np.ndarray) -> Optional[OBB]:
    """Derive minimum-area Oriented Bounding Box (OBB) from a 2D binary mask.

    Uses 2D convex hull and rotating calipers search over candidate edge angles.
    Adheres strictly to the canonical angle convention: 'ccw_from_image_x'.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) < 3:
        # Fewer than 3 pixels: fallback to regular AABB as unrotated OBB
        if len(y_indices) == 0:
            return None
        aabb = mask_to_aabb(mask)
        w = max(1.0, aabb.x_max - aabb.x_min)
        h = max(1.0, aabb.y_max - aabb.y_min)
        return OBB(
            cx=(aabb.x_min + aabb.x_max) / 2.0,
            cy=(aabb.y_min + aabb.y_max) / 2.0,
            width=w,
            height=h,
            angle_deg=0.0,
            angle_convention="ccw_from_image_x",
        )

    # Subsample if mask has many points to speed up convex hull
    points = np.column_stack((x_indices.astype(np.float32), y_indices.astype(np.float32)))
    if len(points) > 1000:
        # Extract boundary points first (pixels with at least one 0 neighbor)
        struct = ndimage.generate_binary_structure(2, 1)
        eroded = ndimage.binary_erosion(mask.astype(bool), structure=struct)
        boundary = np.logical_and(mask.astype(bool), ~eroded)
        by, bx = np.where(boundary > 0)
        if len(by) >= 3:
            points = np.column_stack((bx.astype(np.float32), by.astype(np.float32)))

    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
    except Exception:
        hull_points = points

    # Minimum area rectangle search over hull edges
    min_area = float("inf")
    best_rect: Optional[tuple[float, float, float, float, float]] = None  # (cx, cy, w, h, angle_deg)

    num_pts = len(hull_points)
    for i in range(num_pts):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % num_pts]

        edge = p2 - p1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-6:
            continue

        # Angle of edge relative to X-axis in image space
        theta = math.atan2(edge[1], edge[0])

        # Rotation matrix to project points onto edge coordinate system
        cos_t = math.cos(-theta)
        sin_t = math.sin(-theta)
        rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

        rot_pts = np.dot(hull_points, rot_mat.T)
        min_x, max_x = np.min(rot_pts[:, 0]), np.max(rot_pts[:, 0])
        min_y, max_y = np.min(rot_pts[:, 1]), np.max(rot_pts[:, 1])

        width = max_x - min_x
        height = max_y - min_y
        area = width * height

        if area < min_area:
            min_area = area
            # Center in rotated system
            center_rot = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0], dtype=np.float32)
            # Rotate center back to original coordinate system
            inv_rot_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float32)
            center_orig = np.dot(center_rot, inv_rot_mat.T)

            # Convert theta to degrees CCW from image X
            angle_deg = math.degrees(theta) % 180.0
            best_rect = (float(center_orig[0]), float(center_orig[1]), float(width), float(height), float(angle_deg))

    if best_rect is None:
        aabb = mask_to_aabb(mask)
        return OBB(
            cx=(aabb.x_min + aabb.x_max) / 2.0,
            cy=(aabb.y_min + aabb.y_max) / 2.0,
            width=max(1.0, aabb.x_max - aabb.x_min),
            height=max(1.0, aabb.y_max - aabb.y_min),
            angle_deg=0.0,
            angle_convention="ccw_from_image_x",
        )

    cx, cy, w, h, angle_deg = best_rect
    return OBB(
        cx=cx,
        cy=cy,
        width=max(1.0, w),
        height=max(1.0, h),
        angle_deg=round(angle_deg, 2),
        angle_convention="ccw_from_image_x",
    )


def mask_to_polygon(
    mask: np.ndarray,
    tolerance: float = 1.5,
    max_points: int = 120,
) -> list[list[float]]:
    """Extract simplified boundary polygon from 2D binary mask.

    Args:
        mask: 2D uint8 or boolean array.
        tolerance: Ramer-Douglas-Peucker simplification tolerance in pixels.
        max_points: Maximum vertices in returned polygon.

    Returns:
        List of [x, y] coordinates forming a closed polygon.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    # Extract perimeter boundary pixels
    struct = ndimage.generate_binary_structure(2, 1)
    eroded = ndimage.binary_erosion(mask.astype(bool), structure=struct)
    boundary = np.logical_and(mask.astype(bool), ~eroded)

    by, bx = np.where(boundary > 0)
    if len(by) == 0:
        return []

    # Sort perimeter points by polar angle around centroid for a closed loop
    cx = float(np.mean(bx))
    cy = float(np.mean(by))
    angles = np.arctan2(by - cy, bx - cx)
    sort_idx = np.argsort(angles)

    raw_polygon = np.column_stack((bx[sort_idx], by[sort_idx]))

    # Simplify using Ramer-Douglas-Peucker
    simplified = _ramer_douglas_peucker(raw_polygon, epsilon=tolerance)

    if len(simplified) > max_points:
        # Uniform subsample down to max_points
        step = len(simplified) / max_points
        simplified = [simplified[int(i * step)] for i in range(max_points)]

    # Ensure closed loop
    if len(simplified) >= 3:
        if simplified[0] != simplified[-1]:
            simplified.append(simplified[0])

    return [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in simplified]


def _ramer_douglas_peucker(points: np.ndarray, epsilon: float) -> list[list[float]]:
    """Ramer-Douglas-Peucker 2D curve simplification."""
    if len(points) <= 2:
        return points.tolist()

    dmax = 0.0
    index = 0
    p1 = points[0]
    p2 = points[-1]

    # Line vector
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-6:
        dists = np.linalg.norm(points[1:-1] - p1, axis=1)
        if len(dists) > 0:
            dmax = float(np.max(dists))
            index = int(np.argmax(dists)) + 1
    else:
        # Perpendicular distance formula
        cross_prod = np.abs(
            line_vec[0] * (p1[1] - points[1:-1, 1]) - line_vec[1] * (p1[0] - points[1:-1, 0])
        )
        dists = cross_prod / line_len
        if len(dists) > 0:
            dmax = float(np.max(dists))
            index = int(np.argmax(dists)) + 1

    if dmax > epsilon:
        rec_results1 = _ramer_douglas_peucker(points[: index + 1], epsilon)
        rec_results2 = _ramer_douglas_peucker(points[index:], epsilon)
        return rec_results1[:-1] + rec_results2
    else:
        return [points[0].tolist(), points[-1].tolist()]
