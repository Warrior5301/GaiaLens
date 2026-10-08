"""Unit tests for DBSCAN fragmented mask grouping and spatial clustering."""

from __future__ import annotations

import numpy as np
import pytest
from app.tools.segmentation.clustering import (
    cluster_components_dbscan,
    extract_connected_components,
)


def test_dbscan_groups_nearby_fragmented_components():
    """Verify that fragmented pieces of an object (e.g. ship split in two) merge via DBSCAN."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Piece A: (y: 20-35, x: 20-40)
    mask[20:35, 20:40] = 1
    # Piece B: (y: 40-55, x: 20-40) - separated by only 5 pixels gap
    mask[40:55, 20:40] = 1

    components = extract_connected_components(mask, min_area_pixels=10)
    assert len(components) == 2  # Initially 2 disjoint components

    # Cluster with eps=30 pixels (gap between centroids is ~20 px)
    clusters = cluster_components_dbscan(
        components,
        eps=30.0,
        min_samples=1,
        image_shape=(100, 100),
    )

    assert len(clusters) == 1  # Grouped into 1 coherent candidate cluster
    cluster = clusters[0]
    assert len(cluster.member_components) == 2
    assert cluster.area_pixels == (15 * 20) + (15 * 20)
    assert cluster.bbox_px.x_min == 20.0
    assert cluster.bbox_px.y_min == 20.0
    assert cluster.bbox_px.x_max == 40.0
    assert cluster.bbox_px.y_max == 55.0


def test_dbscan_separates_distant_objects():
    """Verify that distant objects remain in distinct clusters."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Object 1 at top-left
    mask[10:30, 10:30] = 1
    # Object 2 at bottom-right
    mask[160:180, 160:180] = 1

    components = extract_connected_components(mask, min_area_pixels=10)
    assert len(components) == 2

    clusters = cluster_components_dbscan(
        components,
        eps=40.0,  # 40 px eps << 200 px distance
        min_samples=1,
        image_shape=(200, 200),
    )

    assert len(clusters) == 2
    assert clusters[0].cluster_id != clusters[1].cluster_id


def test_dbscan_empty_input():
    clusters = cluster_components_dbscan([], eps=40.0, min_samples=1, image_shape=(100, 100))
    assert clusters == []
