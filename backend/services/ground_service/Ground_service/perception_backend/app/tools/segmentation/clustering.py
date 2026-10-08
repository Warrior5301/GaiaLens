"""Connected component analysis and DBSCAN clustering for fragmented mask grouping."""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np
from scipy import ndimage
from sac_shared_contracts.common import BBoxPx
from app.tools.segmentation.candidate import CandidateRegion, DBSCANCluster

logger = logging.getLogger("app.tools.segmentation.clustering")


def extract_connected_components(
    binary_mask: np.ndarray,
    min_area_pixels: int = 10,
    connectivity: int = 8,
) -> list[CandidateRegion]:
    """Extract connected components from a 2D binary mask.

    Args:
        binary_mask: 2D uint8 or bool array (H, W).
        min_area_pixels: Minimum number of positive pixels to consider a component.
        connectivity: Pixel connectivity: 4 or 8.

    Returns:
        List of CandidateRegion dataclasses.
    """
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {binary_mask.shape}")

    mask = binary_mask.astype(bool)
    if not np.any(mask):
        return []

    # 4- or 8-connectivity structuring element
    struct = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    labeled, num_features = ndimage.label(mask, structure=struct)
    if num_features == 0:
        return []

    slices = ndimage.find_objects(labeled)
    components: list[CandidateRegion] = []

    for i, slc in enumerate(slices):
        if slc is None:
            continue
        comp_id = i + 1
        comp_mask_slice = labeled[slc] == comp_id
        area = int(np.sum(comp_mask_slice))
        if area < min_area_pixels:
            continue

        y_slice, x_slice = slc
        x_min = float(x_slice.start)
        y_min = float(y_slice.start)
        x_max = float(x_slice.stop)
        y_max = float(y_slice.stop)

        # Full-sized component mask
        comp_full_mask = (labeled == comp_id)

        # Centroid
        cy, cx = ndimage.center_of_mass(comp_full_mask)

        bbox = BBoxPx(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

        components.append(
            CandidateRegion(
                region_id=f"comp_{comp_id:04d}",
                component_id=comp_id,
                mask=comp_full_mask,
                area_pixels=area,
                bbox_px=bbox,
                centroid=(float(cx), float(cy)),
            )
        )

    # Sort descending by area
    components.sort(key=lambda c: c.area_pixels, reverse=True)
    return components


def cluster_components_dbscan(
    components: list[CandidateRegion],
    eps: float = 40.0,
    min_samples: int = 1,
    image_shape: tuple[int, int] = (512, 512),
    promote_noise_components: bool = True,
) -> list[DBSCANCluster]:
    """Group fragmented mask components into coherent candidate clusters using DBSCAN.

    Per the GeoNLI research paper:
    RemoteSAM logits produce fragmented predictions for large/occluded objects.
    DBSCAN clustering spatial proximity groups fragmented pieces into unified candidate regions.

    Args:
        components: List of CandidateRegion objects.
        eps: DBSCAN maximum distance between component centroids (in pixels).
        min_samples: DBSCAN minimum samples per core point.
        image_shape: (height, width) of the parent image.
        promote_noise_components: If True, noise points (label -1) are treated as individual clusters.

    Returns:
        List of DBSCANCluster objects.
    """
    if not components:
        return []

    # If only 1 component, return as a single cluster directly
    if len(components) == 1:
        comp = components[0]
        return [
            DBSCANCluster(
                cluster_id=1,
                member_components=[comp],
                mask=comp.mask.copy(),
                bbox_px=comp.bbox_px,
                area_pixels=comp.area_pixels,
                centroid=comp.centroid,
            )
        ]

    # Coordinate matrix of centroids for clustering: shape (N, 2)
    coords = np.array([[c.centroid[0], c.centroid[1]] for c in components], dtype=np.float32)

    try:
        from sklearn.cluster import DBSCAN

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = clustering.labels_
    except Exception as exc:
        logger.warning("sklearn DBSCAN failed or unavailable (%s), using distance-fallback", exc)
        labels = _fallback_spatial_clustering(coords, eps=eps)

    # Group components by label
    cluster_groups: dict[int, list[CandidateRegion]] = {}
    for comp, label in zip(components, labels):
        if label == -1:
            if promote_noise_components:
                # Assign a unique negative key for each standalone noise component
                noise_key = -(len(cluster_groups) + 100)
                cluster_groups[noise_key] = [comp]
            else:
                continue
        else:
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(comp)

    # Build DBSCANCluster objects
    result_clusters: list[DBSCANCluster] = []
    h, w = image_shape

    for cluster_idx, (lbl, members) in enumerate(cluster_groups.items(), start=1):
        merged_mask = np.zeros((h, w), dtype=bool)
        total_area = 0
        weighted_cx = 0.0
        weighted_cy = 0.0

        x_min = float(w)
        y_min = float(h)
        x_max = 0.0
        y_max = 0.0

        for m in members:
            merged_mask = np.logical_or(merged_mask, m.mask)
            total_area += m.area_pixels
            weighted_cx += m.centroid[0] * m.area_pixels
            weighted_cy += m.centroid[1] * m.area_pixels

            x_min = min(x_min, m.bbox_px.x_min)
            y_min = min(y_min, m.bbox_px.y_min)
            x_max = max(x_max, m.bbox_px.x_max)
            y_max = max(y_max, m.bbox_px.y_max)

        if total_area > 0:
            final_centroid = (weighted_cx / total_area, weighted_cy / total_area)
        else:
            final_centroid = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

        cluster_bbox = BBoxPx(
            x_min=max(0.0, x_min),
            y_min=max(0.0, y_min),
            x_max=min(float(w), x_max),
            y_max=min(float(h), y_max),
        )

        result_clusters.append(
            DBSCANCluster(
                cluster_id=cluster_idx,
                member_components=members,
                mask=merged_mask,
                bbox_px=cluster_bbox,
                area_pixels=total_area,
                centroid=final_centroid,
                metadata={
                    "member_count": len(members),
                    "original_label": int(lbl),
                    "eps": float(eps),
                    "min_samples": int(min_samples),
                },
            )
        )

    # Sort descending by area
    result_clusters.sort(key=lambda cl: cl.area_pixels, reverse=True)
    return result_clusters


def _fallback_spatial_clustering(coords: np.ndarray, eps: float) -> np.ndarray:
    """Pure numpy/Python single-linkage Euclidean proximity clustering fallback."""
    n = len(coords)
    labels = -np.ones(n, dtype=int)
    current_label = 0

    for i in range(n):
        if labels[i] != -1:
            continue
        labels[i] = current_label
        # Breadth-first search for neighbors
        queue = [i]
        while queue:
            curr = queue.pop(0)
            dists = np.linalg.norm(coords - coords[curr], axis=1)
            neighbors = np.where(dists <= eps)[0]
            for nb in neighbors:
                if labels[nb] == -1:
                    labels[nb] = current_label
                    queue.append(nb)
        current_label += 1

    return labels
