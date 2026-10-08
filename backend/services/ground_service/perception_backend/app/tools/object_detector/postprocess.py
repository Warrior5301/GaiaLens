"""Pure postprocessing functions for Object_Detector.

Provides:
- confidence_filter()
- apply_nms()
- restore_global_coordinates()
- filter_roi()
- assign_object_ids()
"""

from __future__ import annotations

from typing import Literal, Sequence
from sac_shared_contracts.common import BBoxPx, Confidence, Detection, PointPx


ROIPolicy = Literal["center_inside", "intersection_over_roi_threshold"]


def compute_iou(box_a: BBoxPx, box_b: BBoxPx) -> float:
    """Compute Intersection over Union (IoU) between two axis-aligned bounding boxes."""
    inter_x_min = max(box_a.x_min, box_b.x_min)
    inter_y_min = max(box_a.y_min, box_b.y_min)
    inter_x_max = min(box_a.x_max, box_b.x_max)
    inter_y_max = min(box_a.y_max, box_b.y_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    union_area = box_a.area + box_b.area - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def confidence_filter(
    detections: Sequence[Detection],
    threshold: float,
) -> list[Detection]:
    """Filter detections by confidence score >= threshold."""
    return [d for d in detections if d.confidence.score >= threshold]


def apply_nms(
    detections: Sequence[Detection],
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> list[Detection]:
    """Apply standard greedy Non-Maximum Suppression to detections.

    Args:
        detections: List of detection candidates.
        iou_threshold: Overlap threshold above which lower-scoring boxes are suppressed.
        class_agnostic: If False, NMS is performed independently per object class.

    Returns:
        Suppressed list of surviving detections sorted by score descending.
    """
    if not detections:
        return []

    # Sort descending by confidence score
    sorted_dets = sorted(detections, key=lambda d: d.confidence.score, reverse=True)

    if class_agnostic:
        classes = ["all"]
        grouped = {"all": sorted_dets}
    else:
        grouped: dict[str, list[Detection]] = {}
        for d in sorted_dets:
            grouped.setdefault(d.class_name, []).append(d)

    survivors: list[Detection] = []

    for _, group in grouped.items():
        kept: list[Detection] = []
        for candidate in group:
            suppressed = False
            for existing in kept:
                iou = compute_iou(candidate.bbox_px, existing.bbox_px)
                if iou >= iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(candidate)
        survivors.extend(kept)

    # Final sort by score descending
    survivors.sort(key=lambda d: d.confidence.score, reverse=True)
    return survivors


def restore_global_coordinates(
    local_bbox: BBoxPx,
    tile_offset_x: float,
    tile_offset_y: float,
) -> BBoxPx:
    """Translate local tile coordinates back into original image coordinate space.

    Formula:
        x_global = x_local + tile_offset_x
        y_global = y_local + tile_offset_y
    """
    return BBoxPx(
        x_min=local_bbox.x_min + tile_offset_x,
        y_min=local_bbox.y_min + tile_offset_y,
        x_max=local_bbox.x_max + tile_offset_x,
        y_max=local_bbox.y_max + tile_offset_y,
    )


def filter_roi(
    detections: Sequence[Detection],
    roi: BBoxPx,
    policy: ROIPolicy = "center_inside",
    intersection_ratio_threshold: float = 0.3,
) -> list[Detection]:
    """Filter detections according to their spatial relationship to a specified ROI.

    Policies:
    - 'center_inside' (default): A detection is kept if its centroid (cx, cy)
      lies inside [roi.x_min, roi.x_max] and [roi.y_min, roi.y_max]. Recommended for counting.
    - 'intersection_over_roi_threshold': A detection is kept if the intersection area
      with the ROI relative to the detection's own bounding box area is >= threshold.
    """
    filtered: list[Detection] = []

    for d in detections:
        box = d.bbox_px
        if policy == "center_inside":
            cx = (box.x_min + box.x_max) / 2.0
            cy = (box.y_min + box.y_max) / 2.0
            if roi.x_min <= cx <= roi.x_max and roi.y_min <= cy <= roi.y_max:
                filtered.append(d)
        elif policy == "intersection_over_roi_threshold":
            inter_x_min = max(box.x_min, roi.x_min)
            inter_y_min = max(box.y_min, roi.y_min)
            inter_x_max = min(box.x_max, roi.x_max)
            inter_y_max = min(box.y_max, roi.y_max)
            inter_w = max(0.0, inter_x_max - inter_x_min)
            inter_h = max(0.0, inter_y_max - inter_y_min)
            inter_area = inter_w * inter_h
            if box.area > 0 and (inter_area / box.area) >= intersection_ratio_threshold:
                filtered.append(d)
        else:
            raise ValueError(f"Unknown ROI policy: '{policy}'")

    return filtered


def assign_object_ids(
    detections: Sequence[Detection],
    prefix: str = "det",
) -> list[Detection]:
    """Assign stable, deterministic sequential object IDs (e.g. det_001, det_002) within one tool call."""
    res: list[Detection] = []
    for i, d in enumerate(detections, start=1):
        stable_id = f"{prefix}_{i:03d}"
        res.append(
            Detection(
                object_id=stable_id,
                class_name=d.class_name,
                confidence=d.confidence,
                bbox_px=d.bbox_px,
                center_px=d.center_px or d.bbox_px.center,
                obb=d.obb,
                geolocation=d.geolocation,
            )
        )
    return res
