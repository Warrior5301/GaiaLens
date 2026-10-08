"""Tiled inference strategy for large high-resolution remote-sensing imagery.

Implements sliding window patch generation, local inference execution,
global coordinate restoration, and cross-patch Non-Maximum Suppression (SAHI-style).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
from PIL import Image
from sac_shared_contracts.common import BBoxPx, Detection
from app.tools.object_detector.postprocess import apply_nms, restore_global_coordinates


@dataclass(frozen=True)
class TileWindow:
    """Specification of a single image patch window in global pixel coordinates."""
    tile_index: int
    bbox_px: BBoxPx

    @property
    def x_offset(self) -> float:
        return self.bbox_px.x_min

    @property
    def y_offset(self) -> float:
        return self.bbox_px.y_min


class TilingStrategy:
    """Orchestrates tiled patch generation, batch prediction, and seamless coordinate merging."""

    def __init__(
        self,
        tile_size_px: int = 512,
        overlap_ratio: float = 0.2,
        iou_threshold: float = 0.5,
    ) -> None:
        self.tile_size_px = max(64, tile_size_px)
        self.overlap_ratio = max(0.0, min(0.9, overlap_ratio))
        self.overlap_px = int(self.tile_size_px * self.overlap_ratio)
        self.iou_threshold = iou_threshold

    def generate_tiles(self, image_width: int, image_height: int) -> list[TileWindow]:
        """Generate sliding window tiles covering the entire image canvas."""
        step = max(1, self.tile_size_px - self.overlap_px)
        tiles: list[TileWindow] = []
        idx = 0

        y = 0
        while y < image_height:
            y_max = min(image_height, y + self.tile_size_px)
            # Adjust last tile upwards if it extends beyond image height
            y_min = max(0, y_max - self.tile_size_px) if y_max == image_height else y

            x = 0
            while x < image_width:
                x_max = min(image_width, x + self.tile_size_px)
                # Adjust last tile leftwards if it extends beyond image width
                x_min = max(0, x_max - self.tile_size_px) if x_max == image_width else x

                box = BBoxPx(
                    x_min=float(x_min),
                    y_min=float(y_min),
                    x_max=float(x_max),
                    y_max=float(y_max),
                )
                tiles.append(TileWindow(tile_index=idx, bbox_px=box))
                idx += 1

                if x_max >= image_width:
                    break
                x += step

            if y_max >= image_height:
                break
            y += step

        return tiles

    def run_tiled_inference(
        self,
        image: Image.Image,
        patch_detector_fn: Callable[[Image.Image], Sequence[Detection]],
    ) -> tuple[list[Detection], dict[str, int | float]]:
        """Execute sliding-window inference over the image.

        Args:
            image: Full-scale PIL image.
            patch_detector_fn: Callable that accepts a patch PIL Image and returns local Detections.

        Returns:
            Tuple of:
            - Final merged and NMS-suppressed Detections in global image coordinates.
            - Metrics dict with tile count, total raw boxes, and post-NMS boxes.
        """
        w, h = image.size
        tiles = self.generate_tiles(image_width=w, image_height=h)

        all_global_detections: list[Detection] = []

        for tile in tiles:
            # Crop local image patch
            box = tile.bbox_px
            patch = image.crop((int(box.x_min), int(box.y_min), int(box.x_max), int(box.y_max)))

            # Run detection on local patch
            local_dets = patch_detector_fn(patch)

            # Restore global coordinates
            for d in local_dets:
                global_box = restore_global_coordinates(
                    local_bbox=d.bbox_px,
                    tile_offset_x=tile.x_offset,
                    tile_offset_y=tile.y_offset,
                )
                all_global_detections.append(
                    Detection(
                        object_id=d.object_id,
                        class_name=d.class_name,
                        confidence=d.confidence,
                        bbox_px=global_box,
                        center_px=global_box.center,
                        obb=d.obb,
                        geolocation=d.geolocation,
                    )
                )

        raw_count = len(all_global_detections)

        # Cross-tile NMS to merge duplicate boundary detections
        merged_detections = apply_nms(
            detections=all_global_detections,
            iou_threshold=self.iou_threshold,
            class_agnostic=False,
        )

        metrics = {
            "tile_count": len(tiles),
            "tile_size_px": self.tile_size_px,
            "raw_tile_detections": raw_count,
            "merged_detections": len(merged_detections),
        }

        return merged_detections, metrics
