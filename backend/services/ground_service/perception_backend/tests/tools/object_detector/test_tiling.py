"""Unit tests for sliding window tiling strategy and coordinate merging."""

from __future__ import annotations

from PIL import Image
from app.tools.object_detector.tiling import TilingStrategy
from sac_shared_contracts.common import BBoxPx, Confidence, Detection


def test_tiling_strategy_tile_generation():
    strategy = TilingStrategy(tile_size_px=512, overlap_ratio=0.2)
    # Image 1024x1024
    tiles = strategy.generate_tiles(image_width=1024, image_height=1024)
    assert len(tiles) >= 4

    # Verify tile bounds are all within image canvas
    for t in tiles:
        assert 0.0 <= t.bbox_px.x_min < t.bbox_px.x_max <= 1024.0
        assert 0.0 <= t.bbox_px.y_min < t.bbox_px.y_max <= 1024.0


def test_tiled_inference_restores_global_coordinates():
    strategy = TilingStrategy(tile_size_px=500, overlap_ratio=0.1, iou_threshold=0.5)
    img = Image.new("RGB", (1000, 1000), color=(128, 128, 128))

    # Mock detector returning 1 detection per patch at local coordinate (50, 50, 150, 150)
    def mock_patch_detector(patch: Image.Image) -> list[Detection]:
        box = BBoxPx(x_min=50.0, y_min=50.0, x_max=150.0, y_max=150.0)
        return [
            Detection(
                object_id="patch_det",
                class_name="ship",
                confidence=Confidence(score=0.9, source="patch_test"),
                bbox_px=box,
            )
        ]

    merged_dets, metrics = strategy.run_tiled_inference(img, mock_patch_detector)
    assert len(merged_dets) > 0
    assert metrics["tile_count"] > 1

    # Check that restored global boxes have coordinates reflecting offsets > 0 for non-initial tiles
    max_x = max(d.bbox_px.x_max for d in merged_dets)
    max_y = max(d.bbox_px.y_max for d in merged_dets)
    assert max_x > 150.0
    assert max_y > 150.0
