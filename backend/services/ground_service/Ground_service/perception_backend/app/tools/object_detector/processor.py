"""Input processor and semantic validation for Object_Detector.

Validates input contracts, validates that ROI lies strictly within image bounds,
and prepares the image canvas for full-frame or patch inference.
"""

from __future__ import annotations

from typing import Optional
from PIL import Image
from app.core.vision_runtime import VisionRuntime
from sac_shared_contracts.common import BBoxPx
from sac_shared_contracts.roi import ROI
from sac_shared_contracts.perception import ObjectDetectorInput
from sac_shared_contracts.tool_protocol import ToolError, ToolErrorCode


class ObjectDetectorProcessor:
    """Preprocesses input imagery and performs strict semantic validation."""

    @staticmethod
    def extract_roi_bbox(roi_input: Optional[BBoxPx | ROI]) -> Optional[BBoxPx]:
        """Normalize ROI input into a canonical BBoxPx."""
        if roi_input is None:
            return None
        if isinstance(roi_input, BBoxPx):
            return roi_input
        if hasattr(roi_input, "bbox_px"):
            return roi_input.bbox_px
        raise ValueError(f"Unsupported ROI type: {type(roi_input)}")

    @classmethod
    def validate_inputs(
        cls,
        input_data: ObjectDetectorInput,
        image_width: int,
        image_height: int,
    ) -> Optional[ToolError]:
        """Perform strict semantic input validation.

        Returns:
            ToolError if validation fails, or None if valid.
        """
        # 1. Validate image dimensions
        if image_width <= 0 or image_height <= 0:
            return ToolError(
                code=ToolErrorCode.INVALID_INPUT,
                message=f"Invalid asset dimensions: width={image_width}, height={image_height}",
                retryable=False,
            )

        # 2. Validate confidence threshold
        if not (0.0 <= input_data.confidence_threshold <= 1.0):
            return ToolError(
                code=ToolErrorCode.INVALID_INPUT,
                message=f"Confidence threshold {input_data.confidence_threshold} must be in [0.0, 1.0]",
                retryable=False,
            )

        # 3. Validate tiling overlap
        tiling = input_data.tiling_config
        if tiling.enabled:
            if tiling.tile_size_px <= 0:
                return ToolError(
                    code=ToolErrorCode.INVALID_INPUT,
                    message=f"tile_size_px ({tiling.tile_size_px}) must be > 0",
                    retryable=False,
                )
            if tiling.overlap_px < 0 or tiling.overlap_px >= tiling.tile_size_px:
                return ToolError(
                    code=ToolErrorCode.INVALID_INPUT,
                    message=f"overlap_px ({tiling.overlap_px}) must be in [0, tile_size_px)",
                    retryable=False,
                )

        # 4. Strict ROI bounds validation (Section 6: return structured INVALID_ROI, do NOT silently clip)
        roi_box = cls.extract_roi_bbox(input_data.roi)
        if roi_box is not None:
            if roi_box.x_min < 0.0 or roi_box.y_min < 0.0:
                return ToolError(
                    code=ToolErrorCode.INVALID_ROI,
                    message=(
                        f"ROI exceeds image bounds: [x_min={roi_box.x_min}, y_min={roi_box.y_min}] "
                        f"is less than image origin (0, 0)"
                    ),
                    retryable=False,
                    details={"roi": roi_box.model_dump(), "image_size": [image_width, image_height]},
                )
            if roi_box.x_max > float(image_width) or roi_box.y_max > float(image_height):
                return ToolError(
                    code=ToolErrorCode.INVALID_ROI,
                    message=(
                        f"ROI exceeds image bounds: [x_max={roi_box.x_max}, y_max={roi_box.y_max}] "
                        f"exceeds image dimensions [{image_width}, {image_height}]"
                    ),
                    retryable=False,
                    details={"roi": roi_box.model_dump(), "image_size": [image_width, image_height]},
                )

        return None

    @classmethod
    def prepare_image(
        cls,
        input_data: ObjectDetectorInput,
        runtime: VisionRuntime,
    ) -> tuple[Image.Image, Optional[BBoxPx]]:
        """Load image and apply initial ROI sub-region crop if specified.

        Returns:
            Tuple of (loaded_image, canonical_roi_bbox)
        """
        image = runtime.load_image(input_data.asset)
        roi_box = cls.extract_roi_bbox(input_data.roi)
        return image, roi_box
