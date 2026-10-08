"""Artifact storage and lifecycle manager.

Provides logical abstraction over physical artifact persistence:
- Masks
- Crops
- Detections
- Feature representations
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import uuid
import numpy as np
from app.schemas.common import ArtifactRef, ArtifactType


class ArtifactStore:
    """Manages local and remote artifact serialization, retrieval, and indexing."""

    def __init__(self, root_dir: str | Path = "artifacts"):
        self.root_dir = Path(root_dir)
        self.subdirs = {
            "image": self.root_dir / "images",
            "crop": self.root_dir / "crops",
            "segmentation_mask": self.root_dir / "masks",
            "detection_set": self.root_dir / "detections",
            "polygon": self.root_dir / "polygons",
            "feature_map": self.root_dir / "features",
            "embedding": self.root_dir / "embeddings",
            "change_map": self.root_dir / "change_maps",
        }
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary artifact directory tree."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        for d in self.subdirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def save_mask(
        self,
        mask: np.ndarray,
        source_asset_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> ArtifactRef:
        """Save a binary/segmentation mask array as compressed .npy."""
        art_id = artifact_id or f"mask_{uuid.uuid4().hex[:12]}"
        dest_path = self.subdirs["segmentation_mask"] / f"{art_id}.npy"
        np.save(dest_path, mask.astype(np.uint8))

        meta = {
            "shape": list(mask.shape),
            "dtype": str(mask.dtype),
            "pixel_count": int(mask.sum()),
            **(metadata or {}),
        }

        # Save sidecar metadata JSON
        meta_path = self.subdirs["segmentation_mask"] / f"{art_id}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return ArtifactRef(
            artifact_id=art_id,
            artifact_type="segmentation_mask",
            uri=str(dest_path.as_posix()),
            source_asset_ids=[source_asset_id] if source_asset_id else [],
            metadata=meta,
        )

    def load_mask(self, artifact_ref: ArtifactRef) -> np.ndarray:
        """Load a mask array from an ArtifactRef."""
        if artifact_ref.artifact_type != "segmentation_mask":
            raise ValueError(f"Expected segmentation_mask artifact, got {artifact_ref.artifact_type}")
        path = Path(artifact_ref.uri)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found at uri: {artifact_ref.uri}")
        return np.load(path)

    def save_detection_set(
        self,
        detections_data: list[dict[str, Any]],
        source_asset_id: str | None = None,
        artifact_id: str | None = None,
    ) -> ArtifactRef:
        """Save a collection of detection dicts to JSON."""
        art_id = artifact_id or f"dets_{uuid.uuid4().hex[:12]}"
        dest_path = self.subdirs["detection_set"] / f"{art_id}.json"
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(detections_data, f, indent=2)

        return ArtifactRef(
            artifact_id=art_id,
            artifact_type="detection_set",
            uri=str(dest_path.as_posix()),
            source_asset_ids=[source_asset_id] if source_asset_id else [],
            metadata={"count": len(detections_data)},
        )
