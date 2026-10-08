"""Class vocabulary discovery and controlled normalization mapping for Object_Detector.

Separates three distinct conceptual layers:
1. User Phrase / Prompt Token (e.g. "commercial airplanes", "small cargo vessels")
2. Normalized Semantic Class (e.g. "aircraft", "maritime_vessel")
3. Detector Class (e.g. "airplane", "ship")
"""

from __future__ import annotations

import re
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class ClassMappingResult(BaseModel):
    """Result of mapping a free-form user query/phrase to detector classes."""
    model_config = ConfigDict(extra="forbid")

    original_phrase: str = Field(description="Raw input phrase or expression")
    semantic_target: Optional[str] = Field(default=None, description="Normalized semantic domain concept")
    detector_class: Optional[str] = Field(default=None, description="Physical detector category name")
    is_supported: bool = Field(description="Whether the detector supports localization of this concept")
    confidence: float = Field(ge=0.0, le=1.0, description="Mapping rule confidence")
    notes: Optional[str] = Field(default=None, description="Mapping diagnostics or caveats")


class ClassMapper:
    """Controlled normalization and discovery engine for remote-sensing object detection."""

    # 1. Closed-set categories natively supported by the detector backend
    DETECTOR_CLASSES: set[str] = {
        "airplane",
        "ship",
        "vehicle",
        "storage_tank",
        "baseball_diamond",
        "tennis_court",
        "basketball_court",
        "ground_track_field",
        "harbor",
        "bridge",
        "building",
        "container_crane",
        "helicopter",
    }

    # 2. Semantic Target -> Detector Class
    SEMANTIC_TO_DETECTOR: dict[str, str] = {
        "aircraft": "airplane",
        "airplane": "airplane",
        "fixed_wing": "airplane",
        "rotorcraft": "airplane",
        "maritime_vessel": "ship",
        "ship": "ship",
        "boat": "ship",
        "ground_vehicle": "vehicle",
        "automobile": "vehicle",
        "truck": "vehicle",
        "bus": "vehicle",
        "industrial_facility": "storage_tank",
        "storage_tank": "storage_tank",
        "oil_tank": "storage_tank",
        "fuel_tank": "storage_tank",
        "sports_facility": "tennis_court",
        "tennis_court": "tennis_court",
        "baseball_field": "baseball_diamond",
        "baseball_diamond": "baseball_diamond",
        "athletic_track": "ground_track_field",
        "ground_track_field": "ground_track_field",
        "transportation_infrastructure": "bridge",
        "bridge": "bridge",
        "building": "building",
        "structure": "building",
        "port_crane": "container_crane",
        "harbor": "harbor",
    }

    # 3. Controlled phrase pattern dictionary (regex token patterns -> semantic target)
    PHRASE_PATTERNS: list[tuple[re.Pattern, str, float]] = [
        # Aircraft patterns
        (re.compile(r"\b(commercial\s+)?(airplane|aeroplane|aircraft|airliner|jet|airbus|boeing)s?\b", re.I), "aircraft", 0.98),
        (re.compile(r"\b(cargo|passenger|military)\s+(plane|aircraft)s?\b", re.I), "aircraft", 0.95),
        (re.compile(r"\b(helicopter|chopper)s?\b", re.I), "aircraft", 0.90),

        # Maritime patterns
        (re.compile(r"\b(cargo|container|freight|oil|naval|military|war|fishing)?\s*(ship|vessel|boat|barge)s?\b", re.I), "maritime_vessel", 0.98),
        (re.compile(r"\b(tanker|cruiser|destroyer|frigate|ferry|yacht)s?\b", re.I), "maritime_vessel", 0.95),

        # Vehicle patterns
        (re.compile(r"\b(car|automobile|sedan|suv|truck|lorry|bus|van|vehicle)s?\b", re.I), "ground_vehicle", 0.96),
        (re.compile(r"\b(parked\s+)?vehicles?\b", re.I), "ground_vehicle", 0.98),

        # Storage tanks
        (re.compile(r"\b(oil|fuel|petroleum|gas|water|chemical|storage)\s+(tank|container|reservoir)s?\b", re.I), "storage_tank", 0.97),
        (re.compile(r"\b(storage\s+tank)s?\b", re.I), "storage_tank", 0.98),

        # Sports and fields
        (re.compile(r"\b(tennis\s+court)s?\b", re.I), "tennis_court", 0.98),
        (re.compile(r"\b(baseball\s+(diamond|field))s?\b", re.I), "baseball_diamond", 0.98),
        (re.compile(r"\b(running|ground|athletic)\s+track\s*(field)?s?\b", re.I), "ground_track_field", 0.95),

        # Infrastructure
        (re.compile(r"\b(suspension\s+|highway\s+|rail\s+)?bridge(s)?\b", re.I), "bridge", 0.95),
        (re.compile(r"\b(container\s+crane|gantry\s+crane|harbor\s+crane)s?\b", re.I), "port_crane", 0.95),
        (re.compile(r"\b(building|warehouse|hangar|terminal|house)s?\b", re.I), "structure", 0.92),
    ]

    # Explicit list of unsupported patterns that represent non-detection tasks
    UNSUPPORTED_PATTERNS: list[tuple[re.Pattern, str]] = [
        (re.compile(r"\b(average\s+height|elevation|depth|dem)\b", re.I), "3D/DEM height estimation is not a 2D detector capability"),
        (re.compile(r"\b(land[- ]cover|vegetation\s+type|landuse)\b", re.I), "Land-cover classification requires semantic segmentation"),
        (re.compile(r"\b(cloud\s+cover|fog|haze|weather)\b", re.I), "Atmospheric assessment is routed to Image_Analyzer"),
        (re.compile(r"\b(color|colour|painted|hue)\b", re.I), "Visual attribute question is routed to Image_Analyzer"),
        (re.compile(r"\b(total\s+area|perimeter|length\s+of)\b", re.I), "Metric calculation requires Geo_Measurement"),
    ]

    @classmethod
    def map_phrase(cls, phrase: str) -> ClassMappingResult:
        """Map a free-form phrase or question token into normalized semantic and detector classes."""
        clean_phrase = phrase.strip()

        # Check explicitly unsupported reasoning patterns first
        for pat, reason in cls.UNSUPPORTED_PATTERNS:
            if pat.search(clean_phrase):
                return ClassMappingResult(
                    original_phrase=clean_phrase,
                    semantic_target=None,
                    detector_class=None,
                    is_supported=False,
                    confidence=0.95,
                    notes=reason,
                )

        # 1. Exact match with detector class
        lower_token = clean_phrase.lower()
        if lower_token in cls.DETECTOR_CLASSES:
            return ClassMappingResult(
                original_phrase=clean_phrase,
                semantic_target=lower_token,
                detector_class=lower_token,
                is_supported=True,
                confidence=1.0,
                notes="Exact match against known detector class",
            )

        # 2. Check controlled pattern matches
        for pattern, semantic_target, conf in cls.PHRASE_PATTERNS:
            match = pattern.search(clean_phrase)
            if match:
                detector_class = cls.SEMANTIC_TO_DETECTOR.get(semantic_target)
                is_supported = detector_class in cls.DETECTOR_CLASSES if detector_class else False
                return ClassMappingResult(
                    original_phrase=clean_phrase,
                    semantic_target=semantic_target,
                    detector_class=detector_class,
                    is_supported=is_supported,
                    confidence=conf,
                    notes=f"Matched controlled pattern '{pattern.pattern}'",
                )

        # 3. Fallback: Check if any detector class occurs as a whole word
        for det_class in cls.DETECTOR_CLASSES:
            word_pattern = rf"\b{re.escape(det_class)}s?\b"
            if re.search(word_pattern, clean_phrase, re.I):
                return ClassMappingResult(
                    original_phrase=clean_phrase,
                    semantic_target=det_class,
                    detector_class=det_class,
                    is_supported=True,
                    confidence=0.85,
                    notes="Whole-word match fallback",
                )

        # Unrecognized / unsupported class
        return ClassMappingResult(
            original_phrase=clean_phrase,
            semantic_target=None,
            detector_class=None,
            is_supported=False,
            confidence=0.2,
            notes=f"No known remote-sensing detector mapping for '{clean_phrase}'",
        )

    @classmethod
    def normalize_target_classes(cls, classes: list[str]) -> list[str]:
        """Normalize a list of target class strings to canonical detector classes."""
        normalized: list[str] = []
        for c in classes:
            res = cls.map_phrase(c)
            if res.is_supported and res.detector_class:
                if res.detector_class not in normalized:
                    normalized.append(res.detector_class)
            else:
                # Retain literal if already recognized or pass-through for open evaluation
                clean = c.strip().lower()
                if clean not in normalized:
                    normalized.append(clean)
        return normalized
