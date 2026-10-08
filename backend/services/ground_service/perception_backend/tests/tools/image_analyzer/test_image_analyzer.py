"""Unit tests for Image_Analyzer tool and EarthMind VLM backend."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from PIL import Image
from app.core.evidence import EvidenceBuilder
from app.core.vision_runtime import VisionRuntime
from app.models.earthmind import EarthMindBackend
from app.schemas.common import AssetRef, BBoxPx, ROI
from app.schemas.evidence import EvidenceType
from app.schemas.perception import ImageAnalyzerInput, ImageAnalyzerOutput
from app.tools.image_analyzer.parser import ImageAnalyzerOutputParser
from app.tools.image_analyzer.processor import ImageAnalyzerProcessor
from app.tools.image_analyzer.prompts import EarthMindPrompts
from app.tools.image_analyzer.tool import ImageAnalyzer


@pytest.fixture
def sample_asset() -> AssetRef:
    return AssetRef(
        asset_id="test_airport",
        uri="examples/sample_image.png",
        modality="optical",
        width_px=512,
        height_px=512,
    )


def test_input_semantic_validation(sample_asset: AssetRef):
    # Caption does not require question
    valid_caption_input = ImageAnalyzerInput(asset=sample_asset, task_type="caption")
    assert valid_caption_input.task_type == "caption"
    assert valid_caption_input.question is None

    # VQA requires question - missing question must fail
    with pytest.raises(ValidationError) as exc:
        ImageAnalyzerInput(asset=sample_asset, task_type="vqa")
    assert "question must be provided when task_type is 'vqa'" in str(exc.value)

    # VQA with question succeeds
    valid_vqa_input = ImageAnalyzerInput(
        asset=sample_asset,
        task_type="vqa",
        question="Is there an aircraft visible?",
    )
    assert valid_vqa_input.question == "Is there an aircraft visible?"


def test_earthmind_backend_lifecycle():
    backend = EarthMindBackend(model_id="earthmind_v1", device="cpu")
    assert not backend.is_loaded()

    backend.load()
    assert backend.is_loaded()

    img = Image.new("RGB", (512, 512), color="blue")
    res = backend.predict(image=img, prompt="<image>\nDescribe", task_type="caption")
    assert "raw_text" in res
    assert "confidence_score" in res
    assert res["metadata"]["architecture"] == "EarthMind-4B"

    backend.unload()
    assert not backend.is_loaded()


def test_image_analyzer_caption_execution(sample_asset: AssetRef):
    tool = ImageAnalyzer()
    inp = ImageAnalyzerInput(asset=sample_asset, task_type="caption")
    result = tool.run(inp)

    assert result.status == "success"
    assert result.data is not None

    output = tool.output_schema_class.model_validate(result.data)
    assert output.task_type == "caption"
    assert output.caption is not None
    assert output.answer is None
    assert len(output.observations) > 0
    assert output.raw_text is not None
    assert output.model_metadata["architecture"] == "EarthMind-4B"

    # Test Evidence generation
    evidences = tool.to_evidence(output, asset_id=sample_asset.asset_id)
    assert len(evidences) >= 1
    assert any(ev.evidence_type == EvidenceType.CAPTION for ev in evidences)
    assert evidences[0].payload["model"] == "EarthMind-4B"


def test_image_analyzer_vqa_execution(sample_asset: AssetRef):
    tool = ImageAnalyzer()
    inp = ImageAnalyzerInput(
        asset=sample_asset,
        task_type="vqa",
        question="Is there a runway visible?",
    )
    result = tool.run(inp)

    assert result.status == "success"
    output = tool.output_schema_class.model_validate(result.data)
    assert output.task_type == "vqa"
    assert output.answer is not None
    assert output.caption is None
    assert output.raw_text is not None

    evidences = tool.to_evidence(output, asset_id=sample_asset.asset_id)
    assert len(evidences) >= 1
    assert any(ev.evidence_type == EvidenceType.QUALITATIVE_OBSERVATION for ev in evidences)


def test_image_analyzer_processor_with_roi(sample_asset: AssetRef):
    runtime = VisionRuntime()
    roi = ROI(
        roi_id="roi_1",
        source_asset_id=sample_asset.asset_id,
        bbox_px=BBoxPx(x_min=50.0, y_min=50.0, x_max=200.0, y_max=250.0),
    )
    inp = ImageAnalyzerInput(
        asset=sample_asset,
        task_type="vqa",
        question="What is this facility?",
        roi=roi,
    )
    prepared_img, prompt = ImageAnalyzerProcessor.process(inp, runtime, target_size=(512, 512))
    assert prepared_img.size == (512, 512)
    assert "What is this facility?" in prompt


def test_output_parser():
    raw_text = "Answer: Yes. The facility contains runway infrastructure."
    parsed = ImageAnalyzerOutputParser.parse(
        raw_text=raw_text,
        confidence_score=0.92,
        task_type="vqa",
        model_metadata={"architecture": "EarthMind"},
        question="Is there a runway?",
    )
    assert parsed.answer == "yes"
    assert parsed.task_type == "vqa"
    assert len(parsed.observations) > 0
