import json
import os
from qwen_agent.agents import Assistant
from backend.app.intelli.common import llm_cfg as common_llm_cfg, build_messages, run_messages
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.schema import ContentItem

from backend.app.intelli.grounding.ground import (
    create_agent as create_grounding_agent,
    run_agent as run_grounding_agent,
)
from backend.app.intelli.sar.sar_agent import (
    create_agent as create_sar_agent,
    run_agent as run_sar_agent,
)
from backend.app.intelli.temporal.temporal_agent import (
    create_agent as create_temporal_agent,
    run_agent as run_temporal_agent,
)
from backend.app.intelli.geo_analysis.geo_analyser import (
    create_agent as create_geo_agent,
    run_agent as run_geo_agent,
)
from backend.app.intelli.orchestrator.tools.image_analyzer import ImageAnalyzerTool
from backend.app.intelli.orchestrator.tools.ocr_tool import OCRTool
from backend.app.intelli.orchestrator.tools.image_enhancer import SlimDiffSREnhancer
from backend.app.intelli.orchestrator.tools.image_context import ImageContextTool


# ============================================================================
# Agent-as-a-Tool Wrappers
# ============================================================================

@register_tool('grounding_agent')
class GroundingAgentTool(BaseTool):
    description = (
        "Specialist agent for visual grounding, object detection, and polygon segmentation in satellite/remote sensing imagery. "
        "Accepts a SINGLE image (image_path). Use this tool when the query requires locating specific objects, "
        "detecting predefined classes (airports, ships, bridges, tanks, etc.), segmenting irregular structures (rivers, coastlines, lakes), "
        "or grounding open-vocabulary descriptions with coordinate bounding boxes."
    )
    parameters = [
        {
            'name': 'image_path',
            'type': 'string',
            'description': 'The local file path to the single satellite or remote sensing image to analyze.',
            'required': True,
        },
        {
            'name': 'query',
            'type': 'string',
            'description': 'The specific grounding, object detection, or segmentation instruction.',
            'required': True,
        },
    ]

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.agent = None

    def call(self, params: str | dict, **kwargs) -> list[ContentItem]:
        if isinstance(params, str):
            params = json.loads(params)

        image_path = params.get('image_path')
        query = params.get('query')

        if not image_path:
            return [ContentItem(text="Error: 'image_path' parameter is required for Grounding Agent.")]

        if not os.path.exists(image_path):
            return [ContentItem(text=f"Error: Image file not found at {image_path}")]

        if not query:
            return [ContentItem(text="Error: Query must be provided for Grounding Agent.")]

        try:
            if self.agent is None:
                self.agent = create_grounding_agent()
            result = run_grounding_agent(self.agent, query, image_path=image_path)
            return [ContentItem(text=f"Grounding Agent Response:\n{result}")]
        except Exception as e:
            return [ContentItem(text=f"Error running Grounding Agent: {str(e)}")]


@register_tool('sar_multispectral_agent')
class SARMulitSpectralAgentTool(BaseTool):
    description = (
        "Specialist agent for Synthetic Aperture Radar (SAR) and Multispectral Earth Observation imagery. "
        "Accepts a SINGLE image (image_path). Use this tool when the query involves multispectral band analysis, "
        "SAR backscatter interpretation, spectral indices, or sensor-specific remote sensing reasoning."
    )
    parameters = [
        {
            'name': 'image_path',
            'type': 'string',
            'description': 'The local file path to the single SAR or multispectral image to analyze.',
            'required': True,
        },
        {
            'name': 'query',
            'type': 'string',
            'description': 'The multispectral or SAR analysis query/instruction.',
            'required': True,
        },
    ]

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.agent = None

    def call(self, params: str | dict, **kwargs) -> list[ContentItem]:
        if isinstance(params, str):
            params = json.loads(params)

        image_path = params.get('image_path')
        query = params.get('query')

        if not image_path:
            return [ContentItem(text="Error: 'image_path' parameter is required for SAR/Multispectral Agent.")]

        if not os.path.exists(image_path):
            return [ContentItem(text=f"Error: Image file not found at {image_path}")]

        if not query:
            return [ContentItem(text="Error: Query must be provided for SAR/Multispectral Agent.")]

        try:
            if self.agent is None:
                self.agent = create_sar_agent()
            result = run_sar_agent(self.agent, query, image_path=image_path)
            return [ContentItem(text=f"SAR/Multispectral Agent Response:\n{result}")]
        except Exception as e:
            return [ContentItem(text=f"Error running SAR/Multispectral Agent: {str(e)}")]


@register_tool('temporal_agent')
class TemporalAgentTool(BaseTool):
    description = (
        "Specialist agent for multitemporal analysis and change detection across satellite observations over time. "
        "Accepts MULTIPLE images (image_paths, requires at least 2 images). "
        "Equipped with EarthDial (for arbitrary number of images overall reasoning) and Terra / bi_temporal_analyser (for strictly 2 images pixel-level comparison)."
    )
    parameters = [
        {
            'name': 'image_paths',
            'type': 'array',
            'items': {'type': 'string'},
            'description': 'List of file paths to the images across different time periods (requires at least 2 images).',
            'required': True,
        },
        {
            'name': 'query',
            'type': 'string',
            'description': 'The change detection or temporal comparison question/instruction.',
            'required': True,
        },
    ]

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.agent = None

    def call(self, params: str | dict, **kwargs) -> list[ContentItem]:
        if isinstance(params, str):
            params = json.loads(params)

        image_paths = params.get('image_paths') or params.get('image_path')
        query = params.get('query')

        if isinstance(image_paths, str):
            image_paths = [image_paths]

        if not isinstance(image_paths, (list, tuple)) or len(image_paths) < 2:
            return [ContentItem(text="Error: 'image_paths' must be a list containing at least two image paths for temporal analysis.")]

        for path in image_paths:
            if not os.path.exists(path):
                return [ContentItem(text=f"Error: Image file not found at {path}")]

        if not query:
            return [ContentItem(text="Error: Query must be provided for Temporal Agent.")]

        try:
            if self.agent is None:
                self.agent = create_temporal_agent()
            result = run_temporal_agent(self.agent, query, image_path=list(image_paths))
            return [ContentItem(text=f"Temporal Agent Response:\n{result}")]
        except Exception as e:
            return [ContentItem(text=f"Error running Temporal Agent: {str(e)}")]


@register_tool('geo_analysis_agent')
class GeoAnalysisAgentTool(BaseTool):
    description = (
        "Specialist agent for geometric and geospatial calculations: centroids, polygon area/perimeter, "
        "distance and bearing between objects, bounding box intersections, and pixel-area to m² conversion using GSD. "
        "Pass all coordinates, polygons, bounding boxes and GSD values it needs inside the query."
    )
    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description': 'The calculation to perform, including every coordinate, box, polygon and GSD value required.',
            'required': True,
        },
    ]

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.agent = None

    def call(self, params: str | dict, **kwargs) -> list[ContentItem]:
        if isinstance(params, str):
            params = json.loads(params)

        query = params.get('query')
        if not query:
            return [ContentItem(text="Error: Query must be provided for Geo Analysis Agent.")]

        try:
            if self.agent is None:
                self.agent = create_geo_agent()
            result = run_geo_agent(self.agent, query)
            return [ContentItem(text=f"Geo Analysis Agent Response:\n{result}")]
        except Exception as e:
            return [ContentItem(text=f"Error running Geo Analysis Agent: {str(e)}")]


# ============================================================================
# Master Orchestrator Agent
# ============================================================================

def create_agent():
    """Create and configure the master Orchestrator Agent with all sub-agents as tools."""
    llm_cfg = common_llm_cfg()

    tools = [
        'image_zoom_in_tool',
        'image_search',
        'web_search',
        'image_analyzer',
        'ocr_tool',
        'image_enhancer',
        'image_context',
        'grounding_agent',
        'sar_multispectral_agent',
        'temporal_agent',
        'geo_analysis_agent',
    ]

    SYSTEM_PROMPT = '''
You are the Chief Orchestrator Agent for GaiaLens, an advanced Earth Observation and satellite intelligence system.

You can receive an ARBITRARY number of images from the user. Your role is to understand user queries, plan multi-step workflows, delegate tasks to specialized domain agents, and synthesize the final answer.

### Domain Agents & Image Cardinality Capabilities:

1. **`image_analyzer`** (Single-Image):
   - Accepts a single image (`image_path`).
   - Extracts metadata, aspect ratios, spectral bands, and GSD resolution.

2. **`grounding_agent`** (Single-Image):
   - Accepts a single image (`image_path`).
   - Specialized in object detection, coordinate bounding boxes `[x_min, y_min, x_max, y_max]`, polygon segmentation (e.g. rivers, lakes), and open-vocabulary natural language grounding.
   - Note: If multiple images were provided to the orchestrator, select and pass the specific relevant single image path to this tool.

3. **`sar_multispectral_agent`** (Single-Image):
   - Accepts a single image (`image_path`).
   - Specialized in Synthetic Aperture Radar (SAR) imagery, radar backscatter, multispectral band reasoning, and sensor physics.
   - Note: If multiple images were provided to the orchestrator, select and pass the specific relevant single image path to this tool.

4. **`temporal_agent`** (Multi-Image):
   - The ONLY agent that accepts multiple images (`image_paths`, requires 2 or more images).
   - Within the temporal agent:
     - **`earthdial`**: Handles an arbitrary number of images (2 or more) for overall semantic change analysis and macro-level reasoning.
     - **`bi_temporal_analyser`** (Terra): Handles strictly TWO images for fine-grained pixel-level comparison, localized difference masks/overlays, and change coverage percentages.

5. **`ocr_tool`** (Single-Image):
   - Accepts a single image (`image_path`) and returns the text visible in it (labels, signs, map annotations).

6. **`image_enhancer`** (Single-Image):
   - Accepts a single image (`image_path`) and returns a 4x super-resolved copy with its new path. Use it before other tools when the image is too low-resolution to analyse.

7. **`geo_analysis_agent`** (No image):
   - Performs geometric calculations (centroid, area, perimeter, distance/bearing, box intersection, pixel area to m²). Pass all coordinates and GSD values in its query.

8. **`image_context`** (Single-Image):
   - Accepts a single image (`image_path`). Use it when you have no idea what an image shows or where it is.
   - Returns four images of the same place at different zoom levels plus textual context about it. Use them to understand the image before answering or delegating to other agents.

### Image Paths:
- The user message lists every input image as "Image N path: <absolute path>". Pass these exact paths to tools; never invent paths.
- Tools may return new images (e.g. enhanced images or overlays) along with their paths; you may pass those paths to later tools.

### Orchestration Guidelines:

1. **Task Planning & Delegation**:
   - For queries involving grounding or SAR analysis on multiple observations, call `grounding_agent` or `sar_multispectral_agent` individually for each specific image.
   - For change detection and multitemporal comparisons across dates, pass the list of image paths to `temporal_agent`.

2. **Strict Factual Grounding**:
   - Only report findings supported by the visual evidence and tool outputs. Never fabricate coordinates, spectral data, or change metrics.

3. **Final Response Synthesis**:
   - Synthesize results from all sub-agents into a cohesive, structured, and comprehensive answer answering the user's inquiry.
'''

    agent = Assistant(
        llm=llm_cfg,
        function_list=tools,
        system_message=SYSTEM_PROMPT,
    )

    return agent


create_orchestrator_agent = create_agent


def run_agent(agent, user_input_prompt, image_path=None, num_images=None):
    """
    Execute the master orchestrator on a prompt and optional image path(s).

    `image_path` may be a single path or a list; `num_images` optionally limits how many are used.
    """
    if isinstance(image_path, (list, tuple)):
        image_path = list(image_path)[:num_images]
    # Output images are returned separately (see start_conversation_with_images), not as paths in the text
    return run_messages(agent, build_messages(user_input_prompt, image_path), list_images=False)


run_orchestrator = run_agent
