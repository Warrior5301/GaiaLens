from qwen_agent.agents import Assistant
from backend.app.intelli.common import llm_cfg as common_llm_cfg, build_messages, run_messages
from backend.app.intelli.grounding.tools.draw_poly import Segmentation
from backend.app.intelli.grounding.tools.mdetr_tool import MDETR
from backend.app.intelli.grounding.tools.open_lang_ground import OpenGround
from backend.app.intelli.grounding.tools.yolo_tool import YOLODetection

def create_agent():
    llm_cfg = common_llm_cfg()

    tools = [
        'image_zoom_in_tool',
        'image_search',
        'web_search',
        'object_detection',
        'yolo_object_detection',
        'segmentation',
        'open_language_grounding',
    ]
    SYSTEM_PROMPT='''
        You are a Grounding Agent for remote sensing and satellite imagery. Your task is to locate all objects requested by the user and provide their coordinates.

You have four tools:

- **object_detection**: Detects objects from a predefined set of object classes.
- **yolo_object_detection**: Uses the separately deployed YOLO-World service for its supported classes (including airports, dams, helipads, and windmills). Returns annotated images but no machine-readable boxes.
- **segmentation**: Detects objects with irregular shapes and returns polygons.
- **open_language_grounding**: Performs open-vocabulary, natural-language object grounding and returns bounding boxes.

### Tool Selection

Use tools whenever necessary to answer the user's query accurately.

- If the query can be decomposed **entirely into object classes supported by `object_detection`**, use `object_detection`.
- For a YOLO-supported class that `object_detection` does not support, use `yolo_object_detection`. Its output is visual evidence; do not invent box coordinates from it.
- If any requested object cannot be represented by the supported classes, use `open_language_grounding` for that object.
- You may combine multiple tools when the query contains both supported and unsupported objects.
- Use `segmentation` when polygon-based detection is appropriate for irregular-shaped objects.
- You may use multiple tools and multiple calls if necessary. Your priority is to successfully answer the user's query.

Do not substitute a different object class just because it is available in `object_detection`. Preserve the user's intended meaning.

### Output

Provide locations as **horizontal bounding boxes** in the original image coordinate system when the chosen tool returns coordinates. If YOLO only returns an annotated image, say that coordinates were not supplied.

Format each box as:

`[x_min, y_min, x_max, y_max]`

with `(0, 0)` at the **top-left** of the image.

If a tool returns polygons, convert each polygon to its enclosing horizontal bounding box.

Return the final detections in a clear structured format, for example:

```json
{
  "objects": [
    {"class": "car", "horizontal_bbox": [120, 80, 165, 115]},
    {"class": "car", "horizontal_bbox": [210, 95, 258, 132]}
  ]
}
```

Do not fabricate objects or coordinates. Only report detections supported by the tool results and your spatial reasoning.

'''
    agent = Assistant(
        llm=llm_cfg,
        function_list=tools,
        system_message=SYSTEM_PROMPT,
    )

    return agent


def run_agent(agent, user_input_prompt, image_path=None):
    return run_messages(agent, build_messages(user_input_prompt, image_path))
