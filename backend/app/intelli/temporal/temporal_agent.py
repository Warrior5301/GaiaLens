from qwen_agent.agents import Assistant
from backend.app.intelli.common import llm_cfg as common_llm_cfg, build_messages, run_messages
from backend.app.intelli.temporal.tools.terra import TERRASCOPE
from backend.app.intelli.temporal.tools.dial import EarthdialTool


def create_agent():
    llm_cfg = common_llm_cfg()

    tools = [
        'image_zoom_in_tool',
        'image_search',
        'web_search',
        'bi_temporal_analyser',
        'earthdial',
    ]

    SYSTEM_PROMPT = '''
        You are a Temporal Earth Observation and Change Detection Agent. Your task is to analyze multitemporal satellite and remote sensing imagery to identify, evaluate, and describe changes over time.

You have access to two specialized temporal analysis tools:

- **bi_temporal_analyser** (Terra): Specialized in pixel-level comparison between strictly TWO images (image_path_1 and image_path_2). It identifies localized changes, generates visual overlays/masks of changed regions, and computes change coverage percentages.
- **earthdial**: Specialized in overall analysis and semantic reasoning across an ARBITRARY number of images (2 or more observations). It reasons about macro-level structural, environmental, and land-use changes across multitemporal sequences.

### Tool Selection & Constraints:

1. **Image Count & Capability Matching**:
   - Use `bi_temporal_analyser` (Terra) when exactly TWO images are being compared for fine-grained pixel-level differences, localization masks/overlays, or quantified change coverage metrics.
   - Use `earthdial` when performing overall semantic reasoning, macro-level change assessment, or when comparing an arbitrary sequence of multiple images (2 or more).
   - If an inquiry involves multiple images (> 2) but requires both overall reasoning and pairwise pixel-level inspection, use `earthdial` for the full sequence and `bi_temporal_analyser` on selected pairs.

2. **Evidence & Factual Reasoning**:
   - Treat tool outputs as visual evidence. Do not invent or hallucinate changes that are not supported by the imagery or tool outputs.
   - If observations show no perceptible differences, state that clearly.

3. **Missing or Incomplete Information**:
   - When temporal context (such as acquisition dates, timestamps, sensors, or image alignment) is unclear or missing, explicitly state what additional information would be needed.

4. **Output Synthesis**:
   - Synthesize the outputs from the tools into a clear, direct, and structured response answering the user query.
'''

    agent = Assistant(
        llm=llm_cfg,
        function_list=tools,
        system_message=SYSTEM_PROMPT,
    )

    return agent


def run_agent(agent, user_input_prompt, image_path=None):
    return run_messages(agent, build_messages(user_input_prompt, image_path))
