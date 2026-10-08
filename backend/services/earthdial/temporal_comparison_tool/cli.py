"""
Command-Line Interface (CLI) for EarthDial Temporal Comparison Tool
===================================================================

Run temporal comparison directly from the terminal on local satellite/aerial images.
"""

import argparse
import json
import sys
from pathlib import Path

# Add current folder to sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from comparator import EarthDialTemporalComparator
from schemas import PromptStyle


def main():
    parser = argparse.ArgumentParser(
        description="EarthDial CLI: Multi-temporal remote sensing image comparison and change detection."
    )
    parser.add_argument(
        "--images", "-i",
        nargs="+",
        required=True,
        help="Two or more image file paths or URLs to compare chronologically."
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="Analyze the images and describe any semantic or structural changes detected between the observations.",
        help="Comparison prompt or question."
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="akshaydudhane/EarthDial_4B_RGB",
        help="Model checkpoint path or Hugging Face repo ID."
    )
    parser.add_argument(
        "--style", "-s",
        type=str,
        choices=["evaluation", "conversational", "raw"],
        default="evaluation",
        help="Prompt formatting style."
    )
    parser.add_argument(
        "--task-token",
        type=str,
        default="[changedet] [hr_rgb_temp_0.5]",
        help="Task prefix tokens."
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)."
    )
    parser.add_argument(
        "--max-new-tokens", "-m",
        type=int,
        default=256,
        help="Maximum generation tokens."
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        help="Target device (cuda/cpu)."
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization."
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization."
    )
    parser.add_argument(
        "--remote-worker",
        type=str,
        default=None,
        help="URL of remote EarthDial worker."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in lightweight simulation mode without model weights."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional path to save JSON response."
    )

    args = parser.parse_args()

    if len(args.images) < 2:
        print(f"Error: Temporal comparison requires at least 2 images, got {len(args.images)}.", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print(" EarthDial Temporal Comparison Tool")
    print("=" * 60)
    print(f"Input Images : {len(args.images)} files")
    for idx, img_p in enumerate(args.images, 1):
        print(f"  [{idx}] {img_p}")
    print(f"Prompt Style : {args.style}")
    print(f"Query        : {args.query}")
    print(f"Mode         : {'Mock Simulation' if args.mock else 'Model Inference'}")
    print("=" * 60 + "\n")

    comparator = EarthDialTemporalComparator(
        checkpoint=args.checkpoint,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        remote_worker_url=args.remote_worker,
        mock_mode=args.mock,
        lazy_load=False
    )

    result = comparator.compare(
        images=args.images,
        query=args.query,
        prompt_style=PromptStyle(args.style),
        task_token=args.task_token,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    print("\n" + "-" * 40 + " ANALYSIS RESULT " + "-" * 40)
    print(result.response)
    print("-" * 97)
    print(f"Execution Time : {result.execution_time_seconds}s | Device: {result.device}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        print(f"Full response saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
