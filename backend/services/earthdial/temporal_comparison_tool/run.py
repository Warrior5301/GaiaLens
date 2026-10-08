"""
Runner script for EarthDial FastAPI REST Service.
================================================

Usage:
    python run.py
    python run.py --port 8000 --mock
    python run.py --checkpoint akshaydudhane/EarthDial_4B_RGB
"""

import argparse
import os
import sys
from pathlib import Path
import uvicorn

# Add current folder to sys.path
package_dir = Path(__file__).resolve().parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))


def main():
    parser = argparse.ArgumentParser(description="Start EarthDial FastAPI Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--mock", action="store_true", help="Start server in mock simulation mode")
    parser.add_argument("--checkpoint", type=str, default="akshaydudhane/EarthDial_4B_RGB", help="Model checkpoint path or repo")
    parser.add_argument("--load-in-8bit", action="store_true", help="Enable 8-bit model quantization")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit model quantization")
    args = parser.parse_args()

    if args.mock:
        os.environ["EARTHDIAL_MOCK_MODE"] = "true"
    if args.checkpoint:
        os.environ["EARTHDIAL_CHECKPOINT"] = args.checkpoint
    if args.load_in_8bit:
        os.environ["EARTHDIAL_LOAD_IN_8BIT"] = "true"
    if args.load_in_4bit:
        os.environ["EARTHDIAL_LOAD_IN_4BIT"] = "true"

    print("=" * 65)
    print("   EarthDial REST API Service")
    print("=" * 65)
    print(f" * Server running on: http://{args.host}:{args.port}")
    print(f" * Interactive Docs : http://{args.host}:{args.port}/docs")
    print(f" * Health Endpoint  : http://{args.host}:{args.port}/health")
    print(f" * Model Checkpoint : {args.checkpoint}")
    print(f" * Simulation Mock  : {'ON (Instant test mode)' if args.mock else 'OFF (Real weights)'}")
    print("=" * 65 + "\n")

    uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
