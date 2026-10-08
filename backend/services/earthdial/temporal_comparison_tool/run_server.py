"""
Runner script to start the EarthDial Temporal Comparison FastAPI endpoint server.
"""

import argparse
import os
import sys
import uvicorn
from pathlib import Path

# Add tool directory to sys.path
tool_dir = Path(__file__).resolve().parent
if str(tool_dir) not in sys.path:
    sys.path.insert(0, str(tool_dir))


def main():
    parser = argparse.ArgumentParser(description="Start EarthDial Temporal Comparison API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--mock", action="store_true", help="Start server in mock simulation mode")
    parser.add_argument("--checkpoint", type=str, default="akshaydudhane/EarthDial_4B_RGB", help="Model checkpoint path or repo")
    parser.add_argument("--remote-worker", type=str, default=None, help="Remote EarthDial worker URL")
    parser.add_argument("--load-in-8bit", action="store_true", help="Enable 8-bit model quantization")
    args = parser.parse_args()

    if args.mock:
        os.environ["EARTHDIAL_MOCK_MODE"] = "true"
    if args.checkpoint:
        os.environ["EARTHDIAL_CHECKPOINT"] = args.checkpoint
    if args.remote_worker:
        os.environ["EARTHDIAL_REMOTE_WORKER_URL"] = args.remote_worker
    if args.load_in_8bit:
        os.environ["EARTHDIAL_LOAD_IN_8BIT"] = "true"

    print("=" * 60)
    print(" Starting EarthDial Temporal Comparison REST API")
    print("=" * 60)
    print(f" URL      : http://{args.host}:{args.port}")
    print(f" Docs     : http://{args.host}:{args.port}/docs")
    print(f" Health   : http://{args.host}:{args.port}/health")
    print(f" Mock Mode: {'Enabled' if args.mock else 'Disabled'}")
    print("=" * 60 + "\n")

    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
