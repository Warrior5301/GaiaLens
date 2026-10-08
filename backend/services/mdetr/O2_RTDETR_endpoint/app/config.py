import os
from pathlib import Path


AI4RS_ROOT = Path(os.getenv("AI4RS_ROOT", "/content/ai4rs")).resolve()
CONFIG_PATH = Path(
    os.getenv(
        "O2_RTDETR_CONFIG",
        str(
            AI4RS_ROOT
            / "projects/rotated_rtdetr/configs/"
            / "o2_rtdetr_r18vd_2xb4_72e_dota.py"
        ),
    )
).resolve()
CHECKPOINT_PATH = Path(
    os.getenv(
        "O2_RTDETR_CHECKPOINT",
        str(AI4RS_ROOT / "checkpoints/o2_rtdetr_r18vd_dota_epoch_72.pth"),
    )
).resolve()
DEVICE = os.getenv("O2_RTDETR_DEVICE", "cuda:0")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))
DEFAULT_SCORE_THRESHOLD = float(os.getenv("O2_RTDETR_SCORE_THRESHOLD", "0.25"))
