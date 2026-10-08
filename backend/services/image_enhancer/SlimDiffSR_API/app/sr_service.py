import base64
import io
import threading
from argparse import Namespace

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel

from Student.osunet_student import LEDSR


class SlimDiffSRService:
    """
    Thin API wrapper around the official SlimDiffSR inference pipeline.

    The actual inference procedure mirrors test_ours.py:
      1. Load RGB image
      2. Convert to [-1, 1]
      3. Bicubic upscale
      4. Pad to multiples of 256
      5. Run LEDSR
      6. Crop padding
      7. Match SR mean/std to the padded LR input
      8. Convert back to RGB PNG

    The API layer only handles model lifecycle and base64 I/O.
    """

    def __init__(
        self,
        model_path: str,
        sd_path: str,
        taesd_path: str,
        upscale: int = 4,
        prune_depth: int = 2,
        fp16: bool = False,
    ):
        self.model_path = model_path
        self.sd_path = sd_path
        self.taesd_path = taesd_path
        self.upscale = upscale
        self.prune_depth = prune_depth
        self.fp16 = fp16

        self.model = None
        self.text_encoder = None
        self.tokenizer = None
        self.text_embedding = None

        self._lock = threading.RLock()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _build_args(self):
        return Namespace(
            sd_path=self.sd_path,
            taesd_path=self.taesd_path,
            prune_depth=self.prune_depth,
            fp16=self.fp16,
            upscale=self.upscale,
            model_dir=self.model_path,
        )

    def load(self):
        with self._lock:
            if self.model is not None:
                return

            if self.device != "cuda":
                raise RuntimeError(
                    "SlimDiffSR requires CUDA. "
                    "Start the API on a CUDA-enabled machine."
                )

            args = self._build_args()

            # Same tokenizer used by official test_ours.py
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.sd_path,
                subfolder="tokenizer",
            )

            # Same CLIP text encoder used by official test_ours.py
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.sd_path,
                subfolder="text_encoder",
            ).cuda()

            self.text_encoder.requires_grad_(False)

            with torch.no_grad():
                text_token = self.tokenizer(
                    "",
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.cuda()

                self.text_embedding = self.text_encoder(
                    text_token
                )[0]

            # Same LEDSR model construction
            self.model = LEDSR(args, self.text_embedding).cuda()

            # Same checkpoint loading
            checkpoint = torch.load(
                self.model_path,
                map_location="cpu",
                weights_only=False,
            )

            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Support DataParallel checkpoints if necessary.
            if isinstance(state_dict, dict) and any(
                key.startswith("module.")
                for key in state_dict.keys()
            ):
                state_dict = {
                    key[len("module."):]: value
                    for key, value in state_dict.items()
                }

            self.model.load_state_dict(
                state_dict,
                strict=True,
            )

            # Same FP16 behavior as official script.
            if self.fp16:
                self.model.half()

            self.model.set_eval()

            torch.cuda.empty_cache()

    def unload(self):
        with self._lock:
            self.model = None
            self.text_embedding = None

            if self.text_encoder is not None:
                self.text_encoder.cpu()

            self.text_encoder = None
            self.tokenizer = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _decode_base64_image(image_base64: str) -> Image.Image:
        # Accept raw base64 or data URLs.
        if "," in image_base64 and image_base64.startswith("data:"):
            image_base64 = image_base64.split(",", 1)[1]

        try:
            raw = base64.b64decode(
                image_base64,
                validate=True,
            )
        except Exception as exc:
            raise ValueError(
                "image_base64 is not valid base64"
            ) from exc

        try:
            image = Image.open(
                io.BytesIO(raw)
            ).convert("RGB")
        except Exception as exc:
            raise ValueError(
                "Decoded bytes are not a valid image"
            ) from exc

        return image

    @staticmethod
    def _encode_png(image: Image.Image) -> str:
        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG",
        )

        return base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

    def _infer_official(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """
        Exact official SlimDiffSR inference procedure.
        """

        # Official:
        # LR = Image.open(path).convert("RGB")
        # LR = transforms.ToTensor()(LR).to(device).unsqueeze(0) * 2 - 1

        lr = transforms.ToTensor()(
            image
        ).to(self.device).unsqueeze(0) * 2 - 1

        # Official bicubic pre-upscaling.
        lr = F.interpolate(
            lr,
            scale_factor=self.upscale,
            mode="bicubic",
        )

        # Official padding.
        _, _, h, w = lr.shape

        pad_h = (256 - h % 256) % 256
        pad_w = (256 - w % 256) % 256

        lr = F.pad(
            lr,
            (0, pad_w, 0, pad_h),
            mode="reflect",
        )

        # Official FP16 behavior.
        inp = lr.half() if self.fp16 else lr

        if self.fp16:
            autocast_context = torch.cuda.amp.autocast()
        else:
            from contextlib import nullcontext
            autocast_context = nullcontext()

        # Official inference.
        with torch.no_grad(), autocast_context:
            _, sr = self.model(inp)

        # Official crop.
        sr = sr[:, :, :h, :w]

        # Official brightness normalization.
        sr = (
            (sr - sr.mean(dim=[2, 3], keepdim=True))
            / sr.std(dim=[2, 3], keepdim=True)
            * lr.std(dim=[2, 3], keepdim=True)
            + lr.mean(dim=[2, 3], keepdim=True)
        )

        # Official conversion:
        # (SR / 2 + 0.5).clamp(0, 1)
        sr = transforms.ToPILImage()(
            (sr[0] / 2 + 0.5)
            .clamp(0, 1)
            .float()
            .cpu()
        )

        return sr

    def super_resolve_bytes(
        self,
        image_bytes: bytes,
        filename: str = "image.png",
    ):
        """
        Run the official SlimDiffSR inference directly on image bytes.

        This is used by the multipart/form-data upload endpoint.
        The actual inference path is identical to super_resolve().
        """

        with self._lock:
            if self.model is None:
                raise RuntimeError(
                    "Model is not loaded. "
                    "Call POST /load before /super-resolve."
                )

            if not image_bytes:
                raise ValueError(
                    "Uploaded image is empty."
                )

            try:
                image = Image.open(
                    io.BytesIO(image_bytes)
                ).convert("RGB")
            except Exception as exc:
                raise ValueError(
                    "Uploaded bytes are not a valid image."
                ) from exc

            input_width, input_height = image.size

            result = self._infer_official(image)

            return {
                "image_base64": self._encode_png(result),
                "filename": "super_resolved.png",
                "width": result.width,
                "height": result.height,
                "scale": self.upscale,
                "input_width": input_width,
                "input_height": input_height,
            }

    def super_resolve(
        self,
        image_base64: str,
    ):
        with self._lock:
            if self.model is None:
                raise RuntimeError(
                    "Model is not loaded. "
                    "Call POST /load before /super-resolve."
                )

            image = self._decode_base64_image(
                image_base64
            )

            input_width, input_height = image.size

            result = self._infer_official(image)

            return {
                "image_base64": self._encode_png(result),
                "filename": "super_resolved.png",
                "width": result.width,
                "height": result.height,
                "scale": self.upscale,
                "input_width": input_width,
                "input_height": input_height,
            }
