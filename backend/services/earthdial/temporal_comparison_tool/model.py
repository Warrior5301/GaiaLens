"""
EarthDial Model Wrapper
=======================

A clean, modular PyTorch/Transformers wrapper for the EarthDial vision-language model.
Completely framework-agnostic (runs locally, on cloud VMs, Docker, or any Python environment).
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

# Ensure EarthDial source package is in python path (local self-contained or sibling repo)
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

earthdial_src = current_dir.parent / "EarthDial" / "src"
if earthdial_src.exists() and str(earthdial_src) not in sys.path:
    sys.path.insert(0, str(earthdial_src))

logger = logging.getLogger("EarthDialModel")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class EarthDialModel:
    """
    Lightweight, modular wrapper for EarthDial (InternVL2 + Phi-3 architecture).
    """

    def __init__(
        self,
        checkpoint: str = "akshaydudhane/EarthDial_4B_RGB",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        auto_device_map: bool = True,
        lazy_load: bool = True,
        mock_mode: bool = False,
    ):
        """
        Initialize the EarthDial model.

        Args:
            checkpoint: Hugging Face repo ID or path to local model weights.
            device: 'cuda', 'cpu', or None for auto detection.
            load_in_8bit: Quantize to 8-bit.
            load_in_4bit: Quantize to 4-bit.
            auto_device_map: Use Accelerate device_map='auto'.
            lazy_load: Delay loading large weights until first call.
            mock_mode: Run in simulation mode (ideal for testing without GPU/downloads).
        """
        self.checkpoint = checkpoint
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.auto_device_map = auto_device_map

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.tokenizer = None
        self.transform = None
        self.image_size = 448
        
        env_mock = os.getenv("EARTHDIAL_MOCK_MODE", "false").lower() in ("true", "1", "yes")
        self._mock_mode = mock_mode or env_mock

        if not lazy_load and not self._mock_mode:
            self.load()

    def load(self) -> None:
        """Loads model weights, tokenizer, and image transforms."""
        if self.model is not None and self.tokenizer is not None:
            return

        logger.info(f"Loading EarthDial model from '{self.checkpoint}' on {self.device}...")
        try:
            from transformers import AutoTokenizer
            from earthdial.model.internvl_chat import InternVLChatModel
            from earthdial.train.dataset import build_transform

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint,
                trust_remote_code=True,
                use_fast=False
            )
            # Retain box and reference tokens
            tokens_to_keep = ['<box>', '</box>', '<ref>', '</ref>']
            if hasattr(self.tokenizer, "additional_special_tokens") and self.tokenizer.additional_special_tokens:
                self.tokenizer.additional_special_tokens = [
                    item for item in self.tokenizer.additional_special_tokens
                    if item not in tokens_to_keep
                ]

            kwargs = {}
            if self.auto_device_map and torch.cuda.is_available():
                kwargs["device_map"] = "auto"

            self.model = InternVLChatModel.from_pretrained(
                self.checkpoint,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                load_in_8bit=self.load_in_8bit,
                load_in_4bit=self.load_in_4bit,
                **kwargs
            ).eval()

            if not self.load_in_8bit and not self.load_in_4bit and not kwargs.get("device_map"):
                self.model = self.model.to(self.device)

            self.image_size = getattr(self.model.config, "force_image_size", None) or 448
            self.transform = build_transform(is_train=False, input_size=self.image_size, normalize_type="imagenet")
            logger.info("EarthDial model loaded successfully.")

        except Exception as exc:
            # Never fall back to canned mock output; mock mode is opt-in via EARTHDIAL_MOCK_MODE / --mock
            self.model = self.tokenizer = None
            raise RuntimeError(f"Could not load EarthDial weights from '{self.checkpoint}': {exc}") from exc

    def preprocess_images(
        self,
        images: List[Image.Image],
        dynamic_image_size: bool = True,
        max_tiles_per_image: int = 6
    ) -> Tuple[torch.Tensor, List[int], List[Dict[str, Any]]]:
        """Preprocesses PIL images into dynamic tiles and tensor patches."""
        from earthdial.train.dataset import dynamic_preprocess, build_transform

        if self.transform is None:
            self.transform = build_transform(is_train=False, input_size=self.image_size, normalize_type="imagenet")

        all_tiles = []
        num_patches_list = []
        metadata = []

        for idx, img in enumerate(images):
            w, h = img.size
            if dynamic_image_size:
                tiles = dynamic_preprocess(
                    img,
                    image_size=self.image_size,
                    max_num=max_tiles_per_image,
                    use_thumbnail=True
                )
            else:
                tiles = [img]

            num_patches_list.append(len(tiles))
            all_tiles.extend(tiles)
            metadata.append({
                "index": idx + 1,
                "width": w,
                "height": h,
                "num_tiles": len(tiles)
            })

        pixel_values = [self.transform(tile) for tile in all_tiles]
        pixel_values = torch.stack(pixel_values)
        return pixel_values, num_patches_list, metadata

    def generate(
        self,
        images: Optional[List[Image.Image]],
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        top_p: float = 0.9,
        num_beams: int = 1,
        dynamic_image_size: bool = True,
        max_tiles_per_image: int = 6
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        """
        Executes vision-language inference given images and formatted prompt.

        Returns:
            (generated_text, image_metadata, device_used)
        """
        if self._mock_mode:
            return self._mock_generate(prompt, len(images) if images else 0)

        self.load()
        if self._mock_mode:
            return self._mock_generate(prompt, len(images) if images else 0)

        image_metadata = []
        if images and len(images) > 0:
            pixel_values, num_patches_list, image_metadata = self.preprocess_images(
                images=images,
                dynamic_image_size=dynamic_image_size,
                max_tiles_per_image=max_tiles_per_image
            )
            target_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            pixel_values = pixel_values.to(self.device, dtype=target_dtype)
        else:
            pixel_values = None
            num_patches_list = None

        generation_config = dict(
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
        )

        response = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            question=prompt,
            generation_config=generation_config,
            verbose=False
        )
        return response.strip(), image_metadata, str(self.device)

    def _mock_generate(self, prompt: str, num_images: int) -> Tuple[str, List[Dict[str, Any]], str]:
        q = prompt.lower()
        if "road" in q or "urban" in q or "construct" in q:
            text = "The area was urbanized."
        elif "damage" in q or "disaster" in q:
            text = "Significant structural damage and debris accumulation are evident across the observed sector."
        else:
            text = "Semantic changes detected: new infrastructure developed between the observations."
        meta = [{"index": i + 1, "width": 512, "height": 512, "num_tiles": 1} for i in range(num_images)]
        return text, meta, "mock"

    def get_info(self) -> Dict[str, Any]:
        """Returns runtime diagnostics and GPU status."""
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else None
        gpu_mem = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2) if cuda_avail else 0.0

        return {
            "model_path": self.checkpoint,
            "device": self.device if not self._mock_mode else "mock",
            "model_loaded": (self.model is not None),
            "cuda_available": cuda_avail,
            "gpu_name": gpu_name,
            "gpu_memory_allocated_gb": gpu_mem,
            "image_size": self.image_size
        }
