import json
import re
from pathlib import Path
import modal
from api import app as fastapi_web_app

app = modal.App("terrascope-bf16-test")

TRANSFORMERS_VERSION = "4.42.3"
MODEL_ID = "sy1998/TerraScope"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        f"transformers=={TRANSFORMERS_VERSION}",
        "peft==0.12.0",
        "accelerate",
        "timm",
        "einops",
        "sentencepiece",
        "pillow",
        "numpy<2",
        "opencv-python-headless",
        "requests",
        "huggingface_hub",
        "fastapi",
        "pydantic",
    )
    .add_local_python_source("api")
)

vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,image/*;q=0.9,*/*;q=0.8",
}


def install_flash_attn_stub():
    """Remote code imports flash_attn at module level. We run with use_flash_attn=False,
    so a placeholder module is enough to get past the import check."""
    import sys, types, importlib.machinery

    try:
        import flash_attn  # noqa: F401
        print("flash_attn: real package found")
        return
    except Exception:
        pass

    def make(name):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        m.__path__ = []
        m.__version__ = "0.0.0"

        def _getattr(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)

            def _f(*a, **k):
                raise RuntimeError(f"flash_attn.{attr} is a stub; run with use_flash_attn=False")

            return _f

        m.__getattr__ = _getattr
        return m

    for n in ["flash_attn", "flash_attn.bert_padding", "flash_attn.flash_attn_interface"]:
        sys.modules[n] = make(n)

    print("flash_attn: using stub")


def patch_terrascope_config():
    """Avoid a broken default-config path in TerraScope's remote config class.
    Its constructor requires llm_config['architectures'], even when called with no arguments.
    Transformers may call that no-argument constructor while formatting a loaded config.
    Serializing the actual loaded config directly avoids constructing an invalid default instance.
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config_class = get_class_from_dynamic_module(
        "configuration_sa2va_chat.Sa2VAChatConfig",
        MODEL_ID
    )

    def to_diff_dict(self):
        return self.to_dict()

    config_class.to_diff_dict = to_diff_dict


def load_image(url, raw):
    import io, re, requests
    from urllib.parse import urljoin
    from PIL import Image

    if raw is not None:
        if not raw:
            raise ValueError("The supplied image file is empty.")

        with Image.open(io.BytesIO(raw)) as image:
            return image.convert("RGB")

    if not url:
        raise ValueError("Provide either image bytes or an image URL.")

    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()

    if r.headers.get("content-type", "").startswith("image/"):
        with Image.open(io.BytesIO(r.content)) as image:
            return image.convert("RGB")

    html = r.text

    m = (
        re.search(
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)',
            html
        )
        or re.search(
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
            html
        )
        or re.search(
            r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)',
            html
        )
    )

    if not m:
        raise RuntimeError("No image found on page. Use --image-path with a local file.")

    img_url = urljoin(url, m.group(1))
    print("Resolved image URL:", img_url)

    r2 = requests.get(img_url, headers=HEADERS, timeout=30)
    r2.raise_for_status()

    with Image.open(io.BytesIO(r2.content)) as image:
        return image.convert("RGB")


def make_overlay(img, masks):
    """Blend predicted masks onto the image; returns PNG bytes or None."""
    import io
    import numpy as np
    from PIL import Image

    try:
        base = np.asarray(img).astype(np.float32)
        H, W = base.shape[:2]
        flat = []
        coverages = []

        def walk(x):
            if isinstance(x, (list, tuple)):
                for i in x:
                    walk(i)
            else:
                if hasattr(x, "detach"):
                    x = x.detach().float().cpu().numpy()

                x = np.asarray(x)

                if x.ndim > 2:
                    for item in x:
                        walk(item)
                elif x.ndim == 2:
                    flat.append(x)

        walk(masks)

        colors = [
            (0, 220, 255),
            (255, 0, 220),
            (0, 230, 80),
            (40, 120, 255)
        ]

        labels = np.full((H, W), -1, dtype=np.int16)

        for i, m in enumerate(flat):
            m = np.asarray(m).squeeze()
            # Follow TerraScope's demo: model masks may be probabilities in
            # [0, 1] or byte-valued masks in [0, 255].
            if m.dtype != np.bool_:
                m = np.nan_to_num(m)
                m = m > (0.5 if m.size == 0 or float(m.max()) <= 1.0 else 127)
            else:
                m = m.astype(bool, copy=False)

            if m.shape != (H, W):
                m = np.asarray(
                    Image.fromarray(
                        m.astype(np.uint8) * 255
                    ).resize(
                        (W, H),
                        resample=Image.Resampling.NEAREST
                    )
                ) > 127

            coverage = float(m.mean())
            pixel_count = int(m.sum())
            coverages.append(coverage * 100)
            print(
                f"Mask {i + 1}: {pixel_count} pixels "
                f"({coverage:.1%} image coverage)"
            )

            # Keep the latest mask label at each pixel, then blend once below.
            # Repeated in-place blending made overlapping masks turn orange.
            labels[m] = i % len(colors)

        for i, color in enumerate(colors):
            region = labels == i

            if region.any():
                c = np.asarray(color, dtype=np.float32)
                base[region] = 0.72 * base[region] + 0.28 * c

        if not (labels >= 0).any():
            return None, []

        buf = io.BytesIO()

        # A full-resolution PNG overlay can be several megabytes, making Modal's
        # return-value transfer look stalled before the local JSON is printed.
        Image.fromarray(base.astype(np.uint8)).save(
            buf,
            format="JPEG",
            quality=85,
            optimize=True
        )

        return buf.getvalue(), coverages

    except Exception as e:
        print("Overlay failed (non-fatal):", e)
        return None, []


TERRASCOPE_GROUNDING_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The Assistant first thinks about the reasoning "
    "process in their mind and then provides the user a concise final answer in a "
    "short word or phrase. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>\n\n"
)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    volumes={"/cache": vol},
    env={"HF_HOME": "/cache"},
)
def run_test(image_bytes, prompt, compare_image_bytes=None):
    import time, traceback
    import torch

    install_flash_attn_stub()

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("transformers version:", transformers.__version__)

    patch_terrascope_config()

    img = load_image(None, image_bytes)
    print("Image size:", img.size)

    compare_img = (
        load_image(None, compare_image_bytes)
        if compare_image_bytes
        else None
    )

    if compare_img is not None and compare_img.size != img.size:
        raise ValueError(
            "Comparison images must have the same dimensions for pixel-level change masks."
        )

    if compare_img is not None:
        print("Comparison image size:", compare_img.size)

    # Match the official TerraScope demo's model class and loading path.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="cuda:0",
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=False
    )

    vol.commit()

    print("Param dtypes:", {p.dtype for p in model.parameters()})
    print(
        f"GPU mem after load: "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB"
    )

    t0 = time.time()

    try:
        with torch.no_grad():

            if compare_img is not None:
                if not hasattr(
                    model,
                    "predict_forward_with_grounding_multi"
                ):
                    raise RuntimeError(
                        "This TerraScope model does not expose multi-image grounding."
                    )

                grounded_prompt = (
                    TERRASCOPE_GROUNDING_PROMPT
                    + "<image>\n"
                    + prompt
                    + "\n"
                )

                out = model.predict_forward_with_grounding_multi(
                    image_list=[img, compare_img],
                    text=grounded_prompt,
                    tokenizer=tokenizer,
                    max_tokens_per_seg=8,
                )

            elif hasattr(model, "predict_forward_with_grounding"):
                # Let TerraScope decide whether this question needs pixel
                # grounding; avoid a fixed list of object names or task words.
                grounded_prompt = (
                    TERRASCOPE_GROUNDING_PROMPT
                    + "<image>\n"
                    + prompt
                    + "\n"
                )

                out = model.predict_forward_with_grounding(
                    image=img,
                    text=grounded_prompt,
                    tokenizer=tokenizer,
                    max_tokens_per_seg=8,
                )

            else:
                out = model.predict_forward(
                    image=img,
                    text=f"<image>{prompt}",
                    tokenizer=tokenizer
                )

    except Exception:
        traceback.print_exc()
        raise

    print(
        f"Inference: {time.time()-t0:.1f}s | "
        f"peak mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB"
    )

    if isinstance(out, dict):
        text = out.get("prediction", str(out))
        masks = out.get("prediction_masks", [])
    else:
        text, masks = str(out), []

    if masks is None:
        masks = []

    if compare_img is not None:
        # Match TerraScope's multi-image demo: render the second-frame mask.
        import numpy as np

        post_masks = []

        for mask in masks:
            if hasattr(mask, "detach"):
                mask = mask.detach().float().cpu().numpy()

            mask = np.asarray(mask)

            if mask.ndim == 4 and mask.shape[0] == 1:
                mask = mask[0]

            if mask.ndim == 3 and mask.shape[0] >= 2:
                mask = mask[1]

            post_masks.append(mask)

        masks = post_masks

    mask_count = len(masks)

    print("Num masks:", mask_count)

    overlay_image = compare_img if compare_img is not None else img

    overlay, coverages = (
        make_overlay(overlay_image, masks)
        if mask_count
        else (None, [])
    )

    return text, overlay, coverages


@app.function(
    image=image,
    timeout=1800,
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_app():
    """Serve the TerraScope FastAPI API at a public Modal HTTPS URL."""
    return fastapi_web_app


@app.local_entrypoint()
def main(
    image_path: str = "",
    compare_image_path: str = "",
    prompt: str = "Describe this satellite image.",
    output_path: str = "overlay.jpg",
):
    if not image_path:
        raise ValueError("Provide an image file with --image-path.")

    image_file = Path(image_path)

    if not image_file.is_file():
        raise ValueError(f"Image file not found: {image_file}")

    raw = image_file.read_bytes()
    compare_raw = None

    if compare_image_path:
        compare_file = Path(compare_image_path)

        if not compare_file.is_file():
            raise ValueError(f"Comparison image file not found: {compare_file}")

        compare_raw = compare_file.read_bytes()

    text, overlay, coverages = run_test.remote(
        raw,
        prompt,
        compare_raw
    )

    text = (
        str(text)
        .replace("[SEG]", "")
        .replace("<|im_end|>", "")
        .strip()
    )

    answer_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        re.I | re.S
    )

    if answer_match:
        text = answer_match.group(1).strip()

    text = re.sub(
        r"</?(?:think|answer|p)>",
        "",
        text,
        flags=re.I
    ).strip()

    result = {"answer": text}

    asks_area = bool(
        re.search(
            r"\b(area|proportion|percentage|percent|coverage|how much)\b",
            prompt,
            re.I
        )
    )

    if asks_area and coverages:
        result["answer"] += (
            f" Estimated target coverage: {coverages[0]:.1f}% of image pixels. "
            "This is pixel coverage, not a real-world area measurement."
        )

    if overlay:
        output_file = Path(output_path)
        output_file.write_bytes(overlay)

        result["image_path"] = str(output_file.resolve())

        if coverages:
            result["image_coverage_percent"] = [
                round(value, 1)
                for value in coverages
            ]

    elif asks_area and not coverages:
        result["answer"] += (
            " I couldn't calculate image coverage because no segmentation "
            "mask was returned."
        )

    print(json.dumps(result, ensure_ascii=False))
