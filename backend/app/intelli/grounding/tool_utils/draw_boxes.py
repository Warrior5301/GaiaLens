from PIL import Image, ImageDraw, ImageFont

from backend.app.intelli.common import save_image

# Qwen3-VL expresses boxes in coordinates relative to the image, normalised to 0-1000
QWEN_RANGE = 1000

PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c"]


def to_pixel_box(box, width: int, height: int, coord_system: str = "pixel") -> list[int]:
    """
    Convert [x_min, y_min, x_max, y_max] to ordered integer pixel coordinates of the original image.

    coord_system:
      "pixel" - already in pixels of the original image (what the grounding tools return)
      "qwen"  - Qwen3-VL relative coordinates, 0-1000 regardless of the image size
                (pixel = value / 1000 * width or height)
    """
    if len(box) != 4:
        raise ValueError(f"A box needs 4 numbers, got {box!r}")
    x1, y1, x2, y2 = (float(v) for v in box)

    if coord_system == "qwen":
        x1, x2 = x1 / QWEN_RANGE * width, x2 / QWEN_RANGE * width
        y1, y2 = y1 / QWEN_RANGE * height, y2 / QWEN_RANGE * height
    elif coord_system != "pixel":
        raise ValueError(f"Unknown coord_system {coord_system!r}; use 'pixel' or 'qwen'")

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [
        round(min(max(x1, 0), width)), round(min(max(y1, 0), height)),
        round(min(max(x2, 0), width)), round(min(max(y2, 0), height)),
    ]


def draw_bboxes(image_path: str, objects: list[dict]) -> tuple[str, list[dict]]:
    """
    Draw labelled bounding boxes on a copy of the image.

    objects: [{"label": str, "bbox": [x_min, y_min, x_max, y_max], "coord_system": "pixel" | "qwen"}, ...]
    Returns (path of the saved PNG, the drawn objects with their boxes converted to original-image pixels).
    Boxes that are invalid or have no area after conversion are skipped.
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    line = max(2, round(min(width, height) / 300))
    font = ImageFont.load_default(size=max(12, min(width, height) // 40))
    draw = ImageDraw.Draw(image)

    drawn = []
    for obj in objects:
        try:
            box = to_pixel_box(obj["bbox"], width, height, obj.get("coord_system", "pixel"))
        except (KeyError, TypeError, ValueError):
            continue
        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        label = str(obj.get("label") or "")
        color = PALETTE[len(drawn) % len(PALETTE)]
        draw.rectangle(box, outline=color, width=line)
        if label:
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = right - left, bottom - top
            # Label above the box when there is room, otherwise inside its top edge
            y = box[1] - text_h - 6 if box[1] - text_h - 6 >= 0 else box[1]
            draw.rectangle([box[0], y, box[0] + text_w + 6, y + text_h + 6], fill=color)
            draw.text((box[0] + 3, y + 3 - top), label, fill="white", font=font)
        drawn.append({"label": label, "bbox": box})

    return save_image(image), drawn
