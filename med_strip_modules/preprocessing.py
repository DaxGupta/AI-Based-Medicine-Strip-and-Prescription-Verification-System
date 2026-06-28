"""Stage 2: preprocessing for medicine-strip images.

This module is based on For_Single_image.ipynb. It can be used independently.
"""

from pathlib import Path

import cv2
import numpy as np
from rembg import remove

from .image_utils import crop_to_mask, get_image_paths, straighten_rgba_by_alpha


def odd(value, minimum=3, maximum=None):
    """Round a value to an odd integer inside optional bounds."""
    value = int(round(value))

    if value % 2 == 0:
        value += 1

    value = max(value, minimum)

    if maximum is not None:
        if maximum % 2 == 0:
            maximum -= 1
        value = min(value, maximum)

    return value


def resize_fit_rgba(rgba, out_w=1000, out_h=1000):
    """Resize RGBA image to fit centered on a transparent canvas."""
    h, w = rgba.shape[:2]
    scale = min(out_w / w, out_h / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((out_h, out_w, 4), dtype=np.uint8)

    x = (out_w - new_w) // 2
    y = (out_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def remove_small_black_components(binary, strip_mask, strip_w, strip_h):
    """Remove salt-pepper / foil speckle noise while keeping likely text."""
    foreground = ((binary == 0) & strip_mask).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
    )

    cleaned_foreground = np.zeros_like(foreground)
    strip_area = strip_w * strip_h

    min_area = max(5, int(strip_area * 0.000006))
    max_area = int(strip_area * 0.08)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        if area < min_area:
            continue
        if area > max_area:
            continue

        aspect = width / float(height + 1e-5)

        if area < 25 and (aspect > 8 or aspect < 0.12):
            continue
        if width <= 2 and height <= 2:
            continue

        cleaned_foreground[labels == i] = 255

    cleaned = np.full_like(binary, 255)
    cleaned[cleaned_foreground > 0] = 0
    cleaned[~strip_mask] = 255

    return cleaned


def load_rgba_for_preprocessing(input_path):
    """Load image as RGBA; keep alpha or run rembg when alpha is absent."""
    input_path = Path(input_path)
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Could not load image: {input_path}")

    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA).astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    img_rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    no_bg = remove(img_rgb)

    if no_bg.shape[2] == 4:
        return no_bg.astype(np.uint8)

    alpha = np.full(no_bg.shape[:2], 255, dtype=np.uint8)
    return np.dstack([no_bg, alpha]).astype(np.uint8)


def preprocess_oriented_image(input_path, output_path, output_size=1000):
    """Preprocess one oriented image and save a binary PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgba = load_rgba_for_preprocessing(input_path)

    alpha = rgba[:, :, 3]
    initial_mask = (alpha > 20).astype(np.uint8) * 255
    rgba = crop_to_mask(rgba, initial_mask, pad=12)
    rgba, _ = straighten_rgba_by_alpha(rgba)
    rgba = resize_fit_rgba(rgba, out_w=output_size, out_h=output_size)

    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    strip_mask = alpha > 20

    ys, xs = np.where(strip_mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No medicine strip detected after background removal.")

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    strip_w = x2 - x1 + 1
    strip_h = y2 - y1 + 1
    strip_min = min(strip_w, strip_h)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray[~strip_mask] = 255

    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    bg_kernel_size = odd(strip_min * 0.09, minimum=41, maximum=121)
    bg_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (bg_kernel_size, bg_kernel_size),
    )

    background = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, bg_kernel)
    normalized = cv2.divide(denoised, background, scale=255)
    normalized[~strip_mask] = 255

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)
    enhanced[~strip_mask] = 255

    block_size = odd(strip_min * 0.075, minimum=35, maximum=101)
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        7,
    )
    binary[~strip_mask] = 255

    binary = cv2.medianBlur(binary, 3)
    binary = remove_small_black_components(binary, strip_mask, strip_w, strip_h)

    inv = 255 - binary
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, small_kernel, iterations=1)

    binary = 255 - inv
    binary[~strip_mask] = 255

    if not cv2.imwrite(str(output_path), binary):
        raise ValueError(f"Could not write preprocessed image: {output_path}")

    return str(output_path)


def batch_preprocess(input_folder, output_folder="preprocessed_images", output_size=1000):
    """Run preprocessing on all supported images in a folder."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = get_image_paths(input_folder)
    results = []

    for image_path in image_paths:
        output_path = output_folder / f"{image_path.stem}_preprocessed.png"
        print(f"Preprocessing: {image_path.name}")
        try:
            saved_path = preprocess_oriented_image(image_path, output_path, output_size=output_size)
            results.append({"image_name": image_path.name, "output_path": saved_path})
        except Exception as error:
            print(f"Error preprocessing {image_path.name}: {error}")
            results.append({"image_name": image_path.name, "output_path": str(output_path), "error": str(error)})

    return results
