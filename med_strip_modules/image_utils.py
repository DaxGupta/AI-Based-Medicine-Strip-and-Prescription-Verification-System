"""Shared image utilities for the medicine-strip OCR pipeline."""

from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".avif"
}


def rotate_bound_rgba(image, angle):
    """Rotate an RGBA/BGRA image without cropping the corners."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def crop_to_mask(image, mask, pad=10):
    """Crop image to non-zero mask pixels with padding."""
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return image

    h, w = image.shape[:2]
    x1 = max(0, xs.min() - pad)
    y1 = max(0, ys.min() - pad)
    x2 = min(w, xs.max() + pad)
    y2 = min(h, ys.max() + pad)

    return image[y1:y2 + 1, x1:x2 + 1]


def straighten_rgba_by_alpha(rgba):
    """Fine-straighten a strip using the alpha-mask contour."""
    alpha = rgba[:, :, 3]
    mask = (alpha > 20).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return rgba, 0.0

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = rect[-1]

    if angle < -45:
        angle += 90

    rotated = rotate_bound_rgba(rgba, angle)
    new_alpha = rotated[:, :, 3]
    new_mask = (new_alpha > 20).astype(np.uint8) * 255

    return crop_to_mask(rotated, new_mask, pad=8), angle


def get_image_paths(input_folder):
    """Return supported image paths from a folder."""
    input_folder = Path(input_folder)
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    return sorted(
        path for path in input_folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
