"""Stage 1: orientation correction for medicine-strip images.

This module is based on Part1.ipynb. It can be used independently.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from rembg import remove
from torchvision import models, transforms

from .image_utils import crop_to_mask, get_image_paths, rotate_bound_rgba, straighten_rgba_by_alpha

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)
INV_ANGLE_MAP = {0: 0, 1: 90, 2: 180, 3: 270}


class VGG16RotationClassifier(nn.Module):
    """VGG16-based 0/90/180/270 orientation classifier from Part1.ipynb."""

    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(weights=None)
        self.features = vgg16.features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def load_trained_orientation_model(weights_path):
    """Load VGG16 orientation weights. Return None when weights are missing."""
    weights_path = Path(weights_path)

    if not weights_path.exists():
        print(
            f"Warning: orientation model weights not found at {weights_path}. "
            "VGG 0/90/180/270 correction will be skipped."
        )
        return None

    model = VGG16RotationClassifier().to(DEVICE)
    model.load_state_dict(torch.load(str(weights_path), map_location=DEVICE))
    model.eval()
    return model


def get_vgg_correction_rgba(rgba_img, model):
    """Predict current 0/90/180/270 misalignment using the VGG model."""
    if model is None:
        return 0

    rgb_img = rgba_img[:, :, :3]

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    input_tensor = transform(rgb_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred_idx = torch.max(outputs, 1)

    return INV_ANGLE_MAP[pred_idx.item()]


def orientation_correct_image(input_path, output_path, model=None):
    """Orientation-correct one image and save a transparent PNG."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_image = Image.open(input_path)
    input_image = ImageOps.exif_transpose(input_image)

    no_bg_image = remove(input_image).convert("RGBA")
    rgba = np.array(no_bg_image).astype(np.uint8)

    alpha = rgba[:, :, 3]
    initial_mask = (alpha > 20).astype(np.uint8) * 255
    rgba = crop_to_mask(rgba, initial_mask, pad=12)

    straightened_rgba, fine_angle = straighten_rgba_by_alpha(rgba)

    vgg_angle = get_vgg_correction_rgba(straightened_rgba, model)
    if vgg_angle != 0:
        final_rgba = rotate_bound_rgba(straightened_rgba, -vgg_angle)
        final_alpha = final_rgba[:, :, 3]
        final_rgba = crop_to_mask(final_rgba, (final_alpha > 20).astype(np.uint8) * 255, pad=8)
    else:
        final_rgba = straightened_rgba

    final_bgra = cv2.cvtColor(final_rgba, cv2.COLOR_RGBA2BGRA)
    if not cv2.imwrite(str(output_path), final_bgra):
        raise ValueError(f"Could not write orientation-corrected image: {output_path}")

    return {"fine_angle": float(fine_angle), "vgg_angle": int(vgg_angle), "output_path": str(output_path)}


def batch_orientation_correction(input_folder, output_folder="oriented_images", model_path="quad_vgg16_orientation_classifier.pth"):
    """Run orientation correction on all images in a folder."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = get_image_paths(input_folder)
    model = load_trained_orientation_model(model_path)
    results = []

    for image_path in image_paths:
        output_path = output_folder / f"{image_path.stem}_oriented.png"
        print(f"Orienting: {image_path.name}")
        try:
            info = orientation_correct_image(image_path, output_path, model=model)
            info["image_name"] = image_path.name
            results.append(info)
        except Exception as error:
            print(f"Error orienting {image_path.name}: {error}")
            results.append({"image_name": image_path.name, "output_path": str(output_path), "error": str(error)})

    return results
