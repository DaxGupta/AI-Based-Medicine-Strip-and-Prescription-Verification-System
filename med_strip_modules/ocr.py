"""Stage 3: docTR detection + EasyOCR recognition.

This module is based on Apply_doctr_easyocr_filtered_final.ipynb. It can be used independently.
"""

import csv
import tempfile
from pathlib import Path

import cv2
import easyocr
import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_avif  # noqa: F401  # Registers AVIF support in Pillow.
except Exception:
    pillow_avif = None

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from .image_utils import get_image_paths

MIN_BOX_HEIGHT = 10
HORIZONTAL_RATIO = 0.8
HEIGHT_THRESHOLD_RATIO = 0.6
MIN_CONFIDENCE = 0.0
EASYOCR_DECODER = "greedy"
MIN_EASYOCR_CONFIDENCE = 0.0


def load_image(image_path):
    """Load image as BGR; use Pillow fallback for formats OpenCV cannot decode."""
    image_path = Path(image_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is not None:
        return image

    try:
        with Image.open(image_path) as pil_image:
            pil_image = ImageOps.exif_transpose(pil_image)
            pil_image = pil_image.convert("RGB")
            image_rgb = np.array(pil_image)

        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    except Exception as pillow_error:
        raise ValueError(
            f"Could not read image: {image_path}. "
            "OpenCV could not decode it, and Pillow fallback also failed. "
            "For AVIF files, run: pip install pillow-avif-plugin"
        ) from pillow_error


def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = " ".join(text.split())
    return text.strip()


def prepare_image_for_doctr(image_path):
    """Convert one image to a temporary PNG for docTR."""
    image = load_image(image_path)

    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    if not cv2.imwrite(str(temp_path), image):
        raise ValueError(f"Could not write temporary PNG: {temp_path}")

    return temp_path, image


def extract_doctr_detections(page_result, image_shape, min_confidence=0.0):
    """Extract docTR word boxes as pixel rectangles."""
    img_h, img_w = image_shape[:2]
    detections = []

    for block in page_result.blocks:
        for line in block.lines:
            for word in line.words:
                confidence = getattr(word, "confidence", 1.0)
                if confidence < min_confidence:
                    continue

                (x_min, y_min), (x_max, y_max) = word.geometry

                x1 = int(x_min * img_w)
                y1 = int(y_min * img_h)
                x2 = int(x_max * img_w)
                y2 = int(y_max * img_h)

                if x2 <= x1 or y2 <= y1:
                    continue
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    continue

                pts = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.int32,
                )

                detections.append(
                    {
                        "points": pts,
                        "confidence": confidence,
                        "rect": (x1, y1, x2, y2),
                    }
                )

    return detections


def filter_horizontal_large_boxes(
    detections,
    image_shape,
    min_box_height=10,
    horizontal_ratio=0.8,
    height_threshold_ratio=0.6,
):
    """Keep only large horizontal docTR boxes."""
    valid_boxes = []
    max_height = 0

    for item in detections:
        pts = item["points"]

        if pts.shape != (4, 2):
            continue

        width = np.linalg.norm(pts[0] - pts[1])
        height = np.linalg.norm(pts[0] - pts[3])

        if height >= width * horizontal_ratio:
            continue
        if height < min_box_height:
            continue

        item["height"] = height
        valid_boxes.append(item)
        max_height = max(max_height, height)

    if not valid_boxes or max_height == 0:
        return []

    threshold = max_height * height_threshold_ratio
    filtered_boxes = []
    img_h, img_w = image_shape[:2]

    for item in valid_boxes:
        pts = item["points"]
        height = item["height"]

        if height < threshold:
            continue

        x_min = max(0, int(np.min(pts[:, 0])))
        y_min = max(0, int(np.min(pts[:, 1])))
        x_max = min(img_w, int(np.max(pts[:, 0])))
        y_max = min(img_h, int(np.max(pts[:, 1])))

        if x_max <= x_min or y_max <= y_min:
            continue

        item["rect"] = (x_min, y_min, x_max, y_max)
        filtered_boxes.append(item)

    return sorted(filtered_boxes, key=lambda item: (item["rect"][1], item["rect"][0]))


def expand_box(x1, y1, x2, y2, image_shape, pad_x=10, pad_y=8):
    img_h, img_w = image_shape[:2]

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)

    return x1, y1, x2, y2


def preprocess_crop_for_easyocr(crop):
    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]
    if h < 8 or w < 12:
        return None

    if h < 48:
        scale = 48 / h
        new_w = max(int(w * scale), 80)
        crop = cv2.resize(crop, (new_w, 48), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def recognize_crop_with_easyocr(crop, reader):
    crop = preprocess_crop_for_easyocr(crop)
    if crop is None:
        return ""

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    try:
        results = reader.readtext(
            crop_rgb,
            detail=1,
            paragraph=False,
            decoder=EASYOCR_DECODER,
        )
    except Exception:
        return ""

    texts = []
    for _bbox, text, confidence in results:
        text = clean_text(text)
        if text and confidence >= MIN_EASYOCR_CONFIDENCE:
            texts.append(text)

    return clean_text(" ".join(texts))


def recognize_text_from_filtered_boxes_with_easyocr(image, filtered_boxes, reader):
    extracted_texts = []

    for item in filtered_boxes:
        x1, y1, x2, y2 = item["rect"]
        x1, y1, x2, y2 = expand_box(
            x1,
            y1,
            x2,
            y2,
            image.shape,
            pad_x=10,
            pad_y=8,
        )

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        text = recognize_crop_with_easyocr(crop, reader)
        if text:
            extracted_texts.append(text)

    return clean_text(" ".join(extracted_texts))


def draw_filtered_boxes_only(image, filtered_boxes):
    output_img = image.copy()

    for item in filtered_boxes:
        pts = item["points"]
        cv2.polylines(output_img, [pts], True, (0, 255, 0), 2)

    return output_img


def ocr_single_image(image_path, output_folder, doctr_predictor, easyocr_reader):
    """Run docTR detection + EasyOCR recognition for one image."""
    image_path = Path(image_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    temporary_image_path = None

    try:
        temporary_image_path, original_image = prepare_image_for_doctr(image_path)

        doc = DocumentFile.from_images(str(temporary_image_path))
        result = doctr_predictor(doc)
        page_result = result.pages[0]

        detections = extract_doctr_detections(
            page_result=page_result,
            image_shape=original_image.shape,
            min_confidence=MIN_CONFIDENCE,
        )

        filtered_boxes = filter_horizontal_large_boxes(
            detections=detections,
            image_shape=original_image.shape,
            min_box_height=MIN_BOX_HEIGHT,
            horizontal_ratio=HORIZONTAL_RATIO,
            height_threshold_ratio=HEIGHT_THRESHOLD_RATIO,
        )

        full_extracted_text = recognize_text_from_filtered_boxes_with_easyocr(
            image=original_image,
            filtered_boxes=filtered_boxes,
            reader=easyocr_reader,
        )

        annotated_image = draw_filtered_boxes_only(original_image, filtered_boxes)
        output_image_path = output_folder / f"{image_path.stem}_bbox.png"

        if not cv2.imwrite(str(output_image_path), annotated_image):
            raise ValueError(f"Could not write boxed image: {output_image_path}")

        return full_extracted_text, str(output_image_path)

    finally:
        if temporary_image_path is not None:
            try:
                temporary_image_path.unlink()
            except FileNotFoundError:
                pass


def initialize_ocr_models(languages=None, use_gpu=False):
    """Initialize docTR detector/recognizer and EasyOCR."""
    if languages is None:
        languages = ["en"]

    print("Loading docTR detector...")
    doctr_predictor = ocr_predictor(
        det_arch="db_resnet50",
        reco_arch="crnn_vgg16_bn",
        pretrained=True,
    )

    print("Loading EasyOCR recognizer...")
    easyocr_reader = easyocr.Reader(
        languages,
        gpu=use_gpu,
        detector=True,
        recognizer=True,
        verbose=False,
    )

    return doctr_predictor, easyocr_reader


def batch_ocr_folder(input_folder, output_folder="medicine_strip_ocr_output", csv_path="medicine_strip_ocr_output.csv", languages=None, use_gpu=False):
    """Run OCR on all supported images in a folder and write a CSV."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    csv_path = Path(csv_path)
    output_folder.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = get_image_paths(input_folder)
    doctr_predictor, easyocr_reader = initialize_ocr_models(languages=languages or ["en"], use_gpu=use_gpu)
    rows = []

    for image_path in image_paths:
        print(f"OCR: {image_path.name}")
        try:
            text, bbox_path = ocr_single_image(image_path, output_folder, doctr_predictor, easyocr_reader)
            rows.append({"image_name": image_path.name, "bbox_image": bbox_path, "extracted_text": text})
        except Exception as error:
            print(f"Error running OCR on {image_path.name}: {error}")
            rows.append({"image_name": image_path.name, "bbox_image": "", "extracted_text": "", "error": str(error)})

    fieldnames = ["image_name", "bbox_image", "extracted_text"]
    if any("error" in row for row in rows):
        fieldnames.append("error")

    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows
