"""Optional full pipeline that chains the separate modules."""

import csv
from pathlib import Path

from .image_utils import get_image_paths
from .ocr import initialize_ocr_models, ocr_single_image
from .orientation import load_trained_orientation_model, orientation_correct_image
from .preprocessing import preprocess_oriented_image


def run_full_pipeline(
    input_folder,
    orientation_folder="oriented_images",
    preprocessed_folder="preprocessed_images",
    ocr_output_folder="medicine_strip_ocr_output",
    csv_path="medicine_strip_ocr_output.csv",
    model_path="quad_vgg16_orientation_classifier.pth",
    output_size=1000,
    languages=None,
    use_gpu=False,
):
    """Run orientation -> preprocessing -> OCR using separate modules."""
    input_folder = Path(input_folder)
    orientation_folder = Path(orientation_folder)
    preprocessed_folder = Path(preprocessed_folder)
    ocr_output_folder = Path(ocr_output_folder)
    csv_path = Path(csv_path)

    orientation_folder.mkdir(parents=True, exist_ok=True)
    preprocessed_folder.mkdir(parents=True, exist_ok=True)
    ocr_output_folder.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = get_image_paths(input_folder)
    if not image_paths:
        print("No supported images found.")
        return []

    orientation_model = load_trained_orientation_model(model_path)
    doctr_predictor, easyocr_reader = initialize_ocr_models(
        languages=languages or ["en"],
        use_gpu=use_gpu,
    )

    rows = []

    for image_path in image_paths:
        print(f"\nProcessing: {image_path.name}")
        oriented_path = orientation_folder / f"{image_path.stem}_oriented.png"
        preprocessed_path = preprocessed_folder / f"{image_path.stem}_preprocessed.png"

        try:
            orientation_info = orientation_correct_image(
                input_path=image_path,
                output_path=oriented_path,
                model=orientation_model,
            )
            print(
                "Orientation done "
                f"(fine={orientation_info['fine_angle']:.2f} deg, "
                f"vgg={orientation_info['vgg_angle']} deg): {oriented_path}"
            )

            preprocess_oriented_image(
                input_path=oriented_path,
                output_path=preprocessed_path,
                output_size=output_size,
            )
            print(f"Preprocessing done: {preprocessed_path}")

            text, bbox_path = ocr_single_image(
                image_path=preprocessed_path,
                output_folder=ocr_output_folder,
                doctr_predictor=doctr_predictor,
                easyocr_reader=easyocr_reader,
            )
            print(f"OCR done: {text}")

            rows.append(
                {
                    "image_name": image_path.name,
                    "oriented_image": str(oriented_path),
                    "preprocessed_image": str(preprocessed_path),
                    "bbox_image": str(bbox_path),
                    "extracted_text": text,
                }
            )

        except Exception as error:
            print(f"Error processing {image_path.name}: {error}")
            rows.append(
                {
                    "image_name": image_path.name,
                    "oriented_image": str(oriented_path),
                    "preprocessed_image": str(preprocessed_path),
                    "bbox_image": "",
                    "extracted_text": "",
                    "error": str(error),
                }
            )

    fieldnames = ["image_name", "oriented_image", "preprocessed_image", "bbox_image", "extracted_text"]
    if any("error" in row for row in rows):
        fieldnames.append("error")

    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone.")
    print(f"Orientation-corrected images saved in: {orientation_folder}")
    print(f"Preprocessed images saved in: {preprocessed_folder}")
    print(f"Bounding-box OCR images saved in: {ocr_output_folder}")
    print(f"CSV saved at: {csv_path}")

    return rows
