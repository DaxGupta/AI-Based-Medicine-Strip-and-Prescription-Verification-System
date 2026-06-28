"""Run only Stage 3: OCR."""

import argparse

from med_strip_modules.ocr import batch_ocr_folder


def parse_args():
    parser = argparse.ArgumentParser(description="Run OCR only")
    parser.add_argument("--input_folder", required=True, help="Folder containing preprocessed images")
    parser.add_argument("--output_folder", default="medicine_strip_ocr_output", help="Folder to save OCR bounding-box images")
    parser.add_argument("--csv_path", default="medicine_strip_ocr_output.csv", help="CSV output path")
    parser.add_argument("--languages", nargs="+", default=["en"], help="EasyOCR language list, for example: en")
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU for EasyOCR if available")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_ocr_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        csv_path=args.csv_path,
        languages=args.languages,
        use_gpu=args.use_gpu,
    )
