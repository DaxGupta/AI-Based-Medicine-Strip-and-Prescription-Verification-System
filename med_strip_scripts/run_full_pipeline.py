"""Run orientation -> preprocessing -> OCR using separate modules."""

import argparse

from med_strip_modules.pipeline import run_full_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run full medicine-strip OCR pipeline")
    parser.add_argument("--input_folder", required=True, help="Folder containing raw medicine-strip images")
    parser.add_argument("--orientation_folder", default="oriented_images")
    parser.add_argument("--preprocessed_folder", default="preprocessed_images")
    parser.add_argument("--ocr_output_folder", default="medicine_strip_ocr_output")
    parser.add_argument("--csv_path", default="medicine_strip_ocr_output.csv")
    parser.add_argument("--model_path", default="quad_vgg16_orientation_classifier.pth")
    parser.add_argument("--output_size", type=int, default=1000)
    parser.add_argument("--languages", nargs="+", default=["en"])
    parser.add_argument("--use_gpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_pipeline(
        input_folder=args.input_folder,
        orientation_folder=args.orientation_folder,
        preprocessed_folder=args.preprocessed_folder,
        ocr_output_folder=args.ocr_output_folder,
        csv_path=args.csv_path,
        model_path=args.model_path,
        output_size=args.output_size,
        languages=args.languages,
        use_gpu=args.use_gpu,
    )
