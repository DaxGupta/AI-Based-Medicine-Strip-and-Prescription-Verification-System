"""Run only Stage 1: orientation correction."""

import argparse

from med_strip_modules.orientation import batch_orientation_correction


def parse_args():
    parser = argparse.ArgumentParser(description="Run orientation correction only")
    parser.add_argument("--input_folder", required=True, help="Folder containing raw medicine-strip images")
    parser.add_argument("--output_folder", default="oriented_images", help="Folder to save oriented PNG images")
    parser.add_argument("--model_path", default="quad_vgg16_orientation_classifier.pth", help="Optional VGG orientation weights")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_orientation_correction(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        model_path=args.model_path,
    )
