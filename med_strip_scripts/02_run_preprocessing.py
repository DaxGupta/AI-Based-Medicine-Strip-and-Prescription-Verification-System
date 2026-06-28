"""Run only Stage 2: preprocessing."""

import argparse

from med_strip_modules.preprocessing import batch_preprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Run preprocessing only")
    parser.add_argument("--input_folder", required=True, help="Folder containing oriented images")
    parser.add_argument("--output_folder", default="preprocessed_images", help="Folder to save preprocessed images")
    parser.add_argument("--output_size", type=int, default=1000, help="Square canvas size for preprocessed output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_preprocess(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        output_size=args.output_size,
    )
