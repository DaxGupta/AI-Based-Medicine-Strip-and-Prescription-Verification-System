"""
Map medicine-strip images to prescription medicines with a one-to-one rule.

Rule enforced in final output
-----------------------------
1. One image is mapped to at most one medicine.
2. One medicine_name is mapped to at most one image.
3. No two or more images can have the same medicine_name.

Inputs
------
1. medicine_strip_ocr_output.csv
   Required columns:
       image_name
       extracted_text

2. final_combi.csv
   Required columns:
       medicine_name
       salt_name

Matching logic
--------------
For every image and prescription medicine row, compare image extracted_text with:
    1. medicine_name
    2. salt_name
    3. medicine_name + salt_name

The script first builds all candidate scores, then performs a one-to-one assignment.
If scipy is installed, it uses the Hungarian algorithm for globally best assignment.
If scipy is not installed, it falls back to a deterministic greedy assignment.

Install dependency:
    pip install pandas rapidfuzz

Optional, for globally optimal assignment:
    pip install scipy

Example:
    python map_unique_image_to_unique_medicine.py ^
      --strip_csv medicine_strip_ocr_output.csv ^
      --prescription_csv final_combi.csv ^
      --output_csv final_unique_image_to_medicine_mapping.csv ^
      --all_candidates_csv all_unique_assignment_candidates.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from rapidfuzz import fuzz




FINAL_OUTPUT_COLUMNS = [
    "image_name",
    "extracted_text",
    "prescription_record_number",
    "prescription_medicine_name",
    "prescription_dosage_form_meaning",
    "prescription_strength",
    "prescription_dose_pattern",
    "prescription_dose_pattern_meaning",
    "prescription_dose_frequency",
    "prescription_dose_frequency_meaning",
    "prescription_duration",
    "prescription_instruction_meaning",
    "prescription_salt_name",
    "medicine_unique_key",
    "final_match_score",
    "match_confidence",
]


def keep_only_final_output_columns(final_mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Return the final mapping with exactly the requested columns and order.

    Any missing columns are created as blank values so the output schema remains
    stable even when an image is unmatched or an optional prescription field is
    absent from the input CSV.
    """
    final_output_df = final_mapping_df.copy()

    for column in FINAL_OUTPUT_COLUMNS:
        if column not in final_output_df.columns:
            final_output_df[column] = pd.NA

    return final_output_df[FINAL_OUTPUT_COLUMNS]


# ---------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------
def clean_text(value) -> str:
    """Clean OCR/prescription text for matching."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).lower()

    # Normalize common medicine/OCR separators.
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = text.replace("/", " ")
    text = text.replace("+", " plus ")

    # Keep letters and numbers only.
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def join_non_empty(values: List[str]) -> str:
    """Join only non-empty values."""
    parts = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        value = str(value).strip()
        if value:
            parts.append(value)
    return " ".join(parts)


def make_medicine_unique_key(medicine_name, prescription_row_number: int) -> str:
    """
    Create the uniqueness key used to prevent duplicate medicine assignment.

    Normally this is the cleaned medicine_name. If medicine_name is blank, use
    prescription row number so blank values do not collapse into one key.
    """
    key = clean_text(medicine_name)
    if not key:
        key = f"blank_medicine_row_{prescription_row_number}"
    return key


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------
def fuzzy_score(query_text: str, extracted_text: str) -> Dict[str, float]:
    """
    Score one prescription field against one image OCR text.

    OCR text can be noisy and long, so partial/token scores are often better
    than a simple full-string ratio.
    """
    query = clean_text(query_text)
    target = clean_text(extracted_text)

    if not query or not target:
        return {
            "ratio": 0.0,
            "partial_ratio": 0.0,
            "token_sort_ratio": 0.0,
            "token_set_ratio": 0.0,
            "weighted_ratio": 0.0,
            "score": 0.0,
        }

    ratio = fuzz.ratio(query, target)
    partial_ratio = fuzz.partial_ratio(query, target)
    token_sort_ratio = fuzz.token_sort_ratio(query, target)
    token_set_ratio = fuzz.token_set_ratio(query, target)
    weighted_ratio = fuzz.WRatio(query, target)

    # Main score: best OCR-friendly score.
    score = max(partial_ratio, token_set_ratio, weighted_ratio)

    return {
        "ratio": round(ratio, 2),
        "partial_ratio": round(partial_ratio, 2),
        "token_sort_ratio": round(token_sort_ratio, 2),
        "token_set_ratio": round(token_set_ratio, 2),
        "weighted_ratio": round(weighted_ratio, 2),
        "score": round(score, 2),
    }


def choose_best_match_basis(
    medicine_name_score: float,
    salt_name_score: float,
    combined_score: float,
) -> Tuple[str, float]:
    """Select the best among medicine-name, salt-name, and combined matching."""
    score_map = {
        "medicine_name": medicine_name_score,
        "salt_name": salt_name_score,
        "medicine_name_plus_salt_name": combined_score,
    }
    best_basis = max(score_map, key=score_map.get)
    return best_basis, score_map[best_basis]


def confidence_label(score: float) -> str:
    """Human-readable confidence label."""
    if score >= 85:
        return "High confidence"
    if score >= 70:
        return "Medium confidence"
    if score >= 50:
        return "Low confidence"
    return "Very low confidence"


def assignment_score(row: pd.Series) -> float:
    """
    Score used only for choosing the global one-to-one assignment.

    final_match_score is the main signal. Other fields are tie-breakers.
    """
    return (
        float(row["final_match_score"]) * 1_000_000
        + float(row["medicine_name_plus_salt_name_score"]) * 1_000
        + float(row["medicine_name_score"]) * 10
        + float(row["salt_name_score"]) * 0.1
        - float(row["prescription_row_number"]) * 0.0001
    )


# ---------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------
def build_candidate_scores(strip_df: pd.DataFrame, prescription_df: pd.DataFrame) -> pd.DataFrame:
    """Create all image-vs-medicine candidate match scores."""
    candidate_rows = []

    for _, strip_row in strip_df.iterrows():
        image_name = strip_row["image_name"]
        extracted_text = strip_row["extracted_text"]

        for _, prescription_row in prescription_df.iterrows():
            medicine_name = prescription_row["medicine_name"]
            salt_name = prescription_row["salt_name"]
            medicine_plus_salt = join_non_empty([medicine_name, salt_name])

            medicine_metrics = fuzzy_score(medicine_name, extracted_text)
            salt_metrics = fuzzy_score(salt_name, extracted_text)
            combined_metrics = fuzzy_score(medicine_plus_salt, extracted_text)

            best_basis, final_score = choose_best_match_basis(
                medicine_metrics["score"],
                salt_metrics["score"],
                combined_metrics["score"],
            )

            medicine_unique_key = make_medicine_unique_key(
                medicine_name, int(prescription_row["_prescription_row_number"])
            )

            row = {}

            # Strip/image columns.
            for col in strip_df.columns:
                if col != "_strip_row_number":
                    row[col] = strip_row[col]

            # Prescription columns, prefixed to avoid collisions.
            for col in prescription_df.columns:
                if col != "_prescription_row_number":
                    row[f"prescription_{col}"] = prescription_row[col]

            row["strip_row_number"] = strip_row["_strip_row_number"]
            row["prescription_row_number"] = prescription_row["_prescription_row_number"]
            row["medicine_unique_key"] = medicine_unique_key

            row["clean_extracted_text"] = clean_text(extracted_text)
            row["clean_medicine_name"] = clean_text(medicine_name)
            row["clean_salt_name"] = clean_text(salt_name)
            row["clean_medicine_name_plus_salt_name"] = clean_text(medicine_plus_salt)

            # Main scores for each comparison type.
            row["medicine_name_score"] = medicine_metrics["score"]
            row["salt_name_score"] = salt_metrics["score"]
            row["medicine_name_plus_salt_name_score"] = combined_metrics["score"]

            # Helpful detailed scores.
            row["medicine_name_partial_ratio"] = medicine_metrics["partial_ratio"]
            row["salt_name_partial_ratio"] = salt_metrics["partial_ratio"]
            row["medicine_name_plus_salt_name_partial_ratio"] = combined_metrics[
                "partial_ratio"
            ]

            row["medicine_name_token_set_ratio"] = medicine_metrics["token_set_ratio"]
            row["salt_name_token_set_ratio"] = salt_metrics["token_set_ratio"]
            row["medicine_name_plus_salt_name_token_set_ratio"] = combined_metrics[
                "token_set_ratio"
            ]

            row["best_match_basis"] = best_basis
            row["final_match_score"] = final_score
            row["match_confidence"] = confidence_label(final_score)

            candidate_rows.append(row)

    candidates_df = pd.DataFrame(candidate_rows)
    if candidates_df.empty:
        return candidates_df

    candidates_df["assignment_score"] = candidates_df.apply(assignment_score, axis=1)

    sort_columns = [
        "image_name",
        "final_match_score",
        "medicine_name_plus_salt_name_score",
        "medicine_name_score",
        "salt_name_score",
        "prescription_row_number",
    ]
    ascending = [True, False, False, False, False, True]
    candidates_df = candidates_df.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)

    return candidates_df


# ---------------------------------------------------------------------
# One-to-one assignment
# ---------------------------------------------------------------------
def assign_unique_medicines_optimal(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign images to medicines one-to-one using the Hungarian algorithm.

    This maximizes the total assignment_score and prevents duplicate medicine_name
    use in the final mapping.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return assign_unique_medicines_greedy(candidates_df)

    images = list(candidates_df.sort_values("strip_row_number")["image_name"].drop_duplicates())
    medicines = list(
        candidates_df.sort_values("prescription_row_number")["medicine_unique_key"].drop_duplicates()
    )

    if not images or not medicines:
        return pd.DataFrame()

    image_index = {image: i for i, image in enumerate(images)}
    medicine_index = {medicine: j for j, medicine in enumerate(medicines)}

    # Keep only the best prescription row for each image + medicine_name key.
    best_pairs = (
        candidates_df.sort_values(
            [
                "image_name",
                "medicine_unique_key",
                "assignment_score",
                "final_match_score",
                "prescription_row_number",
            ],
            ascending=[True, True, False, False, True],
        )
        .groupby(["image_name", "medicine_unique_key"], as_index=False, sort=False)
        .head(1)
    )

    # Use negative scores because linear_sum_assignment minimizes cost.
    import numpy as np

    score_matrix = np.full((len(images), len(medicines)), -1e18, dtype=float)
    selected_row_lookup = {}

    for row_id, row in best_pairs.iterrows():
        i = image_index[row["image_name"]]
        j = medicine_index[row["medicine_unique_key"]]
        score_matrix[i, j] = float(row["assignment_score"])
        selected_row_lookup[(i, j)] = row_id

    cost_matrix = -score_matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    chosen_ids = []
    for i, j in zip(row_indices, col_indices):
        row_id = selected_row_lookup.get((i, j))
        if row_id is not None:
            chosen_ids.append(row_id)

    final_mapping_df = best_pairs.loc[chosen_ids].copy().reset_index(drop=True)
    final_mapping_df["assignment_method"] = "optimal_hungarian"

    return final_mapping_df


def assign_unique_medicines_greedy(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback assignment if scipy is unavailable.

    It walks through candidates from best to worst and accepts a pair only if
    both the image and medicine_name have not already been used.
    """
    candidates_df = candidates_df.sort_values(
        [
            "assignment_score",
            "final_match_score",
            "medicine_name_plus_salt_name_score",
            "medicine_name_score",
            "salt_name_score",
            "prescription_row_number",
            "strip_row_number",
        ],
        ascending=[False, False, False, False, False, True, True],
    )

    used_images = set()
    used_medicines = set()
    chosen_rows = []

    for _, row in candidates_df.iterrows():
        image_name = row["image_name"]
        medicine_key = row["medicine_unique_key"]

        if image_name in used_images:
            continue
        if medicine_key in used_medicines:
            continue

        chosen_rows.append(row)
        used_images.add(image_name)
        used_medicines.add(medicine_key)

    final_mapping_df = pd.DataFrame(chosen_rows).reset_index(drop=True)
    if not final_mapping_df.empty:
        final_mapping_df["assignment_method"] = "greedy_fallback"

    return final_mapping_df


def append_unmatched_images(
    final_mapping_df: pd.DataFrame,
    strip_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep one row per input image by appending unmatched images.

    This only matters when there are more images than unique medicine names.
    Unmatched rows do not violate uniqueness because medicine columns are blank.
    """
    matched_images = set(final_mapping_df["image_name"].tolist()) if not final_mapping_df.empty else set()
    unmatched_rows = []

    for _, strip_row in strip_df.iterrows():
        if strip_row["image_name"] in matched_images:
            continue

        row = {}
        for col in strip_df.columns:
            if col != "_strip_row_number":
                row[col] = strip_row[col]

        row["strip_row_number"] = strip_row["_strip_row_number"]
        row["prescription_row_number"] = pd.NA
        row["medicine_unique_key"] = pd.NA
        row["clean_extracted_text"] = clean_text(strip_row["extracted_text"])
        row["clean_medicine_name"] = ""
        row["clean_salt_name"] = ""
        row["clean_medicine_name_plus_salt_name"] = ""
        row["medicine_name_score"] = 0.0
        row["salt_name_score"] = 0.0
        row["medicine_name_plus_salt_name_score"] = 0.0
        row["best_match_basis"] = "unmatched"
        row["final_match_score"] = 0.0
        row["match_confidence"] = "Unmatched - no unused medicine available"
        row["assignment_score"] = 0.0
        row["assignment_method"] = "unmatched"

        unmatched_rows.append(row)

    if unmatched_rows:
        final_mapping_df = pd.concat(
            [final_mapping_df, pd.DataFrame(unmatched_rows)],
            ignore_index=True,
            sort=False,
        )

    return final_mapping_df


def validate_unique_mapping(final_mapping_df: pd.DataFrame) -> None:
    """Raise an error if final output violates one-to-one constraints."""
    duplicate_images = final_mapping_df.loc[
        final_mapping_df["image_name"].duplicated(), "image_name"
    ].tolist()

    mapped_df = final_mapping_df.dropna(subset=["medicine_unique_key"])
    mapped_df = mapped_df[mapped_df["medicine_unique_key"].astype(str).str.strip() != ""]

    duplicate_medicines = mapped_df.loc[
        mapped_df["medicine_unique_key"].duplicated(), "prescription_medicine_name"
    ].astype(str).tolist()

    if duplicate_images:
        raise RuntimeError(
            "Output violates image uniqueness. Duplicate images: "
            f"{duplicate_images}"
        )

    if duplicate_medicines:
        raise RuntimeError(
            "Output violates medicine-name uniqueness. Duplicate medicines: "
            f"{duplicate_medicines}"
        )


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------
def map_unique_image_to_unique_medicine(
    strip_csv: str | Path,
    prescription_csv: str | Path,
    output_csv: str | Path,
    all_candidates_csv: str | Path | None = None,
    keep_unmatched_images: bool = True,
) -> pd.DataFrame:
    """
    Map images to prescription medicines with unique medicine_name assignment.

    The final CSV has no duplicate image_name and no duplicate medicine_name among
    matched rows. If keep_unmatched_images=True, images that cannot receive a
    unique medicine are retained as unmatched rows.
    """
    strip_csv = Path(strip_csv)
    prescription_csv = Path(prescription_csv)
    output_csv = Path(output_csv)

    strip_df = pd.read_csv(strip_csv)
    prescription_df = pd.read_csv(prescription_csv)

    required_strip_columns = {"image_name", "extracted_text"}
    required_prescription_columns = {"medicine_name", "salt_name"}

    missing_strip = required_strip_columns - set(strip_df.columns)
    missing_prescription = required_prescription_columns - set(prescription_df.columns)

    if missing_strip:
        raise ValueError(f"Missing columns in strip CSV: {sorted(missing_strip)}")

    if missing_prescription:
        raise ValueError(
            f"Missing columns in prescription CSV: {sorted(missing_prescription)}"
        )

    # Add stable row numbers for deterministic tie-breaking.
    strip_df = strip_df.copy()
    prescription_df = prescription_df.copy()
    strip_df["_strip_row_number"] = range(1, len(strip_df) + 1)
    prescription_df["_prescription_row_number"] = range(1, len(prescription_df) + 1)

    candidates_df = build_candidate_scores(strip_df, prescription_df)

    if candidates_df.empty:
        final_mapping_df = pd.DataFrame()
    else:
        final_mapping_df = assign_unique_medicines_optimal(candidates_df)

    if keep_unmatched_images:
        final_mapping_df = append_unmatched_images(final_mapping_df, strip_df)

    if not final_mapping_df.empty:
        final_mapping_df = final_mapping_df.sort_values(
            ["strip_row_number"], ascending=[True]
        ).reset_index(drop=True)
        validate_unique_mapping(final_mapping_df)

    # The saved final CSV must contain only the requested user-facing columns.
    final_output_df = keep_only_final_output_columns(final_mapping_df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_output_df.to_csv(output_csv, index=False)

    if all_candidates_csv:
        all_candidates_csv = Path(all_candidates_csv)
        all_candidates_csv.parent.mkdir(parents=True, exist_ok=True)
        candidates_df.to_csv(all_candidates_csv, index=False)

    return final_output_df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map strip images to prescription medicines with one-to-one uniqueness: "
            "no duplicate image_name and no duplicate medicine_name in matched rows."
        )
    )

    parser.add_argument(
        "--strip_csv",
        default="medicine_strip_ocr_output.csv",
        help="Input strip OCR CSV. Default: medicine_strip_ocr_output.csv",
    )
    parser.add_argument(
        "--prescription_csv",
        default="final_combi.csv",
        help="Input prescription OCR CSV. Default: final_combi.csv",
    )
    parser.add_argument(
        "--output_csv",
        default="final_unique_image_to_medicine_mapping.csv",
        help="Final output CSV with unique image-to-medicine mapping.",
    )
    parser.add_argument(
        "--all_candidates_csv",
        default=None,
        help="Optional debug CSV with all image-vs-medicine candidate scores.",
    )
    parser.add_argument(
        "--drop_unmatched_images",
        action="store_true",
        help=(
            "Drop images that cannot get a unique medicine. By default, unmatched "
            "images are kept with blank medicine fields."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    final_df = map_unique_image_to_unique_medicine(
        strip_csv=args.strip_csv,
        prescription_csv=args.prescription_csv,
        output_csv=args.output_csv,
        all_candidates_csv=args.all_candidates_csv,
        keep_unmatched_images=not args.drop_unmatched_images,
    )

    print(f"Saved final unique image-to-medicine mapping: {args.output_csv}")
    if args.all_candidates_csv:
        print(f"Saved all candidate scores: {args.all_candidates_csv}")

    print()
    print(f"Rows in final output: {len(final_df)}")
    if not final_df.empty:
        matched_df = final_df.dropna(subset=["medicine_unique_key"])
        matched_df = matched_df[matched_df["medicine_unique_key"].astype(str).str.strip() != ""]
        print(f"Unique images in final output: {final_df['image_name'].nunique()}")
        print(f"Matched unique medicines: {matched_df['medicine_unique_key'].nunique()}")
        print(f"Unmatched images: {len(final_df) - len(matched_df)}")

        preview_columns = FINAL_OUTPUT_COLUMNS

        print()
        print("Final selected unique mappings:")
        print(final_df[preview_columns].to_string(index=False))


if __name__ == "__main__":
    main()
