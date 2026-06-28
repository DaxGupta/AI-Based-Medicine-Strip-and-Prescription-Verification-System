#!/usr/bin/env python3
"""
combined_prescription_ocr_salt_pipeline.py

Combined pipeline:
1. Runs prescription_ocr_argparse_formats_1_32_tesseract_fixed.py to do the
   normal OCR/extraction.
2. Runs an additional layout-aware parser for wide printed prescriptions that
   use separate visual columns like:
       NAME: ...      DOSE: ...      DURATION: ...
                                      INSTRUCTIONS: ...
   This parser uses Tesseract word coordinates, so it can recover all medicine
   rows even when Tesseract's plain text order merges or drops columns.
3. Resolves the salt name using inference_simple_medicine_salt_altered.py from
   the first matched_medicines item for each medicine.
4. Saves ONLY the requested final columns.

Required files in the same folder:
- prescription_ocr_argparse_formats_1_32_tesseract_fixed.py
- inference_simple_medicine_salt_altered.py

Example:
    python combined_prescription_ocr_salt_pipeline.py \
        --input combi_med_compo.png \
        --index-path filtered_medicines_cleaned_simple_tfidf_index.joblib \
        --output-csv final_prescription_output.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pytesseract

try:
    from prescription_ocr_argparse_formats_1_32_tesseract_fixed import (
        build_record,
        configure_tesseract,
        dedupe_records,
        find_image_paths,
        load_abbreviation_resources,
        preprocess_prescription,
        process_prescription_input,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Could not import the prescription OCR script. Make sure "
        "prescription_ocr_argparse_formats_1_32_tesseract_fixed.py is in the "
        "same folder as this combined script."
    ) from exc

try:
    from inference_simple_medicine_salt_altered import (
        SimpleMedicineSaltResolver,
        ensure_index,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Could not import the medicine salt inference script. Make sure "
        "inference_simple_medicine_salt_altered.py is in the same folder as "
        "this combined script."
    ) from exc


DEFAULT_OCR_MEDICINE_COLUMNS = (
    "medicine_name_normalized",
    "medicine_name",
    "raw_medicine_text",
)

FINAL_OUTPUT_COLUMNS = [
    "source_image",
    "record_number",
    "medicine_name",
    "dosage_form_meaning",
    "strength",
    "dose_pattern",
    "dose_pattern_meaning",
    "times_per_day_from_pattern",
    "dose_frequency",
    "dose_frequency_meaning",
    "dose_frequency_safe_check",
    "duration",
    "instruction_meaning",
    "salt_name",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def clean_cell(value: Any) -> str:
    """Return a safe stripped string for CSV cell values."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def normalize_token(value: Any) -> str:
    """Normalize OCR tokens for robust label matching."""
    text = clean_cell(value).upper()
    return re.sub(r"[^A-Z0-9:+\-/]", "", text)


def normalize_for_dedupe(value: Any) -> str:
    """Normalize a row/text value for duplicate detection."""
    text = clean_cell(value).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def strip_label(text: str, label: str) -> str:
    """Remove a leading OCR label such as NAME:, DOSE:, or INSTRUCTIONS:."""
    if label == "name":
        pattern = r"^\s*(?:N\s*)?A?M?E\s*[:;.-]?\s*|^\s*NAME\s*[:;.-]?\s*"
    elif label == "dose":
        pattern = r"^\s*DOSE\s*[:;.-]?\s*"
    elif label == "duration":
        pattern = r"^\s*DURATION\s*[:;.-]?\s*"
    elif label == "instructions":
        pattern = r"^\s*INSTRUCTIONS?\s*[:;.-]?\s*"
    else:
        pattern = rf"^\s*{re.escape(label)}\s*[:;.-]?\s*"
    return re.sub(pattern, "", clean_cell(text), flags=re.IGNORECASE).strip(" -:;|,.")


def clean_medicine_label_text(text: str) -> str:
    """Clean the visual NAME column before sending it to build_record()."""
    cleaned = clean_cell(text)
    cleaned = re.sub(r"^\s*(?:NAME|NME|AME|NAE|NAM)\s*[:;.-]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(?:Rx|R/x)\s*[:;.-]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;|,.")
    return cleaned


def clean_instruction_text(text: str) -> str:
    cleaned = strip_label(text, "instructions")
    cleaned = re.sub(r"\bIF\s+FEVER\b", "if fever", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;|,.")
    return cleaned


def build_line_text(group: pd.DataFrame) -> str:
    """Join OCR words in left-to-right order."""
    return " ".join(clean_cell(x) for x in group.sort_values("left")["text"].tolist() if clean_cell(x))


# ---------------------------------------------------------------------------
# Layout-aware NAME / DOSE / DURATION / INSTRUCTIONS parser
# ---------------------------------------------------------------------------


def tesseract_lines_for_layout(image_path: str, ocr_timeout: int = 60) -> pd.DataFrame:
    """
    Return line-level OCR boxes from Tesseract sparse text mode.

    PSM 11 keeps separated visual columns as separate lines on the provided image,
    which is exactly what is needed for the NAME / DOSE / DURATION layout.
    """
    processed = preprocess_prescription(image_path)
    all_lines: List[Dict[str, Any]] = []

    # PSM 11 is best for this wide prescription layout and is much faster than
    # running several page-segmentation modes again after the main OCR pass.
    for psm in (11,):
        try:
            data = pytesseract.image_to_data(
                processed,
                config=f"--oem 3 --psm {psm} -l eng",
                output_type=pytesseract.Output.DATAFRAME,
                timeout=ocr_timeout,
            )
        except Exception:
            continue

        if data is None or data.empty or "text" not in data.columns:
            continue

        data = data.dropna(subset=["text"]).copy()
        data["text"] = data["text"].astype(str).str.strip()
        data = data[data["text"] != ""].copy()
        if data.empty:
            continue

        required_cols = {"block_num", "par_num", "line_num", "left", "top", "width", "height"}
        if not required_cols.issubset(data.columns):
            continue

        data["right"] = data["left"].astype(float) + data["width"].astype(float)
        data["bottom"] = data["top"].astype(float) + data["height"].astype(float)
        data["cx"] = data["left"].astype(float) + data["width"].astype(float) / 2.0
        data["cy"] = data["top"].astype(float) + data["height"].astype(float) / 2.0

        for _, group in data.groupby(["block_num", "par_num", "line_num"], sort=False):
            text = build_line_text(group)
            if not text:
                continue
            all_lines.append(
                {
                    "text": text,
                    "norm": normalize_token(text),
                    "left": float(group["left"].min()),
                    "top": float(group["top"].min()),
                    "right": float(group["right"].max()),
                    "bottom": float(group["bottom"].max()),
                    "cy": float(group["cy"].mean()),
                    "psm": psm,
                }
            )

    if not all_lines:
        return pd.DataFrame()

    lines = pd.DataFrame(all_lines)

    # Deduplicate repeated PSM readings. Keep the first/best occurrence for each
    # similar text and visual y coordinate.
    kept: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int]] = set()
    for row in lines.sort_values(["psm", "cy", "left"]).to_dict("records"):
        key = (normalize_for_dedupe(row["text"]), int(round(float(row["cy"]) / 8.0)))
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)

    return pd.DataFrame(kept).sort_values(["cy", "left"]).reset_index(drop=True)


def is_rx_medicine_name_line(line: Dict[str, Any], advice_y: float) -> bool:
    """Detect visual medicine rows, not header/patient/doctor NAME lines."""
    text = clean_cell(line.get("text"))
    upper = text.upper()
    left = float(line.get("left", 0.0))
    cy = float(line.get("cy", 0.0))

    if not text or cy >= advice_y:
        return False
    if left > 650:
        return False

    blocked = (
        "CLINIC NAME",
        "DOCTOR NAME",
        "PATIENT NAME",
        "QUALIFICATION",
        "REGISTRATION",
        "PRESCRIPTION DATE",
        "DIAGNOSIS",
    )
    if any(bad in upper for bad in blocked):
        return False

    # Handle normal NAME: and common OCR variants such as AME:.
    if not re.match(r"^\s*(?:NAME|NME|AME|NAE|NAM)\s*[:;.-]", upper):
        return False

    # A real medicine row almost always contains a dosage form or a strength.
    has_form = bool(re.search(r"\b(?:TAB|TABLET|CAP|CAPSULE|SYP|SYRUP|INJ|OINT|CREAM|GEL|DROPS?)\b", upper))
    has_strength = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:MG|MCG|G|GM|ML|IU|%)\b", upper))
    has_brandish_name = len(strip_label(text, "name")) >= 4
    return has_brandish_name and (has_form or has_strength or left < 260)


def value_after_label(text: str, label_regex: str) -> str:
    """Extract text after a label if present."""
    match = re.search(label_regex + r"\s*[:;.-]?\s*(.+)$", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(" -:;|,.")
    return ""


def extract_layout_records_from_image(
    image_path: str,
    abbrev: Dict[str, Any],
    ocr_timeout: int = 60,
) -> List[Dict[str, Any]]:
    """
    Extract records from coordinate-based wide prescription layouts.

    This specifically fixes cases where plain OCR returns only a partial table,
    e.g. the supplied image where the four visual rows are spread across three
    columns.
    """
    lines_df = tesseract_lines_for_layout(image_path, ocr_timeout=ocr_timeout)
    if lines_df.empty:
        return []

    lines = lines_df.to_dict("records")
    advice_y = float("inf")
    for line in lines:
        if re.match(r"^\s*ADVICE\b", clean_cell(line.get("text")), flags=re.IGNORECASE):
            advice_y = min(advice_y, float(line.get("cy", 0.0)))
    if advice_y == float("inf"):
        # Allow a generous bottom if ADVICE was not detected.
        advice_y = float(lines_df["cy"].max()) + 100.0

    medicine_lines = [line for line in lines if is_rx_medicine_name_line(line, advice_y)]
    medicine_lines = sorted(medicine_lines, key=lambda item: (float(item["cy"]), float(item["left"])))

    # Deduplicate the same medicine row detected by multiple PSMs.
    deduped_medicine_lines: List[Dict[str, Any]] = []
    for line in medicine_lines:
        if deduped_medicine_lines and abs(float(line["cy"]) - float(deduped_medicine_lines[-1]["cy"])) < 16:
            # Prefer the cleaner line: usually one with NAME: and more words.
            old = deduped_medicine_lines[-1]
            if len(clean_cell(line["text"])) > len(clean_cell(old["text"])):
                deduped_medicine_lines[-1] = line
        else:
            deduped_medicine_lines.append(line)
    medicine_lines = deduped_medicine_lines

    if len(medicine_lines) < 2:
        return []

    records: List[Dict[str, Any]] = []
    image_name = Path(image_path).name

    for idx, med_line in enumerate(medicine_lines):
        med_y = float(med_line["cy"])
        prev_y = float(medicine_lines[idx - 1]["cy"]) if idx > 0 else med_y - 90.0
        next_y = float(medicine_lines[idx + 1]["cy"]) if idx + 1 < len(medicine_lines) else min(advice_y, med_y + 95.0)

        upper_y = max(0.0, (prev_y + med_y) / 2.0 if idx > 0 else med_y - 45.0)
        lower_y = (med_y + next_y) / 2.0 if idx + 1 < len(medicine_lines) else min(advice_y, med_y + 95.0)

        row_lines = [
            line for line in lines
            if upper_y <= float(line.get("cy", 0.0)) <= lower_y
            and float(line.get("cy", 0.0)) < advice_y
        ]

        # Include lines slightly above the medicine name when they are clearly in
        # the right-side dose/duration columns, which happens in the supplied image.
        row_lines.extend(
            line for line in lines
            if med_y - 35.0 <= float(line.get("cy", 0.0)) < upper_y
            and float(line.get("left", 0.0)) >= 650.0
            and float(line.get("cy", 0.0)) < advice_y
        )

        # Deduplicate row lines by text/y.
        unique_row_lines: List[Dict[str, Any]] = []
        row_seen: set[Tuple[str, int]] = set()
        for line in sorted(row_lines, key=lambda item: (float(item["cy"]), float(item["left"]))):
            key = (normalize_for_dedupe(line["text"]), int(round(float(line["cy"]) / 8.0)))
            if key not in row_seen:
                row_seen.add(key)
                unique_row_lines.append(line)

        medicine_text = clean_medicine_label_text(clean_cell(med_line["text"]))

        dose_bits: List[str] = []
        duration_bits: List[str] = []
        instruction_bits: List[str] = []

        # Also scan the medicine line itself because some Tesseract modes merge
        # NAME + DOSE + DURATION into one line.
        for line in unique_row_lines:
            text = clean_cell(line.get("text"))
            if not text:
                continue
            upper = text.upper()

            dose_value = value_after_label(text, r"\bDOSE\b")
            if dose_value or re.search(r"\b[0-9OoIl|]{1,2}\s*-\s*[0-9OoIl|]{1,2}\s*-\s*[0-9OoIl|]{1,2}\b", text):
                dose_bits.append("DOSE: " + (dose_value or text))

            duration_value = value_after_label(text, r"\bDURATION\b")
            if duration_value or "DURATION" in upper:
                duration_bits.append("DURATION: " + (duration_value or text))

            instruction_value = value_after_label(text, r"\bINSTRUCTIONS?\b")
            if instruction_value or "INSTRUCTION" in upper:
                instruction_bits.append(clean_instruction_text(instruction_value or text))

        # If the medicine line accidentally contains merged right-side labels, cut
        # them out after extracting their values.
        if re.search(r"\bDOSE\b", medicine_text, flags=re.IGNORECASE):
            dose_value = value_after_label(medicine_text, r"\bDOSE\b")
            if dose_value:
                dose_bits.append("DOSE: " + dose_value)
            medicine_text = re.split(r"\bDOSE\b", medicine_text, maxsplit=1, flags=re.IGNORECASE)[0].strip(" -:;|,.")
        if re.search(r"\bDURATION\b", medicine_text, flags=re.IGNORECASE):
            duration_value = value_after_label(medicine_text, r"\bDURATION\b")
            if duration_value:
                duration_bits.append("DURATION: " + duration_value)
            medicine_text = re.split(r"\bDURATION\b", medicine_text, maxsplit=1, flags=re.IGNORECASE)[0].strip(" -:;|,.")
        if re.search(r"\bINSTRUCTIONS?\b", medicine_text, flags=re.IGNORECASE):
            instruction_value = value_after_label(medicine_text, r"\bINSTRUCTIONS?\b")
            if instruction_value:
                instruction_bits.append(clean_instruction_text(instruction_value))
            medicine_text = re.split(r"\bINSTRUCTIONS?\b", medicine_text, maxsplit=1, flags=re.IGNORECASE)[0].strip(" -:;|,.")

        dose_line = " ".join(dict.fromkeys(bit for bit in dose_bits if bit))
        duration_line = " ".join(dict.fromkeys(bit for bit in duration_bits if bit))
        instruction_line = "; ".join(dict.fromkeys(bit for bit in instruction_bits if bit))
        details_line = " ".join(x for x in [dose_line, duration_line] if x)
        raw_record_text = " ".join(x for x in [medicine_text, details_line, instruction_line] if x)

        if not medicine_text:
            continue

        record = build_record(
            medicine_line=medicine_text,
            dose_line=details_line,
            instruction_line=instruction_line,
            raw_record_text=raw_record_text,
            abbrev=abbrev,
        )
        if not record:
            continue

        record["source_image"] = image_name
        record["record_number_in_image"] = len(records) + 1
        record["ocr_text"] = "\n".join(clean_cell(line.get("text")) for line in lines)
        record["error"] = ""
        records.append(record)

    return dedupe_records(records)


def extract_layout_records_for_input(
    input_path: str,
    abbrev: Dict[str, Any],
    ocr_timeout: int = 60,
) -> pd.DataFrame:
    """Run the coordinate layout parser over one image or a folder of images."""
    rows: List[Dict[str, Any]] = []
    for image_path in find_image_paths(input_path):
        try:
            rows.extend(extract_layout_records_from_image(str(image_path), abbrev, ocr_timeout=ocr_timeout))
        except Exception as exc:
            print(f"Warning: layout-aware parser failed for {image_path.name}: {exc}")

    return pd.DataFrame(rows)


def usable_record_count(df: pd.DataFrame) -> int:
    """Count rows that contain a medicine name."""
    if df is None or df.empty or "medicine_name" not in df.columns:
        return 0
    return int(df["medicine_name"].fillna("").astype(str).str.strip().ne("").sum())



def is_blank_cell(value: Any) -> bool:
    """Return True for empty/None/NaN/None-like OCR cells."""
    text = clean_cell(value)
    return text == "" or text.lower() in {"none", "nan", "null", "na", "n/a"}


def dataframe_field_score(df: pd.DataFrame) -> int:
    """
    Score how complete a prescription extraction dataframe is.

    This is used because the normal OCR and the layout-aware OCR can both find
    the same number of medicines, but one may miss fields like duration. The
    older code only compared row counts, which is why one duration could stay
    blank for this prescription format.
    """
    if df is None or df.empty:
        return 0

    important_columns = [
        "medicine_name",
        "strength",
        "dose_pattern",
        "dose_pattern_meaning",
        "dose_frequency_abbreviation",
        "dose_frequency",
        "dose_frequency_meaning",
        "duration",
        "instruction_meaning",
    ]

    score = 0
    for column in important_columns:
        if column not in df.columns:
            continue
        score += int(df[column].apply(lambda value: not is_blank_cell(value)).sum())
    return score


def fill_missing_fields_from_layout(ocr_part: pd.DataFrame, layout_part: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the normal OCR rows, but fill blank fields from layout-aware rows.

    This is safer than always replacing the whole OCR result. It fixes cases
    like Aceclo-plus where normal OCR found the medicine and instruction but
    missed the duration, while the layout parser can read the visual duration
    column.
    """
    if ocr_part is None or ocr_part.empty:
        return layout_part
    if layout_part is None or layout_part.empty:
        return ocr_part

    output = ocr_part.reset_index(drop=True).copy()
    layout = layout_part.reset_index(drop=True).copy()

    fill_columns = [
        "medicine_name",
        "medicine_name_normalized",
        "raw_medicine_text",
        "dosage_form_meaning",
        "strength",
        "dose_pattern",
        "dose_pattern_meaning",
        "times_per_day_from_pattern",
        "dose_frequency_abbreviation",
        "dose_frequency",
        "dose_frequency_meaning",
        "dose_frequency_safe_check",
        "duration",
        "instruction_meaning",
    ]

    row_count = min(len(output), len(layout))
    for row_idx in range(row_count):
        for column in fill_columns:
            if column not in layout.columns:
                continue
            if column not in output.columns:
                output[column] = ""
            if is_blank_cell(output.at[row_idx, column]) and not is_blank_cell(layout.at[row_idx, column]):
                output.at[row_idx, column] = layout.at[row_idx, column]

    return output


def merge_ocr_with_layout_repairs(ocr_df: pd.DataFrame, layout_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge normal OCR rows with layout-aware repairs.

    Old behavior:
        Use layout parser only when it finds more medicine rows.

    Corrected behavior:
        1. Use layout parser when it finds more medicine rows.
        2. When both parsers find the same number of rows, keep the normal OCR
           result but fill missing fields such as duration/instruction from the
           layout-aware result.
    """
    if layout_df is None or layout_df.empty:
        return ocr_df
    if ocr_df is None or ocr_df.empty:
        return layout_df

    output_parts: List[pd.DataFrame] = []
    all_images = sorted(
        set(ocr_df.get("source_image", pd.Series(dtype=str)).astype(str))
        | set(layout_df.get("source_image", pd.Series(dtype=str)).astype(str))
    )

    for image_name in all_images:
        ocr_part = (
            ocr_df[ocr_df["source_image"].astype(str) == image_name].copy()
            if "source_image" in ocr_df.columns else pd.DataFrame()
        )
        layout_part = (
            layout_df[layout_df["source_image"].astype(str) == image_name].copy()
            if "source_image" in layout_df.columns else pd.DataFrame()
        )

        ocr_count = usable_record_count(ocr_part)
        layout_count = usable_record_count(layout_part)
        ocr_score = dataframe_field_score(ocr_part)
        layout_score = dataframe_field_score(layout_part)

        if layout_count > ocr_count:
            chosen = layout_part
            print(
                f"Layout-aware parser used for {image_name}: "
                f"{layout_count} rows instead of {ocr_count} OCR rows."
            )
        elif layout_count == ocr_count and layout_count > 0:
            chosen = fill_missing_fields_from_layout(ocr_part, layout_part)
            if dataframe_field_score(chosen) > ocr_score:
                print(
                    f"Layout-aware fields merged for {image_name}: "
                    f"field score {ocr_score} -> {dataframe_field_score(chosen)}."
                )
        elif layout_score > ocr_score:
            chosen = layout_part
            print(
                f"Layout-aware parser used for {image_name}: "
                f"field score {layout_score} instead of {ocr_score}."
            )
        else:
            chosen = ocr_part

        if not chosen.empty:
            chosen = chosen.reset_index(drop=True)
            chosen["record_number_in_image"] = range(1, len(chosen) + 1)
            output_parts.append(chosen)

    if not output_parts:
        return ocr_df
    return pd.concat(output_parts, ignore_index=True, sort=False)


def extract_duration_value(text: str) -> str:
    """Extract a clean duration value such as '5 days' from OCR text."""
    text = clean_cell(text)
    text = re.sub(r"\s+", " ", text).strip()

    # Remove the label but keep the value.
    text = re.sub(r"^\s*DURATION\s*[:;.-]?\s*", "", text, flags=re.IGNORECASE)

    # OCR may produce DAY/DAYS, WEEK/WEEKS, MONTH/MONTHS.
    match = re.search(
        r"\b(\d{1,3})\s*(DAY|DAYS|WEEK|WEEKS|MONTH|MONTHS)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""

    number = match.group(1)
    unit = match.group(2).lower()
    if number == "1":
        unit = unit.rstrip("s")
    elif not unit.endswith("s"):
        unit += "s"
    return f"{number} {unit}"


def extract_visual_durations_from_image(image_path: str, ocr_timeout: int = 60) -> List[str]:
    """
    Read durations directly from the visual DURATION column and return them in
    medicine-row order.

    This is a focused fallback for printed prescriptions with columns:
        NAME ...  DOSE ...  DURATION ...
                  ...       INSTRUCTIONS ...
    """
    lines_df = tesseract_lines_for_layout(image_path, ocr_timeout=ocr_timeout)
    if lines_df.empty:
        return []

    lines = lines_df.to_dict("records")

    advice_y = float("inf")
    for line in lines:
        if re.match(r"^\s*ADVICE\b", clean_cell(line.get("text")), flags=re.IGNORECASE):
            advice_y = min(advice_y, float(line.get("cy", 0.0)))
    if advice_y == float("inf"):
        advice_y = float(lines_df["cy"].max()) + 100.0

    medicine_lines = [line for line in lines if is_rx_medicine_name_line(line, advice_y)]
    medicine_lines = sorted(medicine_lines, key=lambda item: (float(item["cy"]), float(item["left"])))

    # Deduplicate near-identical medicine rows.
    deduped_medicine_lines: List[Dict[str, Any]] = []
    for line in medicine_lines:
        if deduped_medicine_lines and abs(float(line["cy"]) - float(deduped_medicine_lines[-1]["cy"])) < 16:
            old = deduped_medicine_lines[-1]
            if len(clean_cell(line["text"])) > len(clean_cell(old["text"])):
                deduped_medicine_lines[-1] = line
        else:
            deduped_medicine_lines.append(line)
    medicine_lines = deduped_medicine_lines

    if not medicine_lines:
        return []

    duration_lines: List[Dict[str, Any]] = []
    seen_duration_keys: set[Tuple[str, int]] = set()
    for line in lines:
        text = clean_cell(line.get("text"))
        upper = text.upper()
        if float(line.get("cy", 0.0)) >= advice_y:
            continue

        # Duration is normally in the right column. Keep this loose enough for
        # resized or slightly shifted images.
        is_right_side = float(line.get("left", 0.0)) >= 500.0
        has_duration = "DURATION" in upper or bool(extract_duration_value(text))
        if is_right_side and has_duration:
            duration = extract_duration_value(text)
            if not duration:
                continue
            key = (duration.lower(), int(round(float(line.get("cy", 0.0)) / 8.0)))
            if key in seen_duration_keys:
                continue
            seen_duration_keys.add(key)
            line_copy = dict(line)
            line_copy["duration_value"] = duration
            duration_lines.append(line_copy)

    duration_lines = sorted(duration_lines, key=lambda item: (float(item["cy"]), float(item["left"])))

    durations_by_row: List[str] = []
    for idx, med_line in enumerate(medicine_lines):
        med_y = float(med_line["cy"])
        prev_y = float(medicine_lines[idx - 1]["cy"]) if idx > 0 else med_y - 90.0
        next_y = float(medicine_lines[idx + 1]["cy"]) if idx + 1 < len(medicine_lines) else min(advice_y, med_y + 95.0)

        upper_y = max(0.0, (prev_y + med_y) / 2.0 if idx > 0 else med_y - 45.0)
        lower_y = (med_y + next_y) / 2.0 if idx + 1 < len(medicine_lines) else min(advice_y, med_y + 95.0)

        candidates = [
            line for line in duration_lines
            if upper_y - 20.0 <= float(line.get("cy", 0.0)) <= lower_y + 20.0
        ]
        if not candidates:
            durations_by_row.append("")
            continue

        best = min(candidates, key=lambda line: abs(float(line.get("cy", 0.0)) - med_y))
        durations_by_row.append(clean_cell(best.get("duration_value")))

    return durations_by_row


def repair_missing_durations_from_input(
    df: pd.DataFrame,
    input_path: str,
    ocr_timeout: int = 60,
) -> pd.DataFrame:
    """
    Fill blank duration cells using direct visual DURATION-column extraction.

    This specifically fixes the Streamlit case where one record has duration
    blank/None even though the prescription clearly contains DURATION: 5 DAYS.
    """
    if df is None or df.empty:
        return df
    if "source_image" not in df.columns:
        return df

    output = df.copy()
    if "duration" not in output.columns:
        output["duration"] = ""

    for image_path in find_image_paths(input_path):
        image_name = Path(image_path).name
        durations = extract_visual_durations_from_image(str(image_path), ocr_timeout=ocr_timeout)
        if not durations:
            continue

        mask = output["source_image"].astype(str) == image_name
        row_indices = list(output[mask].index)
        if not row_indices:
            continue

        for row_position, row_index in enumerate(row_indices):
            if row_position >= len(durations):
                break
            duration_value = clean_cell(durations[row_position])
            if duration_value and is_blank_cell(output.at[row_index, "duration"]):
                output.at[row_index, "duration"] = duration_value

    return output


# ---------------------------------------------------------------------------
# Salt lookup
# ---------------------------------------------------------------------------


def choose_medicine_query(row: pd.Series, medicine_query_column: Optional[str] = None) -> str:
    """
    Choose the medicine text to send to the salt resolver.

    If --medicine-query-column is supplied, that column is used. Otherwise the
    cleaner OCR column `medicine_name_normalized` is preferred, with fallback to
    `medicine_name` and then `raw_medicine_text`.
    """
    if medicine_query_column:
        if medicine_query_column not in row.index:
            raise KeyError(
                f"Medicine query column '{medicine_query_column}' was not found in the OCR CSV. "
                f"Available columns: {list(row.index)}"
            )
        return clean_cell(row.get(medicine_query_column))

    for column in DEFAULT_OCR_MEDICINE_COLUMNS:
        if column in row.index:
            value = clean_cell(row.get(column))
            if value:
                return value
    return ""


def salt_lookup_from_first_matched_medicine(
    resolver: SimpleMedicineSaltResolver,
    medicine_query: str,
) -> Dict[str, Any]:
    """
    Resolve one medicine query and return salt details from the first
    `matched_medicines` item.

    The requested `salt_name` column is the first salt in:
        result["matched_medicines"][0]["salts"]
    """
    empty = {
        "salt_lookup_query": medicine_query,
        "salt_name": "",
        "all_salts_from_first_matched_medicine": "",
        "matched_medicine": "",
        "matched_medicine_score": "",
        "matched_medicine_match_type": "",
        "salt_lookup_input_type": "empty" if not medicine_query else "unknown",
        "matched_medicines_json": "[]",
        "matched_salts_json": "[]",
        "salt_lookup_error": "",
    }

    if not medicine_query:
        return empty

    try:
        result = resolver.resolve(medicine_query)
        matched_medicines = result.get("matched_medicines") or []
        matched_salts = result.get("matched_salts") or []

        if not matched_medicines:
            empty.update(
                {
                    "salt_lookup_input_type": result.get("input_type", "unknown"),
                    "matched_salts_json": json.dumps(matched_salts, ensure_ascii=False),
                }
            )
            return empty

        first_match = matched_medicines[0]
        salts = [clean_cell(salt) for salt in first_match.get("salts", []) if clean_cell(salt)]

        return {
            "salt_lookup_query": medicine_query,
            "salt_name": salts[0] if salts else "",
            "all_salts_from_first_matched_medicine": "; ".join(salts),
            "matched_medicine": clean_cell(first_match.get("medicine")),
            "matched_medicine_score": first_match.get("score", ""),
            "matched_medicine_match_type": clean_cell(first_match.get("match_type")),
            "salt_lookup_input_type": clean_cell(result.get("input_type")),
            "matched_medicines_json": json.dumps(matched_medicines, ensure_ascii=False),
            "matched_salts_json": json.dumps(matched_salts, ensure_ascii=False),
            "salt_lookup_error": "",
        }
    except Exception as exc:  # pragma: no cover - keep batch processing resilient
        empty["salt_lookup_error"] = str(exc)
        return empty


def add_salt_columns_to_ocr_dataframe(
    df: pd.DataFrame,
    resolver: Optional[SimpleMedicineSaltResolver] = None,
    medicine_query_column: Optional[str] = None,
    skip_salt_lookup: bool = False,
) -> pd.DataFrame:
    """Append salt lookup columns to the extracted prescription dataframe."""
    salt_rows = []
    for _, row in df.iterrows():
        medicine_query = choose_medicine_query(row, medicine_query_column)
        if skip_salt_lookup or resolver is None:
            salt_rows.append(
                {
                    "salt_lookup_query": medicine_query,
                    "salt_name": "",
                    "all_salts_from_first_matched_medicine": "",
                    "matched_medicine": "",
                    "matched_medicine_score": "",
                    "matched_medicine_match_type": "",
                    "salt_lookup_input_type": "skipped",
                    "matched_medicines_json": "[]",
                    "matched_salts_json": "[]",
                    "salt_lookup_error": "",
                }
            )
        else:
            salt_rows.append(salt_lookup_from_first_matched_medicine(resolver, medicine_query))

    salt_df = pd.DataFrame(salt_rows)
    return pd.concat([df.reset_index(drop=True), salt_df.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# Final CSV shaping
# ---------------------------------------------------------------------------


def build_final_output_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with exactly the user-requested final CSV columns."""
    final = pd.DataFrame()

    def col(name: str) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series([""] * len(df))

    final["source_image"] = col("source_image")
    final["record_number"] = col("record_number_in_image") if "record_number_in_image" in df.columns else col("record_number")
    final["medicine_name"] = col("medicine_name")
    final["dosage_form_meaning"] = col("dosage_form_meaning")
    final["strength"] = col("strength")
    final["dose_pattern"] = col("dose_pattern")
    final["dose_pattern_meaning"] = col("dose_pattern_meaning")
    final["times_per_day_from_pattern"] = col("times_per_day_from_pattern")
    final["dose_frequency"] = col("dose_frequency_abbreviation") if "dose_frequency_abbreviation" in df.columns else col("dose_frequency")
    final["dose_frequency_meaning"] = col("dose_frequency_meaning")
    final["dose_frequency_safe_check"] = col("dose_frequency_safe_check")
    final["duration"] = col("duration")
    final["instruction_meaning"] = col("instruction_meaning")
    final["salt_name"] = col("salt_name")

    for column in FINAL_OUTPUT_COLUMNS:
        if column not in final.columns:
            final[column] = ""

    final = final[FINAL_OUTPUT_COLUMNS].copy()
    final = final.fillna("")

    # Make numeric record numbers clean in the CSV.
    final["record_number"] = final["record_number"].apply(lambda x: str(int(float(x))) if clean_cell(x) and str(x).replace(".", "", 1).isdigit() else clean_cell(x))
    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract prescription medicines with OCR, repair wide NAME/DOSE/DURATION layouts, "
            "resolve salt names, and save the requested final CSV columns."
        )
    )

    # OCR arguments from prescription_ocr_argparse_formats_1_32_tesseract_fixed.py
    parser.add_argument(
        "--input",
        required=True,
        help="Path to one prescription image or to a folder containing prescription images.",
    )
    parser.add_argument(
        "--dosage-csv",
        default="Abbreviations/Abbreviations_for_dosage.csv",
        help="Path to dosage-form abbreviation CSV.",
    )
    parser.add_argument(
        "--instruction-csv",
        default="Abbreviations/Abbreviations_for_instructions.csv",
        help="Path to instruction abbreviation CSV.",
    )
    parser.add_argument(
        "--duration-csv",
        default="Abbreviations/Duration_abbreviations.csv",
        help="Path to frequency/duration abbreviation CSV.",
    )
    parser.add_argument(
        "--medicine-csv",
        default="Abbreviations/india_medicine_abbreviations.csv",
        help="Path to medicine abbreviation CSV used by the OCR parser.",
    )
    parser.add_argument(
        "--output-csv",
        default="prescription_medicine_extracts_with_salts.csv",
        help="Final CSV path. Only the requested columns are saved here.",
    )
    parser.add_argument(
        "--output-dir",
        default="prescription_ocr_outputs",
        help="Folder where OCR text files and processed images can be saved.",
    )
    parser.add_argument(
        "--no-save-ocr-text",
        action="store_true",
        help="Do not save per-image OCR text files.",
    )
    parser.add_argument(
        "--save-processed-images",
        action="store_true",
        help="Save preprocessed images used for OCR.",
    )
    parser.add_argument(
        "--ocr-timeout",
        type=int,
        default=60,
        help="Tesseract OCR timeout per image in seconds. Default: 60.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default=None,
        help="Optional full path to the Tesseract executable.",
    )

    # Salt resolver arguments from inference_simple_medicine_salt_altered.py
    parser.add_argument(
        "--index-path",
        default=None,
        help=(
            "Path to filtered_medicines_cleaned_simple_tfidf_index.joblib. "
            "If omitted, the inference script tries to find the default index next to itself."
        ),
    )
    parser.add_argument(
        "--knowledge-csv",
        default=None,
        help=(
            "Optional backward-compatible argument. If --index-path is omitted, "
            "the resolver tries <knowledge_csv_stem>_simple_tfidf_index.joblib."
        ),
    )
    parser.add_argument(
        "--medicine-threshold",
        type=float,
        default=None,
        help="Override TF-IDF cosine threshold for medicine matching.",
    )
    parser.add_argument(
        "--salt-threshold",
        type=float,
        default=None,
        help="Override TF-IDF cosine threshold for salt matching.",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=int,
        default=None,
        help="Override RapidFuzz threshold from 0 to 100.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override number of top matches to keep.",
    )
    parser.add_argument(
        "--disable-fuzzy",
        action="store_true",
        help="Disable RapidFuzz matching and use only exact + TF-IDF matching.",
    )
    parser.add_argument(
        "--medicine-query-column",
        default=None,
        help=(
            "OCR CSV column to send to the salt resolver. Default: auto-select "
            "medicine_name_normalized, then medicine_name, then raw_medicine_text."
        ),
    )
    parser.add_argument(
        "--skip-salt-lookup",
        action="store_true",
        help="Skip salt lookup and leave salt_name blank. Useful for testing OCR without the joblib index.",
    )
    parser.add_argument(
        "--disable-layout-repair",
        action="store_true",
        help="Disable the coordinate-based repair parser for wide NAME/DOSE/DURATION layouts.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=20,
        help="Number of final rows to print after processing. Use 0 to skip preview.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    configure_tesseract(args.tesseract_cmd)

    resolver: Optional[SimpleMedicineSaltResolver] = None
    if not args.skip_salt_lookup:
        resolver = SimpleMedicineSaltResolver(
            knowledge_csv=args.knowledge_csv,
            index_path=args.index_path,
            medicine_threshold=args.medicine_threshold,
            salt_threshold=args.salt_threshold,
            fuzzy_threshold=args.fuzzy_threshold,
            top_k=args.top_k,
            use_fuzzy=not args.disable_fuzzy,
        )
        ensure_index(resolver)

    # 1. Run the user's original OCR/extraction pipeline.
    ocr_df = process_prescription_input(
        input_path=args.input,
        dosage_csv=args.dosage_csv,
        instruction_csv=args.instruction_csv,
        duration_csv=args.duration_csv,
        medicine_csv=args.medicine_csv,
        output_csv=args.output_csv,
        output_dir=args.output_dir,
        save_ocr_text=not args.no_save_ocr_text,
        save_processed_images=args.save_processed_images,
        ocr_timeout=args.ocr_timeout,
    )

    # 2. Run layout-aware repair and replace incomplete OCR rows when it finds
    #    more medicines for a given image.
    best_df = ocr_df
    if not args.disable_layout_repair:
        abbrev = load_abbreviation_resources(
            dosage_csv=args.dosage_csv,
            instruction_csv=args.instruction_csv,
            duration_csv=args.duration_csv,
            medicine_csv=args.medicine_csv,
        )
        layout_df = extract_layout_records_for_input(
            input_path=args.input,
            abbrev=abbrev,
            ocr_timeout=args.ocr_timeout,
        )
        best_df = merge_ocr_with_layout_repairs(ocr_df=ocr_df, layout_df=layout_df)

    # Extra repair for printed prescriptions where one DURATION cell is missed
    # even though the visual right-side duration column is present.
    best_df = repair_missing_durations_from_input(
        df=best_df,
        input_path=args.input,
        ocr_timeout=args.ocr_timeout,
    )

    # 3. Add salt columns using the first matched_medicines item.
    combined_df = add_salt_columns_to_ocr_dataframe(
        df=best_df,
        resolver=resolver,
        medicine_query_column=args.medicine_query_column,
        skip_salt_lookup=args.skip_salt_lookup,
    )

    # 4. Save only requested final columns.
    final_df = build_final_output_dataframe(combined_df)

    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv_path, index=False)

    print(f"\nSaved final OCR + salt output CSV to: {output_csv_path.resolve()}")
    print(f"Total rows saved: {len(final_df):,}")
    print(f"Columns saved: {list(final_df.columns)}")

    if args.preview_rows > 0:
        print("\nFinal table preview:")
        with pd.option_context("display.max_columns", None, "display.width", 240):
            print(final_df.head(args.preview_rows).to_string(index=False))


if __name__ == "__main__":
    main()
