"""
Streamlit app for medicine-strip to prescription-record mapping.

Pipeline used by the app
------------------------
1. Medicine strip images:
   orientation correction -> preprocessing -> OCR
   using the Modular medicine-strip pipeline: med_strip_modules.pipeline.run_full_pipeline

2. Printed prescription image:
   prescription OCR -> medicine-salt resolver
   using Final_prescription_ocr_salt_pipeline.py

3. Final mapping:
   one-to-one image-to-medicine assignment
   using mapping_4.py

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# UTF-8 safety for Windows paths containing emoji / special characters
# ---------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


APP_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = APP_ROOT / "streamlit_runs"

# Make local modules importable.
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


SUPPORTED_IMAGE_TYPES = [
    "png",
    "jpg",
    "jpeg",
    "webp",
    "bmp",
    "tif",
    "tiff",
    "avif",
]

FINAL_MAPPING_COLUMNS = [
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


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def clean_display_value(value) -> str:
    """Show blank instead of NaN/None."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()

    if text.lower() in {"nan", "none", "null"}:
        return ""

    return text


def safe_filename(name: str, fallback: str) -> str:
    """Return a filesystem-safe filename while preserving extension."""
    original = Path(name or fallback)
    stem = original.stem or Path(fallback).stem
    suffix = original.suffix or Path(fallback).suffix

    keep = []
    for char in stem:
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")

    cleaned_stem = "".join(keep).strip("_") or Path(fallback).stem
    return f"{cleaned_stem}{suffix.lower()}"


def unique_filename(index: int, uploaded_name: str, prefix: str) -> str:
    """Prefix filenames so duplicate uploads do not overwrite each other."""
    safe = safe_filename(uploaded_name, f"{prefix}_{index:03d}.png")
    return f"{prefix}_{index:03d}_{safe}"


def write_uploaded_files(uploaded_files: Iterable, output_dir: Path, prefix: str) -> List[Path]:
    """Save Streamlit uploaded files to a folder."""
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        filename = unique_filename(index, uploaded_file.name, prefix)
        path = output_dir / filename
        path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(path)

    return saved_paths


def resolve_path(path_text: str | None) -> Optional[Path]:
    """Resolve relative paths from the app folder."""
    if not path_text:
        return None

    path = Path(path_text).expanduser()

    if not path.is_absolute():
        path = APP_ROOT / path

    return path


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    """Read CSV if it exists, otherwise return empty DataFrame."""
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def csv_download_bytes(df: pd.DataFrame) -> bytes:
    """Convert dataframe to UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def zip_directory_to_bytes(directory: Path) -> bytes:
    """Zip a directory into bytes for Streamlit download."""
    buffer = BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(directory))

    buffer.seek(0)
    return buffer.getvalue()


def build_utf8_subprocess_env() -> dict:
    """
    Build UTF-8-safe environment for subprocesses.

    This fixes Windows UnicodeEncodeError caused by paths containing emoji
    or special characters.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run_subprocess_utf8(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """
    Run subprocess using UTF-8-safe settings.

    Prevents errors like:
    UnicodeEncodeError: 'charmap' codec can't encode characters
    """
    return subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=build_utf8_subprocess_env(),
        check=False,
    )


def show_required_file_status(model_path: str, index_path: str, skip_salt_lookup: bool) -> None:
    """Show project file availability in the sidebar."""
    required_items = {
        "med_strip_modules/": APP_ROOT / "med_strip_modules",
        "Final_prescription_ocr_salt_pipeline.py": APP_ROOT / "Final_prescription_ocr_salt_pipeline.py",
        "mapping_4.py": APP_ROOT / "mapping_4.py",
        "prescription_ocr_argparse_formats_1_32_tesseract_fixed.py": APP_ROOT / "prescription_ocr_argparse_formats_1_32_tesseract_fixed.py",
        "inference_simple_medicine_salt_altered.py": APP_ROOT / "inference_simple_medicine_salt_altered.py",
        "Abbreviations/": APP_ROOT / "Abbreviations",
    }

    st.sidebar.markdown("### Project file check")

    for label, path in required_items.items():
        if path.exists():
            st.sidebar.success(f"{label}: found")
        else:
            if label in {
                "med_strip_modules/",
                "Final_prescription_ocr_salt_pipeline.py",
                "mapping_4.py",
            }:
                st.sidebar.error(f"{label}: missing")
            else:
                st.sidebar.warning(f"{label}: missing")

    model_resolved = resolve_path(model_path)

    if model_resolved and model_resolved.exists():
        st.sidebar.success("orientation model: found")
    else:
        st.sidebar.warning("orientation model: missing; VGG orientation step may be skipped")

    index_resolved = resolve_path(index_path)

    if skip_salt_lookup:
        st.sidebar.info("salt index: skipped")
    elif index_resolved and index_resolved.exists():
        st.sidebar.success("salt index: found")
    else:
        st.sidebar.warning("salt index: missing or not set")


# ---------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------
def run_strip_pipeline(
    strip_input_dir: Path,
    run_dir: Path,
    model_path: str,
    output_size: int,
    use_gpu: bool,
) -> Path:
    """
    Run medicine-strip orientation correction, preprocessing, and OCR.

    This uses the modular medicine-strip pipeline directly.
    """
    try:
        from med_strip_modules.pipeline import run_full_pipeline
    except Exception as exc:
        raise RuntimeError(
            "Could not import the Modular medicine-strip pipeline. "
            "Make sure med_strip_modules is in the same folder as streamlit_app.py "
            "and all required packages are installed."
        ) from exc

    strip_csv = run_dir / "medicine_strip_ocr_output.csv"

    resolved_model_path = resolve_path(model_path)
    model_path_to_use = str(resolved_model_path) if resolved_model_path else model_path

    run_full_pipeline(
        input_folder=strip_input_dir,
        orientation_folder=run_dir / "oriented_images",
        preprocessed_folder=run_dir / "preprocessed_images",
        ocr_output_folder=run_dir / "medicine_strip_ocr_output",
        csv_path=strip_csv,
        model_path=model_path_to_use,
        output_size=output_size,
        languages=["en"],
        use_gpu=use_gpu,
    )

    if not strip_csv.exists():
        raise RuntimeError(
            "Medicine-strip OCR pipeline finished, but medicine_strip_ocr_output.csv was not created."
        )

    return strip_csv


def run_prescription_pipeline(
    prescription_input_dir: Path,
    run_dir: Path,
    index_path: str,
    tesseract_cmd: str,
    skip_salt_lookup: bool,
    ocr_timeout: int,
    save_processed_images: bool,
    dosage_csv: str,
    instruction_csv: str,
    duration_csv: str,
    medicine_csv: str,
) -> Path:
    """
    Run prescription OCR + medicine-salt resolver.

    This stage is executed as a subprocess because Final_prescription_ocr_salt_pipeline.py
    is CLI-based.
    """
    script_path = APP_ROOT / "Final_prescription_ocr_salt_pipeline.py"

    if not script_path.exists():
        raise FileNotFoundError(f"Missing required file: {script_path}")

    output_csv = run_dir / "final_combi.csv"
    output_dir = run_dir / "prescription_ocr_outputs"

    command = [
        sys.executable,
        str(script_path),
        "--input",
        str(prescription_input_dir),
        "--output-csv",
        str(output_csv),
        "--output-dir",
        str(output_dir),
        "--ocr-timeout",
        str(ocr_timeout),
        "--dosage-csv",
        dosage_csv,
        "--instruction-csv",
        instruction_csv,
        "--duration-csv",
        duration_csv,
        "--medicine-csv",
        medicine_csv,
    ]

    resolved_index = resolve_path(index_path)
    if resolved_index:
        command.extend(["--index-path", str(resolved_index)])

    resolved_tesseract = resolve_path(tesseract_cmd)
    if resolved_tesseract:
        command.extend(["--tesseract-cmd", str(resolved_tesseract)])

    if skip_salt_lookup:
        command.append("--skip-salt-lookup")

    if save_processed_images:
        command.append("--save-processed-images")

    process = run_subprocess_utf8(command=command, cwd=APP_ROOT)

    stdout_file = run_dir / "prescription_pipeline_stdout.txt"
    stderr_file = run_dir / "prescription_pipeline_stderr.txt"

    stdout_file.write_text(process.stdout or "", encoding="utf-8", errors="replace")
    stderr_file.write_text(process.stderr or "", encoding="utf-8", errors="replace")

    if process.returncode != 0:
        raise RuntimeError(
            "Prescription OCR + salt pipeline failed.\n\n"
            f"Command:\n{' '.join(command)}\n\n"
            f"STDOUT:\n{(process.stdout or '')[-4000:]}\n\n"
            f"STDERR:\n{(process.stderr or '')[-4000:]}"
        )

    if not output_csv.exists():
        raise RuntimeError(
            "Prescription pipeline finished, but final_combi.csv was not created."
        )

    return output_csv


def run_mapping_pipeline(
    strip_csv: Path,
    prescription_csv: Path,
    run_dir: Path,
    keep_unmatched_images: bool,
) -> Path:
    """
    Run one-to-one image-to-prescription mapping.

    This uses mapping_4.py directly as a Python module.
    """
    try:
        from mapping_4 import map_unique_image_to_unique_medicine
    except Exception as exc:
        raise RuntimeError(
            "Could not import mapping_4.py. "
            "Make sure mapping_4.py is in the same folder as streamlit_app.py "
            "and pandas, rapidfuzz, and scipy are installed."
        ) from exc

    final_csv = run_dir / "final_unique_image_to_medicine_mapping.csv"
    candidates_csv = run_dir / "all_unique_assignment_candidates.csv"

    map_unique_image_to_unique_medicine(
        strip_csv=strip_csv,
        prescription_csv=prescription_csv,
        output_csv=final_csv,
        all_candidates_csv=candidates_csv,
        keep_unmatched_images=keep_unmatched_images,
    )

    if not final_csv.exists():
        raise RuntimeError(
            "Mapping pipeline finished, but final_unique_image_to_medicine_mapping.csv was not created."
        )

    return final_csv


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Medicine Strip to Prescription Mapper",
        page_icon="💊",
        layout="wide",
    )

    st.title("Medicine Strip to Prescription Mapper")

    st.caption(
        "Upload multiple medicine-strip images and one printed prescription image. "
        "The app extracts OCR text, resolves prescription salts, and maps each strip image "
        "to one unique prescription medicine record."
    )

    with st.sidebar:
        st.header("Settings")

        model_path = st.text_input(
            "Orientation model path",
            value="quad_vgg16_orientation_classifier.pth",
            help="Place this file in the app folder, or provide an absolute path.",
        )

        index_path = st.text_input(
            "Medicine-salt TF-IDF index path",
            value="filtered_medicines_cleaned_simple_tfidf_index.joblib",
            help="Used by Final_prescription_ocr_salt_pipeline.py.",
        )

        tesseract_cmd = st.text_input(
            "Tesseract executable path, optional",
            value="",
            help=r"Example on Windows: C:\Program Files\Tesseract-OCR\tesseract.exe",
        )

        use_gpu = st.checkbox(
            "Use GPU for strip OCR if available",
            value=False,
        )

        skip_salt_lookup = st.checkbox(
            "Skip salt lookup",
            value=False,
        )

        keep_unmatched_images = st.checkbox(
            "Keep unmatched strip images in final CSV",
            value=True,
        )

        output_size = st.number_input(
            "Preprocessed strip image size",
            min_value=512,
            max_value=1800,
            value=1000,
            step=100,
        )

        ocr_timeout = st.number_input(
            "Prescription OCR timeout per image, seconds",
            min_value=15,
            max_value=300,
            value=60,
            step=15,
        )

        with st.expander("Prescription abbreviation CSV paths"):
            dosage_csv = st.text_input(
                "Dosage CSV",
                value="Abbreviations/Abbreviations_for_dosage.csv",
            )

            instruction_csv = st.text_input(
                "Instruction CSV",
                value="Abbreviations/Abbreviations_for_instructions.csv",
            )

            duration_csv = st.text_input(
                "Duration CSV",
                value="Abbreviations/Duration_abbreviations.csv",
            )

            medicine_csv = st.text_input(
                "Medicine abbreviation CSV",
                value="Abbreviations/india_medicine_abbreviations.csv",
            )

            save_processed_images = st.checkbox(
                "Save processed prescription images",
                value=False,
            )

        show_required_file_status(
            model_path=model_path,
            index_path=index_path,
            skip_salt_lookup=skip_salt_lookup,
        )

    left, right = st.columns(2)

    with left:
        strip_uploads = st.file_uploader(
            "Upload medicine-strip images",
            type=SUPPORTED_IMAGE_TYPES,
            accept_multiple_files=True,
        )

    with right:
        prescription_upload = st.file_uploader(
            "Upload printed prescription image",
            type=SUPPORTED_IMAGE_TYPES,
            accept_multiple_files=False,
        )

    run_button = st.button(
        "Run complete mapping system",
        type="primary",
        use_container_width=True,
    )

    if not run_button:
        st.info(
            "Upload medicine-strip images and one printed prescription image, "
            "then click the run button."
        )
        return

    if not strip_uploads:
        st.error("Please upload at least one medicine-strip image.")
        return

    if prescription_upload is None:
        st.error("Please upload one printed prescription image.")
        return

    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / run_id

    if run_dir.exists():
        shutil.rmtree(run_dir)

    strip_input_dir = run_dir / "input_strip_images"
    prescription_input_dir = run_dir / "input_prescription_image"

    run_dir.mkdir(parents=True, exist_ok=True)

    st.subheader("Pipeline progress")
    progress = st.progress(0)

    strip_df = pd.DataFrame()
    prescription_df = pd.DataFrame()
    final_df = pd.DataFrame()
    saved_strip_paths: List[Path] = []

    try:
        with st.status("Saving uploaded images...", expanded=True) as status:
            saved_strip_paths = write_uploaded_files(
                uploaded_files=strip_uploads,
                output_dir=strip_input_dir,
                prefix="strip",
            )

            saved_prescription_paths = write_uploaded_files(
                uploaded_files=[prescription_upload],
                output_dir=prescription_input_dir,
                prefix="prescription",
            )

            st.write(f"Saved {len(saved_strip_paths)} strip image(s).")
            st.write(f"Saved prescription image: {saved_prescription_paths[0].name}")

            status.update(label="Uploads saved", state="complete")

        progress.progress(10)

        with st.status(
            "Running medicine-strip orientation, preprocessing, and OCR...",
            expanded=False,
        ) as status:
            strip_csv = run_strip_pipeline(
                strip_input_dir=strip_input_dir,
                run_dir=run_dir,
                model_path=model_path,
                output_size=int(output_size),
                use_gpu=use_gpu,
            )

            strip_df = read_csv_if_exists(strip_csv)

            st.write(f"Created strip OCR CSV: {strip_csv.name}")
            st.write(f"Rows: {len(strip_df)}")

            status.update(label="Strip OCR completed", state="complete")

        progress.progress(45)

        with st.status(
            "Running prescription OCR and medicine-salt resolver...",
            expanded=False,
        ) as status:
            prescription_csv = run_prescription_pipeline(
                prescription_input_dir=prescription_input_dir,
                run_dir=run_dir,
                index_path=index_path,
                tesseract_cmd=tesseract_cmd,
                skip_salt_lookup=skip_salt_lookup,
                ocr_timeout=int(ocr_timeout),
                save_processed_images=save_processed_images,
                dosage_csv=dosage_csv,
                instruction_csv=instruction_csv,
                duration_csv=duration_csv,
                medicine_csv=medicine_csv,
            )

            prescription_df = read_csv_if_exists(prescription_csv)

            st.write(f"Created prescription CSV: {prescription_csv.name}")
            st.write(f"Rows: {len(prescription_df)}")

            status.update(
                label="Prescription OCR + salt resolver completed",
                state="complete",
            )

        progress.progress(75)

        with st.status(
            "Mapping each strip image to a unique prescription medicine...",
            expanded=False,
        ) as status:
            final_csv = run_mapping_pipeline(
                strip_csv=strip_csv,
                prescription_csv=prescription_csv,
                run_dir=run_dir,
                keep_unmatched_images=keep_unmatched_images,
            )

            final_df = read_csv_if_exists(final_csv)

            st.write(f"Created final mapping CSV: {final_csv.name}")
            st.write(f"Rows: {len(final_df)}")

            status.update(label="Final mapping completed", state="complete")

        progress.progress(100)

    except Exception as exc:
        progress.empty()
        st.error("Pipeline failed.")
        st.exception(exc)
        st.write("Run folder:", str(run_dir))

        stdout_file = run_dir / "prescription_pipeline_stdout.txt"
        stderr_file = run_dir / "prescription_pipeline_stderr.txt"

        if stdout_file.exists():
            with st.expander("Prescription pipeline STDOUT"):
                st.code(stdout_file.read_text(encoding="utf-8", errors="replace"))

        if stderr_file.exists():
            with st.expander("Prescription pipeline STDERR"):
                st.code(stderr_file.read_text(encoding="utf-8", errors="replace"))

        return

    st.success("Complete system finished successfully.")

    # -----------------------------------------------------------------
    # Final table
    # -----------------------------------------------------------------
    st.subheader("Final image-to-prescription mapping")

    if final_df.empty:
        st.warning("The final mapping CSV is empty.")
    else:
        display_columns = [
            col for col in FINAL_MAPPING_COLUMNS
            if col in final_df.columns
        ]

        st.dataframe(
            final_df[display_columns],
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------------------
    # Visual mapping preview with extra prescription details
    # -----------------------------------------------------------------
    st.subheader("Visual mapping preview")

    image_path_lookup = {
        path.name: path
        for path in saved_strip_paths
    }

    if not final_df.empty:
        for _, row in final_df.iterrows():
            image_name = clean_display_value(row.get("image_name", ""))
            image_path = image_path_lookup.get(image_name)

            mapped_medicine = clean_display_value(
                row.get("prescription_medicine_name", "")
            )
            salt_name = clean_display_value(
                row.get("prescription_salt_name", "")
            )
            record_number = clean_display_value(
                row.get("prescription_record_number", "")
            )
            final_match_score = clean_display_value(
                row.get("final_match_score", "")
            )
            match_confidence = clean_display_value(
                row.get("match_confidence", "")
            )
            extracted_text = clean_display_value(
                row.get("extracted_text", "")
            )

            strength = clean_display_value(
                row.get("prescription_strength", "")
            )
            dose_pattern_meaning = clean_display_value(
                row.get("prescription_dose_pattern_meaning", "")
            )
            dose_frequency_meaning = clean_display_value(
                row.get("prescription_dose_frequency_meaning", "")
            )
            duration = clean_display_value(
                row.get("prescription_duration", "")
            )
            instruction_meaning = clean_display_value(
                row.get("prescription_instruction_meaning", "")
            )

            with st.container(border=True):
                cols = st.columns([1, 2])

                with cols[0]:
                    if image_path and image_path.exists():
                        st.image(
                            str(image_path),
                            caption=image_name,
                            use_container_width=True,
                        )
                    else:
                        st.write(image_name)

                with cols[1]:
                    st.markdown(f"**Mapped medicine:** {mapped_medicine}")
                    st.write(f"**Salt:** {salt_name}")
                    st.write(f"**Record number:** {record_number}")
                    st.write(
                        f"**Score:** {final_match_score} | "
                        f"**Confidence:** {match_confidence}"
                    )

                    st.divider()

                    st.markdown("**Prescription details**")
                    st.write(f"**Strength:** {strength}")
                    st.write(f"**Dose pattern meaning:** {dose_pattern_meaning}")
                    st.write(f"**Dose frequency meaning:** {dose_frequency_meaning}")
                    st.write(f"**Duration:** {duration}")
                    st.write(f"**Instruction meaning:** {instruction_meaning}")

                    st.divider()

    # -----------------------------------------------------------------
    # Intermediate outputs
    # -----------------------------------------------------------------
    st.subheader("Intermediate outputs")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Medicine strip OCR output**")
        st.dataframe(
            strip_df,
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.markdown("**Prescription OCR + salt output**")
        st.dataframe(
            prescription_df,
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------------------
    # Downloads
    # -----------------------------------------------------------------
    st.subheader("Download results")

    candidates_csv = run_dir / "all_unique_assignment_candidates.csv"
    candidates_df = read_csv_if_exists(candidates_csv)

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.download_button(
            "Final mapping CSV",
            data=csv_download_bytes(final_df),
            file_name="final_unique_image_to_medicine_mapping.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "Strip OCR CSV",
            data=csv_download_bytes(strip_df),
            file_name="medicine_strip_ocr_output.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d3:
        st.download_button(
            "Prescription CSV",
            data=csv_download_bytes(prescription_df),
            file_name="final_combi.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d4:
        st.download_button(
            "All outputs ZIP",
            data=zip_directory_to_bytes(run_dir),
            file_name=f"{run_id}_outputs.zip",
            mime="application/zip",
            use_container_width=True,
        )

    if not candidates_df.empty:
        with st.expander("All mapping candidate scores"):
            st.dataframe(
                candidates_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download all candidate scores CSV",
                data=csv_download_bytes(candidates_df),
                file_name="all_unique_assignment_candidates.csv",
                mime="text/csv",
            )

    st.caption(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()