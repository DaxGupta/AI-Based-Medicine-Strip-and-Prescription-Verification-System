# 🧠 AI-Based-Medicine-Strip-and-Prescription-Verification-System ⚕️
An end-to-end computer vision and OCR pipeline deployed via Streamlit that extracts text from medicine strips, parses salt compositions, and verifies them against prescriptions using similarity-based matching.

This Streamlit app combines three parts of your project:

1. **Medicine strip pipeline** from the Modular medicine-strip pipeline ZIP
   - orientation correction (Trained model of orientation correction download from here: https://drive.google.com/file/d/1qQ2smULW888xdNbyJjW8H9susCC27hcZ/view?usp=sharing)
   - preprocessing
   - OCR text extraction

2. **Printed prescription OCR + salt resolver** using:
   - `Final_prescription_ocr_salt_pipeline.py`

3. **Unique image-to-prescription mapping** using:
   - `mapping_4.py`

The final output maps each uploaded medicine-strip image to a unique medicine record from the printed prescription.

---

## 1. Recommended project folder structure

Keep these files/folders together:

```text
streamlit_medicine_mapping_app/
│
├── streamlit_app.py
├── requirements_streamlit_app.txt
├── Final_prescription_ocr_salt_pipeline.py
├── mapping_4.py
│
├── med_strip_modules/
│   ├── __init__.py
│   ├── orientation.py
│   ├── preprocessing.py
│   ├── ocr.py
│   ├── pipeline.py
│   └── image_utils.py
│
├── prescription_ocr_argparse_formats_1_32_tesseract_fixed.py
├── inference_simple_medicine_salt_altered.py
├── quad_vgg16_orientation_classifier.pth
├── filtered_medicines_cleaned_simple_tfidf_index.joblib
│
└── Abbreviations/
    ├── Abbreviations_for_dosage.csv
    ├── Abbreviations_for_instructions.csv
    ├── Duration_abbreviations.csv
    └── india_medicine_abbreviations.csv
```

This ZIP contains the Streamlit app, the modular strip modules, `Final_prescription_ocr_salt_pipeline.py`, and `mapping_4.py`.

You still need to place these project-specific files in the same folder if they are not already present:

```text
prescription_ocr_argparse_formats_1_32_tesseract_fixed.py
inference_simple_medicine_salt_altered.py
quad_vgg16_orientation_classifier.pth
filtered_medicines_cleaned_simple_tfidf_index.joblib
Abbreviations/ folder
```

---

## 2. Create a separate virtual environment

Use **Python 3.10**.

### Windows PowerShell

```powershell
py -3.10 -m venv .venv-medstrip-app
.\.venv-medstrip-app\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-medstrip-app\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python3.10 -m venv .venv-medstrip-app
source .venv-medstrip-app/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

---

## 3. Install dependencies

Install PyTorch first. For CPU-only usage:

```bash
pip install torch torchvision torchaudio
```

Then install the app requirements:

```bash
pip install -r requirements_streamlit_app.txt
```

If you use CUDA/GPU, install the correct PyTorch build from the official PyTorch selector, then run the requirements command.

---

## 4. Install Tesseract OCR

The prescription OCR pipeline uses Tesseract through `pytesseract`.

### Windows

Install Tesseract OCR and keep note of the executable path, commonly:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

Enter that path in the Streamlit sidebar under **Tesseract executable path**.

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

---

## 5. Run the Streamlit app

From inside the app folder:

```bash
streamlit run streamlit_app.py
```

Then upload:

- multiple medicine-strip images
- one printed prescription image

Click:

```text
Run complete mapping system
```

---

## 6. Output files created by the app

Each run is saved under:

```text
streamlit_runs/run_YYYYMMDD_HHMMSS/
```

Main generated files:

```text
medicine_strip_ocr_output.csv
final_combi.csv
final_unique_image_to_medicine_mapping.csv
all_unique_assignment_candidates.csv
```

The final mapping CSV contains these columns:

```text
image_name
extracted_text
prescription_record_number
prescription_medicine_name
prescription_dosage_form_meaning
prescription_strength
prescription_dose_pattern
prescription_dose_pattern_meaning
prescription_dose_frequency
prescription_dose_frequency_meaning
prescription_duration
prescription_instruction_meaning
prescription_salt_name
medicine_unique_key
final_match_score
match_confidence
```

---

## 7. Important notes

- `quad_vgg16_orientation_classifier.pth` should be placed in the app folder, or provide its full path in the sidebar.
- `filtered_medicines_cleaned_simple_tfidf_index.joblib` should also be placed in the app folder, or provide its full path in the sidebar.
- If the salt index is unavailable, enable **Skip salt lookup** in the sidebar for testing.
- The final mapping uses `mapping_4.py`, which enforces unique medicine assignment, so no two images should map to the same medicine name.
