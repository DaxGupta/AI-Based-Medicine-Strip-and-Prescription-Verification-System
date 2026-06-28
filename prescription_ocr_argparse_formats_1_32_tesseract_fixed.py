import argparse
import shutil

import cv2
import pytesseract
import pandas as pd
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Supported image extensions for a single image or folder input
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def configure_tesseract(tesseract_cmd: Optional[str] = None) -> None:
    """Configure the Tesseract executable used by pytesseract.

    The Python package `pytesseract` is only a wrapper. The actual Tesseract OCR
    engine must be installed separately. This function now checks PATH and common
    Windows/Linux install locations before raising an error.
    """
    candidate_paths: List[Path] = []

    if tesseract_cmd:
        candidate_paths.append(Path(tesseract_cmd).expanduser())

    found = shutil.which("tesseract")
    if found:
        candidate_paths.append(Path(found))

    # Common Windows installer locations. These make the script work even when
    # Tesseract is installed but was not added to PATH.
    candidate_paths.extend(
        [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
            Path.home() / r"AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
            Path.home() / r"AppData\Local\Tesseract-OCR\tesseract.exe",
            Path("/usr/bin/tesseract"),
            Path("/usr/local/bin/tesseract"),
        ]
    )

    for path in candidate_paths:
        try:
            if path and path.exists() and path.is_file():
                pytesseract.pytesseract.tesseract_cmd = str(path)
                print(f"Using Tesseract OCR executable: {path}")
                return
        except OSError:
            continue

    install_message = (
        "Tesseract OCR engine was not found. The Python package pytesseract is "
        "installed, but the external tesseract.exe program is missing or not in PATH.\n\n"
        "Windows fix:\n"
        "1. Install Tesseract OCR. The usual install location is:\n"
        r"   C:\Program Files\Tesseract-OCR\tesseract.exe" "\n"
        "2. Then run this script with:\n"
        "   --tesseract-cmd \"C:\\Program Files\\Tesseract-OCR\\tesseract.exe\"\n\n"
        "Colab/Ubuntu fix:\n"
        "   sudo apt-get update && sudo apt-get install -y tesseract-ocr\n"
    )
    raise RuntimeError(install_message)


def natural_key(path: Path):
    """Sort file names naturally: Format_2 before Format_10."""
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def read_csv_safely(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        print(f"Warning: abbreviation file not found: {p}")
        return pd.DataFrame()
    return pd.read_csv(p).fillna("")


def load_abbreviation_resources(
    dosage_csv: str,
    instruction_csv: str,
    duration_csv: str,
    medicine_csv: str,
) -> Dict[str, Any]:
    """Load all abbreviation CSVs into dictionaries used during extraction."""

    # Dosage form abbreviations: TAB -> Tablet, CAP -> Capsule, etc.
    dosage_forms: Dict[str, str] = {
        "TAB": "Tablet",
        "TABLET": "Tablet",
        "CAP": "Capsule",
        "CAPSULE": "Capsule",
        "SYP": "Syrup",
        "SYRUP": "Syrup",
        "OINT": "Ointment",
        "OINTMENT": "Ointment",
        "DROPS": "Liquid drops (usually for eyes, ears, or nose)",
        "DROP": "Liquid drops (usually for eyes, ears, or nose)",
        "INJ": "Injection",
        "INJECTION": "Injection",
        "CREAM": "Topical formulations",
        "GEL": "Topical formulations",
        "SPRAY": "Spray preparation",
        "SACHET": "Sachet / powder preparation",
        "LOTION": "Topical lotion",
        "DROPS": "Liquid drops (usually for eyes, ears, or nose)",
        "DROP": "Liquid drops (usually for eyes, ears, or nose)",
    }
    df_dose_form = read_csv_safely(dosage_csv)
    if not df_dose_form.empty:
        for _, row in df_dose_form.iterrows():
            short_form = str(row.get("Short Form", "")).strip()
            meaning = str(row.get("Dosage Form", "")).strip()
            if short_form and meaning:
                dosage_forms[short_form.upper()] = meaning
                dosage_forms[meaning.upper()] = meaning

    # Instruction abbreviations: AC/PC/BF/AF -> usable instruction text.
    instructions: Dict[str, str] = {
        "AC": "Take on an empty stomach",
        "PC": "Take post-meal",
        "BF": "Take on an empty stomach",
        "AF": "Take post-meal",
        "BEFORE FOOD": "Take on an empty stomach",
        "BEFORE MEAL": "Take on an empty stomach",
        "BEFORE MEALS": "Take on an empty stomach",
        "BEFORE BREAKFAST": "Take on an empty stomach before breakfast",
        "BBF": "Take before breakfast",
        "EMPTY STOMACH": "Take on an empty stomach",
        "AFTER FOOD": "Take post-meal",
        "AFTER MEAL": "Take post-meal",
        "AFTER MEALS": "Take post-meal",
        "AFTER BREAKFAST": "Take post-breakfast",
        "AT NIGHT": "Take at night",
        "NIGHT": "Take at night",
        "BEDTIME": "Take at bedtime",
        "BEFORE SLEEP": "Take before sleep / at bedtime",
        "BEFORE BED": "Take before sleep / at bedtime",
        "AT BEDTIME": "Take at bedtime",
    }
    instruction_timing: Dict[str, str] = {}
    df_inst = read_csv_safely(instruction_csv)
    if not df_inst.empty:
        for _, row in df_inst.iterrows():
            abbr = str(row.get("Abbreviations", "")).strip()
            timing = str(row.get("Timing", "")).strip()
            instruction = str(row.get("Instruction", "")).strip()
            if abbr and instruction:
                instructions[abbr.upper()] = instruction
            if timing and instruction:
                instructions[timing.upper()] = instruction
                instruction_timing[timing.upper()] = instruction

    # The provided Duration_abbreviations.csv contains frequency abbreviations too:
    # OD/BD/TDS/QID/HS/SOS/PRN/STAT.
    frequencies: Dict[str, Dict[str, str]] = {}
    df_freq = read_csv_safely(duration_csv)
    if not df_freq.empty:
        for _, row in df_freq.iterrows():
            abbr = str(row.get("abbreviation", "")).strip().upper()
            if abbr:
                frequencies[abbr] = {
                    "meaning": str(row.get("meaning", "")).strip(),
                    "latin_full_form": str(row.get("latin_full_form", "")).strip(),
                    "safe_check": str(row.get("safe_check", "")).strip(),
                }
                # HS/SOS/PRN/STAT may also function as instruction-like terms.
                if abbr in {"HS", "SOS", "PRN", "STAT"}:
                    instructions[abbr] = str(row.get("meaning", "")).strip()

    # Natural-language schedules commonly found in printed prescriptions.
    # These are added in the same structure as abbreviations so the rest of the
    # extraction code can treat "BD", "Twice daily", and "Two times daily" consistently.
    natural_frequency_phrases = {
        "ONCE DAILY": {
            "meaning": "Once a day",
            "latin_full_form": "",
            "safe_check": "Ask exact time",
        },
        "ONCE A DAY": {
            "meaning": "Once a day",
            "latin_full_form": "",
            "safe_check": "Ask exact time",
        },
        "TWICE DAILY": {
            "meaning": "Twice a day",
            "latin_full_form": "",
            "safe_check": "Confirm approximately 12 hours apart",
        },
        "TWO TIMES DAILY": {
            "meaning": "Twice a day",
            "latin_full_form": "",
            "safe_check": "Confirm approximately 12 hours apart",
        },
        "THRICE DAILY": {
            "meaning": "Thrice a day / Three times daily",
            "latin_full_form": "",
            "safe_check": "Confirm morning, afternoon, and night time slots",
        },
        "THREE TIMES DAILY": {
            "meaning": "Thrice a day / Three times daily",
            "latin_full_form": "",
            "safe_check": "Confirm morning, afternoon, and night time slots",
        },
        "FOUR TIMES DAILY": {
            "meaning": "Four times a day",
            "latin_full_form": "",
            "safe_check": "Confirm four evenly spaced time slots",
        },
        "EVERY NIGHT": {
            "meaning": "Once daily at night",
            "latin_full_form": "",
            "safe_check": "Confirm night/bedtime timing",
        },
        "AT NIGHT": {
            "meaning": "Once daily at night",
            "latin_full_form": "",
            "safe_check": "Confirm night/bedtime timing",
        },
    }

    natural_frequency_abbreviations = {
        "ONCE DAILY": "OD",
        "ONCE A DAY": "OD",
        "TWICE DAILY": "BD",
        "TWO TIMES DAILY": "BD",
        "THRICE DAILY": "TDS",
        "THREE TIMES DAILY": "TDS",
        "FOUR TIMES DAILY": "QID",
        "EVERY NIGHT": "HS",
        "AT NIGHT": "HS",
    }
    for phrase, canonical_abbr in natural_frequency_abbreviations.items():
        if phrase in natural_frequency_phrases:
            natural_frequency_phrases[phrase]["abbreviation"] = canonical_abbr

    for phrase, info in natural_frequency_phrases.items():
        frequencies.setdefault(phrase, info)

    # Indian medicine shorthand/brand-like abbreviations.
    medicines: Dict[str, Dict[str, str]] = {}
    df_med = read_csv_safely(medicine_csv)
    if not df_med.empty:
        for _, row in df_med.iterrows():
            abbr = str(row.get("abbreviation", "")).strip()
            if abbr:
                medicines[abbr.lower()] = {
                    "normalized_names": str(row.get("normalized_names", "")).strip(),
                    "category": str(row.get("category", "")).strip(),
                    "notes": str(row.get("notes", "")).strip(),
                    "ambiguity": str(row.get("ambiguity", "")).strip(),
                }

    # Direct medicine names and common generic terms appearing in the sample images.
    # The CSV mostly contains abbreviations/brand-like shorthand; adding direct
    # names here helps parsing OCR rows that omit dosage form or strength.
    direct_medicine_aliases = {
        "azithromycin": "Azithromycin",
        "amoxicillin": "Amoxicillin",
        "ibuprofen": "Ibuprofen",
        "cetirizine": "Cetirizine",
        "levocetirizine": "Levocetirizine",
        "montelukast": "Montelukast",
        "montelukast levocetirizine": "Montelukast + Levocetirizine",
        "montelukast + levocetirizine": "Montelukast + Levocetirizine",
        "dextromethorphan": "Dextromethorphan",
        "chlorpheniramine": "Chlorpheniramine",
        "paracetamol": "Paracetamol",
        "pantoprazole": "Pantoprazole",
        "domperidone": "Domperidone",
        "drotaverine": "Drotaverine",
        "fexofenadine": "Fexofenadine",
        "saline": "Saline",
        "cough formula": "Cough Formula",
        "cough syrup": "Cough Syrup",
        "nitrofurantoin": "Nitrofurantoin",
        "urinary alkaliser": "Urinary Alkaliser",
        "probiotic": "Probiotic",
        "vitamin c": "Vitamin C",
        "metformin": "Metformin",
        "amlodipine": "Amlodipine",
        "atorvastatin": "Atorvastatin",
        "vitamin d3": "Vitamin D3",
        "hydrocortisone": "Hydrocortisone",
        "calamine": "Calamine",
        "prednisolone": "Prednisolone",
    }
    for alias, normalized in direct_medicine_aliases.items():
        medicines.setdefault(alias.lower(), {
            "normalized_names": normalized,
            "category": "medicine_name",
            "notes": "Direct generic/name alias added for OCR parsing",
            "ambiguity": "low",
        })

    return {
        "dosage_forms": dosage_forms,
        "instructions": instructions,
        "instruction_timing": instruction_timing,
        "frequencies": frequencies,
        "medicines": medicines,
    }


def preprocess_prescription(image_path: str):
    """Preprocess prescription image for OCR."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize for OCR: upscale small photos, but cap very large images so
    # folder processing remains practical.
    height, width = gray.shape[:2]
    max_dim = max(height, width)
    if max_dim < 1400:
        scale = 2.0
    elif max_dim < 2000:
        scale = 1.4
    elif max_dim > 2600:
        scale = 2600.0 / max_dim
    else:
        scale = 1.0
    if abs(scale - 1.0) > 0.05:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Light denoise and improve contrast. Median blur is much faster than
    # non-local-means on large folders and is sufficient for these printed forms.
    gray = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Adaptive threshold generally works well for generated/printed prescription images.
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return thresh


def run_ocr(processed_image, ocr_timeout: int = 60, psm_modes: Tuple[int, ...] = (6, 4, 11)) -> str:
    """Run OCR using multiple Tesseract page segmentation modes.

    Different prescription images in this dataset use paragraph layouts,
    table layouts, and sparse two-column layouts. A single PSM often misses
    one of those. We concatenate the useful OCR outputs and de-duplicate
    repeated lines before parsing.
    """
    outputs: List[str] = []
    for psm in psm_modes:
        config = f"--oem 3 --psm {psm} -l eng"
        try:
            txt = pytesseract.image_to_string(processed_image, config=config, timeout=ocr_timeout)
            if txt and txt.strip():
                outputs.append(txt)
        except RuntimeError as exc:
            print(f"Warning: Tesseract failed with psm {psm}: {exc}")
    seen = set()
    merged_lines = []
    for txt in outputs:
        for line in txt.splitlines():
            cleaned = re.sub(r"\s+", " ", line).strip()
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen.add(key)
                merged_lines.append(cleaned)
    return "\n".join(merged_lines)


def normalize_text(text: str) -> str:
    """Clean common OCR artifacts while preserving prescription line structure."""
    replacements = {
        "–": "-",
        "—": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "Instructions;": "Instructions:",
        "Instruction:": "Instructions:",
        "Food Instruction:": "Instructions:",
        "Medicine Name:": "Medicine:",
        "MEDICINE NAME:": "Medicine:",
        "Dosage:": "Dose:",
        "Frequency:": "Schedule:",
        "Dose & Duration": "Dose and Duration",
        "Dose&Duration": "Dose and Duration",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"Dose\s*(?:&|and)?\s*Duration", "Dose and Duration", text, flags=re.IGNORECASE)
    text = re.sub(r"Food\s+Instr(?:uction|uctions)?\s*[:;]", "Instructions:", text, flags=re.IGNORECASE)
    text = re.sub(r"Instr(?:uction|uctions)?\s*[:;]", "Instructions:", text, flags=re.IGNORECASE)
    text = re.sub(r"Medicine\s+Name\s*[:;]", "Medicine:", text, flags=re.IGNORECASE)
    text = re.sub(r"Med(?:icine|ication)?\s*[:;]", "Medicine:", text, flags=re.IGNORECASE)
    text = re.sub(r"Brand\s*/\s*Form\s*[:;]", "Form:", text, flags=re.IGNORECASE)
    text = re.sub(r"Frequency\s*[:;]", "Schedule:", text, flags=re.IGNORECASE)
    text = re.sub(r"Dosage\s*[:;]", "Dose:", text, flags=re.IGNORECASE)
    # Common OCR mistakes in strengths/durations: wg -> mg, SOO -> 500, 5m! -> 5ml.
    text = re.sub(
        r"\b([0-9OSoIl|]{1,6})\s*w[gq]\b",
        lambda m: normalize_count_token(m.group(1).replace("S", "5").replace("s", "5")) + " mg",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bS([0-9O]{2})\s*mg\b",
        lambda m: "5" + m.group(1).replace("O", "0") + " mg",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b([0-9OoIl|]+)\s*m[!lI|]+\b",
        lambda m: normalize_count_token(m.group(1)) + " ml",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bXS\s*(days?)\b", r"x 5 \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bx\s*S\s*(days?)\b", r"x 5 \1", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_count_token(token: str) -> str:
    """Fix common OCR substitutions in numeric dose patterns: I/l/| -> 1 and O/o -> 0."""
    token = token.strip()
    return token.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"}))


def dose_to_readable(dose_pattern: str) -> str:
    """Convert 1-0-1 into a human-readable schedule."""
    if not dose_pattern:
        return ""
    parts = [normalize_count_token(x) for x in re.split(r"\s*-\s*", dose_pattern) if x.strip()]
    if len(parts) < 3 or not all(x.isdigit() for x in parts[:3]):
        return dose_pattern

    labels = ["morning", "afternoon", "night"]
    readable = []
    for amount, label in zip(parts[:3], labels):
        if int(amount) > 0:
            readable.append(f"{amount} in the {label}")
    return ", ".join(readable) if readable else "No dose indicated"


def dose_to_times_per_day(dose_pattern: str) -> Optional[int]:
    if not dose_pattern:
        return None
    parts = [normalize_count_token(x) for x in re.split(r"\s*-\s*", dose_pattern) if x.strip()]
    if len(parts) >= 3 and all(x.isdigit() for x in parts[:3]):
        return sum(int(x) for x in parts[:3])
    return None


def regex_boundary(term: str) -> str:
    """Boundary regex that also works for terms containing +, -, or digits."""
    return r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"



def known_medicine_terms(abbrev: Dict[str, Any]) -> List[str]:
    """Return medicine names/aliases useful for anchoring noisy OCR rows."""
    terms = set()
    for key, info in abbrev.get("medicines", {}).items():
        if key:
            terms.add(str(key).strip())
        normalized = str(info.get("normalized_names", ""))
        for part in re.split(r"\s*\|\s*|\s+\+\s+|/", normalized):
            part = re.sub(r"\s+", " ", part).strip()
            if part:
                terms.add(part)
        if normalized:
            terms.add(re.sub(r"\s+", " ", normalized.replace("+", " ")).strip())
    # Avoid very short ambiguous tokens such as CA, FE, ML unless they are part of a longer term.
    return sorted([t for t in terms if len(re.sub(r"[^A-Za-z0-9]", "", t)) >= 3], key=len, reverse=True)


def trim_to_medicine_start(text: str, abbrev: Dict[str, Any]) -> str:
    """Remove OCR header/address noise before the actual medicine token.

    OCR sometimes joins page headers or patient details to the first medicine row.
    This function cuts the text at the first credible dosage form or known medicine
    name, e.g. "Park Avenue ... Cap Amoxicillin 500 mg" -> "Cap Amoxicillin 500 mg".
    """
    if not text:
        return text
    candidate = re.sub(r"\s+", " ", text).strip()
    anchors: List[int] = []

    form_keys = sorted(abbrev.get("dosage_forms", {}).keys(), key=len, reverse=True)
    for form in form_keys:
        m = re.search(regex_boundary(form), candidate, flags=re.IGNORECASE)
        if m:
            anchors.append(m.start())

    for term in known_medicine_terms(abbrev):
        m = re.search(regex_boundary(term), candidate, flags=re.IGNORECASE)
        if m:
            anchors.append(m.start())

    if not anchors:
        return candidate

    first = min(i for i in anchors if i >= 0)
    # Do not remove the first one or two characters if the string already starts
    # with a valid medicine row.
    if first <= 2:
        return candidate
    return candidate[first:].strip(" -:;|,.")


def extract_trailing_instruction_phrase(text: str) -> str:
    """Return instruction text after the duration/frequency part in compact rows."""
    if not text:
        return ""
    flat = re.sub(r"\s+", " ", text).strip(" .,;:-")
    # Capture known abbreviation or phrase after the last duration expression.
    m = re.search(
        r"(?:x|for)?\s*\d+(?:\.\d+)?\s*(?:d|day|days|wk|wks|week|weeks|mo|mos|month|months)\b\s*[,;\-]?\s*(.+)$",
        flat,
        flags=re.IGNORECASE,
    )
    if m:
        tail = m.group(1).strip(" .,;:-")
        tail = re.sub(r"\b(?:Advice|Doctor|Signature|Sample|This is sample|Not valid).*$", "", tail, flags=re.IGNORECASE).strip(" .,;:-")
        if tail and not re.fullmatch(r"(?:Dose|Duration|Schedule|Instructions?)", tail, flags=re.IGNORECASE):
            return tail
    # Rows such as "OD (1 tablet) 3 DAYS PC - Empty stomach" often put the
    # instruction after a frequency token without a comma.
    m = re.search(r"\b(AC|PC|AF|BF|BBF|HS|SOS|PRN)\b(?:\s*[-:]\s*[^,;]+)?$", flat, flags=re.IGNORECASE)
    if m:
        return m.group(0).strip()
    return ""


def clean_instruction_value(value: str) -> str:
    """Remove OCR numbering/footer noise from instruction text."""
    if not value:
        return ""
    value = re.sub(r"\s+", " ", str(value)).strip(" -:;|,.~_=<>[]{}")
    # Stop at the next item/medicine marker or footer marker.
    value = re.split(
        r"\s+(?:\d{1,2}\s*[\.)-]?\s*)?(?:Medicine\s*:|Tab\s+[A-Z]|Cap\s+[A-Z]|Syrup\s+[A-Z]|Oint\s+[A-Z]|Lotion\s+[A-Z]|Drop\s+[A-Z]|Spray\s+[A-Z]|Sachet\s+[A-Z]|Advice\s*:|Doctor\s+Signature|Signature|Sample|This is sample|Not valid)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" -:;|,.~_=<>[]{}")
    # Remove trailing item numbers left by merged OCR, e.g. "After food. 2".
    value = re.sub(r"\s+\d{1,2}\s*[\.)-]?\s*$", "", value).strip(" -:;|,.")
    return value

def find_dosage_form(medicine_text: str, dosage_forms: Dict[str, str]) -> Tuple[str, str]:
    choices = sorted(dosage_forms.keys(), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(x) for x in choices) + r")\b", re.IGNORECASE)
    match = pattern.search(medicine_text)
    if not match:
        return "", ""
    raw = match.group(1).strip()
    return raw, dosage_forms.get(raw.upper(), "")


def remove_dosage_form(medicine_text: str, dosage_forms: Dict[str, str]) -> str:
    choices = sorted(dosage_forms.keys(), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(x) for x in choices) + r")\b", re.IGNORECASE)
    return pattern.sub(" ", medicine_text)


def extract_strength(text: str) -> str:
    strengths = re.findall(
        r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|gm|ml|iu|units?|%)\b",
        text,
        flags=re.IGNORECASE,
    )
    return " + ".join(dict.fromkeys(s.strip() for s in strengths))


def remove_strength(text: str) -> str:
    return re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|gm|ml|iu|units?|%)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )


def find_medicine_match(medicine_name: str, medicines: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    cleaned = re.sub(r"[^A-Za-z0-9+\-/ ]+", " ", medicine_name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    best_key = ""
    best_display = ""
    best_info: Dict[str, str] = {}
    for key in sorted(medicines.keys(), key=len, reverse=True):
        match = re.search(regex_boundary(key), cleaned, flags=re.IGNORECASE)
        if match:
            best_key = key
            best_display = match.group(0)
            best_info = medicines[key]
            break

    if best_key:
        return {
            "medicine_abbreviation": best_display,
            "medicine_name_normalized": best_info.get("normalized_names", cleaned),
            "medicine_category": best_info.get("category", ""),
            "medicine_ambiguity": best_info.get("ambiguity", ""),
            "medicine_notes": best_info.get("notes", ""),
        }

    return {
        "medicine_abbreviation": "",
        "medicine_name_normalized": cleaned,
        "medicine_category": "",
        "medicine_ambiguity": "",
        "medicine_notes": "",
    }


def extract_dose_pattern(text: str) -> str:
    """Extract a numeric dose schedule or explicit Dose: quantity.

    Supports both compact prescription patterns such as 1-0-1 and structured
    generated prescriptions such as "Dose: 1 cap" or "Dose: 10 ml".
    """
    schedule_match = re.search(
        r"\b([0-9OoIl|]{1,2})\s*[-]\s*([0-9OoIl|]{1,2})\s*[-]\s*([0-9OoIl|]{1,2})\b",
        text,
    )
    if schedule_match:
        return "-".join(normalize_count_token(x) for x in schedule_match.groups())

    quantity_match = re.search(
        r"\bDose\s*:\s*([0-9OoIl|]+(?:\.\d+)?\s*(?:tabs?|tablets?|caps?|capsules?|ml|mL|drops?|puffs?|units?|spoonfuls?|tsp|teaspoons?))\b",
        text,
        flags=re.IGNORECASE,
    )
    if quantity_match:
        quantity = quantity_match.group(1).strip()
        first_token = quantity.split()[0]
        quantity = quantity.replace(first_token, normalize_count_token(first_token), 1)
        return re.sub(r"\s+", " ", quantity)

    return ""


def extract_frequency(text: str, frequencies: Dict[str, Dict[str, str]]) -> Tuple[str, str, str, str]:
    for abbr in sorted(frequencies.keys(), key=len, reverse=True):
        if re.search(regex_boundary(abbr), text, flags=re.IGNORECASE):
            info = frequencies[abbr]
            return info.get("abbreviation", abbr), info.get("meaning", ""), info.get("latin_full_form", ""), info.get("safe_check", "")
    return "", "", "", ""


def extract_duration(text: str) -> Tuple[str, Optional[float]]:
    """Extract duration such as 3 days, 5D, 2 weeks and normalize it."""
    match = re.search(
        r"(?:\b(?:x|for)\s*)?(\d+(?:\.\d+)?)\s*(d|day|days|wk|wks|week|weeks|mo|mos|month|months)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return "", None

    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"d", "day", "days"}:
        normalized = "day" if value == 1 else "days"
        days = value
    elif unit in {"wk", "wks", "week", "weeks"}:
        normalized = "week" if value == 1 else "weeks"
        days = value * 7
    else:
        normalized = "month" if value == 1 else "months"
        days = value * 30

    value_str = str(int(value)) if value.is_integer() else str(value)
    return f"{value_str} {normalized}", days


def extract_instruction(text: str, instructions: Dict[str, str]) -> Tuple[str, str]:
    """Extract instruction abbreviations/full phrases and expand them."""
    raw_hits: List[str] = []
    meaning_hits: List[str] = []

    # Preserve any explicit Instructions: text first.
    label_match = re.search(r"Instructions?\s*:\s*(.+?)(?=\bMedicine\s*:|\bAdvice\s*:|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if label_match:
        explicit = re.sub(r"\s+", " ", label_match.group(1)).strip()
        if explicit:
            raw_hits.append(explicit)

    for term in sorted(instructions.keys(), key=len, reverse=True):
        pattern = regex_boundary(term) if re.fullmatch(r"[A-Za-z0-9]+", term) else re.escape(term)
        if re.search(pattern, text, flags=re.IGNORECASE):
            # Do not duplicate full phrases already captured from an explicit Instructions: field.
            if not any(term.lower() in existing.lower() for existing in raw_hits):
                raw_hits.append(term)
            meaning_hits.append(instructions[term])

    # Deduplicate while preserving order.
    raw = "; ".join(dict.fromkeys(x.strip() for x in raw_hits if x.strip()))
    meaning = "; ".join(dict.fromkeys(x.strip() for x in meaning_hits if x.strip()))
    return raw, meaning


def split_medicine_and_details(line: str, abbrev: Dict[str, Any]) -> Tuple[str, str]:
    """Split a compact row into medicine part and dose/duration/instruction part."""
    cleaned = re.sub(r"^\s*\d+\s*[\.)-]?\s*", "", line).strip()
    cleaned = re.sub(r"\b(?:Rx|R/x|Medications?|Drugs?)\b\s*[:\-]?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\bMedicine\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()

    starts: List[int] = []

    dose_pattern = re.search(r"\b[0-9OoIl|]{1,2}\s*-\s*[0-9OoIl|]{1,2}\s*-\s*[0-9OoIl|]{1,2}\b", cleaned)
    if dose_pattern:
        starts.append(dose_pattern.start())

    for abbr in abbrev["frequencies"].keys():
        match = re.search(regex_boundary(abbr), cleaned, flags=re.IGNORECASE)
        if match:
            starts.append(match.start())

    duration = re.search(r"\b(?:x|for)\s*\d+", cleaned, flags=re.IGNORECASE)
    if duration:
        starts.append(duration.start())

    label = re.search(r"\bDose\s*(?:and)?\s*Duration\s*:\s*|\bInstructions\s*:\s*", cleaned, flags=re.IGNORECASE)
    if label:
        starts.append(label.start())

    if not starts:
        return cleaned, ""

    idx = min(x for x in starts if x > 0) if any(x > 0 for x in starts) else min(starts)
    return cleaned[:idx].strip(" -:;|"), cleaned[idx:].strip(" -:;|")


def clean_medicine_text(medicine_line: str, abbrev: Dict[str, Any]) -> Tuple[str, str, str, str, Dict[str, str]]:
    """Clean the medication header line and keep only the medicine name/details."""
    medicine_line = re.sub(r"^\s*\d+\s*[\.)-]?\s*", "", medicine_line)
    medicine_line = re.sub(r"\bMedicine(?:\s+Name)?\s*:\s*", "", medicine_line, flags=re.IGNORECASE)
    medicine_line = re.sub(r"\b(?:Rx|R/x|Medications?|Drugs?)\b\s*[:\-]?", "", medicine_line, flags=re.IGNORECASE)
    medicine_line = re.sub(r"\s+", " ", medicine_line).strip(" -:;|")
    medicine_line = trim_to_medicine_start(medicine_line, abbrev)

    # If OCR accidentally merges the Dose/Schedule/Duration text onto the
    # medicine line, remove it before extracting the medicine name.
    medicine_line = re.sub(r"\b(?:Form|Brand\s*/\s*Form|Strength)\s*:.*$", "", medicine_line, flags=re.IGNORECASE).strip()
    medicine_line = re.sub(r"\bDose\s*:.*$", "", medicine_line, flags=re.IGNORECASE).strip()
    medicine_line = re.sub(r"\bSchedule\s*:.*$", "", medicine_line, flags=re.IGNORECASE).strip()
    medicine_line = re.sub(r"\bDuration\s*:.*$", "", medicine_line, flags=re.IGNORECASE).strip()
    medicine_line = re.sub(r"\bInstructions?\s*:.*$", "", medicine_line, flags=re.IGNORECASE).strip()

    dosage_form_raw, dosage_form_meaning = find_dosage_form(medicine_line, abbrev["dosage_forms"])
    without_form = remove_dosage_form(medicine_line, abbrev["dosage_forms"])
    strength = extract_strength(without_form)
    medicine_name = remove_strength(without_form)

    # Remove parenthetical descriptors such as "(Cough Syrup)" that describe
    # the preparation rather than the active medicine name.
    medicine_name = re.sub(r"\([^)]*(?:syrup|tablet|capsule|cough|pain|fever)[^)]*\)", " ", medicine_name, flags=re.IGNORECASE)

    # Remove fragments that are not medicine names.
    medicine_name = re.sub(r"\bDose\s*:.*", "", medicine_name, flags=re.IGNORECASE)
    medicine_name = re.sub(r"\bSchedule\s*:.*", "", medicine_name, flags=re.IGNORECASE)
    medicine_name = re.sub(r"\bDuration\s*:.*", "", medicine_name, flags=re.IGNORECASE)
    medicine_name = re.sub(r"\bDose\s*(?:and|&)?\s*Duration\b.*", "", medicine_name, flags=re.IGNORECASE)
    medicine_name = re.sub(r"\bDose\s*(?:and|&)\b.*", "", medicine_name, flags=re.IGNORECASE)
    medicine_name = re.sub(r"\bInstructions?\b.*", "", medicine_name, flags=re.IGNORECASE)
    medicine_name = re.sub(r"\b(?:After\s+food|Before\s+breakfast|At\s+night|Before\s+sleep)(?:\s*/\s*if\s+fever)?\b.*", "", medicine_name, flags=re.IGNORECASE)
    medicine_name = re.sub(r"^[_=\-<\s]+", "", medicine_name, flags=re.IGNORECASE)
    medicine_name = re.sub(r"^a\s+(?=[A-Z])", "", medicine_name)
    medicine_name = re.sub(r"\s+", " ", medicine_name).strip(" -:;|,()_")

    med_match = find_medicine_match(medicine_name, abbrev["medicines"])
    return medicine_name, dosage_form_raw, dosage_form_meaning, strength, med_match


def build_record(
    medicine_line: str,
    dose_line: str,
    instruction_line: str,
    raw_record_text: str,
    abbrev: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    medicine_line = re.sub(r"\s+", " ", medicine_line or "").strip()
    dose_line = re.sub(r"\s+", " ", dose_line or "").strip()
    instruction_line = re.sub(r"\s+", " ", instruction_line or "").strip()
    raw_record_text = re.sub(r"\s+", " ", raw_record_text or "").strip()

    if not medicine_line:
        medicine_line, compact_details = split_medicine_and_details(raw_record_text, abbrev)
        dose_line = f"{dose_line} {compact_details}".strip()

    medicine_name, dosage_form_raw, dosage_form_meaning, strength, med_match = clean_medicine_text(medicine_line, abbrev)
    if not medicine_name and not med_match.get("medicine_name_normalized"):
        return None

    combined_for_dose = " ".join(x for x in [dose_line, instruction_line, raw_record_text] if x)
    dose_pattern = extract_dose_pattern(combined_for_dose)
    dose_pattern_meaning = dose_to_readable(dose_pattern)
    times_per_day = dose_to_times_per_day(dose_pattern)
    freq_abbr, freq_meaning, freq_latin, freq_safe_check = extract_frequency(combined_for_dose, abbrev["frequencies"])
    duration, duration_days = extract_duration(combined_for_dose)

    # Prefer the explicit instruction line when available. Only scan dose/raw text
    # as a fallback for compact prescriptions that do not have an Instruction(s): label.
    instruction_scan_text = instruction_line if instruction_line else " ".join(x for x in [dose_line, raw_record_text] if x)
    instruction_raw, instruction_meaning = extract_instruction(instruction_scan_text, abbrev["instructions"])
    trailing_instruction = extract_trailing_instruction_phrase(combined_for_dose)
    if instruction_line:
        # Keep the human-readable explicit instruction as the raw instruction
        # and use abbreviation/full-phrase scanning only to fill the meaning.
        instruction_raw = instruction_line
    elif trailing_instruction:
        # Preserve compact-row instructions such as PC, BBF, or After food if fever.
        if not instruction_raw:
            instruction_raw = trailing_instruction
        elif trailing_instruction.upper() not in instruction_raw.upper():
            instruction_raw = f"{instruction_raw}; {trailing_instruction}"
        _, trailing_meaning = extract_instruction(trailing_instruction, abbrev["instructions"])
        if trailing_meaning and trailing_meaning not in instruction_meaning:
            instruction_meaning = "; ".join(x for x in [instruction_meaning, trailing_meaning] if x)

    instruction_raw = clean_instruction_value(instruction_raw)
    instruction_line = clean_instruction_value(instruction_line)

    return {
        "medicine_name": medicine_name,
        "medicine_name_normalized": med_match.get("medicine_name_normalized", medicine_name),
        "medicine_abbreviation": med_match.get("medicine_abbreviation", ""),
        "medicine_category": med_match.get("medicine_category", ""),
        "medicine_ambiguity": med_match.get("medicine_ambiguity", ""),
        "medicine_notes": med_match.get("medicine_notes", ""),
        "dosage_form": dosage_form_raw,
        "dosage_form_meaning": dosage_form_meaning,
        "strength": strength,
        "dose_pattern": dose_pattern,
        "dose_pattern_meaning": dose_pattern_meaning,
        "times_per_day_from_pattern": times_per_day,
        "dose_frequency_abbreviation": freq_abbr,
        "dose_frequency_meaning": freq_meaning,
        "dose_frequency_latin": freq_latin,
        "dose_frequency_safe_check": freq_safe_check,
        "duration": duration,
        "duration_days_estimate": duration_days,
        "instructions": instruction_raw,
        "instruction_meaning": instruction_meaning,
        "raw_medicine_text": medicine_line,
        "raw_dose_text": dose_line,
        "raw_instruction_text": instruction_line,
        "raw_record_text": raw_record_text,
    }


def extract_labeled_blocks(text: str, abbrev: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract records from explicit Medicine/Dose/Instruction blocks.

    Handles both older labels like "Dose and Duration:" and generated formats like:
    "Medicine: Tab Ibuprofen 400 mg", "Dose: 1 tab, Schedule: Twice daily,
    Duration: 5 days", "Instruction: After food".
    """
    records: List[Dict[str, Any]] = []

    # Normalize singular/plural instruction labels just for parsing.
    parse_text = re.sub(r"\bInstruction\s*:", "Instructions:", text, flags=re.IGNORECASE)

    # Split at each numbered or unnumbered Medicine: marker. Keep only blocks after a marker.
    blocks = re.split(r"(?:^|\n)\s*\d*\s*[\.)-]?\s*Medicine\s*:\s*", parse_text, flags=re.IGNORECASE)

    for block in blocks[1:]:
        block = block.strip()
        if not block:
            continue

        # Last medicine block may be followed by Advice/footer/signature text. Cut it off
        # so false hits such as "aC" in registration/footer text do not become AC instructions.
        block = re.split(
            r"\n\s*(?:Advice\s*:|Doctor\s+Signature|This is sample|Not valid|Do not use)",
            block,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

        med_match = re.search(
            r"(.+?)(?=\s*(?:Dose\s*(?:and|&)?\s*Duration\s*:|Dose\s*:|Schedule\s*:|Instructions?\s*:|\d+\s*[\.)-]?\s*Medicine\s*:|Advice\s*:|$))",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        medicine_line = med_match.group(1).strip() if med_match else block.split("\n")[0].strip()

        dose_match = re.search(
            r"(?:Dose\s*(?:and|&)?\s*Duration\s*:|Dose\s*:)(.+?)(?=\s*(?:Instructions?\s*:|\d+\s*[\.)-]?\s*Medicine\s*:|Advice\s*:|$))",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        dose_line = dose_match.group(1).strip(" -:;|\n") if dose_match else ""

        instruction_match = re.search(
            r"Instructions?\s*:\s*(.+?)(?=\s*(?:\d+\s*[\.)-]?\s*Medicine\s*:|Advice\s*:|$))",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        instruction_line = instruction_match.group(1).strip(" -:;|\n") if instruction_match else ""

        # Build a clean per-medicine raw text instead of passing the whole OCR tail.
        raw_parts = [medicine_line]
        if dose_line:
            raw_parts.append("Dose: " + dose_line)
        if instruction_line:
            raw_parts.append("Instructions: " + instruction_line)
        raw_record_text = " ".join(raw_parts)

        record = build_record(medicine_line, dose_line, instruction_line, raw_record_text, abbrev)
        if record:
            records.append(record)

    return records



def extract_anywhere_labeled_blocks(text: str, abbrev: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract Medicine/Dose/Duration/Instruction blocks even when OCR adds noise.

    Unlike extract_labeled_blocks, this does not require the Medicine: marker to
    begin a clean line. This helps with rows like "i 1. Medicine: ..." or
    OCR output where the Rx symbol/header is merged into the first item.
    """
    records: List[Dict[str, Any]] = []
    parse_text = normalize_text(text)
    parse_text = strip_after_non_rx_sections(parse_text)
    markers = list(re.finditer(r"\bMedicine\s*:\s*", parse_text, flags=re.IGNORECASE))
    for idx, marker in enumerate(markers):
        start = marker.end()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(parse_text)
        block = parse_text[start:end].strip()
        block = re.split(
            r"\n\s*(?:Advice|Investigations|Doctor\s+Signature|Signature|Sample|This is sample|Not valid|Do not use)\b",
            block,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        if not block:
            continue
        flat = re.sub(r"\s+", " ", block).strip()
        med_match = re.match(
            r"(.+?)(?=\b(?:Dose\s*(?:and|&)?\s*Duration|Dose|Schedule|Duration|Instructions?)\s*:|$)",
            flat,
            flags=re.IGNORECASE,
        )
        medicine_line = med_match.group(1).strip(" -:;|,.") if med_match else flat
        details = flat[len(med_match.group(1)):] if med_match else ""
        instr_match = re.search(r"\bInstructions?\s*:\s*(.+?)(?=\b(?:Medicine|Advice|Doctor|Signature)\s*:|$)", flat, flags=re.IGNORECASE)
        instruction_line = instr_match.group(1).strip(" -:;|,.") if instr_match else ""
        rec = build_record(medicine_line, details, instruction_line, f"Medicine: {flat}", abbrev)
        if rec:
            records.append(rec)
    return records


def looks_like_medicine_line(line: str, abbrev: Dict[str, Any]) -> bool:
    lower = line.lower().strip()
    if len(lower) < 4:
        return False

    blocked = [
        "clinic",
        "doctor",
        "patient",
        "date",
        "phone",
        "email",
        "diagnosis",
        "advice",
        "review",
        "signature",
        "valid for medical",
        "self-medication",
        "age/sex",
        "uhid",
        "weight",
        "reg. no",
        "provisional",
    ]
    if any(key in lower for key in blocked):
        return False

    has_form = bool(find_dosage_form(line, abbrev["dosage_forms"])[0])
    has_strength = bool(extract_strength(line))
    has_dose_pattern = bool(extract_dose_pattern(line))
    has_frequency = bool(extract_frequency(line, abbrev["frequencies"])[0])
    has_known_medicine = any(re.search(regex_boundary(k), line, flags=re.IGNORECASE) for k in known_medicine_terms(abbrev))

    return has_form or (has_known_medicine and (has_strength or has_dose_pattern or has_frequency))


def extract_line_records(text: str, abbrev: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fallback parser for compact/table-like formats without explicit Medicine blocks."""
    records: List[Dict[str, Any]] = []
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    used_raw = set()

    for i, line in enumerate(lines):
        if not looks_like_medicine_line(line, abbrev):
            continue

        combined = line
        # Add likely continuation lines containing dosage/duration/instructions.
        for nxt in lines[i + 1 : i + 3]:
            continuation = bool(
                re.search(r"Dose|Duration|Instruction", nxt, flags=re.IGNORECASE)
                or extract_dose_pattern(nxt)
                or extract_frequency(nxt, abbrev["frequencies"])[0]
                or extract_duration(nxt)[0]
                or extract_instruction(nxt, abbrev["instructions"])[0]
            )
            if continuation and not looks_like_medicine_line(nxt, abbrev):
                combined += " " + nxt

        if combined in used_raw:
            continue
        used_raw.add(combined)

        medicine_line, details = split_medicine_and_details(combined, abbrev)
        record = build_record(medicine_line, details, "", combined, abbrev)
        if record:
            records.append(record)

    return records




def strip_after_non_rx_sections(text: str) -> str:
    """Keep only the probable prescription area and remove advice/footer text."""
    cut = re.split(
        r"\n\s*(?:Advice|Investigations|Doctor\s*(?:Signature|Signatt)|Signature|Sample|This is sample|Not valid|Do not use)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    # Prefer the text after the prescription marker. This avoids treating
    # address/phone/date numbers as medicine rows while still falling back to
    # the full text when no Rx marker is recognized.
    markers = list(re.finditer(
        r"(?:^|\n)\s*(?:Rx|R/x|Medicines?|Prescription\s*\(\s*Rx\s*\))\s*:?\s*",
        cut,
        flags=re.IGNORECASE,
    ))
    if markers:
        cut = cut[markers[-1].end():]
    return cut


def record_identity(record: Dict[str, Any]) -> Tuple[str, str, str]:
    name = str(record.get("medicine_name_normalized") or record.get("medicine_name") or "").lower()
    name = re.sub(r"\b(?:after\s+food|before\s+breakfast|at\s+night|before\s+sleep)\b.*", "", name)
    name = re.sub(r"\b(?:dose|and|duration|instructions?|schedule)\b.*", "", name)
    name = re.sub(r"\W+", "", name)
    return (
        name,
        str(record.get("strength") or "").lower(),
        str(record.get("duration") or "").lower(),
    )


def dedupe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove parser duplicates and keep the most complete row per medicine."""
    best: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    order: List[str] = []
    known_names = [
        "azithromycin", "paracetamol", "pantoprazole", "montelukast", "levocetirizine",
        "amoxicillin", "ibuprofen", "cetirizine", "dextromethorphan", "chlorpheniramine",
        "domperidone", "drotaverine", "fexofenadine", "saline", "nitrofurantoin",
        "urinary alkaliser", "probiotic", "vitamin c", "metformin", "amlodipine",
        "atorvastatin", "vitamin d3", "hydrocortisone", "calamine", "prednisolone",
        "cough formula", "cough syrup"
    ]
    for rec in records:
        raw_name = str(rec.get("medicine_name") or "")
        lower_name = raw_name.lower()
        name_key = re.sub(r"\b(?:after\s+food|before\s+breakfast|at\s+night|before\s+sleep)\b.*", "", lower_name)
        name_key = re.sub(r"\b(?:dose|and|duration|instructions?|schedule)\b.*", "", name_key)
        name_key = re.sub(r"\W+", "", name_key)
        if not name_key:
            continue
        if any(bad in name_key for bad in ["clinic", "doctor", "patient", "diagnosis", "advice", "phone", "date", "regno"]):
            continue
        if len(raw_name) > 80:
            continue
        found_known = [nm for nm in known_names if nm in lower_name]
        if not found_known:
            # Drop rows created from OCR fragments such as "Schedule: ..." or
            # leftover table cells that contain dose/duration but no medicine.
            if re.search(r"\b(?:dose|duration|instructions?|schedule|doctor|signature)\b", lower_name):
                continue
            if not str(rec.get("strength") or "").strip() and not str(rec.get("dose_frequency_abbreviation") or "").strip():
                continue
        if "+" in str(rec.get("strength") or "") and found_known:
            continue
        # Do not reject known combination medicines such as Montelukast + Levocetirizine.
        has_any_detail = any(str(rec.get(col) or "").strip() for col in ["strength", "dose_pattern", "duration", "instructions", "dose_frequency_abbreviation"])
        if not has_any_detail:
            continue
        # Canonicalize key to a known medicine when possible, so OCR prefixes like "a Paracetamol" dedupe.
        if found_known:
            if "montelukast" in found_known or "levocetirizine" in found_known:
                key = "montelukastlevocetirizine"
            elif "dextromethorphan" in found_known or "chlorpheniramine" in found_known:
                key = "dextromethorphanchlorpheniramine"
            else:
                key = found_known[0]
        else:
            key = name_key
        score = 0
        for col in ["strength", "dose_pattern", "duration", "instructions", "dose_frequency_abbreviation", "dose_frequency_meaning"]:
            if str(rec.get(col) or "").strip():
                score += 2 if col in {"strength", "dose_pattern", "duration"} else 1
        if key not in best:
            order.append(key)
            best[key] = (score, rec)
        else:
            old_score, old = best[key]
            if score > old_score:
                # Preserve any values from the old row that the new row lacks.
                merged = dict(old)
                merged.update({k: v for k, v in rec.items() if str(v or "").strip()})
                best[key] = (score, merged)
    return [best[k][1] for k in order]

def extract_numbered_rx_blocks(text: str, abbrev: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse numbered prescriptions in paragraph/list/table OCR layouts.

    Supports examples such as:
    1. Tab Azithromycin 500 mg, 1-0-0 for 3 days, After food.
    1 Tab Azithromycin 500 mg 1-0-0 3 days After food
    1. Azithromycin 500 mg Tab OD (1 tablet) 3 DAYS PC - Empty stomach.
    """
    records: List[Dict[str, Any]] = []
    rx_text = strip_after_non_rx_sections(text)
    # Remove table headers that otherwise look like medicines.
    rx_text = re.sub(r"\bS\.?\s*No\.?\s+Medicine\b.*", "", rx_text, flags=re.IGNORECASE)
    # Find numbered chunks. Use multiline starts; tolerate OCR spacing.
    matches = list(re.finditer(r"(?:^|\n)\s*(\d{1,2})\s*[\.)-]?\s+", rx_text))
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(rx_text)
        block = rx_text[start:end].strip()
        block = re.split(r"\n\s*(?:Advice|Investigations|Doctor|Signature|Sample|This is sample|Not valid)\b", block, maxsplit=1, flags=re.IGNORECASE)[0]
        block = re.sub(r"\s+", " ", block).strip(" -:;|,. ")
        if not block or len(block) < 6:
            continue
        # If this numbered block contains explicit labels, normalize them to a compact form.
        block = re.sub(r"\bMedicine\s+Name\s*:\s*", "Medicine: ", block, flags=re.IGNORECASE)
        block = re.sub(r"\bFood\s+Instruction\s*:\s*", "Instructions: ", block, flags=re.IGNORECASE)
        block = re.sub(r"\bDosage\s*:\s*", "Dose: ", block, flags=re.IGNORECASE)
        block = re.sub(r"\bFrequency\s*:\s*", "Schedule: ", block, flags=re.IGNORECASE)
        # Remove leading label if present.
        block_wo_label = re.sub(r"^Medicine\s*:\s*", "", block, flags=re.IGNORECASE).strip()
        medicine_line, details = split_medicine_and_details(block_wo_label, abbrev)
        # If medicine line still contains label fields, cut before them.
        med_cut = re.split(r"\b(?:Form|Brand\s*/\s*Form|Strength|Dose|Schedule|Duration|Instructions)\s*:", medicine_line, maxsplit=1, flags=re.IGNORECASE)
        if len(med_cut) > 1:
            details = medicine_line[len(med_cut[0]):] + " " + details
            medicine_line = med_cut[0]
        rec = build_record(medicine_line, details, "", block, abbrev)
        if rec:
            records.append(rec)
    return records


def extract_field_label_records(text: str, abbrev: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse blocks where each medicine has labels like Medicine, Strength, Form, Dose, Duration."""
    records: List[Dict[str, Any]] = []
    parse_text = normalize_text(text)
    parse_text = strip_after_non_rx_sections(parse_text)
    # Split on Medicine: occurrences, including after OCR-normalized Medicine Name.
    parts = re.split(r"(?:^|\n|\s{2,})\s*\d*\s*[\.)-]?\s*Medicine\s*:\s*", parse_text, flags=re.IGNORECASE)
    for part in parts[1:]:
        part = re.split(r"(?=\s*\d+\s*[\.)-]?\s*Medicine\s*:)|\n\s*(?:Advice|Doctor|Signature|Sample|This is sample|Not valid)", part, maxsplit=1, flags=re.IGNORECASE)[0]
        flat = re.sub(r"\s+", " ", part).strip()
        if not flat:
            continue
        med_match = re.match(r"(.+?)(?=\b(?:Form|Brand\s*/\s*Form|Strength|Dose|Schedule|Duration|Instructions)\s*:|$)", flat, flags=re.IGNORECASE)
        medicine_line = med_match.group(1).strip(" -:;|,") if med_match else flat
        details = flat[len(med_match.group(1)):] if med_match else ""
        # Append strength/form to medicine line if they are separate fields.
        strength = re.search(r"\bStrength\s*:\s*([^,;]+?)(?=\b(?:Form|Dose|Schedule|Duration|Instructions)\s*:|$)", flat, flags=re.IGNORECASE)
        form = re.search(r"\bForm\s*:\s*([^,;]+?)(?=\b(?:Strength|Dose|Schedule|Duration|Instructions)\s*:|$)", flat, flags=re.IGNORECASE)
        med_for_build = medicine_line
        if form and not find_dosage_form(med_for_build, abbrev["dosage_forms"])[0]:
            med_for_build = form.group(1).strip() + " " + med_for_build
        if strength and not extract_strength(med_for_build):
            med_for_build = med_for_build + " " + strength.group(1).strip()
        instr = ""
        im = re.search(r"\bInstructions\s*:\s*(.+?)(?=\b(?:Medicine|Advice|Doctor|Signature)\s*:|$)", flat, flags=re.IGNORECASE)
        if im:
            instr = im.group(1).strip()
        rec = build_record(med_for_build, details, instr, flat, abbrev)
        if rec:
            records.append(rec)
    return records


def extract_matrix_table_records(text: str, abbrev: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse table-like OCR where rows contain medicine, dose columns and instructions."""
    records: List[Dict[str, Any]] = []
    rx_text = strip_after_non_rx_sections(text)
    lines = [re.sub(r"\s+", " ", x).strip() for x in rx_text.splitlines() if x.strip()]
    # Combine wrapped numbered rows.
    chunks: List[str] = []
    current = ""
    for line in lines:
        if re.match(r"^\d{1,2}\s*[\.)-]?\s+", line):
            if current:
                chunks.append(current)
            current = line
        elif current and not re.search(r"\b(?:Clinic|Doctor|Patient|Diagnosis|UHID|Date)\b", line, flags=re.IGNORECASE):
            # Continuation if it has known medicine/detail terms or starts with strength continuation.
            if (extract_strength(line) or extract_duration(line)[0] or extract_dose_pattern(line) or
                extract_frequency(line, abbrev["frequencies"])[0] or extract_instruction(line, abbrev["instructions"])[0] or
                any(re.search(regex_boundary(k), line, flags=re.IGNORECASE) for k in known_medicine_terms(abbrev))):
                current += " " + line
    if current:
        chunks.append(current)
    for chunk in chunks:
        # Skip obvious non-medicine numbered text.
        if not looks_like_medicine_line(chunk, abbrev) and not any(re.search(regex_boundary(k), chunk, flags=re.IGNORECASE) for k in known_medicine_terms(abbrev)):
            continue
        body = re.sub(r"^\d{1,2}\s*[\.)-]?\s*", "", chunk).strip()
        medicine_line, details = split_medicine_and_details(body, abbrev)
        rec = build_record(medicine_line, details, "", body, abbrev)
        if rec:
            records.append(rec)
    return records

def extract_medications(text: str, abbrev: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract medication records from normalized OCR text across many prescription layouts."""
    records: List[Dict[str, Any]] = []
    # Run several complementary parsers. The final de-duplication keeps the best
    # unique medicine rows while allowing each image layout to succeed.
    for parser in (
        extract_anywhere_labeled_blocks,
        extract_labeled_blocks,
        extract_field_label_records,
        extract_numbered_rx_blocks,
        extract_matrix_table_records,
        extract_line_records,
    ):
        try:
            records.extend(parser(text, abbrev))
        except Exception as exc:
            print(f"Warning: {parser.__name__} failed: {exc}")
    return dedupe_records(records)



def find_image_paths(input_path: str) -> List[Path]:
    """Return prescription image paths from one image file or a folder of images.

    This function intentionally does NOT support ZIP files.

    Parameters
    ----------
    input_path:
        Path to either:
        - one prescription image file, for example /content/prescription_1.jpg
        - one folder containing prescription images, for example /content/prescription_images/

    Returns
    -------
    List[Path]
        One or more image paths. Folder inputs are searched recursively, so images
        inside subfolders are included too.
    """
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input path does not exist: {source}")

    if source.is_file():
        if source.suffix.lower() == ".zip":
            raise ValueError(
                "ZIP input is intentionally not supported in this version. "
                "Please provide a single image file or a folder containing images."
            )
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            return [source]
        raise ValueError(
            f"Unsupported file type: {source.suffix}. Provide an image file with one of: "
            f"{', '.join(sorted(IMAGE_EXTENSIONS))}"
        )

    if source.is_dir():
        image_paths = sorted(
            [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
            key=natural_key,
        )
        if not image_paths:
            raise ValueError(
                f"No supported prescription images found in folder: {source}. "
                f"Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}"
            )
        return image_paths

    raise ValueError(
        f"Unsupported input: {source}. Provide a single prescription image or a folder containing images."
    )


def make_empty_record(
    source_image: str,
    raw_record_text: str = "",
    ocr_text: str = "",
    error: str = "",
) -> Dict[str, Any]:
    """Create a consistent empty/error row when no medicine is detected or OCR fails."""
    return {
        "source_image": source_image,
        "record_number_in_image": "",
        "medicine_name": "",
        "medicine_name_normalized": "",
        "medicine_abbreviation": "",
        "medicine_category": "",
        "medicine_ambiguity": "",
        "medicine_notes": "",
        "dosage_form": "",
        "dosage_form_meaning": "",
        "strength": "",
        "dose_pattern": "",
        "dose_pattern_meaning": "",
        "times_per_day_from_pattern": "",
        "dose_frequency_abbreviation": "",
        "dose_frequency_meaning": "",
        "dose_frequency_latin": "",
        "dose_frequency_safe_check": "",
        "duration": "",
        "duration_days_estimate": "",
        "instructions": "",
        "instruction_meaning": "",
        "raw_medicine_text": "",
        "raw_dose_text": "",
        "raw_instruction_text": "",
        "raw_record_text": raw_record_text,
        "ocr_text": ocr_text,
        "error": error,
    }


def records_to_dataframe(all_records: List[Dict[str, Any]], output_csv: str) -> pd.DataFrame:
    """Convert extracted records to a stable-column dataframe and save it."""
    df = pd.DataFrame(all_records)

    preferred_columns = [
        "source_image",
        "record_number_in_image",
        "medicine_name",
        "medicine_name_normalized",
        "medicine_abbreviation",
        "medicine_category",
        "medicine_ambiguity",
        "medicine_notes",
        "dosage_form",
        "dosage_form_meaning",
        "strength",
        "dose_pattern",
        "dose_pattern_meaning",
        "times_per_day_from_pattern",
        "dose_frequency_abbreviation",
        "dose_frequency_meaning",
        "dose_frequency_latin",
        "dose_frequency_safe_check",
        "duration",
        "duration_days_estimate",
        "instructions",
        "instruction_meaning",
        "raw_medicine_text",
        "raw_dose_text",
        "raw_instruction_text",
        "raw_record_text",
        "ocr_text",
        "error",
    ]
    for col in preferred_columns:
        if col not in df.columns:
            df[col] = ""
    df = df[preferred_columns]

    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved extracted medication table to: {output_csv_path}")
    return df

def prescription_image_to_records(
    image_path: str,
    abbrev: Dict[str, Any],
    save_processed_dir: Optional[str] = None,
    ocr_timeout: int = 60,
) -> Tuple[List[Dict[str, Any]], str]:
    """Image -> OCR text -> medication records."""
    processed = preprocess_prescription(image_path)

    if save_processed_dir:
        out_dir = Path(save_processed_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{Path(image_path).stem}_processed.png"), processed)

    ocr_text = run_ocr(processed, ocr_timeout=ocr_timeout, psm_modes=(6, 4, 11))
    clean_text = normalize_text(ocr_text)
    records = extract_medications(clean_text, abbrev)

    for idx, record in enumerate(records, start=1):
        record["source_image"] = Path(image_path).name
        record["record_number_in_image"] = idx
        record["ocr_text"] = clean_text

    return records, clean_text



def process_prescription_input(
    input_path: str,
    dosage_csv: str,
    instruction_csv: str,
    duration_csv: str,
    medicine_csv: str,
    output_csv: str = "prescription_medicine_extracts.csv",
    output_dir: str = "prescription_ocr_outputs",
    save_ocr_text: bool = True,
    save_processed_images: bool = False,
    ocr_timeout: int = 60,
) -> pd.DataFrame:
    """Extract medicines from one prescription image or every image in a folder.

    input_path can be either:
    - Single image: /content/prescription_1.jpg
    - Folder:       /content/prescription_images/

    ZIP files are not supported in this version.

    The function processes each prescription image independently. If OCR or parsing
    fails for one image, the error is saved in the output table and the remaining
    images continue processing.
    """

    abbrev = load_abbreviation_resources(
        dosage_csv=dosage_csv,
        instruction_csv=instruction_csv,
        duration_csv=duration_csv,
        medicine_csv=medicine_csv,
    )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    ocr_text_dir = output_dir_path / "ocr_texts"
    processed_dir = output_dir_path / "processed_images" if save_processed_images else None
    if save_ocr_text:
        ocr_text_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_image_paths(input_path)
    print(f"Found {len(image_paths)} prescription image(s) in input: {input_path}")

    all_records: List[Dict[str, Any]] = []
    for image_path in image_paths:
        print(f"Processing: {image_path.name}")
        try:
            records, clean_text = prescription_image_to_records(
                str(image_path),
                abbrev,
                save_processed_dir=str(processed_dir) if processed_dir else None,
                ocr_timeout=ocr_timeout,
            )

            if save_ocr_text:
                (ocr_text_dir / f"{image_path.stem}.txt").write_text(clean_text, encoding="utf-8")

            if records:
                all_records.extend(records)
            else:
                all_records.append(
                    make_empty_record(
                        source_image=image_path.name,
                        raw_record_text="No medication rows detected; review OCR text.",
                        ocr_text=clean_text,
                        error="",
                    )
                )
        except Exception as exc:
            all_records.append(
                make_empty_record(
                    source_image=image_path.name,
                    raw_record_text="",
                    ocr_text="",
                    error=str(exc),
                )
            )
            print(f"Warning: failed to process {image_path.name}: {exc}")

    return records_to_dataframe(all_records, output_csv=output_csv)


def process_single_prescription_image(
    image_path: str,
    dosage_csv: str,
    instruction_csv: str,
    duration_csv: str,
    medicine_csv: str,
    output_csv: str = "single_prescription_medicine_extracts.csv",
    output_dir: str = "prescription_ocr_outputs",
    save_ocr_text: bool = True,
    save_processed_images: bool = False,
    ocr_timeout: int = 60,
) -> pd.DataFrame:
    """Convenience wrapper for processing one prescription image."""
    return process_prescription_input(
        input_path=image_path,
        dosage_csv=dosage_csv,
        instruction_csv=instruction_csv,
        duration_csv=duration_csv,
        medicine_csv=medicine_csv,
        output_csv=output_csv,
        output_dir=output_dir,
        save_ocr_text=save_ocr_text,
        save_processed_images=save_processed_images,
        ocr_timeout=ocr_timeout,
    )


def process_prescription_folder(
    folder_path: str,
    dosage_csv: str,
    instruction_csv: str,
    duration_csv: str,
    medicine_csv: str,
    output_csv: str = "folder_prescription_medicine_extracts.csv",
    output_dir: str = "prescription_ocr_outputs",
    save_ocr_text: bool = True,
    save_processed_images: bool = False,
    ocr_timeout: int = 60,
) -> pd.DataFrame:
    """Convenience wrapper for processing every image in a folder."""
    return process_prescription_input(
        input_path=folder_path,
        dosage_csv=dosage_csv,
        instruction_csv=instruction_csv,
        duration_csv=duration_csv,
        medicine_csv=medicine_csv,
        output_csv=output_csv,
        output_dir=output_dir,
        save_ocr_text=save_ocr_text,
        save_processed_images=save_processed_images,
        ocr_timeout=ocr_timeout,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for single-image or folder processing."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract medicine name, dose pattern, schedule/frequency, instruction, "
            "and duration from one prescription image or from a folder of images. "
            "ZIP files are intentionally not supported."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to one prescription image or to a folder containing prescription images.",
    )
    parser.add_argument(
        "--dosage-csv",
        default="Abbreviations/Abbreviations_for_dosage.csv",
        help="Path to dosage-form abbreviation CSV. Default: Abbreviations_for_dosage.csv",
    )
    parser.add_argument(
        "--instruction-csv",
        default="Abbreviations/Abbreviations_for_instructions.csv",
        help="Path to instruction abbreviation CSV. Default: Abbreviations_for_instructions.csv",
    )
    parser.add_argument(
        "--duration-csv",
        default="Abbreviations/Duration_abbreviations.csv",
        help=(
            "Path to frequency/duration abbreviation CSV. "
            "Default: Duration_abbreviations.csv"
        ),
    )
    parser.add_argument(
        "--medicine-csv",
        default="Abbreviations/india_medicine_abbreviations.csv",
        help="Path to medicine abbreviation CSV. Default: india_medicine_abbreviations.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="prescription_medicine_extracts.csv",
        help="Where to save the extracted medication table CSV.",
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
        help=(
            "Optional full path to the Tesseract executable. Useful on Windows, "
            "for example: C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        ),
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=20,
        help="Number of extracted rows to print after processing. Use 0 to skip preview.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_tesseract(args.tesseract_cmd)

    df = process_prescription_input(
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

    print("\nExtracted medication table preview:")
    if args.preview_rows > 0:
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(df.head(args.preview_rows).to_string(index=False))
    print(f"\nTotal rows extracted: {len(df)}")
    print(f"Output CSV: {Path(args.output_csv).resolve()}")


if __name__ == "__main__":
    main()

