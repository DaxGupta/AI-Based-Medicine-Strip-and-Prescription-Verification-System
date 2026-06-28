#!/usr/bin/env python3
"""
inference_simple_medicine_salt_altered.py

Self-contained inference script for the SimpleMedicineSaltResolver TF-IDF index.

This version is meant for the supplied joblib file:

    filtered_medicines_cleaned_simple_tfidf_index.joblib

It does NOT require simple_medicine_salt_matcher.py or the original knowledge CSV,
because the joblib already contains the vectorizer, matrices, medicines, salts, and
medicine-to-salts mapping needed for inference.

Examples:

1. Single query:
    python inference_simple_medicine_salt_altered.py \
        --index-path filtered_medicines_cleaned_simple_tfidf_index.joblib \
        --query "Augmentin 625 Duo"

2. Batch CSV:
    python inference_simple_medicine_salt_altered.py \
        --index-path filtered_medicines_cleaned_simple_tfidf_index.joblib \
        --input-csv medicine_salt_test_10000.csv \
        --input-column query \
        --output-csv simple_predictions_10000.csv

3. Backward-compatible call using --knowledge-csv:
    python inference_simple_medicine_salt_altered.py \
        --knowledge-csv filtered_medicines_cleaned.csv \
        --query "Augmentin 625 Duo"
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover - optional dependency fallback
    fuzz = None
    process = None


INDEX_FILENAME = "filtered_medicines_cleaned_simple_tfidf_index.joblib"
REQUIRED_INDEX_KEYS = {
    "version",
    "unique_medicines",
    "unique_salts",
    "medicine_to_salts",
    "vectorizer",
    "medicine_matrix",
    "salt_matrix",
}


def normalize_text(value: Any) -> str:
    """Normalize text for exact/fuzzy lookup while leaving TF-IDF to its own analyzer."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def ordered_unique(values: Iterable[str]) -> List[str]:
    """Return values in first-seen order after dropping duplicates and blanks."""
    seen = set()
    output: List[str] = []
    for value in values:
        value_str = str(value).strip()
        key = normalize_text(value_str)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(value_str)
    return output


def split_possible_salt_components(query: str) -> List[str]:
    """
    Split multi-salt inputs such as "Amoxycillin + Clavulanic Acid".

    The full query is also used by the resolver; this helper only adds component-level
    matching for common composition separators.
    """
    parts = re.split(r"\s*(?:\+|,|;|/|&|\band\b)\s*", query, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part and part.strip()]


@dataclass
class SimpleMedicineSaltResolver:
    """
    Inference-only resolver that loads the supplied simple TF-IDF joblib index.

    The saved index is a dictionary containing all inference artifacts. This class
    recreates the resolver interface expected by the original inference script:
    load_index(), resolve(), and resolve_many().
    """

    knowledge_csv: Optional[str] = None
    index_path: Optional[str] = None
    medicine_threshold: Optional[float] = None
    salt_threshold: Optional[float] = None
    fuzzy_threshold: Optional[int] = None
    top_k: Optional[int] = None
    use_fuzzy: bool = True

    def __post_init__(self) -> None:
        self.index_path = str(self._resolve_index_path())
        self.index: Dict[str, Any] = {}
        self.loaded = False

        self.unique_medicines: List[str] = []
        self.unique_salts: List[str] = []
        self.medicine_to_salts: Dict[str, List[str]] = {}
        self.vectorizer = None
        self.medicine_matrix = None
        self.salt_matrix = None

        self._medicine_exact: Dict[str, str] = {}
        self._salt_exact: Dict[str, str] = {}
        self._medicine_choices_norm: List[str] = []
        self._salt_choices_norm: List[str] = []

    def _resolve_index_path(self) -> Path:
        """Find the index from --index-path, --knowledge-csv, or the script folder."""
        if self.index_path:
            return Path(self.index_path).expanduser()

        if self.knowledge_csv:
            knowledge_path = Path(self.knowledge_csv).expanduser()
            derived = knowledge_path.with_name(f"{knowledge_path.stem}_simple_tfidf_index.joblib")
            if derived.exists():
                return derived

        local_index = Path(__file__).resolve().with_name(INDEX_FILENAME)
        return local_index

    def load_index(self) -> None:
        """Load and validate the supplied joblib index."""
        path = Path(self.index_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Index file not found: {path}\n"
                "Pass it explicitly with --index-path, for example:\n"
                f"  python {Path(__file__).name} --index-path {INDEX_FILENAME} --query \"Augmentin 625 Duo\""
            )

        index = joblib.load(path)
        if not isinstance(index, dict):
            raise TypeError(
                f"Expected the joblib index to contain a dict, but got {type(index)!r}."
            )

        missing = sorted(REQUIRED_INDEX_KEYS - set(index.keys()))
        if missing:
            raise KeyError(f"Index is missing required keys: {missing}")

        self.index = index
        self.unique_medicines = list(index["unique_medicines"])
        self.unique_salts = list(index["unique_salts"])
        self.medicine_to_salts = {
            str(medicine): list(salts or [])
            for medicine, salts in dict(index["medicine_to_salts"]).items()
        }
        self.vectorizer = index["vectorizer"]
        self.medicine_matrix = index["medicine_matrix"]
        self.salt_matrix = index["salt_matrix"]

        # Use index defaults unless the CLI explicitly overrides them.
        if self.medicine_threshold is None:
            self.medicine_threshold = float(index.get("medicine_threshold", 0.72))
        if self.salt_threshold is None:
            self.salt_threshold = float(index.get("salt_threshold", 0.70))
        if self.fuzzy_threshold is None:
            self.fuzzy_threshold = int(index.get("fuzzy_threshold", 92))
        if self.top_k is None:
            self.top_k = int(index.get("top_k", 5))

        self._medicine_exact = {
            normalize_text(medicine): medicine for medicine in self.unique_medicines
        }
        self._salt_exact = {normalize_text(salt): salt for salt in self.unique_salts}
        self._medicine_choices_norm = [normalize_text(m) for m in self.unique_medicines]
        self._salt_choices_norm = [normalize_text(s) for s in self.unique_salts]
        self.loaded = True

    def _require_loaded(self) -> None:
        if not self.loaded:
            self.load_index()

    def _top_tfidf_matches(
        self,
        query: str,
        choices: Sequence[str],
        matrix: Any,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """Return top TF-IDF cosine matches above threshold."""
        if not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])
        scores_matrix = query_vector @ matrix.T
        scores = np.asarray(scores_matrix.toarray()).ravel()

        if scores.size == 0:
            return []

        top_k = min(int(self.top_k or 5), scores.size)
        if top_k <= 0:
            return []

        candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
        candidate_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

        matches: List[Dict[str, Any]] = []
        for idx in candidate_indices:
            score = float(scores[idx])
            if score >= threshold:
                matches.append(
                    {
                        "name": choices[int(idx)],
                        "score": round(score, 6),
                        "match_type": "tfidf",
                    }
                )
        return matches

    def _fuzzy_matches(
        self,
        query: str,
        choices: Sequence[str],
        normalized_choices: Sequence[str],
        threshold: int,
    ) -> List[Dict[str, Any]]:
        """Return RapidFuzz matches above threshold, if RapidFuzz is available."""
        if not self.use_fuzzy or process is None or fuzz is None or not query.strip():
            return []

        normalized_query = normalize_text(query)
        raw_matches = process.extract(
            normalized_query,
            normalized_choices,
            scorer=fuzz.WRatio,
            score_cutoff=threshold,
            limit=int(self.top_k or 5),
        )

        matches: List[Dict[str, Any]] = []
        for _, score, idx in raw_matches:
            matches.append(
                {
                    "name": choices[int(idx)],
                    "score": round(float(score) / 100.0, 6),
                    "match_type": "fuzzy",
                }
            )
        return matches

    def _merge_matches(self, matches: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate matches by normalized name, keeping the highest-priority result."""
        priority = {"exact": 3, "fuzzy": 2, "tfidf": 1}
        by_name: Dict[str, Dict[str, Any]] = {}

        for match in matches:
            name = str(match["name"])
            key = normalize_text(name)
            existing = by_name.get(key)
            if existing is None:
                by_name[key] = dict(match)
                continue

            current_rank = priority.get(str(match.get("match_type")), 0)
            existing_rank = priority.get(str(existing.get("match_type")), 0)
            if (current_rank, float(match.get("score", 0.0))) > (
                existing_rank,
                float(existing.get("score", 0.0)),
            ):
                by_name[key] = dict(match)

        return sorted(
            by_name.values(),
            key=lambda item: (
                priority.get(str(item.get("match_type")), 0),
                float(item.get("score", 0.0)),
            ),
            reverse=True,
        )[: int(self.top_k or 5)]

    def _match_medicines(self, query: str) -> List[Dict[str, Any]]:
        normalized_query = normalize_text(query)
        matches: List[Dict[str, Any]] = []

        exact = self._medicine_exact.get(normalized_query)
        if exact:
            matches.append({"name": exact, "score": 1.0, "match_type": "exact"})

        tfidf_matches = self._top_tfidf_matches(
            query=query,
            choices=self.unique_medicines,
            matrix=self.medicine_matrix,
            threshold=float(self.medicine_threshold or 0.72),
        )
        matches.extend(tfidf_matches)

        # Full fuzzy search over 147k+ medicine names can be expensive. Only use it
        # as a fallback when exact/TF-IDF did not find a medicine.
        if not exact and not tfidf_matches:
            matches.extend(
                self._fuzzy_matches(
                    query=query,
                    choices=self.unique_medicines,
                    normalized_choices=self._medicine_choices_norm,
                    threshold=int(self.fuzzy_threshold or 92),
                )
            )
        return self._merge_matches(matches)

    def _match_salts(self, query: str) -> List[Dict[str, Any]]:
        candidates = [query]
        for component in split_possible_salt_components(query):
            if normalize_text(component) != normalize_text(query):
                candidates.append(component)

        matches: List[Dict[str, Any]] = []
        for candidate in ordered_unique(candidates):
            normalized_candidate = normalize_text(candidate)

            exact = self._salt_exact.get(normalized_candidate)
            if exact:
                matches.append({"name": exact, "score": 1.0, "match_type": "exact"})

            tfidf_matches = self._top_tfidf_matches(
                query=candidate,
                choices=self.unique_salts,
                matrix=self.salt_matrix,
                threshold=float(self.salt_threshold or 0.70),
            )
            matches.extend(tfidf_matches)

            # Salt vocabulary is small, but use the same conservative fallback rule.
            if not exact and not tfidf_matches:
                matches.extend(
                    self._fuzzy_matches(
                        query=candidate,
                        choices=self.unique_salts,
                        normalized_choices=self._salt_choices_norm,
                        threshold=int(self.fuzzy_threshold or 92),
                    )
                )

        return self._merge_matches(matches)

    def resolve(self, query: str) -> Dict[str, Any]:
        """Resolve one input string to matching medicines and/or salts."""
        self._require_loaded()

        original_query = "" if query is None else str(query)
        clean_query = original_query.strip()

        if not clean_query:
            return {
                "input": original_query,
                "input_type": "empty",
                "output_salts": [],
                "matched_medicines": [],
                "matched_salts": [],
            }

        medicine_matches_raw = self._match_medicines(clean_query)
        salt_matches_raw = self._match_salts(clean_query)

        matched_medicines = []
        for match in medicine_matches_raw:
            medicine_name = str(match["name"])
            salts = ordered_unique(self.medicine_to_salts.get(medicine_name, []))
            matched_medicines.append(
                {
                    "medicine": medicine_name,
                    "score": match["score"],
                    "match_type": match["match_type"],
                    "salts": salts,
                }
            )

        matched_salts = [
            {
                "salt": str(match["name"]),
                "score": match["score"],
                "match_type": match["match_type"],
            }
            for match in salt_matches_raw
        ]

        # Keep output_salts conservative: if an exact match exists, do not add salts
        # from lower-confidence lookalike medicines/salts. This prevents examples like
        # "Allegra" from returning salts from "Allegra-M".
        primary_medicine_matches = [
            match for match in matched_medicines if match["match_type"] == "exact"
        ] or matched_medicines
        primary_salt_matches = [
            match for match in matched_salts if match["match_type"] == "exact"
        ] or matched_salts

        salts_from_medicines: List[str] = []
        for match in primary_medicine_matches:
            salts_from_medicines.extend(match["salts"])

        salts_from_salt_matches = [match["salt"] for match in primary_salt_matches]
        output_salts = ordered_unique([*salts_from_medicines, *salts_from_salt_matches])

        if matched_medicines and matched_salts:
            input_type = "medicine_and_salt"
        elif matched_medicines:
            input_type = "medicine"
        elif matched_salts:
            input_type = "salt"
        else:
            input_type = "unknown"

        return {
            "input": original_query,
            "input_type": input_type,
            "output_salts": output_salts,
            "matched_medicines": matched_medicines,
            "matched_salts": matched_salts,
        }

    def resolve_many(self, queries: Sequence[str]) -> List[Dict[str, Any]]:
        """Resolve many input strings."""
        self._require_loaded()
        return [self.resolve(query) for query in queries]


def ensure_index(resolver: SimpleMedicineSaltResolver) -> None:
    """Load the supplied inference index."""
    resolver.load_index()


def run_single_query(resolver: SimpleMedicineSaltResolver, query: str) -> Dict[str, Any]:
    """Run inference for one input string."""
    return resolver.resolve(query)


def run_csv_inference(
    resolver: SimpleMedicineSaltResolver,
    input_csv: str,
    output_csv: str,
    input_column: Optional[str] = None,
) -> None:
    """
    Run inference on a CSV file.

    If input_column is not provided, the first column is used.
    """
    df = pd.read_csv(input_csv)

    if df.empty:
        raise ValueError("Input CSV is empty.")

    selected_column = input_column or df.columns[0]

    if selected_column not in df.columns:
        raise ValueError(
            f"Input column '{selected_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    queries = df[selected_column].fillna("").astype(str).tolist()
    results = resolver.resolve_many(queries)

    prediction_df = pd.DataFrame(
        {
            "input": [r["input"] for r in results],
            "input_type": [r["input_type"] for r in results],
            "output_salts": ["; ".join(r["output_salts"]) for r in results],
            "output_salts_json": [
                json.dumps(r["output_salts"], ensure_ascii=False) for r in results
            ],
            "matched_medicines_json": [
                json.dumps(r["matched_medicines"], ensure_ascii=False)
                for r in results
            ],
            "matched_salts_json": [
                json.dumps(r["matched_salts"], ensure_ascii=False) for r in results
            ],
        }
    )

    final_df = pd.concat([df.reset_index(drop=True), prediction_df], axis=1)
    final_df.to_csv(output_csv, index=False)

    print(f"Saved predictions to: {output_csv}")
    print(f"Rows processed: {len(final_df):,}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Self-contained inference for the supplied SimpleMedicineSaltResolver "
            "simple TF-IDF index. Supports single-line and CSV batch inference."
        )
    )

    parser.add_argument(
        "--index-path",
        default=None,
        help=(
            "Path to the supplied saved index joblib. Defaults to "
            f"{INDEX_FILENAME} next to this script, or to a path derived from --knowledge-csv."
        ),
    )

    parser.add_argument(
        "--knowledge-csv",
        default=None,
        help=(
            "Optional backward-compatible argument. If --index-path is omitted, "
            "the script tries <knowledge_csv_stem>_simple_tfidf_index.joblib. "
            "The CSV itself is not needed for inference."
        ),
    )

    parser.add_argument(
        "--query",
        default=None,
        help="Single input query, such as a medicine name, salt, or both.",
    )

    parser.add_argument(
        "--input-csv",
        default=None,
        help="CSV file containing input queries for batch inference.",
    )

    parser.add_argument(
        "--input-column",
        default=None,
        help="Column name in --input-csv containing queries. Defaults to the first column.",
    )

    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path for batch inference results.",
    )

    parser.add_argument(
        "--medicine-threshold",
        type=float,
        default=None,
        help="Override TF-IDF cosine threshold for medicine match. Defaults to index value.",
    )

    parser.add_argument(
        "--salt-threshold",
        type=float,
        default=None,
        help="Override TF-IDF cosine threshold for salt match. Defaults to index value.",
    )

    parser.add_argument(
        "--fuzzy-threshold",
        type=int,
        default=None,
        help="Override RapidFuzz threshold from 0 to 100. Defaults to index value.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override number of top matches to keep. Defaults to index value.",
    )

    parser.add_argument(
        "--disable-fuzzy",
        action="store_true",
        help="Disable RapidFuzz matching and use only exact + TF-IDF matching.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.query is None and args.input_csv is None:
        parser.error("Provide either --query for single inference or --input-csv for batch inference.")

    if args.input_csv is not None and args.output_csv is None:
        parser.error("--output-csv is required when using --input-csv.")

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

    if args.query is not None:
        result = run_single_query(resolver, args.query)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.input_csv is not None:
        run_csv_inference(
            resolver=resolver,
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            input_column=args.input_column,
        )


if __name__ == "__main__":
    main()
