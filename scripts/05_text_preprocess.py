"""
05_text_preprocess.py — NLP text preprocessing for transcribed radio clips.

Implements Section 6.2.4 of the methodology:
  - Lowercase + normalisation (strip ASR artefacts)
  - Tokenisation
  - Stop word tracking (not removed — frequency is a signal)
  - Lemmatisation

Reads JSON transcripts from data/transcripts/, enriches each with preprocessed
text fields, and writes a combined dataset to annotations/preprocessed_text.csv.

Usage:
    python scripts/05_text_preprocess.py
    python scripts/05_text_preprocess.py --input data/transcripts/clip_0001.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRANSCRIPTS_DIR, ANNOTATIONS_DIR

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
except ImportError:
    print("[ERROR] nltk not installed. Run: pip install nltk")
    sys.exit(1)


def ensure_nltk_data():
    """Download required NLTK corpora if not already present."""
    required = [
        ("tokenizers/punkt",          "punkt"),
        ("tokenizers/punkt_tab",      "punkt_tab"),
        ("corpora/wordnet",           "wordnet"),
        ("corpora/stopwords",         "stopwords"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    for path, pkg in required:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"  Downloading NLTK data: {pkg}")
            nltk.download(pkg, quiet=True)


# ── Cleaning helpers ──────────────────────────────────────────────────────────

# ASR artefacts: hesitation tokens, repeated punctuation, etc.
_ASR_ARTEFACTS = re.compile(
    r"\[.*?\]"              # bracketed annotations e.g. [inaudible]
    r"|\(.*?\)"             # parenthetical e.g. (sic)
    r"|<.*?>"               # XML-like tags
    r"|\.{2,}"              # ellipsis / multiple dots
    r"|[^a-z0-9\s',.?!-]", # non-speech characters (keep apostrophes for contractions)
    re.IGNORECASE,
)


def clean_text(raw: str) -> str:
    """Lowercase, strip ASR artefacts, collapse whitespace."""
    text = raw.lower()
    text = _ASR_ARTEFACTS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise(text: str) -> list[str]:
    """Tokenise using NLTK word tokeniser."""
    return word_tokenize(text)


def track_stopwords(tokens: list[str], stop_set: set[str]) -> dict[str, int]:
    """
    Return a frequency dict of stop words found in tokens.
    These are NOT removed — high frequency of 'no', 'why', 'what' is a signal.
    """
    return {t: tokens.count(t) for t in set(tokens) if t in stop_set}


def lemmatise(tokens: list[str], lemmatizer: WordNetLemmatizer) -> list[str]:
    """Lemmatise each token (verb form, then noun form fallback)."""
    return [lemmatizer.lemmatize(t, pos="v") for t in tokens]


def count_negations(tokens: list[str]) -> int:
    """Count negation words — elevated under frustration/stress."""
    negations = {"no", "not", "never", "neither", "nothing", "nobody", "none",
                 "nowhere", "nor", "cannot", "can't", "won't", "don't",
                 "doesn't", "didn't", "haven't", "hasn't", "hadn't"}
    return sum(1 for t in tokens if t in negations)


def count_questions(text: str) -> int:
    """Count question marks as a proxy for interrogative sentences."""
    return text.count("?")


def avg_word_length(tokens: list[str]) -> float:
    """Average word length (pure alpha tokens only)."""
    alpha = [t for t in tokens if t.isalpha()]
    if not alpha:
        return 0.0
    return sum(len(t) for t in alpha) / len(alpha)


# ── Per-transcript processing ─────────────────────────────────────────────────

def process_transcript(json_path: Path, stop_set: set, lemmatizer: WordNetLemmatizer) -> dict:
    """Load a transcript JSON and return an enriched dict."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    raw_text = data.get("text", "")
    if not raw_text.strip():
        return {
            "clip_id":          data.get("clip_id", json_path.stem),
            "raw_text":         "",
            "clean_text":       "",
            "tokens":           "[]",
            "lemmas":           "[]",
            "stopword_counts":  "{}",
            "word_count":       0,
            "negation_count":   0,
            "question_count":   0,
            "avg_word_length":  0.0,
            "is_empty":         True,
        }

    clean  = clean_text(raw_text)
    tokens = tokenise(clean)
    lemmas = lemmatise(tokens, lemmatizer)
    sw_counts = track_stopwords(tokens, stop_set)

    return {
        "clip_id":          data.get("clip_id", json_path.stem),
        "raw_text":         raw_text,
        "clean_text":       clean,
        "tokens":           json.dumps(tokens),
        "lemmas":           json.dumps(lemmas),
        "stopword_counts":  json.dumps(sw_counts),
        "word_count":       len([t for t in tokens if t.isalpha()]),
        "negation_count":   count_negations(tokens),
        "question_count":   count_questions(raw_text),
        "avg_word_length":  round(avg_word_length(tokens), 3),
        "is_empty":         False,
    }


def main():
    parser = argparse.ArgumentParser(description="NLP text preprocessing for transcripts.")
    parser.add_argument("--input", type=str, default=None,
                        help="Single transcript JSON to process (default: all in data/transcripts/)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process transcripts already in the output CSV")
    args = parser.parse_args()

    print("Checking NLTK data...")
    ensure_nltk_data()

    stop_set   = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = ANNOTATIONS_DIR / "preprocessed_text.csv"

    if args.input:
        json_files = [Path(args.input)]
    else:
        json_files = sorted(TRANSCRIPTS_DIR.glob("*.json"))

    if not json_files:
        print(f"[ERROR] No transcript JSON files found in {TRANSCRIPTS_DIR}")
        print("  Run 04_transcribe.py first.")
        sys.exit(1)

    print(f"Found {len(json_files)} transcript(s) to process.")

    # Skip already-processed clips
    existing_ids: set[str] = set()
    if out_csv.exists() and not args.overwrite:
        existing_df = pd.read_csv(out_csv)
        existing_ids = set(existing_df["clip_id"].astype(str))

    rows = []
    for i, jp in enumerate(json_files, 1):
        clip_id = jp.stem
        if clip_id in existing_ids:
            continue
        print(f"  [{i:4d}/{len(json_files)}] {jp.name}", end="  ", flush=True)
        try:
            row = process_transcript(jp, stop_set, lemmatizer)
            rows.append(row)
            marker = "[EMPTY]" if row["is_empty"] else f"{row['word_count']}w"
            print(f"✓  {marker}  \"{row['clean_text'][:55]}{'…' if len(row['clean_text']) > 55 else ''}\"")
        except Exception as e:
            print(f"✗  ERROR: {e}")

    if rows:
        new_df = pd.DataFrame(rows)
        if out_csv.exists() and not args.overwrite:
            old_df = pd.read_csv(out_csv)
            combined = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(out_csv, index=False)

        print(f"\n{'─'*50}")
        print(f"Text preprocessing complete.")
        print(f"  Processed   : {len(rows)}")
        empty = sum(1 for r in rows if r["is_empty"])
        print(f"  Empty clips : {empty}")
        wc = [r["word_count"] for r in rows if not r["is_empty"]]
        if wc:
            print(f"  Word count  : min={min(wc)}, max={max(wc)}, mean={sum(wc)/len(wc):.1f}")
        print(f"  Output      : {out_csv}")
    else:
        print("\nNo new transcripts to process.")


if __name__ == "__main__":
    main()
