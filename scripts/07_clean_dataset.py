"""
07_clean_dataset.py — Remove garbage clips from the dataset.

Removes clips that are not useful for modelling:
  - Empty transcripts
  - Too few words (< MIN_WORDS)
  - Whisper hallucinations (repeated characters/words)
  - YouTube noise (outros, music labels, subscribe prompts)

Does NOT remove clips just because they are flagged for review —
those are real radio clips that need human annotation, not garbage.

Input:
  annotations/labels.csv
  annotations/preprocessed_text.csv
  data/clips/*.wav
  data/cleaned/*.wav
  data/transcripts/*.json

Output:
  annotations/labels_clean.csv
  annotations/preprocessed_text_clean.csv
  annotations/cleaning_report.csv
  (optionally deletes source WAV/JSON files with --delete-files)

Usage:
    python scripts/07_clean_dataset.py                 # dry run — shows what would be removed
    python scripts/07_clean_dataset.py --apply          # removes from CSVs, keeps WAV files
    python scripts/07_clean_dataset.py --apply --delete-files  # removes from CSVs + deletes WAVs
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANNOTATIONS_DIR, CLIPS_DIR, CLEANED_DIR, TRANSCRIPTS_DIR


# ── Cleaning rules ───────────────────────────────────────────────────────────

MIN_WORDS = 4  # clips with fewer words are removed

# YouTube outro / non-radio patterns (matched against lowercased clean_text)
YOUTUBE_NOISE = [
    "thank you for watching",
    "thanks for watching",
    "subscribe to the channel",
    "subscribe to our channel",
    "like and subscribe",
    "the end",
    "the end.",
    "music",
]

# Whisper hallucination: single filler words that appear as entire transcript
FILLER_ONLY = {"i", "the", "a", "and", "is", "oh", "um", "uh", "hmm", "so", "it"}


def classify_garbage(clip_id: str, raw_text: str, clean_text: str, word_count: int) -> str | None:
    """
    Return a reason string if the clip is garbage, or None if it should be kept.
    """
    # 1. Empty transcript
    if word_count == 0 or not clean_text or not clean_text.strip():
        return "empty_transcript"

    lower = clean_text.strip().lower()

    # 2. Single filler word
    if lower in FILLER_ONLY:
        return f"filler_only: '{lower}'"

    # 3. YouTube noise — exact or near-exact match
    for pattern in YOUTUBE_NOISE:
        if lower == pattern or lower.rstrip(".!") == pattern:
            return f"youtube_noise: '{lower}'"

    # 4. Too few words (after checking it's not a known pattern above)
    if word_count < MIN_WORDS:
        # But keep short genuine radio calls (e.g. "box box box")
        # by only removing if it doesn't contain F1 radio keywords
        radio_keywords = ["box", "copy", "p1", "p2", "p3", "p4", "p5", "mode",
                          "safety", "pit", "push", "tyre", "tire", "gap", "delta"]
        has_radio_word = any(kw in lower for kw in radio_keywords)
        if not has_radio_word:
            return f"too_short: {word_count} words"

    # 5. Repeated character hallucination (e.g. "aaaaaaa...")
    # Check if >60% of the text is the same character repeated
    alpha_only = re.sub(r"[^a-z]", "", lower)
    if len(alpha_only) > 5:
        most_common = max(set(alpha_only), key=alpha_only.count)
        ratio = alpha_only.count(most_common) / len(alpha_only)
        if ratio > 0.6:
            return f"repeated_char: '{most_common}' ({ratio:.0%})"

    # 6. Repeated word hallucination (e.g. "the the the the the")
    words = lower.split()
    if len(words) >= 3:
        unique_words = set(words)
        if len(unique_words) <= 2 and len(words) >= 3:
            return f"repeated_words: '{' '.join(unique_words)}' x{len(words)}"

    # 7. Whisper silence hallucination patterns
    hallucination_patterns = [
        r"^i'm not sure",
        r"^i'm not gonna",
        r"^(my,?\s*){4,}",  # "my, my, my, my..."
    ]
    for pat in hallucination_patterns:
        if re.match(pat, lower):
            # Only flag if it's very repetitive (>50% repeated phrases)
            # Check for high repetition
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            if bigrams:
                most_common_bg = max(set(bigrams), key=bigrams.count)
                bg_ratio = bigrams.count(most_common_bg) / len(bigrams)
                if bg_ratio > 0.4:
                    return f"whisper_hallucination: repeating '{most_common_bg}'"

    return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clean garbage clips from the dataset.")
    parser.add_argument("--apply", action="store_true",
                        help="Actually remove clips. Without this flag, just prints a report.")
    parser.add_argument("--delete-files", action="store_true",
                        help="Also delete WAV and JSON files (requires --apply)")
    args = parser.parse_args()

    # Load data
    text_csv = ANNOTATIONS_DIR / "preprocessed_text.csv"
    labels_csv = ANNOTATIONS_DIR / "labels.csv"

    if not text_csv.exists():
        print(f"[ERROR] {text_csv} not found. Run 05_text_preprocess.py first.")
        sys.exit(1)

    text_df = pd.read_csv(text_csv)
    labels_df = pd.read_csv(labels_csv) if labels_csv.exists() else None

    # Classify each clip
    garbage = []
    keep = []

    for row in text_df.itertuples():
        clip_id = str(row.clip_id)
        raw_text = str(row.raw_text) if not pd.isna(row.raw_text) else ""
        clean_text = str(row.clean_text) if not pd.isna(row.clean_text) else ""
        word_count = int(row.word_count) if not pd.isna(row.word_count) else 0

        reason = classify_garbage(clip_id, raw_text, clean_text, word_count)
        if reason:
            garbage.append({"clip_id": clip_id, "reason": reason, "text": clean_text[:80]})
        else:
            keep.append(clip_id)

    # Report
    print(f"Dataset cleaning analysis")
    print(f"{'─' * 50}")
    print(f"  Total clips     : {len(text_df)}")
    print(f"  Garbage clips   : {len(garbage)}")
    print(f"  Clean clips     : {len(keep)}")
    print()

    if garbage:
        # Group by reason type
        reason_types = {}
        for g in garbage:
            rtype = g["reason"].split(":")[0]
            reason_types[rtype] = reason_types.get(rtype, 0) + 1

        print(f"  Removal reasons:")
        for rtype, count in sorted(reason_types.items(), key=lambda x: -x[1]):
            print(f"    {rtype:<25}: {count}")
        print()

        print(f"  Clips to remove:")
        for g in garbage:
            print(f"    {g['clip_id']:<15} {g['reason']:<40} \"{g['text']}\"")
        print()

    if not args.apply:
        print(f"  ** DRY RUN — no changes made. Use --apply to remove garbage clips. **")
        # Save report anyway
        report_df = pd.DataFrame(garbage)
        report_path = ANNOTATIONS_DIR / "cleaning_report.csv"
        report_df.to_csv(report_path, index=False)
        print(f"  Report saved to: {report_path}")
        return

    # ── Apply cleaning ────────────────────────────────────────────────────
    garbage_ids = {g["clip_id"] for g in garbage}

    # Clean text CSV
    text_clean = text_df[~text_df["clip_id"].isin(garbage_ids)]
    text_clean_path = ANNOTATIONS_DIR / "preprocessed_text_clean.csv"
    text_clean.to_csv(text_clean_path, index=False)
    print(f"  Saved clean text     : {text_clean_path} ({len(text_clean)} clips)")

    # Clean labels CSV
    if labels_df is not None:
        labels_clean = labels_df[~labels_df["clip_id"].isin(garbage_ids)]
        labels_clean_path = ANNOTATIONS_DIR / "labels_clean.csv"
        labels_clean.to_csv(labels_clean_path, index=False)
        print(f"  Saved clean labels   : {labels_clean_path} ({len(labels_clean)} clips)")

        # Show clean label distribution
        dist = labels_clean["final_label"].value_counts()
        print(f"\n  Clean label distribution:")
        for lbl, cnt in dist.items():
            pct = cnt / len(labels_clean) * 100
            print(f"    {lbl:<14}: {cnt:4d}  ({pct:.1f}%)")

        flagged = labels_clean["flagged_for_review"].sum()
        print(f"\n  Still flagged for review: {flagged} / {len(labels_clean)} clips")

    # Delete files
    if args.delete_files:
        deleted = 0
        for g in garbage:
            cid = g["clip_id"]
            for directory in [CLIPS_DIR, CLEANED_DIR]:
                wav = directory / f"{cid}.wav"
                if wav.exists():
                    wav.unlink()
                    deleted += 1
            json_file = TRANSCRIPTS_DIR / f"{cid}.json"
            if json_file.exists():
                json_file.unlink()
                deleted += 1
        print(f"\n  Deleted {deleted} files from disk.")

    # Save report
    report_df = pd.DataFrame(garbage)
    report_path = ANNOTATIONS_DIR / "cleaning_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"\n  Cleaning report: {report_path}")
    print(f"\nDone. Use *_clean.csv files for downstream modelling.")


if __name__ == "__main__":
    main()
