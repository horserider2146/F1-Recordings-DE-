"""
06_auto_label.py — Automated emotion pre-labelling combining acoustic + text signals.

Target categories (Section 6.1.3):
  Calm | Frustrated | High Stress | Urgent

Strategy:
  Text channel  — HuggingFace emotion classifier (j-hartmann/emotion-english-distilroberta-base)
                  + keyword lexicon matching
  Acoustic channel — pitch, energy, speech rate extracted with librosa
  Fusion         — weighted vote; clips with low confidence are flagged for manual review

Input:
  annotations/preprocessed_text.csv   (from script 05)
  data/cleaned/*.wav                   (for acoustic features)

Output:
  annotations/labels.csv   (clip_id, text_label, acoustic_label, final_label,
                             confidence, flagged_for_review)

Usage:
    python scripts/06_auto_label.py
    python scripts/06_auto_label.py --clip-id clip_0001
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CLEANED_DIR, ANNOTATIONS_DIR, SAMPLE_RATE,
    EMOTION_MODEL, LABEL_CONFIDENCE_MIN,
    URGENCY_KEYWORDS, FRUSTRATION_MARKERS,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ── Emotion model mapping ─────────────────────────────────────────────────────
# j-hartmann model outputs: anger, disgust, fear, joy, neutral, sadness, surprise
_HARTMANN_TO_PROJECT = {
    "anger":    "Frustrated",
    "disgust":  "Frustrated",
    "fear":     "High Stress",
    "sadness":  "High Stress",
    "surprise": "Urgent",        # will be refined by acoustic energy
    "joy":      "Calm",
    "neutral":  "Calm",
}

# Weight for text vs acoustic channel in final fusion (must sum to 1.0)
TEXT_WEIGHT     = 0.60
ACOUSTIC_WEIGHT = 0.40


# ── Text-based labelling ──────────────────────────────────────────────────────

def load_emotion_classifier():
    """Load the HuggingFace emotion pipeline (downloads model on first run)."""
    try:
        from transformers import pipeline
        print(f"  Loading emotion model: {EMOTION_MODEL}")
        clf = pipeline(
            "text-classification",
            model=EMOTION_MODEL,
            top_k=None,         # return scores for all classes
            device=-1,          # CPU; change to 0 for CUDA GPU
        )
        return clf
    except Exception as e:
        print(f"  [WARN] Could not load emotion classifier: {e}")
        return None


def text_label(clf, text: str, clean_text: str) -> tuple[str, float]:
    """
    Classify text emotion and return (label, confidence).
    Falls back to keyword matching if classifier is unavailable or text is empty.
    """
    if not text or not text.strip():
        return "Calm", 0.5

    # ── Keyword override (high precision rules) ────────────────────────────
    lower_text = clean_text.lower()

    urgency_score = sum(1 for kw in URGENCY_KEYWORDS if kw in lower_text)
    frustration_score = sum(1 for m in FRUSTRATION_MARKERS if m in lower_text)

    if urgency_score >= 1:
        base_conf = min(0.5 + 0.15 * urgency_score, 0.95)
        if clf is not None:
            pass  # let model boost confidence below
        else:
            return "Urgent", base_conf

    if frustration_score >= 2:
        base_conf = min(0.5 + 0.10 * frustration_score, 0.90)
        if clf is None:
            return "Frustrated", base_conf

    # ── Model-based classification ─────────────────────────────────────────
    if clf is not None:
        try:
            results = clf(text[:512])  # Whisper outputs ≤ 512 tokens easily
            # results is a list of lists: [[{label, score}, ...]]
            scores_list = results[0] if results else []
            scores = {r["label"].lower(): r["score"] for r in scores_list}

            top_label = max(scores, key=scores.get)
            top_score = scores[top_label]
            project_label = _HARTMANN_TO_PROJECT.get(top_label, "Calm")

            # Keyword overrides can bump the label
            if urgency_score >= 1 and top_score < 0.70:
                project_label = "Urgent"
                top_score = max(top_score, base_conf if frustration_score == 0 else 0.65)
            elif frustration_score >= 2 and project_label == "Calm":
                project_label = "Frustrated"
                top_score = max(top_score, 0.65)

            return project_label, float(top_score)
        except Exception as e:
            print(f"    [WARN] Model inference failed: {e} — falling back to keywords")

    # ── Pure keyword fallback ──────────────────────────────────────────────
    if urgency_score >= 1:
        return "Urgent", min(0.5 + 0.15 * urgency_score, 0.85)
    if frustration_score >= 1:
        return "Frustrated", min(0.45 + 0.12 * frustration_score, 0.80)
    return "Calm", 0.55


# ── Acoustic-based labelling ──────────────────────────────────────────────────

def acoustic_label(clip_path: Path) -> tuple[str, float, dict]:
    """
    Extract pitch, energy, and speech rate; map to a project label.
    Returns (label, confidence, feature_dict).
    """
    try:
        audio, sr = librosa.load(clip_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        return "Calm", 0.4, {"error": str(e)}

    duration = len(audio) / sr
    if duration < 0.5:
        return "Calm", 0.4, {"duration": duration}

    # Pitch (F0) — use pyin for robustness
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None and f0 is not None else np.array([])
        mean_pitch  = float(np.mean(voiced_f0))  if len(voiced_f0) > 0 else 0.0
        pitch_range = float(np.ptp(voiced_f0))   if len(voiced_f0) > 1 else 0.0
        pitch_std   = float(np.std(voiced_f0))   if len(voiced_f0) > 1 else 0.0
    except Exception:
        mean_pitch = pitch_range = pitch_std = 0.0

    # RMS energy
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    mean_energy = float(np.mean(rms))
    energy_std  = float(np.std(rms))

    # Approximate speech rate: zero-crossing rate as a voice activity proxy
    # then count syllable-like energy peaks
    zcr   = librosa.feature.zero_crossing_rate(audio, hop_length=512)[0]
    mean_zcr = float(np.mean(zcr))

    features = {
        "mean_pitch":   round(mean_pitch,   2),
        "pitch_range":  round(pitch_range,  2),
        "pitch_std":    round(pitch_std,    2),
        "mean_energy":  round(mean_energy,  5),
        "energy_std":   round(energy_std,   5),
        "mean_zcr":     round(mean_zcr,     4),
        "duration":     round(duration,     3),
    }

    # ── Decision heuristics ────────────────────────────────────────────────
    # These thresholds are approximate — they will be refined once real data
    # is collected. They reflect published clinical norms for stress speech.

    high_energy  = mean_energy > 0.05
    high_pitch   = mean_pitch  > 200.0    # Hz — elevated above typical male F0
    high_range   = pitch_range > 150.0
    high_zcr     = mean_zcr    > 0.15     # lots of high-freq content / fast speech

    # Urgent: high energy + high pitch + fast rate
    if high_energy and high_pitch and high_zcr:
        return "Urgent", 0.75, features

    # High Stress: high energy + high pitch range (strained voice) but not ultra-fast
    if high_energy and high_range:
        return "High Stress", 0.70, features

    # Frustrated: moderate energy, varied pitch, no urgency markers
    if high_pitch and not high_energy:
        return "Frustrated", 0.60, features

    # Calm: low energy, low pitch variation
    return "Calm", 0.65, features


# ── Fusion ────────────────────────────────────────────────────────────────────

_LABEL_INDEX = {"Calm": 0, "Frustrated": 1, "High Stress": 2, "Urgent": 3}
_INDEX_LABEL = {v: k for k, v in _LABEL_INDEX.items()}


def fuse_labels(
    text_lbl: str, text_conf: float,
    acoustic_lbl: str, acoustic_conf: float,
) -> tuple[str, float]:
    """
    Weighted vote: combine text and acoustic predictions into one final label.
    """
    n_classes = len(_LABEL_INDEX)
    text_vec     = np.zeros(n_classes)
    acoustic_vec = np.zeros(n_classes)

    text_vec[_LABEL_INDEX[text_lbl]]         = text_conf
    acoustic_vec[_LABEL_INDEX[acoustic_lbl]] = acoustic_conf

    combined = TEXT_WEIGHT * text_vec + ACOUSTIC_WEIGHT * acoustic_vec
    final_idx = int(np.argmax(combined))
    final_conf = float(combined[final_idx])

    return _INDEX_LABEL[final_idx], round(final_conf, 4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auto-label clips with emotional state.")
    parser.add_argument("--clip-id", type=str, default=None,
                        help="Process a single clip by ID (e.g. clip_0001)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-label clips already present in labels.csv")
    args = parser.parse_args()

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Load preprocessed text
    text_csv = ANNOTATIONS_DIR / "preprocessed_text.csv"
    if not text_csv.exists():
        print(f"[ERROR] {text_csv} not found. Run 05_text_preprocess.py first.")
        sys.exit(1)

    text_df = pd.read_csv(text_csv)
    if args.clip_id:
        text_df = text_df[text_df["clip_id"] == args.clip_id]
        if text_df.empty:
            print(f"[ERROR] clip_id '{args.clip_id}' not found in {text_csv}")
            sys.exit(1)

    labels_csv = ANNOTATIONS_DIR / "labels.csv"
    existing_ids: set[str] = set()
    if labels_csv.exists() and not args.overwrite:
        existing_df = pd.read_csv(labels_csv)
        existing_ids = set(existing_df["clip_id"].astype(str))

    # Load emotion classifier once
    clf = load_emotion_classifier()

    rows = []
    total = len(text_df)

    for i, row in enumerate(text_df.itertuples(), 1):
        clip_id = str(row.clip_id)
        if clip_id in existing_ids:
            continue

        print(f"  [{i:4d}/{total}] {clip_id}", end="  ", flush=True)

        raw_text   = str(row.raw_text)   if not pd.isna(row.raw_text)   else ""
        clean_text_val = str(row.clean_text) if not pd.isna(row.clean_text) else ""

        # Text channel
        t_label, t_conf = text_label(clf, raw_text, clean_text_val)

        # Acoustic channel
        clip_path = CLEANED_DIR / f"{clip_id}.wav"
        if clip_path.exists():
            a_label, a_conf, features = acoustic_label(clip_path)
        else:
            a_label, a_conf, features = "Calm", 0.4, {}

        # Fuse
        final_label, final_conf = fuse_labels(t_label, t_conf, a_label, a_conf)
        flagged = final_conf < LABEL_CONFIDENCE_MIN

        print(f"text={t_label}({t_conf:.2f})  acou={a_label}({a_conf:.2f})  "
              f"→ {final_label}({final_conf:.2f})"
              + ("  [REVIEW]" if flagged else ""))

        rows.append({
            "clip_id":            clip_id,
            "text_label":         t_label,
            "text_confidence":    round(t_conf, 4),
            "acoustic_label":     a_label,
            "acoustic_confidence": round(a_conf, 4),
            "final_label":        final_label,
            "confidence":         final_conf,
            "flagged_for_review": flagged,
            "acoustic_features":  json.dumps(features),
        })

    if rows:
        new_df = pd.DataFrame(rows)
        if labels_csv.exists() and not args.overwrite:
            old_df = pd.read_csv(labels_csv)
            combined = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(labels_csv, index=False)

        print(f"\n{'─'*50}")
        print(f"Auto-labelling complete.")
        print(f"  Labelled this run : {len(rows)}")

        dist = new_df["final_label"].value_counts()
        print(f"\n  Label distribution (new clips):")
        for lbl, cnt in dist.items():
            pct = cnt / len(new_df) * 100
            print(f"    {lbl:<14}: {cnt:4d}  ({pct:.1f}%)")

        flagged = new_df["flagged_for_review"].sum()
        print(f"\n  Flagged for review : {flagged} clips (confidence < {LABEL_CONFIDENCE_MIN})")
        print(f"  Output             : {labels_csv}")
    else:
        print("\nNo new clips to label.")


if __name__ == "__main__":
    main()
