"""
04_transcribe.py — Transcribe cleaned audio clips using local OpenAI Whisper (free, no API key needed).

For each clip in data/cleaned/, runs whisper locally and saves:
  - data/transcripts/<clip_id>.json   (text + word timestamps)
  - data/transcription_log.csv        (summary per clip)

Models (downloaded automatically on first run):
  tiny   — fastest, least accurate  (~75MB)
  base   — good balance             (~145MB)
  small  — better accuracy          (~480MB)  ← default
  medium — high accuracy, slow      (~1.5GB)
  large  — best accuracy, very slow (~3GB)

Usage:
    python scripts/04_transcribe.py
    python scripts/04_transcribe.py --model base
    python scripts/04_transcribe.py --input data/cleaned/clip_0001.wav
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLEANED_DIR, TRANSCRIPTS_DIR, SAMPLE_RATE

try:
    import whisper
except ImportError:
    print("[ERROR] openai-whisper not installed. Run: python -m pip install openai-whisper")
    sys.exit(1)

try:
    import librosa
except ImportError:
    print("[ERROR] librosa not installed. Run: python -m pip install librosa")
    sys.exit(1)


def transcribe_clip(model, clip_path: Path) -> dict:
    """Transcribe a single clip and return a result dict."""
    # Load audio with librosa (avoids ffmpeg PATH dependency on Windows)
    audio, _ = librosa.load(str(clip_path), sr=16000, mono=True)
    audio = audio.astype(np.float32)

    result = model.transcribe(
        audio,
        language="en",
        word_timestamps=True,
        verbose=False,
        temperature=0.0,
        fp16=False,   # CPU-safe
    )

    text      = result.get("text", "").strip()
    segments  = result.get("segments", [])
    word_count = len(text.split()) if text else 0

    # Flatten word-level timestamps from segments
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            words.append({
                "word":  w.get("word", "").strip(),
                "start": round(w.get("start", 0), 3),
                "end":   round(w.get("end",   0), 3),
            })

    return {
        "clip_id":       clip_path.stem,
        "text":          text,
        "words":         words,
        "language":      result.get("language", "en"),
        "word_count":    word_count,
        "is_empty":      word_count == 0,
        "is_very_short": word_count <= 2,
    }


def main():
    parser = argparse.ArgumentParser(description="Transcribe clips using local Whisper (free).")
    parser.add_argument("--model",  type=str, default="small",
                        help="Whisper model size: tiny/base/small/medium/large (default: small)")
    parser.add_argument("--input",  type=str, default=None,
                        help="Single WAV clip to transcribe (default: all in data/cleaned/)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-transcribe clips that already have a transcript")
    args = parser.parse_args()

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model (downloads automatically on first run)
    print(f"Loading Whisper '{args.model}' model (downloads on first run)...")
    model = whisper.load_model(args.model)
    print(f"Model loaded.\n")

    if args.input:
        clip_files = [Path(args.input)]
    else:
        clip_files = sorted(CLEANED_DIR.glob("*.wav"))

    if not clip_files:
        print(f"[ERROR] No cleaned clips found in {CLEANED_DIR}")
        sys.exit(1)

    print(f"Found {len(clip_files)} clip(s) to transcribe.")

    log_path = CLEANED_DIR.parent / "transcription_log.csv"
    existing_ids: set = set()
    if log_path.exists() and not args.overwrite:
        existing_ids = set(pd.read_csv(log_path)["clip_id"].astype(str))
        print(f"Skipping {len(existing_ids)} already-transcribed clips.")

    log_rows = []
    errors   = []

    for i, clip_path in enumerate(clip_files, 1):
        if clip_path.stem in existing_ids:
            continue

        print(f"  [{i:4d}/{len(clip_files)}] {clip_path.name}", end="  ", flush=True)
        try:
            result = transcribe_clip(model, clip_path)

            # Save JSON
            out_path = TRANSCRIPTS_DIR / f"{result['clip_id']}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            flag = "[EMPTY]" if result["is_empty"] else ""
            preview = result["text"][:60] + ("…" if len(result["text"]) > 60 else "")
            print(f"✓  {flag}  \"{preview}\"")

            log_rows.append({
                "clip_id":       result["clip_id"],
                "text":          result["text"],
                "word_count":    result["word_count"],
                "language":      result["language"],
                "is_empty":      result["is_empty"],
                "is_very_short": result["is_very_short"],
            })

        except Exception as e:
            print(f"✗  ERROR: {e}")
            errors.append({"clip_id": clip_path.stem, "error": str(e)})

    if log_rows:
        new_df = pd.DataFrame(log_rows)
        if log_path.exists() and not args.overwrite:
            combined = pd.concat([pd.read_csv(log_path), new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(log_path, index=False)

    print(f"\n{'─'*50}")
    print(f"Transcription complete.")
    print(f"  Transcribed : {len(log_rows)}")
    print(f"  Errors      : {len(errors)}")
    if log_rows:
        df = pd.DataFrame(log_rows)
        print(f"  Empty clips : {df['is_empty'].sum()}")
        print(f"  Output      : {TRANSCRIPTS_DIR}")
        print(f"  Log         : {log_path}")


if __name__ == "__main__":
    main()
