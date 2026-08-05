"""
run_new_batch.py — Safely run the full pipeline on new URLs only.

Handles the fact that script 02 has no skip logic by:
  1. Snapshotting existing raw_audio files before script 01
  2. Passing only the newly created WAVs to script 02

Usage:
    python scripts/run_new_batch.py --urls high_octane_urls.txt
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import RAW_AUDIO_DIR


def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n[ERROR] {label} failed with exit code {result.returncode}. Stopping.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", required=True, help="URL file with new videos")
    args = parser.parse_args()

    url_file = Path(args.urls)
    if not url_file.exists():
        print(f"[ERROR] URL file not found: {url_file}")
        sys.exit(1)

    # Snapshot existing WAV files before download
    existing_wavs = set(RAW_AUDIO_DIR.glob("*.wav")) if RAW_AUDIO_DIR.exists() else set()
    print(f"Existing WAV files in raw_audio: {len(existing_wavs)}")

    # Step 01 — Download and extract audio for new URLs only
    run([sys.executable, "scripts/01_extract_audio.py", "--urls", str(url_file)],
        "01 — Extract audio from new URLs")

    # Identify newly created WAV files
    new_wavs = sorted(set(RAW_AUDIO_DIR.glob("*.wav")) - existing_wavs)
    print(f"\nNew WAV files created: {len(new_wavs)}")
    for w in new_wavs:
        print(f"  {w.name}")

    if not new_wavs:
        print("\n[WARN] No new WAV files were created. Check if downloads succeeded.")
        sys.exit(0)

    # Step 02 — Segment only the new WAV files
    for wav in new_wavs:
        run([sys.executable, "scripts/02_segment_clips.py", "--input", str(wav)],
            f"02 — Segment: {wav.name}")

    # Steps 03-06 — all have skip logic, run on full dataset safely
    run([sys.executable, "scripts/03_preprocess_audio.py"],
        "03 — Preprocess audio (skips existing)")

    run([sys.executable, "scripts/04_transcribe.py"],
        "04 — Transcribe (skips existing)")

    run([sys.executable, "scripts/05_text_preprocess.py"],
        "05 — Text preprocess (skips existing)")

    run([sys.executable, "scripts/06_auto_label.py"],
        "06 — Auto-label (skips existing)")

    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"{'='*60}")
    print("\nNext step: run scripts/07_clean_dataset.py when ready to clean,")
    print("then scripts/build_splits.py to rebuild train/val/test splits.")


if __name__ == "__main__":
    main()
