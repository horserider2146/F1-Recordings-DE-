"""
post_pipeline.py — Waits for auto-labelling to finish, then runs:
  07_clean_dataset -> build_splits -> train_focal

Run this as a background process immediately after run_new_batch.py starts.
It polls labels.csv until it stops growing, then proceeds autonomously.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import ANNOTATIONS_DIR

LABELS_CSV    = ANNOTATIONS_DIR / "labels.csv"
POLL_INTERVAL = 30   # seconds between checks
STABLE_AFTER  = 3    # consecutive unchanged checks = done


def run(cmd, label):
    print(f"\n{'='*60}", flush=True)
    print(f"  STEP: {label}", flush=True)
    print(f"{'='*60}\n", flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n[ERROR] {label} failed (exit {result.returncode}). Stopping.", flush=True)
        sys.exit(result.returncode)


def wait_for_labels():
    print("Waiting for 06_auto_label to finish (polling labels.csv)...", flush=True)
    last_count  = -1
    stable_ticks = 0

    while True:
        if LABELS_CSV.exists():
            with open(LABELS_CSV, encoding="utf-8") as f:
                count = sum(1 for _ in f) - 1  # subtract header
            print(f"  labels.csv rows: {count}", flush=True)

            if count == last_count:
                stable_ticks += 1
                print(f"  Stable tick {stable_ticks}/{STABLE_AFTER}", flush=True)
                if stable_ticks >= STABLE_AFTER:
                    print(f"  labels.csv stable at {count} rows — proceeding.", flush=True)
                    return
            else:
                stable_ticks = 0
            last_count = count
        else:
            print("  labels.csv not found yet...", flush=True)

        time.sleep(POLL_INTERVAL)


def main():
    print("post_pipeline.py started.", flush=True)
    print(f"Will poll every {POLL_INTERVAL}s until labels.csv stabilises.\n", flush=True)

    wait_for_labels()

    run([sys.executable, "scripts/07_clean_dataset.py"],
        "07 — Clean dataset")

    run([sys.executable, "scripts/build_splits.py"],
        "Build train/val/test splits")

    run([sys.executable, "-u", "train_focal.py"],
        "Retrain (focal loss)")

    print("\n" + "="*60, flush=True)
    print("  All done! Check models/best_model_focal.pt for results.", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()
