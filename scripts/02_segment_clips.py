"""
02_segment_clips.py — VAD-based segmentation of full audio files into individual radio clips.

Reads WAV files from data/raw_audio/, detects speech regions using energy-based
Voice Activity Detection, and writes each segment to data/clips/.
A clip_manifest.csv is written to data/ with metadata for every clip.

Usage:
    python scripts/02_segment_clips.py
    python scripts/02_segment_clips.py --input data/raw_audio/my_file.wav
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_AUDIO_DIR, CLIPS_DIR, SAMPLE_RATE,
    VAD_FRAME_LENGTH, VAD_HOP_LENGTH, VAD_ENERGY_THRESH,
    VAD_GAP_TOLERANCE, VAD_MIN_DURATION, VAD_MAX_DURATION,
)


def detect_speech_regions(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """
    Return a list of (start_sec, end_sec) tuples for detected speech regions.

    Algorithm:
    1. Compute frame-level RMS energy.
    2. Threshold at VAD_ENERGY_THRESH * max_rms to produce a binary speech mask.
    3. Convert frame mask to time segments.
    4. Merge segments separated by gaps shorter than VAD_GAP_TOLERANCE.
    """
    # 1. Frame-level RMS
    rms = librosa.feature.rms(
        y=audio,
        frame_length=VAD_FRAME_LENGTH,
        hop_length=VAD_HOP_LENGTH,
    )[0]

    max_rms = rms.max()
    if max_rms == 0:
        return []  # silence-only file

    # 2. Binary speech mask
    threshold = VAD_ENERGY_THRESH * max_rms
    speech_mask = rms > threshold  # shape: (n_frames,)

    # 3. Convert frame indices to time segments
    frame_times = librosa.frames_to_time(
        np.arange(len(speech_mask)),
        sr=sr,
        hop_length=VAD_HOP_LENGTH,
    )

    segments = []
    in_speech = False
    seg_start = 0.0

    for i, is_speech in enumerate(speech_mask):
        if is_speech and not in_speech:
            seg_start = float(frame_times[i])
            in_speech = True
        elif not is_speech and in_speech:
            seg_end = float(frame_times[i])
            segments.append((seg_start, seg_end))
            in_speech = False

    if in_speech:
        segments.append((seg_start, float(frame_times[-1])))

    # 4. Merge segments with short gaps
    merged = []
    for seg in segments:
        if merged and (seg[0] - merged[-1][1]) <= VAD_GAP_TOLERANCE:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(list(seg))

    return [(s[0], s[1]) for s in merged]


def segment_file(wav_path: Path, clip_id_offset: int) -> list[dict]:
    """
    Segment a single WAV file and save clips to CLIPS_DIR.
    Returns a list of manifest row dicts.
    """
    print(f"\n  Processing: {wav_path.name}")

    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    total_duration = len(audio) / sr
    print(f"    Duration: {total_duration:.1f}s")

    regions = detect_speech_regions(audio, sr)
    print(f"    Speech regions detected: {len(regions)}")

    rows = []
    clip_idx = clip_id_offset

    for start_sec, end_sec in regions:
        duration = end_sec - start_sec

        if duration < VAD_MIN_DURATION:
            continue
        if duration > VAD_MAX_DURATION:
            # Split long segments into VAD_MAX_DURATION chunks
            sub_start = start_sec
            while sub_start < end_sec:
                sub_end = min(sub_start + VAD_MAX_DURATION, end_sec)
                if (sub_end - sub_start) >= VAD_MIN_DURATION:
                    _save_clip(audio, sr, sub_start, sub_end, clip_idx, wav_path, rows)
                    clip_idx += 1
                sub_start = sub_end
            continue

        _save_clip(audio, sr, start_sec, end_sec, clip_idx, wav_path, rows)
        clip_idx += 1

    print(f"    Clips saved: {len(rows)}")
    return rows


def _save_clip(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    clip_idx: int,
    source_path: Path,
    rows: list,
):
    start_sample = int(start_sec * sr)
    end_sample   = int(end_sec   * sr)
    clip_audio   = audio[start_sample:end_sample]

    clip_id   = f"clip_{clip_idx:04d}"
    clip_path = CLIPS_DIR / f"{clip_id}.wav"

    sf.write(clip_path, clip_audio, sr, subtype="PCM_16")

    rows.append({
        "clip_id":     clip_id,
        "source_file": source_path.name,
        "start_sec":   round(start_sec, 3),
        "end_sec":     round(end_sec,   3),
        "duration_sec": round(end_sec - start_sec, 3),
    })


def main():
    parser = argparse.ArgumentParser(description="Segment audio files into individual radio clips.")
    parser.add_argument("--input", type=str, default=None,
                        help="Single WAV file to segment (default: all files in data/raw_audio/)")
    args = parser.parse_args()

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    if args.input:
        wav_files = [Path(args.input)]
    else:
        wav_files = sorted(RAW_AUDIO_DIR.glob("*.wav"))

    if not wav_files:
        print(f"[ERROR] No WAV files found in {RAW_AUDIO_DIR}")
        print("  Run 01_extract_audio.py first.")
        sys.exit(1)

    print(f"Found {len(wav_files)} WAV file(s) to segment.")

    # Load existing manifest to continue clip numbering
    manifest_path = RAW_AUDIO_DIR.parent / "clip_manifest.csv"
    existing_clips = 0
    if manifest_path.exists():
        existing_df = pd.read_csv(manifest_path)
        existing_clips = len(existing_df)
        print(f"Existing manifest: {existing_clips} clips — new clips will continue from there.")

    all_rows = []
    clip_offset = existing_clips

    for wav_path in wav_files:
        rows = segment_file(wav_path, clip_id_offset=clip_offset)
        all_rows.extend(rows)
        clip_offset += len(rows)

    if not all_rows:
        print("\n[WARN] No clips were produced. Try lowering VAD_ENERGY_THRESH in config.py.")
        sys.exit(0)

    new_df = pd.DataFrame(all_rows)

    if manifest_path.exists():
        old_df = pd.read_csv(manifest_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(manifest_path, index=False)

    print(f"\n{'─'*50}")
    print(f"Segmentation complete.")
    print(f"  New clips this run : {len(all_rows)}")
    print(f"  Total clips        : {len(combined)}")
    print(f"  Clips directory    : {CLIPS_DIR}")
    print(f"  Manifest           : {manifest_path}")

    # Summary stats
    durations = new_df["duration_sec"]
    print(f"\n  Clip duration stats (new):")
    print(f"    Min    : {durations.min():.1f}s")
    print(f"    Max    : {durations.max():.1f}s")
    print(f"    Mean   : {durations.mean():.1f}s")
    print(f"    Median : {durations.median():.1f}s")


if __name__ == "__main__":
    main()
