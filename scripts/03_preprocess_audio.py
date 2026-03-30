"""
03_preprocess_audio.py — Noise reduction + signal cleaning for raw radio clips.

Applies the four-stage cleaning pipeline from Section 6.2.2 of the methodology:
  1. Spectral Subtraction (via noisereduce)
  2. Bandpass Filtering     (80 Hz – 8000 Hz, scipy Butterworth)
  3. VAD Trim               (re-trim silence after denoising)
  4. Amplitude Normalisation (target –20 dBFS)

Input : data/clips/
Output: data/cleaned/

Usage:
    python scripts/03_preprocess_audio.py
    python scripts/03_preprocess_audio.py --input data/clips/clip_0001.wav
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CLIPS_DIR, CLEANED_DIR, SAMPLE_RATE,
    BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ,
    TARGET_DBFS, NOISE_PROFILE_SEC,
)


# ── Stage 1: Spectral Subtraction ─────────────────────────────────────────────

def spectral_subtraction(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Estimate noise from the first NOISE_PROFILE_SEC of the clip and subtract
    its spectral profile from the full signal using noisereduce.
    """
    noise_samples = int(NOISE_PROFILE_SEC * sr)
    if len(audio) <= noise_samples:
        # Clip too short for a noise profile — use the whole clip as reference
        noise_clip = audio
    else:
        noise_clip = audio[:noise_samples]

    reduced = nr.reduce_noise(
        y=audio,
        y_noise=noise_clip,
        sr=sr,
        stationary=False,   # non-stationary mode handles changing noise (engine rev)
        prop_decrease=0.85, # aggressiveness: 0=none, 1=full suppression
    )
    return reduced.astype(np.float32)


# ── Stage 2: Bandpass Filter ──────────────────────────────────────────────────

def bandpass_filter(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply a 4th-order Butterworth bandpass filter from BANDPASS_LOW_HZ to
    BANDPASS_HIGH_HZ to isolate the human speech frequency range.
    """
    nyquist = sr / 2.0
    low  = BANDPASS_LOW_HZ  / nyquist
    high = BANDPASS_HIGH_HZ / nyquist

    # Clamp to valid range (0, 1) exclusive
    low  = max(1e-4, min(low,  0.9999))
    high = max(1e-4, min(high, 0.9999))

    sos = butter(4, [low, high], btype="band", output="sos")
    filtered = sosfilt(sos, audio)
    return filtered.astype(np.float32)


# ── Stage 3: VAD Trim ─────────────────────────────────────────────────────────

def vad_trim(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Re-trim leading and trailing silence after denoising using librosa.
    A top_db of 30 means anything 30 dB below the max is treated as silence.
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=30, frame_length=2048, hop_length=512)
    if len(trimmed) == 0:
        return audio  # fallback — do not return empty array
    return trimmed


# ── Stage 4: Amplitude Normalisation ─────────────────────────────────────────

def normalise_amplitude(audio: np.ndarray) -> np.ndarray:
    """
    Normalise RMS loudness to TARGET_DBFS (e.g. –20 dBFS).
    Avoids clipping by capping at ±1.0 after normalisation.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio  # silence — nothing to normalise

    target_rms = 10 ** (TARGET_DBFS / 20.0)
    gain = target_rms / rms
    normalised = audio * gain

    # Prevent clipping
    peak = np.abs(normalised).max()
    if peak > 1.0:
        normalised = normalised / peak

    return normalised.astype(np.float32)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def preprocess_clip(clip_path: Path, output_path: Path) -> dict:
    """
    Run the full 4-stage cleaning pipeline on a single WAV clip.
    Returns a dict with diagnostic information.
    """
    audio, sr = librosa.load(clip_path, sr=SAMPLE_RATE, mono=True)
    original_rms = float(np.sqrt(np.mean(audio ** 2)))

    # Stage 1
    audio = spectral_subtraction(audio, sr)

    # Stage 2
    audio = bandpass_filter(audio, sr)

    # Stage 3
    audio = vad_trim(audio, sr)

    # Stage 4
    audio = normalise_amplitude(audio)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sr, subtype="PCM_16")

    final_rms     = float(np.sqrt(np.mean(audio ** 2)))
    final_duration = len(audio) / sr

    return {
        "clip_id":          clip_path.stem,
        "input_rms":        round(original_rms, 5),
        "output_rms":       round(final_rms,     5),
        "output_duration":  round(final_duration, 3),
        "status":           "ok",
    }


def main():
    parser = argparse.ArgumentParser(description="Apply noise reduction and cleaning to audio clips.")
    parser.add_argument("--input", type=str, default=None,
                        help="Single WAV clip to process (default: all clips in data/clips/)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process clips that already have a cleaned version")
    args = parser.parse_args()

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    if args.input:
        clip_files = [Path(args.input)]
    else:
        clip_files = sorted(CLIPS_DIR.glob("*.wav"))

    if not clip_files:
        print(f"[ERROR] No WAV clips found in {CLIPS_DIR}")
        print("  Run 02_segment_clips.py first.")
        sys.exit(1)

    print(f"Found {len(clip_files)} clip(s) to process.")

    results = []
    skipped = 0

    for i, clip_path in enumerate(clip_files, 1):
        out_path = CLEANED_DIR / clip_path.name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        print(f"  [{i:4d}/{len(clip_files)}] {clip_path.name}", end="  ", flush=True)
        try:
            info = preprocess_clip(clip_path, out_path)
            results.append(info)
            print(f"✓  {info['output_duration']:.1f}s")
        except Exception as e:
            print(f"✗  ERROR: {e}")
            results.append({
                "clip_id":         clip_path.stem,
                "input_rms":       None,
                "output_rms":      None,
                "output_duration": None,
                "status":          f"error: {e}",
            })

    print(f"\n{'─'*50}")
    print(f"Preprocessing complete.")
    print(f"  Processed : {len(results)}")
    print(f"  Skipped   : {skipped} (already exist — use --overwrite to redo)")
    errors = [r for r in results if r["status"] != "ok"]
    if errors:
        print(f"  Errors    : {len(errors)}")
        for e in errors:
            print(f"    {e['clip_id']}: {e['status']}")
    print(f"  Output    : {CLEANED_DIR}")


if __name__ == "__main__":
    main()
