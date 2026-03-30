"""
config.py — Central configuration for the F1 radio preprocessing pipeline.

Set your OpenAI API key either here or via the environment variable OPENAI_API_KEY.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
RAW_AUDIO_DIR   = DATA_DIR / "raw_audio"
CLIPS_DIR       = DATA_DIR / "clips"
CLEANED_DIR     = DATA_DIR / "cleaned"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
ANNOTATIONS_DIR = BASE_DIR / "annotations"

# ── OpenAI / Whisper ───────────────────────────────────────────────────────────
# Leave as None to read from the OPENAI_API_KEY environment variable instead.
OPENAI_API_KEY = "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"  # e.g. "sk-..."

def get_openai_key() -> str:
    key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError(
            "OpenAI API key not found. Set OPENAI_API_KEY in config.py "
            "or as an environment variable."
        )
    return key

# ── Audio extraction ───────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000      # Hz — industry standard for speech processing
CHANNELS    = 1           # mono

# ── VAD segmentation ──────────────────────────────────────────────────────────
VAD_FRAME_LENGTH  = 2048  # samples per RMS frame
VAD_HOP_LENGTH    = 512   # hop between frames
VAD_ENERGY_THRESH = 0.05  # fraction of max RMS — below = silence (raised for F1 radio)
VAD_GAP_TOLERANCE = 0.8   # seconds — gaps shorter than this are bridged
VAD_MIN_DURATION  = 2.0   # seconds — discard clips shorter than this
VAD_MAX_DURATION  = 20.0  # seconds — discard clips longer than this

# ── Audio preprocessing ────────────────────────────────────────────────────────
BANDPASS_LOW_HZ   = 80    # Hz
BANDPASS_HIGH_HZ  = 8000  # Hz
TARGET_DBFS       = -20.0 # normalisation target (dBFS)
NOISE_PROFILE_SEC = 0.5   # seconds of leading audio used as noise profile

# ── Transcription ──────────────────────────────────────────────────────────────
WHISPER_MODEL       = "whisper-1"
WHISPER_LANGUAGE    = "en"
WHISPER_TEMPERATURE = 0.0  # deterministic output

# ── Auto-labelling ─────────────────────────────────────────────────────────────
EMOTION_MODEL        = "j-hartmann/emotion-english-distilroberta-base"
LABEL_CONFIDENCE_MIN = 0.6  # clips below this are flagged for manual review

# Keyword lexicons
URGENCY_KEYWORDS = [
    "box now", "box box", "safety car", "virtual safety car", "vsc",
    "red flag", "brake failure", "brake issue", "brake problem",
    "puncture", "blowout", "crash", "accident", "retire", "stop the car",
    "drs failure", "engine failure", "hydraulics", "fire",
]

FRUSTRATION_MARKERS = [
    "come on", "what is this", "this is ridiculous", "unbelievable",
    "are you kidding", "seriously", "what happened", "why", "i don't understand",
    "terrible", "disaster", "nightmare", "rubbish", "pathetic",
    "again", "every time", "always", "never works",
]
