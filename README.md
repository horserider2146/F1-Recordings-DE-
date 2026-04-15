# F1 Radio Emotion Classification Pipeline

A 7-step pipeline that processes Formula 1 race radio recordings and classifies driver/engineer speech by emotional state: **Calm**, **Frustrated**, **High Stress**, or **Urgent**.

---

## Pipeline Overview
Raw Audio → Extract → Segment → Denoise → Transcribe → Preprocess → Label → Clean

| Step | Script | What It Does |
|------|--------|--------------|
| 01 | `01_extract_audio.py` | Extracts 16 kHz mono audio from video files or YouTube URLs |
| 02 | `02_segment_clips.py` | Uses VAD to split audio into 2–20 second speech clips |
| 03 | `03_preprocess_audio.py` | Noise reduction, bandpass filter (80–8000 Hz), normalization |
| 04 | `04_transcribe.py` | Transcribes clips via OpenAI Whisper (local, no API key) |
| 05 | `05_text_preprocess.py` | Tokenizes, lemmatizes, extracts linguistic features |
| 06 | `06_auto_label.py` | Fuses text (60%) + acoustic (40%) signals into emotion label |
| 07 | `07_clean_dataset.py` | Removes junk clips, outputs clean CSVs for analysis |

---

## Outputs

**For analysis/modelling**, use the `*_clean.csv` files:

- `annotations/labels_clean.csv` — Final emotion labels with confidence scores
- `annotations/preprocessed_text_clean.csv` — Linguistic features per clip

**For debugging**, check:
- `annotations/cleaning_report.csv` — Audit log of removed clips and why
- `data/transcripts/*.json` — Word-level timestamps per clip

---

## Key Output Schema

### `labels_clean.csv`

| Column | Description | Example |
|--------|-------------|---------|
| `clip_id` | Unique clip identifier | `clip_0001` |
| `text_label` | Emotion from transcript (distilroberta-base + keyword rules) | `Frustrated` |
| `text_confidence` | Model certainty (0–1) | `0.82` |
| `acoustic_label` | Emotion from pitch/energy/ZCR analysis | `High Stress` |
| `acoustic_confidence` | Acoustic certainty (0–1) | `0.70` |
| `final_label` | **Primary label** — 60% text + 40% acoustic weighted vote | `High Stress` |
| `confidence` | Combined confidence | `0.76` |
| `flagged_for_review` | True if confidence < 0.6 | `False` |
| `acoustic_features` | JSON of mean_pitch, pitch_range, mean_energy, mean_zcr, duration | `{mean_pitch: 145.2…}` |

### `preprocessed_text_clean.csv`

| Column | Description |
|--------|-------------|
| `clip_id` | Links to `labels_clean.csv` |
| `raw_text` | Original Whisper transcript |
| `clean_text` | Lowercased, punctuation-cleaned |
| `tokens` / `lemmas` | Tokenized and lemmatized word lists |
| `word_count` | Number of alphabetic tokens |
| `negation_count` | Count of no/not/never/can't/won't/don't/hasn't |
| `question_count` | Count of `?` characters |
| `avg_word_length` | Mean characters per word |

---

## Demo Commands

### Show existing results instantly
```python
import pandas as pd
labels = pd.read_csv("annotations/labels_clean.csv")
text = pd.read_csv("annotations/preprocessed_text_clean.csv")
result = labels.merge(text, on="clip_id")
print(result[["clip_id", "raw_text", "final_label", "confidence"]].head(10))
```

### Run pipeline on a single clip (~20 seconds)
```bash
python scripts/03_preprocess_audio.py --input data/clips/clip_0000.wav --overwrite
python scripts/04_transcribe.py --input data/cleaned/clip_0000.wav --overwrite
python scripts/05_text_preprocess.py --input data/transcripts/clip_0000.json --overwrite
python scripts/06_auto_label.py --clip-id clip_0000 --overwrite
```

> **Tip:** Warm up Whisper before presenting to avoid cold-start lag:
> ```bash
> python scripts/04_transcribe.py --model small
> ```

---

## Cleaning Rules

Clips are removed by `07_clean_dataset.py` if they match any of:

- `empty_transcript` — No words detected
- `filler_only` — Single word repeated (Whisper hallucination)
- `repeated_words` — Same phrase repeated 100+ times
- `repeated_char` — Same character makes up 80%+ of transcript
- `youtube_noise` — Non-radio content (e.g. "subscribe…")
- `too_short` — Fewer than 4 words (unless F1 radio keyword detected)
- `whisper_hallucination` — Known Whisper error pattern

Clips flagged for human review (`flagged_for_review = True`) are **kept** in the clean dataset.

---

## File Flow
Raw Audio
└─ 02_segment_clips.py      → clip_manifest.csv + data/clips/.wav
└─ 03_preprocess_audio.py  → data/cleaned/.wav
└─ 04_transcribe.py       → transcription_log.csv + data/transcripts/*.json
└─ 05_text_preprocess.py → preprocessed_text.csv
└─ 06_auto_label.py      → labels.csv
└─ 07_clean_dataset.py   → labels_clean.csv ★
preprocessed_text_clean.csv ★
cleaning_report.csv
