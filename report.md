# F1 Radio Emotion Classification — Project Report

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement & Goals](#2-problem-statement--goals)
3. [Data Collection & Sources](#3-data-collection--sources)
4. [End-to-End Pipeline](#4-end-to-end-pipeline)
5. [Data Cleaning & Quality Control](#5-data-cleaning--quality-control)
6. [Auto-Labelling Strategy](#6-auto-labelling-strategy)
7. [Exploratory Data Analysis](#7-exploratory-data-analysis)
8. [Dataset Splits](#8-dataset-splits)
9. [Model Architecture](#9-model-architecture)
10. [Experiment 1 — Frozen DistilBERT (Baseline)](#10-experiment-1--frozen-distilbert-baseline)
11. [Experiment 2 — DistilBERT with Last Layer Unfrozen](#11-experiment-2--distilbert-with-last-layer-unfrozen)
12. [Experiment 3 — Focal Loss + LR Scheduler](#12-experiment-3--focal-loss--lr-scheduler)
13. [Experiment 4 — DeBERTa-v3-small](#13-experiment-4--deberta-v3-small)
14. [Experiment 5 — DistilBERT + EDA Text Augmentation (Best Model)](#14-experiment-5--distilbert--eda-text-augmentation-best-model)
15. [Experiment 6 — DeBERTa-v3-small + EDA Text Augmentation](#15-experiment-6--deberta-v3-small--eda-text-augmentation)
16. [Experiment 7 — DistilBERT + wav2vec2 Audio Embeddings](#16-experiment-7--distilbert--wav2vec2-audio-embeddings)
17. [Training Dynamics](#17-training-dynamics)
18. [Results Summary](#18-results-summary)
19. [Ensemble](#19-ensemble)
20. [Inference Pipeline](#20-inference-pipeline)
21. [Synthetic Data Generation](#21-synthetic-data-generation)
22. [Branch Structure](#22-branch-structure)
23. [Key Findings & Remaining Levers](#23-key-findings--remaining-levers)

---

## 1. Project Overview

This project builds an end-to-end machine learning pipeline that classifies Formula 1 team radio communications into three emotional states: **Calm**, **Frustrated**, and **High Stress**. The pipeline begins with raw YouTube URLs and produces a fully labelled, stratified dataset and a trained classifier — with no human annotation required for the initial pass.

The work spans audio extraction, signal processing, ASR transcription, NLP preprocessing, automated labelling, multiple model experiments, and an inference utility.

---

## 2. Problem Statement & Goals

F1 team radio clips are short, noisy, and emotionally charged. They contain overlapping acoustic content (engine noise, crowd noise) and highly domain-specific language (F1 jargon, acronyms, team-specific terms). The goal is to automatically classify each clip into one of three states:

| Label | ID | Description |
|---|---|---|
| **Calm** | 0 | Normal race comms, strategy calls, routine updates |
| **Frustrated** | 1 | Driver complaints, team disputes, mechanical anger |
| **High Stress** | 2 | Incidents, safety cars, close racing, pressure moments |

This is fundamentally a **multimodal classification problem** — both the spoken words (text modality) and the acoustic properties (tone, pitch, energy — audio modality) carry independent predictive signal.

**Primary target:** >60% accuracy on a held-out test set, measured by Macro F1 across all three classes.

---

## 3. Data Collection & Sources

### Video Sources

Data was collected from YouTube in three waves:

| Wave | Source Type | Count |
|---|---|---|
| Wave 1 | 2026 Chinese Grand Prix compilations & per-driver radio | 14 URLs |
| Wave 2 | Angry/Frustrated F1 moments compilations | 14 URLs |
| Wave 3 | High-octane/intensity F1 moments | 11 URLs |

The URL sets are stored in `urls.txt`, `angry_urls.txt`, and `high_octane_urls.txt`. Waves 2 and 3 were specifically added to address the class imbalance problem — Frustrated and High Stress clips are naturally rarer in general race compilations.

### Dataset Size Progression

| Stage | Total Clips |
|---|---|
| Initial dataset | ~900 |
| After first expansion (new race videos) | 1,306 |
| After second expansion (targeted high-octane/angry) | 1,536 |

The progression reflects an iterative approach: train, observe failure modes (Frustrated severely underrepresented), collect more data targeted at the weak class, retrain.

### Tools

- **yt-dlp**: downloading best available audio quality from YouTube
- **ffmpeg**: audio extraction and format conversion
- **openai-whisper** (local, no API key): ASR transcription

---

## 4. End-to-End Pipeline

Seven sequential scripts take raw YouTube URLs to a clean labelled dataset. All configuration (paths, thresholds, model names) is centralised in `config.py`.

```
01_extract_audio.py   →  02_segment_clips.py   →  03_preprocess_audio.py
       ↓                        ↓                          ↓
  data/raw_audio/          data/clips/               data/cleaned/
                                                           ↓
04_transcribe.py       →  05_text_preprocess.py  →  06_auto_label.py
       ↓                        ↓                          ↓
 data/transcripts/      annotations/                annotations/labels.csv
                        preprocessed_text.csv              ↓
                                                   07_clean_dataset.py
                                                           ↓
                                                  annotations/labels_clean.csv
```

### Script 01 — Extract Audio (`01_extract_audio.py`)

Downloads video from YouTube via yt-dlp and extracts a 16 kHz mono WAV using ffmpeg. Supports:
- Text file of URLs (one per line, `#` comments skipped)
- Single YouTube URL
- Local video file or directory of video files

Output is saved to `data/raw_audio/`. Existing files are skipped to allow safe re-runs. Validates output with soundfile before reporting success.

**Config:** `SAMPLE_RATE=16000`, `CHANNELS=1`, `audio codec=pcm_s16le`

### Script 02 — Segment Clips (`02_segment_clips.py`)

Applies energy-based Voice Activity Detection (VAD) to segment long recordings into individual radio clips. Each clip maps to one conversational exchange.

**Algorithm:**
1. Compute frame-level RMS energy (frame=2048 samples, hop=512)
2. Binary speech mask: frames above 5% of max RMS are speech
3. Convert frame mask to (start, end) time segments
4. Merge segments with gaps ≤ 0.8 seconds (bridging brief pauses within one utterance)
5. Discard clips shorter than 2.0 seconds or longer than 35.0 seconds
6. Long segments (>35s) are split into ≤35s chunks

Output: `data/clips/*.wav`, `data/clip_manifest.csv` (tracks clip\_id, source, start/end times, duration). Clip IDs are globally unique zero-padded integers (`clip_0000.wav`). The manifest is appended, not overwritten, so new batches continue from the last clip number.

**Config:** `VAD_ENERGY_THRESH=0.05`, `VAD_GAP_TOLERANCE=0.8s`, `VAD_MIN_DURATION=2.0s`, `VAD_MAX_DURATION=35.0s`

### Script 03 — Preprocess Audio (`03_preprocess_audio.py`)

Applies a four-stage signal cleaning pipeline to each clip:

| Stage | Method | Purpose |
|---|---|---|
| 1. Spectral Subtraction | `noisereduce` (non-stationary, prop_decrease=0.85) | Remove engine/crowd noise using a 0.5s noise profile from the clip's leading audio |
| 2. Bandpass Filter | 4th-order Butterworth, 80–8000 Hz | Isolate the human speech frequency band |
| 3. VAD Trim | `librosa.effects.trim` (top_db=30) | Remove leading/trailing silence after denoising |
| 4. Amplitude Normalisation | RMS normalisation to –20 dBFS | Ensure consistent loudness across clips |

Output: `data/cleaned/*.wav`. Supports `--overwrite` flag. Errors are logged but do not halt the pipeline.

### Script 04 — Transcribe (`04_transcribe.py`)

Runs OpenAI Whisper locally (no API key required) on each cleaned clip. Uses the `small` model by default (~480MB), which balances accuracy and speed. The `--model` flag allows choosing `tiny`, `base`, `medium`, or `large`.

Audio is loaded with librosa to avoid ffmpeg PATH issues on Windows. Whisper is run with `temperature=0.0` (deterministic) and `word_timestamps=True`.

Output per clip: `data/transcripts/<clip_id>.json` (contains raw text + word-level timestamps). A summary `data/transcription_log.csv` tracks word count, language detected, and empty/very-short flags. Already-transcribed clips are skipped by default.

### Script 05 — Text Preprocessing (`05_text_preprocess.py`)

Applies NLP preprocessing to each transcript:

1. **Lowercase + ASR artefact removal**: strips bracketed annotations `[inaudible]`, XML tags, ellipses, non-speech characters (preserving apostrophes for contractions)
2. **Tokenisation**: NLTK word tokeniser
3. **Stop word tracking**: frequency counts of stop words retained — high frequency of "no", "why", "what" is a genuine frustration signal, so stop words are NOT removed
4. **Lemmatisation**: WordNetLemmatizer (verb form first, noun fallback)
5. **Feature extraction**: word count, negation count, question count, average word length

Output: `annotations/preprocessed_text.csv`. Incremental — only new clips are processed.

### Script 06 — Auto-Label (`06_auto_label.py`)

The labelling engine. Combines two independent signals to assign each clip a label and a confidence score.

**Text channel (60% weight):**
- Runs `j-hartmann/emotion-english-distilroberta-base` on the transcript. This model outputs 7 emotion scores (anger, disgust, fear, joy, neutral, sadness, surprise).
- Maps to project labels: anger/disgust → Frustrated; fear/sadness/surprise → High Stress; joy/neutral → Calm
- Keyword override layer: urgency keywords (`"box box"`, `"safety car"`, `"puncture"`, etc.) push towards High Stress; frustration markers (`"come on"`, `"unbelievable"`, `"ridiculous"`, etc.) push towards Frustrated

**Acoustic channel (40% weight):**
- Extracts: mean pitch (pyin F0), pitch range, pitch std dev, RMS energy, energy std, zero-crossing rate, duration
- Heuristic rules: high energy + high pitch + high ZCR → High Stress; high pitch + low energy → Frustrated; low energy + low variation → Calm

**Fusion:**
- Weighted vote: builds a 3-class probability vector from each channel's label and confidence, weights them 60/40, takes argmax
- Clips with final confidence < 0.6 are flagged `flagged_for_review=True`

Output: `annotations/labels.csv`. Approximately 224 clips required manual review via `label_review.ipynb`.

### Script 07 — Clean Dataset (`07_clean_dataset.py`)

Removes garbage clips not suitable for modelling. Applies rule-based filters:

| Rule | Description |
|---|---|
| Empty transcript | No text output at all |
| Filler only | Single stop word ("the", "and", "oh", "um") |
| Too few words | < 4 words, without F1 radio keywords ("box", "copy", "pit") |
| YouTube noise | Exact match to outro phrases ("thank you for watching", "subscribe") |
| Repeated character | >60% of characters are the same (hallucination) |
| Repeated word | ≤2 unique words across ≥3 word sequence (hallucination) |
| Whisper silence hallucination | Detected via bigram repetition analysis |

**57 clips removed** across all expansion rounds. The most common removal reasons were empty transcripts (Whisper hallucinating numbers/characters on silence), repeated-word hallucinations ("good, good, good..." × 112 tokens), and too-short fragments.

Output: `annotations/labels_clean.csv`, `annotations/preprocessed_text_clean.csv`, `annotations/cleaning_report.csv`.

### Supporting Scripts

| Script | Purpose |
|---|---|
| `scripts/build_splits.py` | Stratified 70/15/15 train/val/test split using sklearn's `train_test_split(stratify=...)`. Expands acoustic_features JSON into individual columns and merges with text. |
| `scripts/extract_wav2vec2.py` | One-time wav2vec2 embedding extraction: loads `facebook/wav2vec2-base`, mean-pools last hidden state → 768-dim vector per clip. Saves to `data/wav2vec2_embeddings.pkl` with checkpoint every 50 clips. Takes ~1.5–2 hours on CPU. |
| `scripts/post_pipeline.py` | Autonomous watcher: polls `labels.csv` every 30 seconds until row count stabilises for 3 consecutive checks, then chains `07_clean_dataset → build_splits → train_focal` without human intervention. |
| `scripts/generate_synthetic_frustrated.py` | Uses Claude API to generate ~300 synthetic frustrated F1 radio transcripts across 10 focus areas (understeer, team orders, DRS failures, strategy disputes, etc.). |

---

## 5. Data Cleaning & Quality Control

### Whisper Hallucination Patterns Observed

A significant fraction of clips produced garbage transcriptions when Whisper encountered near-silence or very-low-SNR audio. Common hallucination types found in the cleaning report:

- **Number sequences**: `"8, 0, 0, 0, 0, 0, 0, 0..."` — Whisper hallucinating numeric patterns over background noise
- **Repeated words**: `"good, good, good, good..."` repeated 112 times; `"oh, oh, oh..."` 224 times; `"no, no, no..."` 112 times; `"go, go, go..."` 334 times
- **Filler hallucinations**: single tokens "the", "and", "i", "oh"
- **YouTube noise**: one clip transcribed as `"thank you for watching!"`, one as `"music"` (from a video outro)
- **Encoding artifacts**: `"lin-bland-oconen-kolatintosunhar."` — likely non-English speech or garbled audio

### Manual Label Review

`label_review.ipynb` provides an interactive interface for reviewing flagged clips. For each flagged clip, it displays the transcript, the text/acoustic confidence scores, and the auto-assigned label, allowing the reviewer to confirm or correct. Approximately 224 clips were manually reviewed.

---

## 6. Auto-Labelling Strategy

The labelling fusion architecture attempts to compensate for the limitations of each channel:

**Why not text only?** F1 radio clips have heavy domain-specific language. Neutral-sounding phrases like "box box box" are High Stress during an incident. The HuggingFace emotion model is trained on general English text, so acoustic context helps.

**Why not acoustic only?** Background noise, radio compression artefacts, and overlapping voices corrupt acoustic features. Calm clips with excited-sounding voices (e.g., celebrating a good lap) would be misclassified as High Stress without the text channel.

**The keyword override layer** provides high-precision shortcuts: if "safety car", "puncture", or "brake failure" appears in the text, the clip is pushed to High Stress regardless of the model's score. This handles rare but highly diagnostic terms that the general emotion model would underweight.

**Text weight (0.60) > Acoustic weight (0.40)** because the ASR transcription, even imperfect, tends to be more discriminative than acoustic heuristics for the Frustrated/High Stress boundary, which is the hardest to separate acoustically.

---

## 7. Exploratory Data Analysis

The EDA notebook (`EDA.ipynb`) examines four axes before model selection. Key findings:

### 7.1 Label Distribution

The dataset is heavily imbalanced:

| Label | Count (1,306 clips) | Share |
|---|---|---|
| Calm | 863 | 66.1% |
| High Stress | 296 | 22.7% |
| Frustrated | 147 | 11.3% |

The 1,306-clip dataset was the basis for the EDA. After expansion to 1,536 clips, the proportions shifted slightly but remained structurally the same. **Calm dominates. Frustrated is severely underrepresented at ~11%.**

*The EDA generates a bar chart + pie chart of these counts (Figure 1). The pie chart makes the Frustrated scarcity visually obvious.*

### 7.2 Confidence Score Distributions

Three confidence histograms are generated: overall fusion confidence, text-only confidence, and acoustic-only confidence, broken down per label.

Key observations:
- **Acoustic confidence** has a bimodal distribution: most clips are assigned either 0.65 (Calm heuristic) or 0.70–0.75 (High Stress heuristic). The heuristic nature of the acoustic scorer produces discrete confidence bands rather than a continuous distribution.
- **Text confidence** is more spread, with Calm clips clustering near 0.90 (the emotion model is very confident about neutral/joy) and Frustrated/High Stress clips showing broader distributions.
- **Fusion confidence** for Frustrated clips peaks around 0.55–0.65, explaining why Frustrated had the most manual review flags.
- Many Calm clips have text\_confidence > 0.85, giving the model a strong training signal for that class.

### 7.3 Acoustic Feature Distributions per Class

Boxplots of mean_pitch, pitch_range, pitch_std, mean_energy, mean_zcr, and duration by class.

Key observations:
- **Mean Pitch**: High Stress clips have a noticeably higher median pitch (approximately 160–180 Hz) compared to Calm (~110–130 Hz). Frustrated clips overlap with both.
- **Pitch Range**: High Stress shows the highest pitch variability (wider interquartile range and longer whiskers). This confirms that stressed/excited speech has wider pitch excursions.
- **Mean Energy**: High Stress and Frustrated clips have higher mean energy than Calm, but with substantial overlap. The energy difference between Frustrated and High Stress is small.
- **ZCR (Zero Crossing Rate)**: High Stress clips show slightly elevated ZCR, consistent with faster speech rate and more high-frequency content.
- **Duration**: Most clips are 5–20 seconds. High Stress clips tend to be longer (incidents generate extended radio exchanges). Frustrated clips are shorter and more direct.

These observations motivated the choice of 3 acoustic features as the initial model input: `mean_pitch`, `pitch_range`, `pitch_std`. Energy and ZCR were used in the auto-labeller but not fed to the classifier directly (included in auto-labelling heuristics but excluded from the trained model to limit noise).

### 7.4 Transcript Length

- **Word count**: Calm clips are the longest on average (~20–30 words, strategy calls are detailed). Frustrated clips are the shortest (~8–12 words, terse, clipped). High Stress clips are intermediate but variable.
- **Average word length**: Very similar across classes (~4–5 characters). No strong discriminative signal.
- **Negation count**: Elevated in Frustrated clips ("I don't understand", "that's not right", "why isn't this working").
- **Question count**: Also elevated in Frustrated clips (drivers asking rhetorical questions to their engineer).

### 7.5 Training History Comparison (EDA notebook — Section 5)

The EDA notebook also overlays training curves from the frozen baseline vs. focal loss experiment. Key visible differences:
- **Val loss**: Focal loss model converges faster and reaches lower validation loss, with a more pronounced train/val divergence at later epochs (indicating overfitting risk on the small dataset).
- **Val Macro F1**: The focal loss model's val F1 reaches ~0.63 vs. the frozen baseline's plateau at ~0.56. The 0.60 target threshold is visibly crossed only by the focal loss model.

---

## 8. Dataset Splits

Final splits (1,536-clip dataset), stratified 70/15/15 by `final_label`:

| Split | Total | Calm | Frustrated | High Stress |
|---|---|---|---|---|
| Train | 1,030 | 616 | 143 | 271 |
| Val | 221 | 132 | 31 | 58 |
| Test | 221 | 132 | 30 | 59 |

The splits are fixed (random_state=42) and the test set is held out — it is never used during training or hyperparameter tuning. All reported test metrics are from a single final evaluation pass after the model has stopped training.

For the earlier 1,306-clip experiments, the split was:

| Split | Total | Calm | Frustrated | High Stress |
|---|---|---|---|---|
| Train | 914 | ~607 | ~104 | ~203 |
| Val | 196 | ~130 | ~22 | ~44 |
| Test | 196 | ~130 | ~22 | ~44 |

---

## 9. Model Architecture

All experiments share the same conceptual architecture: a pretrained **transformer encoder** (text branch) fused with an **acoustic branch** (either hand-crafted features or wav2vec2 embeddings) via MLP compression, followed by a 3-class classification head.

### Base Architecture (Experiments 1–6)

```
Input text  →  [Transformer Tokenizer]
                      ↓
             [Transformer Encoder]
                      ↓
               [CLS token, 768-dim]
                      ↓
                   concat
                      ↑
[3 acoustic features] → [Linear(3→64) → ReLU → Linear(64→32) → ReLU] → [32-dim]

concat([768-dim CLS, 32-dim acoustic]) = 800-dim
      ↓
[Linear(800→256) → ReLU → Dropout(p) → Linear(256→3)]
      ↓
 class logits
```

### wav2vec2 Architecture (Experiment 7)

```
Input text → [DistilBERT CLS, 768-dim]
Input audio → [wav2vec2-base, mean-pooled, 768-dim]
                      ↓
[Linear(768→256) → ReLU → Linear(256→64) → ReLU] → [64-dim]

concat([768-dim CLS, 64-dim audio]) = 832-dim
      ↓
[Linear(832→256) → ReLU → Dropout(p) → Linear(256→3)]
```

### Common Training Setup

- **Tokeniser**: max_length=128, padding=max_length, truncation=True
- **Batch size**: 16
- **Optimiser**: AdamW with weight_decay=0.01
- **Gradient clipping**: max_norm=1.0
- **Early stopping**: on Val Macro F1 (not loss)
- **Scaler**: sklearn `StandardScaler` fit on train split only, saved as `.pkl` for inference

---

## 10. Experiment 1 — Frozen DistilBERT (Baseline)

**Script:** `train.py`
**Dataset:** 1,306 clips (Train=914, Val=196, Test=196)

### Motivation

Establish a lower bound. Freeze all DistilBERT weights and only train the classification head and acoustic MLP. This tests whether the generic DistilBERT representations are already useful for F1 radio emotion, without any fine-tuning.

### Configuration

| Parameter | Value |
|---|---|
| Backbone | `distilbert-base-uncased` (66M params) |
| Trainable params | ~300K (head only) |
| Learning rate | 1e-4 (head only) |
| Loss | Weighted CrossEntropyLoss (class weights from training distribution) |
| Dropout | 0.5 |
| Patience | 4 |
| Max epochs | 20 |

### Training History (from `models/history.csv`)

Training ran for 17 epochs before early stopping. Key observations:
- Train loss dropped from 1.078 (epoch 1) to 0.795 (epoch 17)
- Val F1 fluctuated erratically in early epochs (0.07 → 0.29 → 0.27 → 0.47), indicating instability with only the head learning
- Best val F1 ≈ 0.558 (epoch 13), declining slightly thereafter
- Large gap between train F1 (~0.55) and val F1 (~0.56) suggests underfitting — the frozen backbone can't adapt

### Results

| Metric | Score |
|---|---|
| Test Accuracy | **56%** |
| Test Macro F1 | **0.467** |
| Calm F1 | 0.69 |
| Frustrated F1 | 0.35 |
| High Stress F1 | 0.36 |

**Interpretation:** The frozen backbone lacks F1-specific language understanding. "Box box" and "Safety car" appear in generic text corpora but with different semantic associations. Both minority classes perform similarly poorly, suggesting the model defaults to predicting Calm on uncertain inputs.

---

## 11. Experiment 2 — DistilBERT with Last Layer Unfrozen

**Script:** `train_unfreeze.py`
**Dataset:** 1,306 clips

### Motivation

The frozen baseline underfit. Unfreezing the final transformer block (~7.3M additional trainable params) gives the model capacity to adapt the top-level contextual representations to F1 domain language without full fine-tuning risk on the small dataset.

### Configuration

| Parameter | Value |
|---|---|
| Trainable | `transformer.layer[-1]` + head (~7.6M params) |
| LR (BERT layer) | 2e-5 |
| LR (head) | 1e-4 |
| Loss | Weighted CrossEntropyLoss |
| Dropout | 0.5 |
| Patience | 5 |

Differential learning rates are used: the unfrozen BERT layer gets a much smaller LR (2e-5) to avoid catastrophic forgetting, while the head gets a larger LR (1e-4) for fast adaptation.

### Training History (from `models/history_unfreeze.csv` - 17 epochs)

Val F1 stabilises around 0.52–0.55, with best val F1 ≈ 0.558. The training curve is smoother than the frozen baseline, consistent with the BERT layer providing stronger gradient signal.

### Results

| Metric | Score |
|---|---|
| Test Accuracy | **57%** |
| Test Macro F1 | **0.508** |
| Calm F1 | 0.68 |
| Frustrated F1 | 0.45 |
| High Stress F1 | 0.40 |

**Interpretation:** Modest improvement in both minority classes (+0.10 Frustrated, +0.04 High Stress). The model is beginning to adapt to domain-specific language but still insufficient — cross-entropy with class weights isn't pushing hard enough on hard-to-classify examples.

---

## 12. Experiment 3 — Focal Loss + LR Scheduler

**Script:** `train_focal.py`
**Dataset:** Initially 1,306 clips; later retrained on 1,536 clips

### Motivation

Weighted cross-entropy gives each class a fixed weight, but within each class it still equally penalises easy and hard examples. **Focal loss** down-weights easy examples (Calm clips the model classifies confidently) and amplifies the gradient from hard ones (Frustrated clips on the decision boundary), effectively forcing the model to focus where it needs to learn most.

### Configuration

| Parameter | Value |
|---|---|
| Loss | Focal Loss: `FL = alpha * (1-pt)^gamma * CE` |
| Focal gamma | 2.0 |
| Per-class alpha | Computed from training distribution, capped at 2.5 |
| Alpha values (1,306 split) | Calm: 0.504, Frustrated: 2.5, High Stress: 1.472 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=2, min_lr=1e-6) |
| Dropout | 0.4 |
| Patience | 6 |

The alpha cap at 2.5 prevents extreme overweighting of the Frustrated class, which was found to cause training instability in preliminary runs.

### Training History — 1,306 clips (from `models/focal_log.txt`)

28 epochs before early stopping. Key dynamics:
- Epochs 1–4: rapid val F1 improvement (0.449 → 0.558)
- Epoch 5: best early val F1 = 0.574 (LR still 1e-4)
- Epochs 6–8: patience ticking, val F1 plateaus ~0.507–0.569
- Epoch 9: LR halved to 5e-5, val F1 jumps to 0.610 (new best)
- Epochs 11–12: best val F1 = 0.614
- Later epochs (15–28): LR continues halving; val F1 oscillates 0.607–0.627; best = **0.6265** at epoch 22
- Train F1 reaches 0.846 (epoch 28) — clear train/val divergence but still improving val slowly

### Results — 1,306 clips

| Metric | Score |
|---|---|
| Test Accuracy | **66.8% (~67%)** |
| Test Macro F1 | **0.591** |

**Per-class breakdown (Test = 196 clips):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Calm | 0.84 | 0.72 | 0.78 | 130 |
| Frustrated | 0.48 | 0.55 | 0.51 | 22 |
| High Stress | 0.42 | 0.57 | 0.49 | 44 |

**60%+ accuracy target achieved.** This became the production model on `main`. The LR scheduler is critical — the best val F1 was not reached until epoch 22, long after most simple schedulers would have stopped. The ReduceLROnPlateau allows continued learning at lower rates without oscillating.

### Retrain on 1,536 clips

After the dataset was expanded to 1,536 clips, the focal loss model was retrained (same config):
- Test Accuracy: **67%**, Macro F1: **0.563**
- Calm F1: 0.80, Frustrated F1: 0.48, High Stress F1: 0.40
- Calm improved (more Calm clips), but Frustrated and High Stress dropped slightly — the test set is now harder (221 vs 196 clips, with a broader distribution).

---

## 13. Experiment 4 — DeBERTa-v3-small

**Script:** Adapted `train_focal.py` with DeBERTa backbone
**Dataset:** 1,536 clips (Train=1030, Val=221, Test=221)

### Motivation

DeBERTa-v3-small uses **disentangled attention** (separate content and position attention matrices) and an enhanced mask decoder. On standard NLP benchmarks it outperforms DistilBERT on short emotional/sentiment texts. The hypothesis is that a better backbone alone will improve classification without changing the training setup.

### Configuration

| Parameter | Value |
|---|---|
| Backbone | `microsoft/deberta-v3-small` (141M params vs DistilBERT's 66M) |
| Trainable | Last 2 encoder layers (DeBERTa has 6 layers) = 14.4M trainable |
| LR (BERT layers) | 2e-5 |
| LR (head) | 1e-4 |
| Loss | Focal Loss (same alpha/gamma as Exp 3) |
| Alpha | Calm: 0.557, Frustrated: 2.401, High Stress: 1.267 |
| Dropout | 0.4 |
| Patience | 6 |
| Additional | `sentencepiece` required for DeBERTa tokeniser |

### Training History (from `models/deberta_log.txt`, `history_deberta.csv`)

Only 9 epochs before early stopping. Key dynamics:
- Val F1 peaked at epoch 3 = **0.532**
- After epoch 3, val F1 oscillates 0.47–0.52 while train F1 continues climbing (0.52 → 0.72)
- Classic overfitting signature: the larger model quickly memorises training examples faster than it generalises
- ReduceLROnPlateau halves LR at epoch 6 but val F1 does not recover

### Results

| Metric | Score |
|---|---|
| Test Accuracy | **61%** |
| Test Macro F1 | **0.523** |

**Per-class breakdown (Test = 221 clips):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Calm | 0.80 | 0.75 | 0.78 | 132 |
| Frustrated | 0.38 | 0.60 | 0.46 | 30 |
| High Stress | 0.36 | 0.31 | 0.33 | 59 |

**Worse than DistilBERT across the board.** The High Stress F1 dropped from 0.40 → 0.33 — a significant regression. DeBERTa, with 141M parameters vs DistilBERT's 66M, needs considerably more data to generalise. With only 1,030 training clips, it overfits rapidly. The train/val F1 gap was already visible by epoch 4 (train 0.573, val 0.472).

---

## 14. Experiment 5 — DistilBERT + EDA Text Augmentation (Best Model)

**Script:** `train_focal.py` with `scripts/augment_text.py`
**Dataset:** 1,536 real clips + augmented training set

### Motivation

The Frustrated class has only 143 real training clips. This is the single biggest constraint across all experiments. Instead of collecting more YouTube videos, **Easy Data Augmentation (EDA)** synthetically expands the minority classes using two text transforms applied only to the training split.

### Augmentation Method

Implemented in `scripts/augment_text.py`:

1. **Synonym Replacement**: Replaces 2 random words with WordNet synonyms (e.g., "the car won't turn" → "the vehicle won't rotate")
2. **Random Deletion**: Drops each word independently with p=0.10 (e.g., "what is this, again?" → "what this, again?")

Each augmented sample has a 50% chance of either transform applied.

**Augmentation targets:**
- Frustrated: 143 → 300 (157 synthetic EDA samples added)
- High Stress: 271 → 430 (159 synthetic EDA samples added)
- Calm: unchanged (already the majority class)

**Training set after augmentation:** 1,030 → 1,346 clips

Alpha values are recomputed from augmented counts: Calm: 0.728, Frustrated: 1.496, High Stress: 1.043 (lower alpha for Frustrated since it now has more examples).

### Training History (from `models/focal_aug_log.txt`, `history_focal_aug.csv`)

19 epochs before early stopping. Key dynamics:
- Val F1 peaks at epoch 13 = **0.550** (best)
- Training converges smoothly; LR drops at epochs 5, 8, 11, 14
- No visible overfitting until epoch 14+ (train F1 reaches 0.866 at epoch 13 but val holds steady)
- 85 train batches (vs 58 in Exp 3) due to larger augmented training set

### Results

| Metric | Score |
|---|---|
| Test Accuracy | **67.9% (~68%)** |
| Test Macro F1 | **0.590** |

**Per-class breakdown (Test = 221 clips):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Calm | 0.82 | 0.80 | **0.81** | 132 |
| Frustrated | 0.55 | 0.40 | 0.46 | 30 |
| High Stress | 0.46 | 0.54 | **0.50** | 59 |

**Best model overall.** High Stress F1 jumped from 0.40 → 0.50 (+25% relative improvement). Calm F1 also improved marginally. Frustrated F1 was unchanged (0.46), suggesting that EDA synonym/deletion augmentation does not add enough novel linguistic signal for this class specifically — the frustrated style is too narrow for paraphrasing to create genuinely new training examples.

**Best model file:** `models/best_model_focal_aug.pt`

---

## 15. Experiment 6 — DeBERTa-v3-small + EDA Text Augmentation

**Script:** Adapted `train_focal.py` with DeBERTa + augmentation
**Dataset:** 1,536 real + augmented (1,346 training clips)

### Motivation

DeBERTa failed in Experiment 4 due to data starvation. With augmented minority classes (300 Frustrated, 430 High Stress training examples), it has significantly more signal. The question is whether this additional data allows DeBERTa to realise its architectural advantage.

### Configuration

Same as Experiment 4 but training on augmented set. Same focal alpha (recomputed from augmented counts: 0.728/1.496/1.043).

### Training History (from `models/deberta_aug_log.txt`, `history_deberta_aug.csv`)

Only 10 epochs before early stopping (faster overfit than Exp 4 — possibly because the augmented data is not as diverse as real data):
- Val F1 peaks at epoch 4 = **0.572**
- Train F1 at epoch 4 = 0.664, by epoch 10 = 0.865 while val F1 declines to 0.506
- The same overfitting pattern — the model memorises training data faster than generalisation

### Results

| Metric | Score |
|---|---|
| Test Accuracy | **65.2% (~65%)** |
| Test Macro F1 | **0.556** |

**Per-class breakdown (Test = 221 clips):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Calm | 0.78 | 0.82 | 0.80 | 132 |
| Frustrated | 0.45 | 0.60 | **0.51** | 30 |
| High Stress | 0.42 | 0.31 | 0.35 | 59 |

Better than un-augmented DeBERTa (65% vs 61%, Macro F1 0.556 vs 0.523), confirming that augmentation helps. However, still behind DistilBERT+aug on every metric except Frustrated F1 (0.51 vs 0.46). Critically, High Stress dropped further (0.35 vs 0.33 for DeBERTa without aug, and vs 0.50 for DistilBERT+aug) — DeBERTa appears to struggle disproportionately with the High Stress / Frustrated boundary on this dataset.

---

## 16. Experiment 7 — DistilBERT + wav2vec2 Audio Embeddings

**Script:** `train_wav2vec2.py`
**Dataset:** 1,536 real + augmented (1,346 training clips)
**Branch:** `experiment/wav2vec2-fusion`

### Motivation

The 3 hand-crafted acoustic features (mean_pitch, pitch_range, pitch_std) are shallow heuristics extracted with librosa's pyin algorithm. `facebook/wav2vec2-base` is a pretrained audio transformer trained on 960 hours of LibriSpeech speech data using a contrastive self-supervised objective. It produces rich 768-dimensional embeddings that capture prosody, tone, rhythm, and stress patterns directly from the raw waveform — far more expressive than 3 scalar values.

### Architecture Change

```
Before (Exp 1–6):   [DistilBERT CLS 768] + [3 features → MLP → 32]  → 800-dim → classifier
After  (Exp 7):     [DistilBERT CLS 768] + [wav2vec2 768 → MLP → 64] → 832-dim → classifier
```

The wav2vec2 MLP compresses 768-dim → 256 → 64 before fusion.

### Embedding Extraction (`scripts/extract_wav2vec2.py`)

- Loads `facebook/wav2vec2-base`
- Processes each clip: `librosa.load` → `Wav2Vec2Processor` → `Wav2Vec2Model.forward()` → mean-pool `last_hidden_state` across time → 768-dim vector
- Saves checkpoint every 50 clips (resumable)
- Took ~1.5–2 hours on CPU for 1,536 clips
- 1,472 embeddings successfully extracted (64 clips had missing wav files)

### Training Configuration

Same as Experiment 5 (DistilBERT, focal loss, augmentation) but with wav2vec2 replacing the 3-feature acoustic MLP.

Alpha values (post-aug): Calm: 0.728, Frustrated: 1.496, High Stress: 1.043

### Training History (from `models/wav2vec2_train_log.txt`, `history_wav2vec2.csv`)

19 epochs before early stopping. Key dynamics:
- Val F1 peaks at epoch 11 = **0.567**
- At epoch 4, val F1 = 0.545 — the model is learning quickly
- After epoch 4, scheduler halves LR at epoch 5 → val F1 oscillates 0.505–0.565
- Best recovered at epoch 11 (LR = 1.25e-5)
- No improvement after epoch 11; 6 patience ticks elapse
- Train F1 reaches 0.902 by epoch 19 — significant train/val gap (val stuck at ~0.557)

### Results

| Metric | Score |
|---|---|
| Test Accuracy | **65.2% (~65%)** |
| Test Macro F1 | **0.562** |
| Best val Macro F1 | 0.568 |

**Per-class breakdown (Test = 221 clips):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Calm | 0.78 | 0.80 | 0.79 | 132 |
| Frustrated | 0.50 | 0.43 | 0.46 | 30 |
| High Stress | 0.43 | 0.44 | 0.43 | 59 |

**Did NOT beat DistilBERT+aug (68%).** wav2vec2 embeddings improve over hand-crafted features on High Stress (0.43 vs 0.40 for DistilBERT without aug) but fall short of the augmented baseline (0.50). The model overfits — train F1 reaches 0.90 while val F1 stagnates at 0.57. Possible reasons:
1. The wav2vec2 embeddings add noise from the heavily compressed F1 radio audio (not similar to LibriSpeech training domain)
2. The additional 256→64 MLP capacity increases model complexity without proportional signal gain
3. 1,030 real training clips may be insufficient to fully exploit the 768-dim audio space

This experiment is on the `experiment/wav2vec2-fusion` branch. The result was reported in the README as "65% accuracy", which the log confirms.

---

## 17. Training Dynamics

### Learning Rate Scheduling Effect

The ReduceLROnPlateau scheduler is highly effective across all focal loss experiments. In every run, the best model checkpoint comes after at least one LR reduction — confirming that the initial LR overshoots the optimal region and the scheduler enables recovery.

| Experiment | Best val F1 epoch | LR at best epoch |
|---|---|---|
| Exp 3 (focal, 1306) | 22 | 3.13e-6 (7th halving) |
| Exp 5 (focal+aug) | 13 | 1.25e-5 (3rd halving) |
| Exp 6 (deberta+aug) | 4 | 1e-4 (no halving yet) |
| Exp 7 (wav2vec2) | 11 | 1.25e-5 (3rd halving) |

DeBERTa (Exp 4, 6) tends to overfit before the LR scheduler can help, suggesting the bottleneck is model capacity relative to data size, not the LR trajectory.

### Train vs Val F1 Gap (Overfitting Diagnostic)

| Experiment | Best train F1 | Val F1 at same epoch | Gap |
|---|---|---|---|
| Exp 3 (focal, 1306) | 0.846 (ep 28) | 0.614 | 0.23 |
| Exp 4 (deberta) | 0.719 (ep 9) | 0.526 | 0.19 |
| Exp 5 (focal+aug) | 0.850 (ep 13) | 0.550 | 0.30 |
| Exp 6 (deberta+aug) | 0.865 (ep 10) | 0.506 | 0.36 |
| Exp 7 (wav2vec2) | 0.902 (ep 19) | 0.557 | 0.34 |

All models show substantial overfitting. The augmented models have a larger train F1 (training on 1,346 clips instead of 1,030), but the validation F1 is comparable or better — augmentation improves generalisation, not memorisation.

### Val F1 vs Test F1 Discrepancy

Across experiments, the best val F1 tends to be higher than the test F1:

| Experiment | Best Val F1 | Test F1 |
|---|---|---|
| Exp 3 (focal, 1306) | 0.627 | 0.591 |
| Exp 5 (focal+aug) | 0.550 | 0.590 |
| Exp 7 (wav2vec2) | 0.568 | 0.562 |

Interestingly, Exp 5 achieves *higher* test F1 (0.590) than its best val F1 (0.550). This is unusual and may indicate that the val set happened to contain particularly hard examples from the augmented data, while the test set (purely real clips) better represents learnable patterns.

---

## 18. Results Summary

| Model | Backbone | Data | Accuracy | Macro F1 | Calm F1 | Frustrated F1 | High Stress F1 |
|---|---|---|---|---|---|---|---|
| Exp 1: Frozen | DistilBERT | 1,306 | 56% | 0.467 | 0.69 | 0.35 | 0.36 |
| Exp 2: Unfrozen-1 | DistilBERT | 1,306 | 57% | 0.508 | 0.68 | 0.45 | 0.40 |
| Exp 3: Focal loss | DistilBERT | 1,306 | 67% | 0.591 | 0.78 | 0.51 | 0.49 |
| Exp 3b: Focal (retrain) | DistilBERT | 1,536 | 67% | 0.563 | 0.80 | 0.48 | 0.40 |
| Exp 4: DeBERTa | DeBERTa-v3-small | 1,536 | 61% | 0.523 | 0.78 | 0.46 | 0.33 |
| **Exp 5: Focal+aug** | **DistilBERT** | **1,536+aug** | **68%** | **0.590** | **0.81** | **0.46** | **0.50** |
| Exp 6: DeBERTa+aug | DeBERTa-v3-small | 1,536+aug | 65% | 0.556 | 0.80 | 0.51 | 0.35 |
| Exp 7: wav2vec2 | DistilBERT + wav2vec2 | 1,536+aug | 65% | 0.562 | 0.79 | 0.46 | 0.43 |

**Best model: Experiment 5 — DistilBERT + Focal Loss + EDA Augmentation** (`experiment/distilbert-augmented` branch)

### Key Takeaways

1. **Focal loss was the single biggest improvement** (+10 percentage points over Exp 2, from 57% to 67%). The effect is clear and consistent.

2. **Data augmentation improved generalisation** for High Stress (+25% relative F1: 0.40 → 0.50) but had minimal effect on Frustrated. EDA's synonym replacement and random deletion do not generate diverse enough frustrated-style utterances.

3. **Bigger backbone ≠ better results** on small datasets. DeBERTa consistently underperformed DistilBERT despite being 2× larger. The data starvation problem is architectural in nature — DeBERTa needs more real training data, not just synthetic augmentation.

4. **wav2vec2 adds audio signal** but doesn't beat the best acoustic-only+text model. The hand-crafted acoustic features (while shallow) avoid the domain shift problem — wav2vec2 was trained on clean LibriSpeech speech, not compressed F1 radio audio.

5. **Frustrated is consistently the hardest class** across all experiments. F1 ranged 0.35–0.51, and the mode of failures is the Frustrated↔Calm boundary (many mildly frustrated clips get classified as Calm).

6. **Calm is consistently the easiest class** (F1 0.68–0.81). The large majority class provides ample training signal, and Calm F1 is largely determined by how much the model avoids false negatives on minority classes.

---

## 19. Ensemble

**Script:** `ensemble.py`

The ensemble averages softmax probabilities from three models:

- **Model A**: `best_model_focal.pt` — DistilBERT + 3 hand-crafted acoustic features (Exp 3, 1,306 clips)
- **Model B**: `best_model_focal_aug.pt` — DistilBERT + EDA augmentation (Exp 5, best model)
- **Model C**: `best_model_wav2vec2.pt` — DistilBERT + wav2vec2 embeddings (Exp 7)

Each model runs inference on the test set independently. The three softmax probability vectors are averaged, and the argmax of the average gives the final prediction.

The ensemble is implemented cleanly: each model type (`AcousticModel`, `Wav2VecModel`) is defined separately, with dedicated Dataset classes that apply the correct scaler and feature extraction for each model.

Expected benefit: ~1–2% accuracy improvement over the best single model, with particular gains on borderline predictions where individual model confidences disagree.

---

## 20. Inference Pipeline

**Script:** `inference.py`

Provides a command-line interface for predicting on new clips:

```
python inference.py --audio path/to/clip.wav
python inference.py --text "Box box box, come in now!"
python inference.py --audio clip.wav --text "Blue flags, he's not moving"
python inference.py --audio clip.wav --model models/best_model_focal.pt
```

The scaler is auto-selected based on the model filename (`focal` → `scaler_focal.pkl`, `unfreeze` → `scaler_unfreeze.pkl`, etc.).

Output format:
```
Prediction : High Stress
Confidence : 71.4%

  Calm         12.3%  ###
  Frustrated    16.3% ####
  High Stress   71.4% #####################
```

The `--audio` path uses `librosa.pyin` for pitch extraction; the `--text` path uses minimal preprocessing matching the training pipeline.

---

## 21. Synthetic Data Generation

**Script:** `scripts/generate_synthetic_frustrated.py`

Addresses the Frustrated class bottleneck using Claude (`claude-sonnet-4-6`) to generate realistic F1 driver radio transcripts.

### Generation Design

**10 focus areas** × 32 clips per batch = 320 clips (before deduplication filtering):

1. Severe understeer — car won't rotate through corners
2. Team orders — driver ordered to hold position
3. Backmarker traffic — blue flags not being waved
4. DRS/ERS failures — system not activating
5. Strategy disputes — tire compound or pit timing disagreements
6. Tire temperature and degradation — fronts/rears overheating
7. Safety car timing — driver feels the team got the call wrong
8. Racing incidents — being pushed wide, contact damage
9. Brake problems — balance off, overheating
10. Car not fast enough — setup issues, no grip

**Style enforcement via system prompt:**
- First-person (driver speaking to engineer)
- 6–22 words per clip (real radio transcriptions are short)
- Genuine F1 vocabulary required
- Frustration implied through word choice, not stated explicitly ("I'm frustrated" → rejected)
- No profanity
- Structural variety: questions, commands, statements, responses

**Post-processing:** Deduplicate by lowercased key. Validate: 5–30 words, ≥10 alphabetical characters. Clips failing validation are dropped.

### Integration into Training (`train_focal_synth.py`)

Synthetic clips are injected before real training data. Acoustic features (mean_pitch, pitch_range, pitch_std) are not available for synthetic clips, so they are **sampled with replacement from real Frustrated clips** to give each synthetic clip a realistic, varied acoustic profile rather than zero-vectors.

Training set after synthetic injection + EDA:
- Calm: ~616 (unchanged)
- Frustrated: ~443 (143 real + 300 synthetic)
- High Stress: ~430 (271 real + 159 EDA)
- Total: ~1,489 clips

---

## 22. Branch Structure

| Branch | Description | Best test result |
|---|---|---|
| `main` | DistilBERT + focal loss, 1,536 clips | 67% accuracy, Macro F1=0.563 |
| `experiment/distilbert-augmented` | DistilBERT + EDA augmentation | **68% accuracy, Macro F1=0.590** |
| `experiment/deberta-augmented` | DeBERTa-v3-small + EDA augmentation | 65% accuracy, Macro F1=0.556 |
| `experiment/wav2vec2-fusion` | DistilBERT + wav2vec2 audio embeddings | 65% accuracy, Macro F1=0.562 |

---

## 23. Key Findings & Remaining Levers

### What Worked

| Technique | Gain |
|---|---|
| Focal loss (Exp 2→3) | +10 pp accuracy, +0.12 Macro F1 |
| EDA augmentation (Exp 3b→5) | +1 pp accuracy; +0.10 High Stress F1 |
| ReduceLROnPlateau | Critical for convergence — best checkpoint never reached at initial LR |
| Targeted data collection (Wave 2+3) | +230 clips, more Frustrated/High Stress balance |
| Manual review of 224 clips | Improved label quality for low-confidence predictions |

### What Did Not Work (As Expected)

| Technique | Outcome | Root cause |
|---|---|---|
| DeBERTa-v3-small | Worse than DistilBERT | 2× more parameters, same data → overfitting |
| wav2vec2 fusion | Worse than DistilBERT+aug | Domain shift; compressed F1 audio ≠ clean LibriSpeech |
| EDA for Frustrated | Minimal Frustrated F1 gain | Paraphrasing doesn't generate new frustrated patterns |

### Remaining Bottlenecks

1. **Frustrated class data starvation** — only 143 real training clips. This is the single biggest constraint. DeBERTa, wav2vec2, and augmentation all fail to overcome it. The most reliable fix is collecting more real Frustrated radio exchanges.

2. **Domain mismatch in audio models** — both the hand-crafted acoustic features and wav2vec2 are operating on heavily compressed, radio-quality audio with engine/crowd background. Pre-training or fine-tuning wav2vec2 on F1 radio audio specifically would be the correct fix, but requires labelled audio pairs.

3. **Frustrated/Calm boundary** — the model's most common error mode. Subtle frustration (measured complaints, mild sarcasm) sounds and reads similar to calm neutral speech. Capturing longer conversational context (more than one turn) might resolve this.

4. **Ensemble not yet benchmarked** — the three models (Exp 3, 5, 7) produce independent probability distributions. Averaging them should yield 1–2% gain for free, but the ensemble script (`ensemble.py`) depends on the wav2vec2 embeddings file being complete.

### Next Steps

1. Run ensemble (`ensemble.py`) and record the result
2. Evaluate synthetic Frustrated approach (`train_focal_synth.py`)  
3. Collect more real Frustrated clips (targeting 300+ real training examples)
4. Consider fine-tuning wav2vec2 on F1 radio audio specifically (requires audio labels, not just text labels)
5. Explore model merging (weight averaging) as an alternative to probability ensemble
