# F1 Radio Emotion Classification

An end-to-end ML pipeline that classifies Formula 1 team radio communications into three emotional states: **Calm**, **Frustrated**, and **High Stress**.

---

## Table of Contents
1. [Project Goal](#project-goal)
2. [Dataset](#dataset)
3. [Pipeline](#pipeline)
4. [Model Experiments](#model-experiments)
5. [Results Summary](#results-summary)
6. [Branch Structure](#branch-structure)
7. [Current Work](#current-work)

---

## Project Goal

F1 team radio clips are short, noisy, and emotionally charged. The goal is to automatically classify each clip into one of three states:

| Label | Description |
|---|---|
| **Calm (0)** | Normal race comms, strategy calls, routine updates |
| **Frustrated (1)** | Driver complaints, team disputes, mechanical anger |
| **High Stress (2)** | Incidents, safety cars, close racing, pressure moments |

This is a multimodal classification problem — both the **spoken words** (text) and the **acoustic properties** (tone, pitch, energy) carry signal.

---

## Dataset

### Collection
- Source: YouTube F1 team radio compilations
- Tool: `yt-dlp` for downloading, `ffmpeg` for audio extraction
- Audio segmented into ~2–15 second clips using silence detection

### Size Progression
| Stage | Clips |
|---|---|
| Initial dataset | ~900 |
| After first expansion (new videos) | 1,306 |
| After second expansion (high-octane videos) | 1,536 |

### Final Split (1,536 clips)
| Split | Total | Calm | Frustrated | High Stress |
|---|---|---|---|---|
| Train | 1,030 | 616 | 143 | 271 |
| Val | 221 | 132 | 31 | 58 |
| Test | 221 | 132 | 30 | 59 |

Stratified 70/15/15 split. The dataset is **imbalanced** — Calm dominates, Frustrated is severely underrepresented.

### Labelling
Each clip is labelled by a fusion of two signals:
- **Text confidence**: NLP model run on the Whisper transcript
- **Acoustic confidence**: Heuristic scoring on pitch, energy, and ZCR

Clips below a confidence threshold are flagged for manual review. ~224 clips were manually reviewed and corrected using `label_review.ipynb`.

---

## Pipeline

Seven scripts run sequentially to take raw YouTube URLs to a clean labelled dataset:

```
01_extract_audio.py     — Download and extract WAV from YouTube via yt-dlp
02_segment_clips.py     — Segment each WAV into clips using silence detection
03_preprocess_audio.py  — Extract acoustic features (pitch, energy, ZCR, duration)
04_transcribe.py        — Transcribe clips using OpenAI Whisper (local, no API key)
05_text_preprocess.py   — Clean transcripts, compute word count, negation, etc.
06_auto_label.py        — Fuse text + acoustic signals into a label + confidence score
07_clean_dataset.py     — Remove garbage clips (too short, silent, transcription failures)
```

Supporting scripts:
- `scripts/build_splits.py` — Stratified train/val/test split
- `scripts/augment_text.py` — EDA text augmentation (synonym replacement + random deletion)
- `scripts/extract_wav2vec2.py` — wav2vec2 audio embedding extraction (one-time)
- `scripts/run_new_batch.py` — Safely adds new videos without re-processing existing clips
- `scripts/post_pipeline.py` — Autonomous watcher: polls labels until stable, then chains 07 → build_splits → retrain

---

## Model Experiments

All models share the same base architecture: a **transformer encoder** fused with **acoustic features** via an MLP, followed by a classification head. Training always evaluates on an untouched held-out test set.

---

### Experiment 1 — Frozen DistilBERT (Baseline)

**Why:** Establish a baseline. Freeze all DistilBERT weights and only train the classification head. Low risk of overfitting, fast to train.

**How:**
- `distilbert-base-uncased` fully frozen (66M params, 0 trainable in backbone)
- 3 acoustic features (mean_pitch, pitch_range, pitch_std) → 32-dim MLP
- DistilBERT CLS (768-dim) + acoustic (32-dim) → Linear(256) → ReLU → Dropout → Linear(3)
- Weighted cross-entropy loss
- Dataset: 1,306 clips

**Results:**
| Metric | Score |
|---|---|
| Accuracy | 56% |
| Macro F1 | 0.467 |
| Calm F1 | 0.69 |
| Frustrated F1 | 0.35 |
| High Stress F1 | 0.36 |

Both minority classes performed poorly — the frozen backbone couldn't adapt to F1-specific language.

---

### Experiment 2 — DistilBERT with Last Layer Unfrozen

**Why:** The frozen baseline underfit. Unfreezing the last transformer block gives the model more capacity to adapt its representations to F1 radio language without full fine-tuning risk on a small dataset.

**How:**
- All layers frozen except `transformer.layer[-1]` (~7.3M trainable params)
- Differential learning rates: LR_BERT=2e-5, LR_HEAD=1e-4
- Same architecture as Experiment 1
- Patience=5 early stopping
- Dataset: 1,306 clips

**Results:**
| Metric | Score |
|---|---|
| Accuracy | 57% |
| Macro F1 | 0.508 |
| Calm F1 | 0.68 |
| Frustrated F1 | 0.45 |
| High Stress F1 | 0.40 |

Modest improvement across minority classes. Not enough to hit the 60% target.

---

### Experiment 3 — Focal Loss + LR Scheduler

**Why:** Cross-entropy with class weights wasn't handling class imbalance well enough. Focal loss down-weights easy examples (Calm) and forces the model to focus on hard ones (Frustrated, High Stress). ReduceLROnPlateau prevents the model from stagnating at a local minimum.

**How:**
- **Focal loss**: `FL = alpha * (1 - pt)^gamma * CE`, with gamma=2.0 and per-class alpha capped at 2.5
- **ReduceLROnPlateau** scheduler: factor=0.5, patience=2, min_lr=1e-6
- Last BERT layer unfrozen, same differential LRs as Experiment 2
- Dropout=0.4, Patience=6
- Trained on 1,306 clips, then retrained on expanded 1,536 clips after dataset expansion

**Results (1,306 clips):**
| Metric | Score |
|---|---|
| Accuracy | 67% |
| Macro F1 | 0.591 |
| Calm F1 | 0.78 |
| Frustrated F1 | 0.51 |
| High Stress F1 | 0.49 |

**Results (1,536 clips):**
| Metric | Score |
|---|---|
| Accuracy | 67% |
| Macro F1 | 0.563 |
| Calm F1 | 0.80 |
| Frustrated F1 | 0.48 |
| High Stress F1 | 0.40 |

**60%+ accuracy target achieved.** This became the production model on `main`. Calm F1 improved further with more data, but Frustrated remained the hardest class.

---

### Experiment 4 — DeBERTa-v3-small

**Why:** DeBERTa-v3-small consistently outperforms DistilBERT on short, emotion-heavy text in benchmarks due to its disentangled attention mechanism. Tested to see if a better backbone alone would improve results without changing anything else.

**How:**
- Swapped `distilbert-base-uncased` → `microsoft/deberta-v3-small` (141M params vs 66M)
- Unfroze last **2** encoder layers (DeBERTa has 6 total) — 14.4M trainable params
- Same focal loss, same differential LRs
- Required installing `sentencepiece` for the tokenizer
- Dataset: 1,536 clips

**Results:**
| Metric | Score |
|---|---|
| Accuracy | 61% |
| Macro F1 | 0.523 |
| Calm F1 | 0.78 |
| Frustrated F1 | 0.46 |
| High Stress F1 | 0.33 |

**Worse than DistilBERT.** DeBERTa is a larger model that needs more data to generalise. With only 1,030 training clips it overfitted — train F1 reached 0.72 while val F1 stagnated at 0.53. High Stress dropped significantly (0.40 → 0.33).

---

### Experiment 5 — DistilBERT + Text Augmentation

**Why:** The Frustrated class has only 143 training clips — the single biggest bottleneck. Instead of collecting more videos, EDA (Easy Data Augmentation) synthetically expands minority classes using two text transforms applied only to the training split.

**How:**
- `scripts/augment_text.py` implements:
  - **Synonym replacement**: replaces 2 random words with WordNet synonyms
  - **Random deletion**: drops each word with probability p=0.10
- Augmentation targets: Frustrated 143→300, High Stress 271→430
- Training set grows: 1,030 → 1,346 clips
- Same DistilBERT focal architecture (last layer unfrozen)
- Alpha recomputed from augmented class counts

**Results:**
| Metric | Score |
|---|---|
| Accuracy | **68%** |
| Macro F1 | **0.590** |
| Calm F1 | 0.81 |
| Frustrated F1 | 0.46 |
| High Stress F1 | **0.50** |

**Best model overall.** High Stress F1 jumped from 0.40 → 0.50. Augmentation helped generalisation without requiring new video collection.

---

### Experiment 6 — DeBERTa-v3-small + Text Augmentation

**Why:** DeBERTa failed without augmentation due to data starvation. With augmented minority classes (300 Frustrated, 430 High Stress), it has more training signal. Testing whether it can now beat DistilBERT.

**How:**
- Same augmentation as Experiment 5 (1,030 → 1,346 training clips)
- DeBERTa-v3-small, last 2 encoder layers unfrozen
- Same focal loss and differential LRs

**Results:**
| Metric | Score |
|---|---|
| Accuracy | 65% |
| Macro F1 | 0.556 |
| Calm F1 | 0.80 |
| Frustrated F1 | 0.51 |
| High Stress F1 | 0.35 |

Better than un-augmented DeBERTa (65% vs 61%), but still behind DistilBERT+aug. High Stress worsened relative to Experiment 5. DeBERTa needs more real data, not just synthetic augmentation, to beat the smaller model on this dataset size.

---

### Experiment 7 — DistilBERT + wav2vec2 Audio Embeddings (In Progress)

**Why:** The 3 hand-crafted acoustic features (mean_pitch, pitch_range, pitch_std) are shallow heuristics. `facebook/wav2vec2-base` is a pretrained audio transformer trained on 960 hours of speech that produces rich 768-dim embeddings capturing prosody, tone, and stress directly from the waveform — far more expressive than 3 numbers.

**Architecture change:**
```
Before:  [DistilBERT CLS 768] + [3 hand-crafted features → MLP → 32]  → 800 → classifier
After:   [DistilBERT CLS 768] + [wav2vec2 768 → MLP → 64]             → 832 → classifier
```

**How:**
- `scripts/extract_wav2vec2.py`: runs wav2vec2-base over all 1,536 clips, mean-pools the last hidden state → 768-dim embedding per clip, saved as `data/wav2vec2_embeddings.pkl`. Includes checkpoint resuming every 50 clips.
- `train_wav2vec2.py`: loads embeddings, normalises with StandardScaler, fuses with DistilBERT CLS via a 768→256→64 MLP before the classifier head. Also applies text augmentation (same as Experiment 5).
- Pipeline is fully autonomous: extraction runs first (~1.5-2 hrs), training auto-starts via a watcher process.

**Status:** Feature extraction running (~1,472 clips). Results pending.

---

## Results Summary

| Model | Data | Accuracy | Macro F1 | Calm | Frustrated | High Stress |
|---|---|---|---|---|---|---|
| DistilBERT frozen | 1,306 | 56% | 0.467 | 0.69 | 0.35 | 0.36 |
| DistilBERT unfrozen (1 layer) | 1,306 | 57% | 0.508 | 0.68 | 0.45 | 0.40 |
| DistilBERT focal loss | 1,306 | 67% | 0.591 | 0.78 | 0.51 | 0.49 |
| DistilBERT focal loss | 1,536 | 67% | 0.563 | 0.80 | 0.48 | 0.40 |
| DeBERTa-v3-small | 1,536 | 61% | 0.523 | 0.78 | 0.46 | 0.33 |
| **DistilBERT focal + aug** | **1,536+aug** | **68%** | **0.590** | **0.81** | 0.46 | **0.50** |
| DeBERTa-v3-small + aug | 1,536+aug | 65% | 0.556 | 0.80 | 0.51 | 0.35 |
| DistilBERT + wav2vec2 + aug | 1,536+aug | TBD | TBD | — | — | — |

**Best model:** `experiment/distilbert-augmented` — 68% accuracy, Macro F1=0.590.

---

## Branch Structure

| Branch | Description |
|---|---|
| `main` | Production — DistilBERT focal loss, 67% accuracy, 1,536 clips |
| `experiment/distilbert-augmented` | DistilBERT + EDA text augmentation — **68% accuracy** |
| `experiment/deberta-augmented` | DeBERTa-v3-small + EDA text augmentation — 65% accuracy |
| `experiment/wav2vec2-fusion` | DistilBERT + wav2vec2 audio embeddings — in progress |

---

## Remaining Levers to Improve Accuracy

1. **More Frustrated clips** — only 143 real training examples; single biggest bottleneck across all experiments
2. **wav2vec2 fusion results** — pending (current run); higher ceiling than hand-crafted features
3. **Ensemble** — average predictions from best 3 models; free 1-2% gain with no new training
