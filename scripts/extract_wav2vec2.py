"""
extract_wav2vec2.py
Runs wav2vec2-base over every clip in labels_clean.csv and saves
mean-pooled 768-dim embeddings to data/wav2vec2_embeddings.pkl.

Run once — subsequent training scripts load the pkl directly.
Takes ~1.5-2 hrs on CPU.
"""

import os, pickle, sys
import numpy as np
import pandas as pd
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

BASE      = r'C:\Users\Ritarshi Roy\OneDrive\Desktop\Projects\F1 Recordings'
CLIPS_DIR = os.path.join(BASE, 'data', 'clips')
OUT_FILE  = os.path.join(BASE, 'data', 'wav2vec2_embeddings.pkl')
LABELS    = os.path.join(BASE, 'annotations', 'labels_clean.csv')
TARGET_SR = 16000
MODEL_NAME = 'facebook/wav2vec2-base'

print(f'Loading {MODEL_NAME}...')
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model     = Wav2Vec2Model.from_pretrained(MODEL_NAME)
model.eval()
print('Model loaded.')

df = pd.read_csv(LABELS)
clip_ids = df['clip_id'].tolist()
print(f'Clips to process: {len(clip_ids)}')

# Load existing embeddings so we can resume if interrupted
if os.path.exists(OUT_FILE):
    with open(OUT_FILE, 'rb') as f:
        embeddings = pickle.load(f)
    print(f'Resuming — {len(embeddings)} embeddings already done.')
else:
    embeddings = {}

missing = []
for cid in clip_ids:
    path = os.path.join(CLIPS_DIR, f'{cid}.wav')
    if not os.path.exists(path):
        missing.append(cid)

if missing:
    print(f'WARNING: {len(missing)} clip files not found — will be skipped.')

todo = [c for c in clip_ids if c not in embeddings]
print(f'Remaining: {len(todo)}')

for i, cid in enumerate(todo):
    path = os.path.join(CLIPS_DIR, f'{cid}.wav')
    if not os.path.exists(path):
        embeddings[cid] = np.zeros(768, dtype=np.float32)
        continue

    try:
        audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
        inputs   = processor(audio, sampling_rate=TARGET_SR, return_tensors='pt',
                             padding=True)
        with torch.no_grad():
            out = model(**inputs)
        # Mean-pool across time → (768,)
        emb = out.last_hidden_state.squeeze(0).mean(dim=0).numpy()
        embeddings[cid] = emb.astype(np.float32)
    except Exception as e:
        print(f'  ERROR on {cid}: {e}')
        embeddings[cid] = np.zeros(768, dtype=np.float32)

    if (i + 1) % 50 == 0 or (i + 1) == len(todo):
        with open(OUT_FILE, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f'  [{i+1}/{len(todo)}] saved checkpoint.', flush=True)

print(f'\nDone. {len(embeddings)} embeddings saved to {OUT_FILE}')
