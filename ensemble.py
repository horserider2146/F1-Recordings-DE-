"""
ensemble.py
Averages softmax probabilities from 3 models on the test set:
  1. best_model_focal.pt     (DistilBERT + 3 acoustic features)
  2. best_model_focal_aug.pt (DistilBERT + 3 acoustic features, aug training)
  3. best_model_wav2vec2.pt  (DistilBERT + wav2vec2 768-dim embeddings, aug training)
"""

import os, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.metrics import f1_score, classification_report

BASE        = r'C:\Users\Ritarshi Roy\OneDrive\Desktop\Projects\F1 Recordings'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABEL_ORDER = ['Calm', 'Frustrated', 'High Stress']
NUM_CLASSES = 3
MAX_LEN     = 128
BATCH_SIZE  = 16
DROPOUT     = 0.4
ACOUSTIC_COLS = ['mean_pitch', 'pitch_range', 'pitch_std']
WAV2VEC_DIM = 768

print(f'Using device: {DEVICE}')

# -- Load test data ------------------------------------------------------------
test_df = pd.read_csv(os.path.join(BASE, 'data', 'splits', 'test.csv'))
test_df['clean_text'] = test_df['clean_text'].fillna('')
true_labels = test_df['label_id'].values
print(f'Test set: {len(test_df)} clips')

# -- Load wav2vec2 embeddings --------------------------------------------------
print('Loading wav2vec2 embeddings...')
with open(os.path.join(BASE, 'data', 'wav2vec2_embeddings.pkl'), 'rb') as f:
    wav2vec_embeddings = pickle.load(f)

# -- Tokenizer (shared) --------------------------------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# =============================================================================
# Model A & B: DistilBERT + 3 acoustic features
# =============================================================================
class AcousticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.acoustic_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 256), nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask, acoustic):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.classifier(torch.cat([cls, self.acoustic_mlp(acoustic)], dim=1))


class AcousticDataset(Dataset):
    def __init__(self, df, scaler):
        self.texts    = df['clean_text'].tolist()
        self.acoustic = scaler.transform(df[ACOUSTIC_COLS].values).astype(np.float32)
        self.labels   = df['label_id'].values.astype(np.int64)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], max_length=MAX_LEN,
                        padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'acoustic':       torch.tensor(self.acoustic[idx]),
            'label':          torch.tensor(self.labels[idx])
        }


def get_probs_acoustic(model_path, scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    dataset = AcousticDataset(test_df, scaler)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model   = AcousticModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['input_ids'].to(DEVICE),
                           batch['attention_mask'].to(DEVICE),
                           batch['acoustic'].to(DEVICE))
            all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(all_probs)


# =============================================================================
# Model C: DistilBERT + wav2vec2 embeddings
# =============================================================================
class Wav2VecModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.wav_mlp = nn.Sequential(
            nn.Linear(WAV2VEC_DIM, 256), nn.ReLU(),
            nn.Linear(256, 64),          nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 + 64, 256), nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask, wav_emb):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.classifier(torch.cat([cls, self.wav_mlp(wav_emb)], dim=1))


class Wav2VecDataset(Dataset):
    def __init__(self, df, wav_scaler):
        self.texts    = df['clean_text'].tolist()
        self.clip_ids = df['clip_id'].tolist()
        self.labels   = df['label_id'].values.astype(np.int64)
        self.scaler   = wav_scaler

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], max_length=MAX_LEN,
                        padding='max_length', truncation=True, return_tensors='pt')
        raw = wav2vec_embeddings.get(self.clip_ids[idx], np.zeros(WAV2VEC_DIM, dtype=np.float32))
        emb = self.scaler.transform(raw.reshape(1, -1)).squeeze(0).astype(np.float32)
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'wav_emb':        torch.tensor(emb),
            'label':          torch.tensor(self.labels[idx])
        }


def get_probs_wav2vec():
    with open(os.path.join(BASE, 'models', 'scaler_wav2vec2.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    dataset = Wav2VecDataset(test_df, scaler)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model   = Wav2VecModel().to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(BASE, 'models', 'best_model_wav2vec2.pt'),
        map_location=DEVICE, weights_only=True))
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['input_ids'].to(DEVICE),
                           batch['attention_mask'].to(DEVICE),
                           batch['wav_emb'].to(DEVICE))
            all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(all_probs)


# =============================================================================
# Run all 3 and ensemble
# =============================================================================
print('\nRunning Model A: best_model_focal.pt ...')
probs_a = get_probs_acoustic(
    os.path.join(BASE, 'models', 'best_model_focal.pt'),
    os.path.join(BASE, 'models', 'scaler_focal.pkl')
)

print('Running Model B: best_model_focal_aug.pt ...')
probs_b = get_probs_acoustic(
    os.path.join(BASE, 'models', 'best_model_focal_aug.pt'),
    os.path.join(BASE, 'models', 'scaler_focal_aug.pkl')
)

print('Running Model C: best_model_wav2vec2.pt ...')
probs_c = get_probs_wav2vec()

# Average probabilities
ensemble_probs = (probs_a + probs_b + probs_c) / 3.0
preds = ensemble_probs.argmax(axis=1)

# =============================================================================
# Results
# =============================================================================
print('\n' + '='*60)
print('  INDIVIDUAL MODEL RESULTS')
print('='*60)
for name, probs in [('focal', probs_a), ('focal_aug', probs_b), ('wav2vec2', probs_c)]:
    p = probs.argmax(axis=1)
    acc = np.mean(p == true_labels)
    f1  = f1_score(true_labels, p, average='macro', zero_division=0)
    print(f'  {name:<12}  Accuracy={acc:.4f}  Macro F1={f1:.4f}')

print('\n' + '='*60)
print('  ENSEMBLE (average probabilities)')
print('='*60)
acc = np.mean(preds == true_labels)
f1  = f1_score(true_labels, preds, average='macro', zero_division=0)
print(f'  Accuracy : {acc:.4f}  ({acc*100:.1f}%)')
print(f'  Macro F1 : {f1:.4f}')
print()
print(classification_report(true_labels, preds, target_names=LABEL_ORDER, zero_division=0))
