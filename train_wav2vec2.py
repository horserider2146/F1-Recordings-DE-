import os, sys, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from scripts.augment_text import augment_dataframe

BASE        = r'C:\Users\Ritarshi Roy\OneDrive\Desktop\Projects\F1 Recordings'
EMBED_FILE  = os.path.join(BASE, 'data', 'wav2vec2_embeddings.pkl')
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

MAX_LEN     = 128
BATCH_SIZE  = 16
EPOCHS      = 30
LR_HEAD     = 1e-4
LR_BERT     = 2e-5
DROPOUT     = 0.4
PATIENCE    = 6
FOCAL_GAMMA = 2.0
LABEL_ORDER = ['Calm', 'Frustrated', 'High Stress']
NUM_CLASSES = 3
WAV2VEC_DIM = 768

# -- 1. Load wav2vec2 embeddings -----------------------------------------------
print('Loading wav2vec2 embeddings...')
with open(EMBED_FILE, 'rb') as f:
    wav2vec_embeddings = pickle.load(f)
print(f'Loaded {len(wav2vec_embeddings)} embeddings.')

# -- 2. Load splits ------------------------------------------------------------
train_df = pd.read_csv(os.path.join(BASE, 'data', 'splits', 'train.csv'))
val_df   = pd.read_csv(os.path.join(BASE, 'data', 'splits', 'val.csv'))
test_df  = pd.read_csv(os.path.join(BASE, 'data', 'splits', 'test.csv'))

for df in [train_df, val_df, test_df]:
    df['clean_text'] = df['clean_text'].fillna('')

print(f'Original train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}')

# -- 3. Augment training set ---------------------------------------------------
train_df = augment_dataframe(train_df, random_seed=42)

counts = train_df['label_id'].value_counts().sort_index()
total  = counts.sum()
alpha  = torch.tensor(
    [total / (NUM_CLASSES * counts[i]) for i in range(NUM_CLASSES)],
    dtype=torch.float
).to(DEVICE)
alpha = torch.clamp(alpha, max=2.5)
print(f'Focal alpha (post-aug): {dict(zip(LABEL_ORDER, alpha.cpu().numpy().round(3)))}')

# -- 4. Normalise wav2vec2 embeddings ------------------------------------------
# Fit scaler on train clip_ids (original only, not augmented duplicates)
original_ids = pd.read_csv(os.path.join(BASE, 'data', 'splits', 'train.csv'))['clip_id'].tolist()
train_embs   = np.stack([wav2vec_embeddings.get(c, np.zeros(WAV2VEC_DIM)) for c in original_ids])
wav_scaler   = StandardScaler()
wav_scaler.fit(train_embs)

os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)
with open(os.path.join(BASE, 'models', 'scaler_wav2vec2.pkl'), 'wb') as f:
    pickle.dump(wav_scaler, f)
print('Scaler saved.')

# -- 5. Dataset ----------------------------------------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def get_wav_emb(clip_id):
    raw = wav2vec_embeddings.get(clip_id, np.zeros(WAV2VEC_DIM, dtype=np.float32))
    return wav_scaler.transform(raw.reshape(1, -1)).squeeze(0).astype(np.float32)

class F1RadioDataset(Dataset):
    def __init__(self, df):
        self.texts    = df['clean_text'].tolist()
        self.clip_ids = df['clip_id'].tolist()
        self.labels   = df['label_id'].values.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], max_length=MAX_LEN,
                        padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'wav_emb':        torch.tensor(get_wav_emb(self.clip_ids[idx])),
            'label':          torch.tensor(self.labels[idx])
        }

train_loader = DataLoader(F1RadioDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(F1RadioDataset(val_df),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(F1RadioDataset(test_df),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'Train batches: {len(train_loader)}  Val batches: {len(val_loader)}')

# -- 6. Model ------------------------------------------------------------------
class F1EmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.transformer.layer[-1].parameters():
            param.requires_grad = True

        # Compress wav2vec2 768-dim → 64-dim
        self.wav_mlp = nn.Sequential(
            nn.Linear(WAV2VEC_DIM, 256), nn.ReLU(),
            nn.Linear(256, 64),          nn.ReLU()
        )
        # DistilBERT 768 + wav2vec2 compressed 64 = 832
        self.classifier = nn.Sequential(
            nn.Linear(768 + 64, 256), nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask, wav_emb):
        cls_emb = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        wav_out = self.wav_mlp(wav_emb)
        return self.classifier(torch.cat([cls_emb, wav_out], dim=1))

model = F1EmotionModel().to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p   = sum(p.numel() for p in model.parameters())
print(f'Params: {total_p:,} total / {trainable:,} trainable')

# -- 7. Focal loss -------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, reduction='none')
        pt   = torch.exp(-ce)
        at   = self.alpha[targets]
        loss = at * (1 - pt) ** self.gamma * ce
        return loss.mean()

criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA)

# -- 8. Optimiser + scheduler --------------------------------------------------
bert_params = list(model.bert.transformer.layer[-1].parameters())
head_params = list(model.wav_mlp.parameters()) + list(model.classifier.parameters())

optimizer = torch.optim.AdamW([
    {'params': bert_params, 'lr': LR_BERT},
    {'params': head_params, 'lr': LR_HEAD},
], weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6, verbose=False
)

# -- 9. Train/eval loop --------------------------------------------------------
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            ids    = batch['input_ids'].to(DEVICE)
            mask   = batch['attention_mask'].to(DEVICE)
            wav    = batch['wav_emb'].to(DEVICE)
            lbls   = batch['label'].to(DEVICE)
            logits = model(ids, mask, wav)
            loss   = criterion(logits, lbls)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
    return total_loss / len(loader), f1_score(all_labels, all_preds, average='macro', zero_division=0), all_labels, all_preds

# -- 10. Training loop ---------------------------------------------------------
best_val_f1     = 0.0
patience_count  = 0
best_model_path = os.path.join(BASE, 'models', 'best_model_wav2vec2.pt')
history         = []

print()
print(f"{'Ep':<4} {'TrLoss':<10} {'TrF1':<8} {'VlLoss':<10} {'VlF1':<8} {'LR_head':<10} Status")
print('-' * 65)

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, train=True)
    vl_loss, vl_f1, _, _ = run_epoch(val_loader,   train=False)

    scheduler.step(vl_loss)
    current_lr = optimizer.param_groups[1]['lr']

    history.append({'epoch': epoch, 'train_loss': tr_loss, 'train_f1': tr_f1,
                    'val_loss': vl_loss, 'val_f1': vl_f1, 'lr': current_lr})

    if vl_f1 > best_val_f1:
        best_val_f1    = vl_f1
        patience_count = 0
        torch.save(model.state_dict(), best_model_path)
        status = 'saved'
    else:
        patience_count += 1
        status = f'p {patience_count}/{PATIENCE}'

    print(f"{epoch:<4} {tr_loss:<10.4f} {tr_f1:<8.4f} {vl_loss:<10.4f} {vl_f1:<8.4f} {current_lr:<10.2e} {status}", flush=True)

    if patience_count >= PATIENCE:
        print(f'\nEarly stop at epoch {epoch}.')
        break

print(f'\nBest val macro F1: {best_val_f1:.4f}')
pd.DataFrame(history).to_csv(os.path.join(BASE, 'models', 'history_wav2vec2.csv'), index=False)

# -- 11. Test evaluation -------------------------------------------------------
print('\nRunning test evaluation...')
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
te_loss, te_f1, te_labels, te_preds = run_epoch(test_loader, train=False)

print(f'Test Loss:     {te_loss:.4f}')
print(f'Test Macro F1: {te_f1:.4f}')
print(f'Accuracy:      {np.mean(np.array(te_preds) == np.array(te_labels)):.4f}')
print()
print(classification_report(te_labels, te_preds, target_names=LABEL_ORDER, zero_division=0))
print('Done. History saved to models/history_wav2vec2.csv')
