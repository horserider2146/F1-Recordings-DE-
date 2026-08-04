import os, json, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

BASE   = r'C:\Users\Ritarshi Roy\OneDrive\Desktop\Projects\F1 Recordings'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# Hyperparameters
MAX_LEN       = 128
BATCH_SIZE    = 16
EPOCHS        = 20
LR            = 1e-4
DROPOUT       = 0.5
PATIENCE      = 4
LABEL_ORDER   = ['Calm', 'Frustrated', 'High Stress']
NUM_CLASSES   = 3
ACOUSTIC_COLS = ['mean_pitch', 'pitch_range', 'pitch_std']

# ── 1. Load data ──────────────────────────────────────────────────────────────
train_df = pd.read_csv(os.path.join(BASE, 'data', 'splits', 'train.csv'))
val_df   = pd.read_csv(os.path.join(BASE, 'data', 'splits', 'val.csv'))
test_df  = pd.read_csv(os.path.join(BASE, 'data', 'splits', 'test.csv'))

for df in [train_df, val_df, test_df]:
    df['clean_text'] = df['clean_text'].fillna('')

with open(os.path.join(BASE, 'annotations', 'class_weights.json')) as f:
    cw_data = json.load(f)
weight_list   = [cw_data['by_id'][str(i)] for i in range(NUM_CLASSES)]
class_weights = torch.tensor(weight_list, dtype=torch.float).to(DEVICE)

print(f'Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}')
print(f'Class weights: {dict(zip(LABEL_ORDER, weight_list))}')

# ── 2. Scale acoustic features ────────────────────────────────────────────────
scaler = StandardScaler()
train_df[ACOUSTIC_COLS] = scaler.fit_transform(train_df[ACOUSTIC_COLS])
val_df[ACOUSTIC_COLS]   = scaler.transform(val_df[ACOUSTIC_COLS])
test_df[ACOUSTIC_COLS]  = scaler.transform(test_df[ACOUSTIC_COLS])

os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)
with open(os.path.join(BASE, 'models', 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print('Scaler saved.')

# ── 3. Dataset ────────────────────────────────────────────────────────────────
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

class F1RadioDataset(Dataset):
    def __init__(self, df):
        self.texts    = df['clean_text'].tolist()
        self.acoustic = df[ACOUSTIC_COLS].values.astype(np.float32)
        self.labels   = df['label_id'].values.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], max_length=MAX_LEN,
                        padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'acoustic':       torch.tensor(self.acoustic[idx]),
            'label':          torch.tensor(self.labels[idx])
        }

train_loader = DataLoader(F1RadioDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(F1RadioDataset(val_df),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(F1RadioDataset(test_df),  batch_size=BATCH_SIZE, shuffle=False)
print(f'Train batches: {len(train_loader)}  Val batches: {len(val_loader)}')

# ── 4. Model (BERT frozen) ────────────────────────────────────────────────────
class F1EmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False

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
        cls_emb      = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        acoustic_emb = self.acoustic_mlp(acoustic)
        return self.classifier(torch.cat([cls_emb, acoustic_emb], dim=1))

model = F1EmotionModel().to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Params: {total:,} total / {trainable:,} trainable (BERT frozen)')

# ── 5. Loss & optimiser ───────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01
)

# ── 6. Train/eval loop ────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            ids   = batch['input_ids'].to(DEVICE)
            mask  = batch['attention_mask'].to(DEVICE)
            acou  = batch['acoustic'].to(DEVICE)
            lbls  = batch['label'].to(DEVICE)
            logits = model(ids, mask, acou)
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

# ── 7. Training loop with early stopping ─────────────────────────────────────
best_val_f1     = 0.0
patience_count  = 0
best_model_path = os.path.join(BASE, 'models', 'best_model.pt')
history         = []

print()
print(f"{'Epoch':<6} {'Train Loss':<12} {'Train F1':<10} {'Val Loss':<12} {'Val F1':<10} Status")
print('-' * 65)

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, train=True)
    vl_loss, vl_f1, _, _ = run_epoch(val_loader,   train=False)
    history.append({'epoch': epoch, 'train_loss': tr_loss, 'train_f1': tr_f1,
                    'val_loss': vl_loss, 'val_f1': vl_f1})

    if vl_f1 > best_val_f1:
        best_val_f1 = vl_f1
        torch.save(model.state_dict(), best_model_path)
        status = 'saved'
        patience_count = 0
    else:
        patience_count += 1
        status = f'patience {patience_count}/{PATIENCE}'

    print(f"{epoch:<6} {tr_loss:<12.4f} {tr_f1:<10.4f} {vl_loss:<12.4f} {vl_f1:<10.4f} {status}")

    if patience_count >= PATIENCE:
        print(f'\nEarly stop at epoch {epoch}.')
        break

print(f'\nBest val macro F1: {best_val_f1:.4f}')

# ── 8. Test evaluation ────────────────────────────────────────────────────────
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
te_loss, te_f1, te_labels, te_preds = run_epoch(test_loader, train=False)

print(f'\nTest Loss:     {te_loss:.4f}')
print(f'Test Macro F1: {te_f1:.4f}')
print()
print(classification_report(te_labels, te_preds, target_names=LABEL_ORDER, zero_division=0))

# Save history
pd.DataFrame(history).to_csv(os.path.join(BASE, 'models', 'history.csv'), index=False)
print('Done. history saved to models/history.csv')
