import os, pickle
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

BASE          = r'C:\Users\Ritarshi Roy\OneDrive\Desktop\Projects\F1 Recordings'
DEVICE        = torch.device('cpu')
MAX_LEN       = 128
BATCH_SIZE    = 16
LABEL_ORDER   = ['Calm', 'Frustrated', 'High Stress']
NUM_CLASSES   = 3
ACOUSTIC_COLS = ['mean_pitch', 'pitch_range', 'pitch_std']
DROPOUT       = 0.5
MODEL_PATH    = os.path.join(BASE, 'models', 'best_model_unfreeze.pt')
SCALER_PATH   = os.path.join(BASE, 'models', 'scaler_unfreeze.pkl')

class F1EmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.transformer.layer[-1].parameters():
            param.requires_grad = True
        self.acoustic_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 256), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask, acoustic):
        cls_emb      = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        acoustic_emb = self.acoustic_mlp(acoustic)
        return self.classifier(torch.cat([cls_emb, acoustic_emb], dim=1))

class F1RadioDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts    = df['clean_text'].tolist()
        self.acoustic = df[ACOUSTIC_COLS].values.astype(np.float32)
        self.labels   = df['label_id'].values.astype(np.int64)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=MAX_LEN,
                             padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'acoustic':       torch.tensor(self.acoustic[idx]),
            'label':          torch.tensor(self.labels[idx])
        }

print('Loading data...')
test_df = pd.read_csv(os.path.join(BASE, 'data', 'splits', 'test.csv'))
test_df['clean_text'] = test_df['clean_text'].fillna('')

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
test_df[ACOUSTIC_COLS] = scaler.transform(test_df[ACOUSTIC_COLS])

tokenizer   = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
test_loader = DataLoader(F1RadioDataset(test_df, tokenizer), batch_size=BATCH_SIZE, shuffle=False)

print('Loading model...')
model = F1EmotionModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print('Model loaded. Running test eval...')

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        logits = model(batch['input_ids'].to(DEVICE),
                       batch['attention_mask'].to(DEVICE),
                       batch['acoustic'].to(DEVICE))
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(batch['label'].cpu().numpy())

f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
print(f'\nTest Macro F1 (unfreeze): {f1:.4f}')
print(f'Test Macro F1 (frozen baseline): 0.4669')
print(f'Delta: {f1 - 0.4669:+.4f}')
print()
print(classification_report(all_labels, all_preds, target_names=LABEL_ORDER, zero_division=0))
