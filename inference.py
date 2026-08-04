"""
inference.py  —  Predict emotion from a .wav file or raw text.

Default model: best_model_focal.pt (67% accuracy, macro F1=0.591)

Usage examples:
    python inference.py --audio data/cleaned/clip_0001.wav
    python inference.py --text "Box box box, come in now!"
    python inference.py --audio data/cleaned/clip_0001.wav --text "Box box box"
    python inference.py --audio data/cleaned/clip_0001.wav --model models/best_model_focal.pt
"""

import argparse, json, os, pickle, sys
import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertModel

BASE = r'C:\Users\Ritarshi Roy\OneDrive\Desktop\Projects\F1 Recordings'

LABEL_ORDER   = ['Calm', 'Frustrated', 'High Stress']
NUM_CLASSES   = 3
MAX_LEN       = 128
ACOUSTIC_COLS = ['mean_pitch', 'pitch_range', 'pitch_std']
DROPOUT       = 0.4   # matches train_focal.py

# ── Model definition (must match train.py) ────────────────────────────────────
class F1EmotionModel(nn.Module):
    def __init__(self, unfreeze_last=False):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        if unfreeze_last:
            for param in self.bert.transformer.layer[-1].parameters():
                param.requires_grad = True

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


def extract_acoustic(wav_path):
    """Extract mean_pitch, pitch_range, pitch_std from a .wav file."""
    try:
        import librosa
        y, sr = librosa.load(wav_path, sr=None)
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) == 0:
            return [0.0, 0.0, 0.0]
        return [float(np.mean(f0_valid)),
                float(np.max(f0_valid) - np.min(f0_valid)),
                float(np.std(f0_valid))]
    except Exception as e:
        print(f'[warn] acoustic extraction failed ({e}), using zeros')
        return [0.0, 0.0, 0.0]


def preprocess_text(raw):
    """Minimal text cleaning matching scripts/05_text_preprocess.py."""
    import re
    text = raw.lower()
    text = re.sub(r"[^a-z0-9\s']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_model(model_path, device):
    unfreeze_last = 'unfreeze' in os.path.basename(model_path)
    model = F1EmotionModel(unfreeze_last=unfreeze_last).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(text, acoustic_raw, model, tokenizer, scaler, device):
    clean = preprocess_text(text)
    enc = tokenizer(clean, max_length=MAX_LEN, padding='max_length',
                    truncation=True, return_tensors='pt')
    input_ids   = enc['input_ids'].to(device)
    attn_mask   = enc['attention_mask'].to(device)

    acoustic_scaled = scaler.transform([acoustic_raw])
    acoustic_tensor = torch.tensor(acoustic_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(input_ids, attn_mask, acoustic_tensor)
        probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_label = LABEL_ORDER[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_label, confidence, probs


def main():
    parser = argparse.ArgumentParser(description='F1 Radio Emotion Inference')
    parser.add_argument('--audio', type=str, default=None, help='Path to .wav file')
    parser.add_argument('--text',  type=str, default=None, help='Radio transcript text')
    parser.add_argument('--model', type=str,
                        default=os.path.join(BASE, 'models', 'best_model_focal.pt'),
                        help='Path to model checkpoint')
    parser.add_argument('--scaler', type=str, default=None,
                        help='Path to scaler pickle (auto-detected from model name if omitted)')
    args = parser.parse_args()

    if args.audio is None and args.text is None:
        parser.error('Provide at least --audio or --text (or both).')

    # Auto-select matching scaler based on model name
    if args.scaler is None:
        name = os.path.basename(args.model)
        if 'focal' in name:
            args.scaler = os.path.join(BASE, 'models', 'scaler_focal.pkl')
        elif 'unfreeze' in name:
            args.scaler = os.path.join(BASE, 'models', 'scaler_unfreeze.pkl')
        else:
            args.scaler = os.path.join(BASE, 'models', 'scaler.pkl')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    print(f'Model  : {args.model}')

    with open(args.scaler, 'rb') as f:
        scaler = pickle.load(f)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model     = load_model(args.model, device)

    text         = args.text if args.text else ''
    acoustic_raw = extract_acoustic(args.audio) if args.audio else [0.0, 0.0, 0.0]

    if args.audio and not args.text:
        print('[info] No --text provided; text branch will use empty string.')

    pred_label, confidence, probs = predict(text, acoustic_raw, model, tokenizer, scaler, device)

    print()
    print('─' * 40)
    print(f'Prediction : {pred_label}')
    print(f'Confidence : {confidence:.1%}')
    print()
    for label, p in zip(LABEL_ORDER, probs):
        bar = '#' * int(p * 30)
        print(f'  {label:<12} {p:.1%}  {bar}')
    print('─' * 40)


if __name__ == '__main__':
    main()
