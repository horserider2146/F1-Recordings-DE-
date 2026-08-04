"""
Rebuild stratified train/val/test splits from labels_clean.csv + preprocessed_text_clean.csv.
Split: 70% train / 15% val / 15% test (stratified by final_label).
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANNOTATIONS_DIR

LABEL_ORDER = ["Calm", "Frustrated", "High Stress"]
LABEL_ID    = {l: i for i, l in enumerate(LABEL_ORDER)}
SPLITS_DIR  = Path("data/splits")
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

import json

labels = pd.read_csv(ANNOTATIONS_DIR / "labels_clean.csv")
text   = pd.read_csv(ANNOTATIONS_DIR / "preprocessed_text_clean.csv")

# Expand acoustic_features JSON column into individual columns
acoustic_expanded = labels["acoustic_features"].apply(
    lambda x: json.loads(x) if pd.notna(x) and x else {}
)
labels = pd.concat([labels.drop(columns=["acoustic_features"]),
                    pd.json_normalize(acoustic_expanded)], axis=1)

df = labels.merge(text[["clip_id", "clean_text", "word_count"]], on="clip_id", how="left")
df["label_id"] = df["final_label"].map(LABEL_ID)

# 70 / 15 / 15 stratified split
train, temp = train_test_split(df, test_size=0.30, stratify=df["final_label"], random_state=42)
val,   test = train_test_split(temp, test_size=0.50, stratify=temp["final_label"], random_state=42)

for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
    split_df.to_csv(SPLITS_DIR / f"{split_name}.csv", index=False)

print("Splits rebuilt")
print()
print(f"{'Split':<8} {'Total':>6}  {'Calm':>6}  {'Frustrated':>10}  {'High Stress':>11}")
print("-" * 50)
for name, split in [("train", train), ("val", val), ("test", test)]:
    dist = split["final_label"].value_counts()
    print(f"{name:<8} {len(split):>6}  {dist.get('Calm',0):>6}  {dist.get('Frustrated',0):>10}  {dist.get('High Stress',0):>11}")
print()
print("Saved to data/splits/")
