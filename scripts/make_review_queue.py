import pandas as pd

labels = pd.read_csv('annotations/labels_clean.csv')
text   = pd.read_csv('annotations/preprocessed_text_clean.csv')

flagged = labels[labels['flagged_for_review'] == True].copy()
merged  = flagged.merge(text[['clip_id','raw_text','clean_text','word_count']], on='clip_id', how='left')

review = merged[['clip_id','final_label','confidence','text_label','text_confidence',
                 'acoustic_label','acoustic_confidence','word_count','raw_text']].copy()
review.columns = ['clip_id','auto_label','confidence','text_model_label','text_conf',
                  'acoustic_label','acoustic_conf','word_count','transcript']
review.insert(1, 'your_label', '')

review.to_csv('annotations/review_queue.csv', index=False)
print(f'Saved {len(review)} clips to annotations/review_queue.csv')
print()

print('Auto-label breakdown of flagged clips:')
print(review['auto_label'].value_counts().to_string())
print()
print('Sample clips:')
for _, r in review.head(15).iterrows():
    print(f"  {r['clip_id']:<15} auto={r['auto_label']:<12} conf={r['confidence']:.2f}  '{str(r['transcript'])[:65]}'")
