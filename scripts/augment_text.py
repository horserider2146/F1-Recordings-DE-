"""
Text augmentation for minority classes (Frustrated, High Stress).
Uses EDA techniques: synonym replacement and random deletion.
Only applied to the training split — val/test are never touched.
"""

import random
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

LABEL_ORDER = ['Calm', 'Frustrated', 'High Stress']

# How many clips each class should have after augmentation.
# Calm stays as-is (it's the majority); minority classes are upsampled.
AUG_TARGETS = {
    'Calm':        None,   # do not augment
    'Frustrated':  300,    # 143 → 300
    'High Stress': 430,    # 271 → 430
}


def _synonym_replace(text, n=2):
    words = text.split()
    if not words:
        return text
    new_words = words.copy()
    candidates = [i for i, w in enumerate(words) if wordnet.synsets(w)]
    random.shuffle(candidates)
    replaced = 0
    for i in candidates:
        if replaced >= n:
            break
        syns = wordnet.synsets(words[i])
        lemmas = [
            l.name().replace('_', ' ')
            for s in syns for l in s.lemmas()
            if l.name().lower() != words[i].lower()
        ]
        if lemmas:
            new_words[i] = random.choice(lemmas)
            replaced += 1
    return ' '.join(new_words)


def _random_deletion(text, p=0.10):
    words = text.split()
    if len(words) <= 1:
        return text
    kept = [w for w in words if random.random() > p]
    return ' '.join(kept) if kept else words[0]


def augment_one(text):
    """Apply one random EDA transform to a single text string."""
    if random.random() < 0.5:
        return _synonym_replace(text, n=2)
    return _random_deletion(text, p=0.10)


def augment_dataframe(df, random_seed=42):
    """
    Takes the training DataFrame and returns an augmented copy.
    Minority class rows are duplicated with augmented text until
    each class reaches its AUG_TARGET count.
    The original rows are always kept unchanged.
    """
    import pandas as pd
    random.seed(random_seed)

    label_map = {i: l for i, l in enumerate(LABEL_ORDER)}
    extras = []

    for label_id, label_name in label_map.items():
        target = AUG_TARGETS.get(label_name)
        if target is None:
            continue

        subset = df[df['label_id'] == label_id].copy()
        current = len(subset)
        needed  = max(0, target - current)

        if needed == 0:
            continue

        # Sample with replacement and augment text
        sampled = subset.sample(n=needed, replace=True, random_state=random_seed)
        sampled = sampled.copy()
        sampled['clean_text'] = sampled['clean_text'].apply(augment_one)
        extras.append(sampled)

    if extras:
        import pandas as pd
        augmented = pd.concat([df] + extras, ignore_index=True)
    else:
        augmented = df.copy()

    augmented = augmented.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    counts = augmented['label_id'].value_counts().sort_index()
    print('Augmented training set:')
    for lid, lname in label_map.items():
        print(f'  {lname}: {counts.get(lid, 0)}')
    print(f'  Total: {len(augmented)}')

    return augmented
