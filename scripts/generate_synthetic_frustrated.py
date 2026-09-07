"""
scripts/generate_synthetic_frustrated.py

Generates synthetic frustrated F1 driver radio transcripts using Claude.
Saves ~300 unique clips to data/synthetic_frustrated.csv.

Requirements:
    pip install anthropic
    Set ANTHROPIC_API_KEY in your environment before running.

Run time: ~2 minutes (10 API calls).
"""

import os
import sys
import json
import time
import pandas as pd

try:
    import anthropic
except ImportError:
    print('ERROR: anthropic package not installed. Run: pip install anthropic')
    sys.exit(1)

BASE     = r'C:\Users\Ritarshi Roy\OneDrive\Desktop\Projects\F1 Recordings'
OUT_FILE = os.path.join(BASE, 'data', 'synthetic_frustrated.csv')

# 10 focus areas × 32 per batch = 320 clips before filtering
TARGET_PER_FOCUS = 32

FOCUS_AREAS = [
    "severe understeer — the car won't rotate through corners, front end pushing off exits, losing traction",
    "team orders — driver ordered to hold position or let teammate past, driver disagrees with the call",
    "backmarker traffic — losing lap time, blue flags not being waved in time, being blocked by lapped cars",
    "DRS or ERS not working — system failing to activate, losing significant straight-line speed each lap",
    "strategy disputes — questioning the tire compound chosen, disagreeing with pit timing or fuel targets",
    "tire temperature and degradation — fronts or rears overheating or graining, outside the working window",
    "safety car and virtual safety car timing — driver feels the team got the call wrong and lost positions",
    "racing incident frustration — being pushed wide on track, losing positions unfairly, contact damage",
    "brake problems — balance off, overheating, locking up repeatedly, pedal feeling inconsistent",
    "car simply not fast enough — setup wrong, no grip, losing time in specific corners, asking for changes",
]

SYSTEM = """\
You are an expert Formula 1 team radio transcription specialist. Your task is to generate \
realistic, authentic frustrated F1 driver radio clips — exactly as a real driver would say \
them over the radio during a race or qualifying.

The three emotion categories in our dataset are:
  - Calm: normal strategy calls, routine updates, straightforward exchanges
  - Frustrated: driver complaints, disputes with the engineer, mechanical anger, strategy disagreements
  - High Stress: incidents, safety cars, close wheel-to-wheel racing, accidents

You are generating FRUSTRATED clips only. These are clipped, direct expressions of frustration — \
not generic anger, not panic (that would be High Stress).

Style rules:
  - First person: the driver is speaking to their engineer
  - Short and clipped: 6 to 22 words per clip (real radio transcriptions are never long essays)
  - Use genuine F1 vocabulary: understeer, push, snap, deg, graining, blistering, ERS, DRS, \
    SC, VSC, stint, delta, window, undercut, overcut, tyre-limited, front-limited, rear-limited, \
    balance, overheating, marbles, warm-up lap, out-lap, in-lap, sector time, gap, margin
  - Frustration is implied through word choice and urgency — never state "I'm frustrated"
  - No profanity — F1 radio is broadcast and transcribed professionally
  - Vary the structure: questions, commands, statements, responses to engineer instructions
  - Every clip must be unique — no two clips should be near-duplicates
  - Include a mix: some clips are subtle (measured frustration), some more direct
  - Real examples of good clips (for tone reference only — do NOT copy these):
      "The DRS isn't opening, I'm losing two tenths on every straight"
      "Why are we holding position? We had the pace on those tires"
      "Blue flags! He's not moving, I'm losing time here"
      "The fronts are completely gone, I'm understeering off every exit"
      "That was the wrong call, we should have pitted under the safety car"
      "Copy, but I completely disagree with that strategy"
      "These brakes are on fire, the balance is all wrong"
      "I can't keep up in the high-speed stuff, there's no front grip"
"""


def generate_batch(client, focus, n):
    prompt = (
        f"Generate exactly {n} frustrated F1 driver radio clips.\n"
        f"Focus area for this batch: {focus}.\n\n"
        f"Return ONLY a valid JSON array of strings. No explanation, no numbering, "
        f"no markdown formatting. Just the raw JSON array.\n"
        f"Example format: [\"clip one here\", \"clip two here\", \"clip three here\"]"
    )

    message = client.messages.create(
        model='claude-sonnet-4-6',
        max_tokens=2048,
        temperature=1.0,
        system=SYSTEM,
        messages=[{'role': 'user', 'content': prompt}]
    )

    raw = message.content[0].text.strip()

    # Strip markdown code fences if the model adds them
    if raw.startswith('```'):
        lines = raw.split('\n')
        raw = '\n'.join(lines[1:])
        if raw.endswith('```'):
            raw = raw[:-3].strip()

    clips = json.loads(raw)
    return [c.strip() for c in clips if isinstance(c, str) and c.strip()]


def is_valid(text):
    words = text.split()
    if len(words) < 5 or len(words) > 30:
        return False
    # Must contain at least some alphabetical content
    if sum(c.isalpha() for c in text) < 10:
        return False
    return True


def main():
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print('ERROR: ANTHROPIC_API_KEY environment variable is not set.')
        print('Set it with: $env:ANTHROPIC_API_KEY = "sk-ant-..."')
        sys.exit(1)

    os.makedirs(os.path.join(BASE, 'data'), exist_ok=True)
    client = anthropic.Anthropic(api_key=api_key)

    all_clips = []
    seen = set()

    print('Generating synthetic frustrated F1 radio clips...')
    print(f'Model: claude-sonnet-4-6')
    print(f'Plan: {TARGET_PER_FOCUS} clips × {len(FOCUS_AREAS)} focus areas\n')

    for i, focus in enumerate(FOCUS_AREAS):
        short_focus = focus[:65] + ('...' if len(focus) > 65 else '')
        print(f'[{i+1}/{len(FOCUS_AREAS)}] {short_focus}')

        try:
            batch = generate_batch(client, focus, TARGET_PER_FOCUS)
        except Exception as e:
            print(f'  ERROR generating batch: {e}')
            batch = []

        added = 0
        for clip in batch:
            key = clip.lower().strip()
            if is_valid(clip) and key not in seen:
                seen.add(key)
                all_clips.append(clip)
                added += 1

        print(f'  +{added} clips (running total: {len(all_clips)})')

        # Brief pause between calls to avoid rate limits
        if i < len(FOCUS_AREAS) - 1:
            time.sleep(1.5)

    print(f'\nTotal unique valid clips generated: {len(all_clips)}')

    if len(all_clips) < 100:
        print('WARNING: fewer than 100 clips generated — check API key and retry.')
        sys.exit(1)

    # Build DataFrame with just the columns the training script needs
    rows = []
    for j, text in enumerate(all_clips):
        rows.append({
            'clip_id':    f'synth_{j+1:04d}',
            'clean_text': text,
            'label_id':   1,
            'label':      'Frustrated',
            'word_count': len(text.split()),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)
    print(f'Saved {len(df)} clips to {OUT_FILE}')

    print('\n--- Sample clips (random 12) ---')
    for _, row in df.sample(min(12, len(df)), random_state=7).iterrows():
        print(f'  [{row["clip_id"]}] {row["clean_text"]}')

    print(f'\nNext step: run python train_focal_synth.py')


if __name__ == '__main__':
    main()
