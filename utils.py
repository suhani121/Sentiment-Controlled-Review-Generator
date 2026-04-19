import re
import json
import os
from collections import Counter

# BUG FIX: Removed unused `import torch` (was only needed for perplexity, now removed)

FEEDBACK_FILE = "feedback/feedback.json"

def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return []
    with open(FEEDBACK_FILE, "r") as f:
        return json.load(f)

def save_feedback(text, sentiment, rating):
    data = load_feedback()
    data.append({
        "text": text,
        "sentiment": sentiment,
        "rating": rating
    })
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)

def feedback_score(text):
    data = load_feedback()
    return sum(d["rating"] for d in data if d["text"] == text)

def diversity(text):
    words = text.split()
    return len(set(words)) / len(words) if words else 0

# BUG FIX: perplexity() function removed entirely as requested.
# It was also causing a crash in app.py because it was imported but
# the import itself would fail if torch wasn't available or just
# caused unnecessary overhead.

def get_feedback_stats():
    data = load_feedback()
    total = len(data)
    positive = sum(1 for d in data if d["rating"] == 1)
    negative = sum(1 for d in data if d["rating"] == -1)
    sentiment_count = Counter(d["sentiment"] for d in data)
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "sentiment_dist": sentiment_count
    }