import re
import json
import os
import torch
from collections import Counter

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

def perplexity(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    return torch.exp(loss).item()

# 🔥 ANALYTICS FUNCTIONS

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