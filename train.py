# ==========================================
# GPT-2 + ML Sentiment Controlled Generator (FINAL FIXED + SAVING)
# ==========================================

import torch
import re
import os
import pickle
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# =========================
# CREATE FOLDERS (IMPORTANT)
# =========================
os.makedirs("model", exist_ok=True)
os.makedirs("feedback", exist_ok=True)

# create empty feedback file if not exists
if not os.path.exists("feedback/feedback.json"):
    with open("feedback/feedback.json", "w") as f:
        f.write("[]")

# =========================
# CONFIG
# =========================
MODEL_NAME = "gpt2"
MAX_LEN = 128
TRAIN_SAMPLES = 10000
EPOCHS = 3
BATCH_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# LOAD DATA
# =========================
print("\nLoading dataset...")
dataset = load_dataset("imdb")
train_data = dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))

texts = [clean_text(x["text"]) for x in train_data]
labels = [x["label"] for x in train_data]

# =========================
# TRAIN SENTIMENT MODEL
# =========================
print("\nTraining sentiment classifier...")

vectorizer = CountVectorizer(max_features=3000)
X_vec = vectorizer.fit_transform(texts)

clf = LogisticRegression(max_iter=200)
clf.fit(X_vec, labels)

def sentiment_score(sentence):
    vec = vectorizer.transform([sentence])
    return clf.predict_proba(vec)[0][1]

# =========================
# SAVE SENTIMENT MODEL 🔥
# =========================
with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump((clf, vectorizer), f)

print("✅ sentiment_model.pkl saved")

# =========================
# ADD SENTIMENT TOKENS
# =========================
def add_sentiment(example):
    label = "positive" if example["label"] == 1 else "negative"
    text = clean_text(example["text"])
    example["text"] = f"<{label}> {text}"
    return example

train_data = train_data.map(add_sentiment)

# =========================
# TOKENIZER
# =========================
print("\nLoading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

tokenizer.add_special_tokens({
    "additional_special_tokens": ["<positive>", "<negative>"]
})

tokenizer.pad_token = tokenizer.eos_token

# =========================
# TOKENIZATION
# =========================
def tokenize(example):
    encoding = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

print("\nTokenizing dataset...")
tokenized_data = train_data.map(tokenize, batched=True)

tokenized_data.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# =========================
# MODEL
# =========================
print("\nLoading model...")
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# =========================
# TRAINING
# =========================
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=5e-5,
    logging_steps=100,
    save_steps=500,
    fp16=torch.cuda.is_available(),
    report_to="none",
    optim="adamw_torch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
)

print("\nTraining started...\n")
trainer.train()

# =========================
# GENERATION
# =========================
def generate_with_sentiment(prompt, sentiment="positive"):
    candidates = []

    for _ in range(20):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        if sentiment == "positive":
            bad_words = [
                tokenizer.encode("bad"),
                tokenizer.encode("terrible"),
                tokenizer.encode("awful"),
                tokenizer.encode("worst")
            ]
        else:
            bad_words = [
                tokenizer.encode("good"),
                tokenizer.encode("great"),
                tokenizer.encode("amazing"),
                tokenizer.encode("excellent")
            ]

        output = model.generate(
            **inputs,
            max_length=50,
            do_sample=True,
            top_k=40,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            bad_words_ids=bad_words
        )

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        text = clean_text(text)
        candidates.append(text)

    def score(text):
        s = sentiment_score(text)

        if sentiment == "positive" and s < 0.6:
            return -1
        if sentiment == "negative" and s > 0.4:
            return -1

        return s if sentiment == "positive" else (1 - s)

    best = max(candidates, key=score)

    if score(best) == -1:
        return "Retry generation"

    return best

# =========================
# TEST OUTPUT
# =========================
print("\n========== GENERATED REVIEWS ==========\n")

print("Positive Reviews:\n")
for _ in range(3):
    print(generate_with_sentiment("<positive> This movie was amazing because", "positive"))
    print()

print("Negative Reviews:\n")
for _ in range(3):
    print(generate_with_sentiment("<negative> This movie was terrible because", "negative"))
   