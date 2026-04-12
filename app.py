import streamlit as st
import torch
import pickle
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import (
    clean_text, feedback_score, save_feedback,
    diversity, perplexity, load_feedback, get_feedback_stats
)

# =========================
# LOAD MODEL
# =========================
model_path = "model/gpt2_sentiment_model"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# =========================
# LOAD SENTIMENT MODEL
# =========================
with open("model/sentiment_model.pkl", "rb") as f:
    clf, vectorizer = pickle.load(f)

def sentiment_score(text):
    vec = vectorizer.transform([text])
    return clf.predict_proba(vec)[0][1]

# =========================
# GENERATION FUNCTION
# =========================
def generate(prompt, sentiment):
    candidates = []

    for _ in range(10):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output = model.generate(
            **inputs,
            max_length=50,
            do_sample=True,
            top_k=40,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3
        )

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        text = clean_text(text)
        candidates.append(text)

    def score(text):
        s = sentiment_score(text)
        fb = feedback_score(text)
        base = s if sentiment == "positive" else (1 - s)
        return base + 0.3 * fb

    return max(candidates, key=score)

# =========================
# UI TABS
# =========================
st.set_page_config(page_title="AI Review Generator", layout="centered")

tab1, tab2 = st.tabs(["🎬 Generate", "📊 Analytics"])

# =========================
# TAB 1: GENERATION
# =========================
with tab1:
    st.title("🎬 AI Movie Review Generator")

    sentiment = st.radio("Select Sentiment", ["positive", "negative"])
    user_input = st.text_input("Enter movie/topic", "This movie")

    if st.button("Generate Review"):
        prompt = (
            f"<positive> {user_input} was amazing because"
            if sentiment == "positive"
            else f"<negative> {user_input} was terrible because"
        )

        result = generate(prompt, sentiment)

        st.success(result)

        # Metrics
        st.subheader("📊 Metrics")
        st.write("Sentiment Score:", round(sentiment_score(result), 3))
        st.write("Diversity:", round(diversity(result), 3))
        st.write("Perplexity:", round(perplexity(result, model, tokenizer, device), 2))

        # Feedback
        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Good"):
                save_feedback(result, sentiment, 1)
                st.success("Saved!")

        with col2:
            if st.button("👎 Bad"):
                save_feedback(result, sentiment, -1)
                st.warning("Saved!")

# =========================
# TAB 2: ANALYTICS
# =========================
with tab2:
    st.title("📊 Analytics Dashboard")

    data = load_feedback()

    if len(data) == 0:
        st.warning("No feedback yet!")
    else:
        df = pd.DataFrame(data)

        stats = get_feedback_stats()

        # 🔥 KPIs
        st.metric("Total Feedback", stats["total"])
        st.metric("👍 Positive", stats["positive"])
        st.metric("👎 Negative", stats["negative"])

        # 🔥 PIE CHART
        st.subheader("Sentiment Distribution")
        st.bar_chart(df["sentiment"].value_counts())

        # 🔥 FEEDBACK TREND
        st.subheader("Feedback Trend")
        df["index"] = range(len(df))
        st.line_chart(df["rating"])

        # 🔥 RAW DATA
        st.subheader("Recent Feedback")
        st.dataframe(df.tail(10))