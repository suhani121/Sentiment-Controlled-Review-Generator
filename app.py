import streamlit as st
import torch
import pickle
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import (
    clean_text, feedback_score, save_feedback,
    diversity, load_feedback, get_feedback_stats
)


st.set_page_config(page_title="Sentiment Controlled Review Generator", layout="centered")


# SESSION STATE INIT
if "result" not in st.session_state:
    st.session_state.result = None
if "result_sentiment" not in st.session_state:
    st.session_state.result_sentiment = None
if "feedback_msg" not in st.session_state:
    st.session_state.feedback_msg = None


# LOAD GPT-2 MODEL

@st.cache_resource
def load_gpt2():
    model_path = "model/gpt2_sentiment_model"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_gpt2()


# LOAD SENTIMENT MODEL

@st.cache_resource
def load_sentiment_model():
    with open("model/sentiment_model.pkl", "rb") as f:
        clf, vectorizer = pickle.load(f)
    return clf, vectorizer

clf, vectorizer = load_sentiment_model()

def sentiment_score(text):
    vec = vectorizer.transform([text])
    return clf.predict_proba(vec)[0][1]


# GENERATION FUNCTION
def generate(prompt, sentiment):
    candidates = []

    for _ in range(10):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output = model.generate(
            **inputs,
            max_length=100,
            do_sample=True,
            top_k=40,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id  # BUG FIX: suppresses pad warning
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


# UI TABS

tab1, tab2 = st.tabs(["🎬 Generate", "📊 Analytics"])


# TAB 1: GENERATION

with tab1:
    st.title("🎬 Sentiment Controlled Review Generator")

    sentiment = st.radio("Select Sentiment", ["positive", "negative"])
    user_input = st.text_input("Enter movie/topic", "This movie")

    if st.button("Generate Review"):
        prompt = (
            f"<positive> {user_input} was amazing because"
            if sentiment == "positive"
            else f"<negative> {user_input} was terrible because"
        )

        with st.spinner("Generating review..."):
            result = generate(prompt, sentiment)

        
        st.session_state.result = result
        st.session_state.result_sentiment = sentiment
        st.session_state.feedback_msg = None  # reset old feedback message

    
    if st.session_state.result:
        result = st.session_state.result
        result_sentiment = st.session_state.result_sentiment

        st.success(result)

        st.subheader("📊 Metrics")
        st.write("Sentiment Score:", round(sentiment_score(result), 3))
        st.write("Diversity:", round(diversity(result), 3))

        st.subheader("Was this review good?")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Good"):
                save_feedback(result, result_sentiment, 1)
                st.session_state.feedback_msg = "✅ Positive feedback saved!"

        with col2:
            if st.button("👎 Bad"):
                save_feedback(result, result_sentiment, -1)
                st.session_state.feedback_msg = "⚠️ Negative feedback saved!"

        if st.session_state.feedback_msg:
            st.info(st.session_state.feedback_msg)


with tab2:
    st.title("📊 Analytics Dashboard")

    data = load_feedback()

    if len(data) == 0:
        st.warning("No feedback yet! Generate some reviews and rate them.")
    else:
        df = pd.DataFrame(data)
        stats = get_feedback_stats()

        # KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Feedback", stats["total"])
        col2.metric("👍 Positive", stats["positive"])
        col3.metric("👎 Negative", stats["negative"])

        # Sentiment Distribution
        st.subheader("Sentiment Distribution")
        st.bar_chart(df["sentiment"].value_counts())

        # Feedback Trend
        st.subheader("Feedback Trend")
        df["index"] = range(len(df))
        st.line_chart(df.set_index("index")["rating"])

        # Recent Feedback Table
        st.subheader("Recent Feedback")
        st.dataframe(df[["text", "sentiment", "rating"]].tail(10))