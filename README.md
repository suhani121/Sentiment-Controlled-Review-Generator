# 🎬 Sentiment-Controlled Review Generator

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

**An end-to-end AI system that generates movie reviews with controlled sentiment using GPT-2,  
enhanced with ML-based sentiment reranking, user feedback learning, and an analytics dashboard.**

[Features](#-key-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Dashboard](#-analytics-dashboard) • [Roadmap](#-future-improvements)

</div>

---

## 🚀 Overview

This project combines **Generative AI + Machine Learning + Feedback Loop** to produce high-quality, sentiment-aligned text.

Unlike basic generators, this system:
- Generates **multiple candidate outputs** simultaneously
- Filters them using a **trained sentiment classifier**
- **Improves over time** using real user feedback

---

## 🧠 Key Features

| Feature | Description |
|--------|-------------|
| 🎯 **Sentiment-Controlled Generation** | Generate strictly **positive** or **negative** reviews on demand |
| 🤖 **GPT-2 Based Model** | Fine-tuned GPT-2 with special tokens `<positive>` and `<negative>` |
| 📊 **ML Sentiment Reranking** | Logistic Regression + CountVectorizer ensures output matches target sentiment |
| 🔁 **Feedback Loop (RLHF-lite)** | Users give 👍 / 👎 — feedback is stored and reused to improve results |
| 📈 **Analytics Dashboard** | Tracks feedback trends, sentiment distribution, and performance insights |
| 🧪 **Evaluation Metrics** | Sentiment Score, Perplexity, and Diversity reported per generation |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      USER INPUT                         │
│              (Movie name + Sentiment choice)            │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│               GPT-2 GENERATOR                           │
│        Fine-tuned with <positive> / <negative>          │
│             Generates N candidate reviews               │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│            ML SENTIMENT CLASSIFIER                      │
│        Logistic Regression + CountVectorizer            │
│         Scores each candidate for alignment             │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│           FEEDBACK-BASED RERANKING                      │
│     Historical 👍 / 👎 data boosts/penalizes outputs    │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   BEST OUTPUT                           │
│          Displayed with evaluation metrics              │
└───────────────────────────┬─────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │       USER FEEDBACK        │
              │         👍  /  👎          │
              └─────────────┬──────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │    feedback/feedback.json   │
              │  Stored → Improves system   │
              └─────────────────────────────┘
```

---

## 📁 Project Structure

```
sentiment-controlled-review-generator/
│
├── 📄 app.py                        # Streamlit UI — main entry point
├── 📄 utils.py                      # Helper functions
├── 📄 requirements.txt              # Python dependencies
│
├── 📂 model/
│   ├── 📂 gpt2_sentiment_model/     # Fine-tuned GPT-2 weights & config
│   └── 📄 sentiment_model.pkl       # Trained ML classifier (Logistic Regression)
│
└── 📂 feedback/
    └── 📄 feedback.json             # Persistent user feedback storage
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/sentiment-controlled-review-generator.git
cd sentiment-controlled-review-generator
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the app

```bash
streamlit run app.py
```

---

## 💻 Usage

```
1. 🎭  Select sentiment        →  Positive or Negative
2. 🎬  Enter a movie / topic   →  e.g., "Inception", "a sci-fi thriller"
3. ⚡  Click "Generate Review" →  System generates & reranks candidates
4. 📋  View output + metrics   →  Sentiment score, perplexity, diversity
5. 🗳️  Give feedback           →  👍 or 👎 to improve future outputs
```

---

## 🧪 Example Output

**Input:** `Inception` | Sentiment: `Positive`

```
✅ POSITIVE REVIEW
──────────────────────────────────────────────────────────
"This movie was amazing — the story was deeply engaging and
the performances were nothing short of outstanding. Every
frame felt intentional and the ending left me speechless."

📊 Sentiment Score : 0.94  |  Perplexity: 32.1  |  Diversity: 0.87
```

---

**Input:** `Inception` | Sentiment: `Negative`

```
❌ NEGATIVE REVIEW
──────────────────────────────────────────────────────────
"This movie was terrible — the plot was needlessly convoluted
and the pacing made it hard to stay engaged. The acting
lacked emotional depth and the ending felt unearned."

📊 Sentiment Score : 0.11  |  Perplexity: 38.4  |  Diversity: 0.81
```

---

## 📊 Analytics Dashboard

The built-in dashboard tracks:

```
┌────────────────────────────────────────────┐
│  📦 Total Feedback Collected               │
│  ✅ Positive vs ❌ Negative Ratio          │
│  📈 Feedback Trends Over Time              │
│  🗂️  Recent Generated Outputs              │
└────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language Model** | GPT-2 (HuggingFace Transformers) |
| **Deep Learning** | PyTorch |
| **Sentiment Classifier** | Scikit-learn (Logistic Regression + CountVectorizer) |
| **UI / Frontend** | Streamlit |
| **Training Dataset** | IMDB Movie Reviews |
| **Feedback Storage** | JSON (local) |

---

## 🚀 Future Improvements

- [ ] 🌐 Deploy on **HuggingFace Spaces**
- [ ] 🎭 Add **multi-emotion support** (angry, excited, neutral, etc.)
- [ ] 🧠 Upgrade to **GPT-2 Medium / LLaMA** for richer outputs
- [ ] 🗄️ Migrate feedback storage to **MongoDB / Firebase**
- [ ] 💅 Improve UI with custom theming and animations
- [ ] 📤 Add **export to PDF / CSV** for generated reviews

---

## 🎯 Use Cases

- 🤖 AI content generation pipelines
- 🎬 Movie review simulation & testing
- 💬 Sentiment-aware chatbot prototyping
- ✍️ Tone-controlled writing assistants

---


