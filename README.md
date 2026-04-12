# 🎬 Sentiment-Controlled Review Generator

An end-to-end AI system that generates **movie reviews with controlled sentiment (positive/negative)** using GPT-2, enhanced with **ML-based sentiment reranking, user feedback learning, and an analytics dashboard**.

---

## 🚀 Overview

This project combines **Generative AI + Machine Learning + Feedback Loop** to produce high-quality, sentiment-aligned text.

Unlike basic generators, this system:
- Generates multiple outputs  
- Filters them using a sentiment classifier  
- Improves over time using user feedback  

---

## 🧠 Key Features

### ✅ Sentiment-Controlled Generation
- Generate strictly **positive** or **negative** reviews  

### ✅ GPT-2 Based Model
- Fine-tuned GPT-2  
- Uses special tokens: `<positive>`, `<negative>`  

### ✅ ML Sentiment Reranking
- Logistic Regression + CountVectorizer  
- Ensures output matches sentiment  

### ✅ Feedback Loop (RLHF-lite)
- Users give 👍 / 👎  
- Feedback stored and reused  

### ✅ Analytics Dashboard 📊
- Feedback trends  
- Sentiment distribution  
- Performance insights  

### ✅ Evaluation Metrics
- Sentiment Score  
- Perplexity  
- Diversity  

---

## 🏗️ Architecture
User Input
↓
GPT-2 Generator (multiple outputs)
↓
Sentiment Classifier (ML)
↓
Feedback-based Reranking
↓
Best Output
↓
User Feedback → Stored → Improves system
