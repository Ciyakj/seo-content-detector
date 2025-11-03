# SEO Content & Duplicate Analyzer

This project is a complete data science pipeline. It includes a Jupyter Notebook for analysis and a fully functional Streamlit web app for real-time predictions.

## 🚀 Live Demo

**[Link to your deployed Streamlit App]** <-- *PASTE YOUR LIVE URL HERE*

## Overview

The goal was to build a system that can:
1.  Parse raw HTML to extract clean article text.
2.  Analyze the text for SEO quality using an advanced Machine Learning model.
3.  Detect near-duplicate content using TF-IDF and Cosine Similarity.

---

## 🛠️ Key Decisions & Features

* **Robust Parsing:** Uses the `trafilatura` library for advanced, ML-based text extraction, with a `BeautifulSoup` fallback to handle complex or non-standard HTML.
* **Advanced NLP Features:** The model doesn't just use `word_count`. It is trained on 8 features, including **Readability** (`textstat`), **Sentiment**, **Entity Count**, and **Part-of-Speech Ratios** (`nltk`) for a much more nuanced prediction.
* **Performance:** The advanced model achieved **90.5% accuracy**, a significant improvement over the baseline (word-count only) model, which scored **61.9%**.
* **Bonus Features:** Includes a fully interactive Streamlit app, bonus data visualizations, and advanced NLP feature engineering.

---

## 📂 Project Structure

```
seo-content-detector/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/               # Contains all output CSVs
│   ├── duplicates.csv
│   ├── extracted_content.csv
│   └── features_advanced.csv
│
├── models/             # Contains all saved artifacts
│   ├── embeddings.npz
│   ├── quality_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── seo_pipeline.ipynb  # Main notebook for analysis & model training
│
└── streamlit_app/
    └── app.py              # The live web application
```

---

## How to Run

### 1. Setup Environment

```bash
# Clone the repository
git clone [https://github.com/your-username/seo-content-detector.git](https://github.com/your-username/seo-content-detector.git)
cd seo-content-detector

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # (or venv\Scripts\activate on Windows)

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook

To see the full analysis and model training process, run the notebook:
```bash
jupyter notebook notebooks/seo_pipeline.ipynb
```

### 3. Run the Streamlit App Locally

To run the interactive web app:
```bash
streamlit run streamlit_app/app.py
```