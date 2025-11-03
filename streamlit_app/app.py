import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import re
import os
from scipy.sparse import load_npz

# --- NLP & Parsing Imports ---
from bs4 import BeautifulSoup
import trafilatura
import textstat
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import ne_chunk, pos_tag, word_tokenize
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Configuration ---
st.set_page_config(
    page_title="SEO Content Analyzer",
    page_icon="🔍",
    layout="wide"
)

# --- Download NLTK Data ---
@st.cache_resource
def download_nltk_data():
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    return True

with st.spinner("Loading NLTK data..."):
    download_nltk_data()

# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    """
    Loads all saved ML artifacts and dataset references.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "quality_model.pkl")
    VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
    EMBEDDINGS_PATH = os.path.join(BASE_DIR, "models", "embeddings.npz")
    DATA_PATH = os.path.join(BASE_DIR, "data", "features_advanced.csv")

    try:
        model = pickle.load(open(MODEL_PATH, 'rb'))
        vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
        tfidf_matrix = load_npz(EMBEDDINGS_PATH)
        all_urls = pd.read_csv(DATA_PATH)['url'].tolist()
        return model, vectorizer, tfidf_matrix, all_urls
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}. Ensure all model/data files are present.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None, None

with st.spinner("Loading models and vectorizers..."):
    (model, vectorizer, tfidf_matrix, all_urls) = load_artifacts()

# --- HTTP Fetch ---
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
HEADERS = {'User-Agent': USER_AGENT}

def fetch_html(url, timeout=10):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"⚠️ Failed to fetch {url}: {e}")
        return None

# --- HTML Extraction with Fallback ---
def extract_title_and_body(html):
    """
    Extracts title & main content.
    Uses trafilatura first, falls back to BeautifulSoup if it fails.
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        # Try trafilatura
        body_text = trafilatura.extract(html, include_comments=False, include_tables=False, deduplicate=True)
        method_used = "trafilatura"

        # Fallback if empty or too short
        if not body_text or len(body_text) < 200:
            paragraphs = soup.find_all(['p', 'article', 'main'])
            body_text = " ".join([p.get_text(strip=True) for p in paragraphs])
            method_used = "BeautifulSoup (fallback)"

        if not body_text or len(body_text.strip()) < 100:
            return title, None, method_used

        body_text = re.sub(r'\s+', ' ', body_text).strip()
        return title, body_text, method_used

    except Exception as e:
        print(f"Extraction failed: {e}")
        return None, None, "Error"

# --- Advanced NLP Analysis ---
def advanced_nlp_analysis(text):
    results = {}
    try:
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(text)
        results['sentiment_compound'] = round(sentiment['compound'], 3)
        results['sentiment_label'] = (
            'Positive' if sentiment['compound'] > 0.05
            else 'Negative' if sentiment['compound'] < -0.05
            else 'Neutral'
        )

        tokens = word_tokenize(text[:5000])
        pos_tags = pos_tag(tokens)
        chunks = ne_chunk(pos_tags)
        entities = [chunk.label() for chunk in chunks if hasattr(chunk, 'label')]
        results['entity_count'] = len(entities)

        pos_counts = Counter([tag for _, tag in pos_tags])
        total_tags = max(1, len(pos_tags))
        results['noun_ratio'] = round(sum(v for k, v in pos_counts.items() if k.startswith('NN')) / total_tags, 3)
        results['verb_ratio'] = round(sum(v for k, v in pos_counts.items() if k.startswith('VB')) / total_tags, 3)
        results['adj_ratio'] = round(sum(v for k, v in pos_counts.items() if k.startswith('JJ')) / total_tags, 3)

    except Exception:
        results = {'sentiment_compound': 0, 'sentiment_label': 'Unknown',
                   'entity_count': 0, 'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0}
    return pd.Series(results)

# --- Main Analysis Pipeline ---
def analyze_url_streamlit(url):
    if not model:
        return {'error': 'Model not loaded. Please check artifacts.'}

    try:
        html = fetch_html(url)
        if not html:
            return {'error': 'Failed to fetch URL (timeout, redirect, or block).'}

        title, body, method_used = extract_title_and_body(html)
        if not body:
            return {'error': 'Failed to extract main article content.'}

        clean_text = re.sub(r'\s+', ' ', body).strip().lower()
        wc = len(clean_text.split())
        sc = len(nltk.sent_tokenize(clean_text)) if clean_text else 0
        flesch = textstat.flesch_reading_ease(clean_text)
        nlp_features = advanced_nlp_analysis(clean_text)

        features_vector = [[
            wc, sc, flesch,
            nlp_features['sentiment_compound'],
            nlp_features['entity_count'],
            nlp_features['noun_ratio'],
            nlp_features['verb_ratio'],
            nlp_features['adj_ratio']
        ]]

        quality_label = model.predict(features_vector)[0]

        url_tfidf_vector = vectorizer.transform([clean_text])
        sim_scores = cosine_similarity(url_tfidf_vector, tfidf_matrix)[0]
        top_indices = sim_scores.argsort()[-10:][::-1]

        similar_to = [
            {'url': all_urls[i], 'similarity': round(float(sim_scores[i]), 4)}
            for i in top_indices if sim_scores[i] > 0.1
        ]

        return {
            "url": url,
            "title": title,
            "quality_label": quality_label,
            "word_count": wc,
            "readability": round(flesch, 2),
            "is_thin": bool(wc < 500),
            "sentiment": nlp_features['sentiment_label'],
            "entity_count": nlp_features['entity_count'],
            "similar_to": similar_to,
            "extraction_method": method_used
        }

    except Exception as e:
        return {'error': str(e), 'message': 'Pipeline failed.'}

# --- Streamlit UI ---
st.title("🔍 SEO Content & Duplicate Analyzer")
st.markdown("Analyze any webpage’s content quality, readability, and similarity to known articles.")

with st.expander("ℹ️ About This Project"):
    st.write("""
        This app analyzes web pages for **SEO content quality** and **duplicate detection** using NLP and ML.
        This project was completed for a Data Science job placement assignment.
        
        **Pipeline Overview:**
        - **Text Extraction:** Uses `trafilatura` with a `BeautifulSoup` fallback for robust parsing.
        - **Feature Engineering:** Calculates 8 features (Word Count, Readability, Sentiment, Entity Count, POS Ratios).
        - **Model:** A Random Forest model classifies content as High, Medium, or Low quality.
        - **Duplicate Check:** Uses TF-IDF + Cosine Similarity to find similar articles.

        ---

        **Model Performance:**
        * **Advanced Model (8 features): 90.5% Accuracy**
        * **Baseline Model (Word count only): 61.9% Accuracy**
    """)
   

url_input = st.text_input("Enter a URL", placeholder="https://example.com/article")

if st.button("Analyze", type="primary"):
    if url_input and model:
        with st.spinner("Analyzing..."):
            result = analyze_url_streamlit(url_input)

        if "error" in result:
            st.error(f"❌ {result['error']}")
        else:
            st.success("✅ Analysis Complete!")
            method_color = "#27ae60" if "trafilatura" in result['extraction_method'].lower() else "#f39c12"
            st.markdown(
                f"<p style='color:grey;'>Extraction method: <b style='color:{method_color};'>{result['extraction_method']}</b></p>",
                unsafe_allow_html=True
            )

            # --- Summary Metrics ---
            col1, col2, col3, col4 = st.columns(4)
            color_map = {"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"}
            quality_color = color_map.get(result['quality_label'], "grey")
            col1.markdown(
                f"<h3 style='color:{quality_color}; margin-top:-10px;'>{result['quality_label']}</h3>"
                f"<p style='color:grey; margin-top:-10px;'>Quality Label</p>", unsafe_allow_html=True
            )
            col2.metric("Word Count", result['word_count'])
            col3.metric("Readability", result['readability'])
            col4.metric("Sentiment", result['sentiment'])

            st.markdown("---")

            # --- Similar Content Section ---
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Similar Content Found")
                scores_df = pd.DataFrame(result["similar_to"])
                if not scores_df.empty:
                    top_n = st.slider("Show Top N Similar", 1, len(scores_df), 3)
                    chart_df = scores_df.head(top_n).copy()
                    chart_df['similarity_pct'] = chart_df['similarity'] * 100
                    chart_df['short_url'] = chart_df['url'].apply(lambda x: x.split('//')[-1][:40] + '...')
                    st.bar_chart(chart_df, x='short_url', y='similarity_pct', use_container_width=True)
                else:
                    st.info("No similar content found.")

            with col2:
                st.subheader("Full Analysis (JSON)")
                st.json(result)

    elif not model:
        st.error("❌ Model not loaded properly.")
    else:
        st.warning("⚠️ Please enter a valid URL.")
