# -------------------- app.py --------------------
import streamlit as st
import re
import joblib
from transformers import pipeline  # Make sure transformers >= 4.30 is installed

# Optional: import torch to avoid internal errors
import torch

# -------------------- LOAD SAVED ARTIFACTS --------------------
# Ensure these files are in the same folder as app.py
model = joblib.load('email_model.pkl')           # Trained classifier
vectorizer = joblib.load('tfidf_vectorizer.pkl') # TF-IDF vectorizer
encoder = joblib.load('label_encoder.pkl')       # Label encoder for categories

# -------------------- INITIALIZE SUMMARIZATION PIPELINE --------------------
@st.cache_resource
def load_summarizer():
    """
    Load a small transformer summarization model.
    Cached to prevent reloading on each app refresh.
    """
    return pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

with st.spinner("Loading AI summarizer — this may take a few seconds..."):
    summarizer = load_summarizer()

# -------------------- TEXT CLEANING FUNCTION --------------------
def clean_email(text):
    """
    Preprocess email text by lowercasing, removing special characters, and extra spaces.
    """
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------- SUMMARIZATION FUNCTION --------------------
def generate_summary(email_text):
    """
    Generate a concise summary of the given email text using a local summarizer model.
    """
    try:
        summary = summarizer(email_text, max_length=60, min_length=20, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Summary could not be generated: {str(e)}"

# -------------------- STREAMLIT INTERFACE --------------------
st.title("📩 AI-Powered Email Categorizer & Summarizer")
st.write("Paste an email below to classify it into a category and generate a short summary.")

# Input box for email text
email_input = st.text_area("✉️ Enter Email Text Here:", height=180)

# Button to trigger prediction
if st.button("🔍 Analyze Email"):
    if email_input.strip():
        # Step 1: Clean & classify
        processed_text = clean_email(email_input)
        vectorized_text = vectorizer.transform([processed_text])
        predicted_label = model.predict(vectorized_text)
        predicted_category = encoder.inverse_transform(predicted_label)[0]

        # Step 2: Generate summary
        summary_text = generate_summary(email_input)

        # -------------------- DISPLAY RESULTS --------------------
        st.subheader("📂 Predicted Category")
        st.success(predicted_category)

        st.subheader("🧠 AI-Generated Summary")
        st.info(summary_text)

        # Optional download button
        st.download_button(
            label="📄 Download Summary",
            data=summary_text,
            file_name="email_summary.txt",
            mime="text/plain"
        )
    else:
        st.warning("Please enter some email text to classify and summarize.")
