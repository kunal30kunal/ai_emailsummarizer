import streamlit as st
import re
import joblib
from transformers import pipeline
import torch

model = joblib.load('email_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
encoder = joblib.load('label_encoder.pkl')

@st.cache_resource
def load_summarizer():

    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=device)

with st.spinner():
    summarizer = load_summarizer()

def clean_email(text):


    text = text.lower()
    text = re.sub(r'[^a-z0-9@.\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_summary(email_text):

    try:
        max_len = min(150, len(email_text)//2)
        summary = summarizer(email_text, max_length=max_len, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Summary could not be generated: {str(e)}"


st.set_page_config(page_title="AI Email Categorizer & Summarizer", layout="wide")
st.title(" AI-Powered Email Categorizer & Summarizer")
st.write("Paste an email below to classify it into categories and generate a short summary.")

email_input = st.text_area("Enter Email Text Here:", height=180)

if st.button(" Analyze Email"):
    if email_input.strip():
        with st.spinner("Processing email..."):
            processed_text = clean_email(email_input)
            vectorized_text = vectorizer.transform([processed_text])

            probs = model.predict_proba(vectorized_text)[0]
            top3_idx = probs.argsort()[::-1][:3]
            top3_categories = [(encoder.inverse_transform([i])[0], probs[i]) for i in top3_idx]

            summary_text = generate_summary(email_input)

        st.subheader(" Top 3 Predicted Categories")
        for i, (category, score) in enumerate(top3_categories, start=1):
            st.write(f"{i}. {category} — {score*100:.2f}%")

        st.subheader(" Summary")
        with st.expander("View Summary"):
            st.write(summary_text)
    else:
        st.warning("Please enter some email text to analyze.")


