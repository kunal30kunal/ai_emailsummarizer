import streamlit as st
import re
import joblib
import torch
from transformers import pipeline

# --- Load ML Models (Email Categorizer) ---
@st.cache_resource
def load_classification_models():
    model = joblib.load('email_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    encoder = joblib.load('label_encoder.pkl')
    return model, vectorizer, encoder

# --- Load Smaller Summarizer ---
@st.cache_resource
def load_summarizer():
    device = 0 if torch.cuda.is_available() else -1
    # smaller summarizer model (~250 MB)
    return pipeline("summarization", model="t5-small", tokenizer="t5-small", device=device)

# --- Load Lightweight Generator (for chatbot + reports) ---
@st.cache_resource
def load_generator():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text2text-generation", model="google/flan-t5-small", device=device)

# --- Initialize ---
with st.spinner("Loading lightweight models..."):
    model, vectorizer, encoder = load_classification_models()
    summarizer = load_summarizer()
    generator = load_generator()

# --- Utility Functions ---
def clean_email(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9@.\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_summary(email_text):
    try:
        text = "summarize: " + email_text
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"⚠️ Summary could not be generated: {str(e)}"

def generate_business_report(prompt):
    try:
        text = f"Generate a short business report about: {prompt}"
        output = generator(text, max_length=200, temperature=0.7)
        return output[0]['generated_text']
    except Exception as e:
        return f"⚠️ Error generating report: {str(e)}"

def chatbot_response(query):
    try:
        text = f"Answer briefly and helpfully: {query}"
        output = generator(text, max_length=150, temperature=0.7)
        return output[0]['generated_text']
    except Exception as e:
        return f"⚠️ Chatbot error: {str(e)}"

# --- Streamlit UI ---
st.set_page_config(page_title="Free AI Workspace (Light)", layout="wide")
st.title("🧠 Lightweight AI Workspace: Email + Reports + Chatbot")

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Choose a feature:", [
    "📧 Email Categorizer & Summarizer",
    "📊 Business Report Generator",
    "💬 Internal AI Chatbot"
])

# --- PAGE 1 ---
if page == "📧 Email Categorizer & Summarizer":
    st.header("📧 Email Categorizer & Summarizer (Light Models)")
    email_input = st.text_area("✉️ Enter Email Text Here:", height=180)

    if st.button("🔍 Analyze Email"):
        if email_input.strip():
            with st.spinner("Analyzing email..."):
                processed_text = clean_email(email_input)
                vectorized_text = vectorizer.transform([processed_text])
                probs = model.predict_proba(vectorized_text)[0]
                top3_idx = probs.argsort()[::-1][:3]
                top3_categories = [(encoder.inverse_transform([i])[0], probs[i]) for i in top3_idx]
                summary_text = generate_summary(email_input)

            st.subheader("🏷️ Top 3 Predicted Categories")
            for i, (category, score) in enumerate(top3_categories, start=1):
                st.write(f"**{i}. {category}** — {score*100:.2f}%")

            st.subheader("📝 Summary")
            st.write(summary_text)
        else:
            st.warning("⚠️ Please enter some email text to analyze.")

# --- PAGE 2 ---
elif page == "📊 Business Report Generator":
    st.header("📊 Business Report Generator (Small Model)")
    report_prompt = st.text_area("🧠 Describe the report you need:",
                                 placeholder="Example: Monthly marketing performance...")

    if st.button("📄 Generate Report"):
        if report_prompt.strip():
            with st.spinner("Generating report..."):
                report = generate_business_report(report_prompt)
            st.subheader("🧾 Generated Business Report")
            st.write(report)
        else:
            st.warning("⚠️ Please provide a topic or prompt for the report.")

# --- PAGE 3 ---
elif page == "💬 Internal AI Chatbot":
    st.header("💬 Internal AI Chatbot (Small Model)")
    user_query = st.text_input("🗣️ Ask a question:",
                               placeholder="e.g. What are the project approval steps?")

    if st.button("💡 Ask Chatbot"):
        if user_query.strip():
            with st.spinner("Generating answer..."):
                answer = chatbot_response(user_query)
            st.subheader("🤖 Assistant’s Reply")
            st.write(answer)
        else:
            st.warning("⚠️ Please type a question first.")

