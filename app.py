import streamlit as st
import pandas as pd
import os
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
os.makedirs("results", exist_ok=True)

# --- Extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# --- Clean and preprocess text
def clean_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num]
    return " ".join(tokens)

# --- TF-IDF similarity
def rank_resumes(resumes, jd_text):
    cleaned_resumes = [clean_text(resume) for resume in resumes]
    cleaned_jd = clean_text(jd_text)

    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([cleaned_jd] + cleaned_resumes)
    similarity_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return similarity_scores

# --- Streamlit UI Setup
st.set_page_config(page_title="Resume Screening Tool", layout="wide")

# --- Pine Green Theme Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #01796F;
        font-family: 'Segoe UI', sans-serif;
    }

    h1 {
        text-align: center;
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin-top: 30px;
        text-shadow: 1px 1px 3px #004D45;
    }

    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #DFF9F7;
        margin-bottom: 2rem;
    }

    label, .css-1cpxqw2 {
        color: #F0F0F0 !important;
        font-weight: 500;
    }

    textarea, .stTextInput > div > div, .stFileUploader {
        background-color: #E8F5F4 !important;
        border: 1px solid #004D45 !important;
        border-radius: 8px !important;
        color: #013934 !important;
    }

    .stButton {
        margin-top: 20px;
    }

    .stButton > button {
        background-color: #015F55;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
    }

    .stButton > button:hover {
        background-color: #014741;
    }

    .css-1d391kg td {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        vertical-align: top;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title
st.markdown("<h1>Resume Screening Tool</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Paste a job description and upload multiple resumes. Get ranked candidates instantly!</div>", unsafe_allow_html=True)

# --- Inputs
job_description = st.text_area("Paste Job Description", height=200)
uploaded_files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

# --- Evaluate Button
if st.button("Evaluate Candidates"):
    if not uploaded_files or not job_description:
        st.warning("Please upload at least one resume and enter a job description.")
    else:
        resume_texts = []
        filenames = []

        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resume_texts.append(text)
            filenames.append(file.name)

        scores = rank_resumes(resume_texts, job_description)
        jd_keywords = set(clean_text(job_description).split())

        result_data = []
        for i in range(len(filenames)):
            resume_clean = clean_text(resume_texts[i])
            resume_words = set(resume_clean.split())
            missing = list(jd_keywords - resume_words)
            match_score = round(scores[i] * 100, 2)

            if match_score >= 80:
                rating = "Excellent"
            elif match_score >= 50:
                rating = "Good"
            else:
                rating = "Low"

            result_data.append({
                "Filename": filenames[i],
                "Match %": match_score,
                "Rating": rating,
                "Missing Keywords": ", ".join(missing[:10]),
                "Suggestions": "Add more job-specific keywords." if match_score < 80 else "Resume matches well."
            })

        df = pd.DataFrame(result_data)
        df = df.sort_values(by="Match %", ascending=False).reset_index(drop=True)
        df.insert(3, "Rank", df.index + 1)
        df = df[["Filename", "Match %", "Rating", "Rank", "Missing Keywords", "Suggestions"]]

        # --- Show Final Table
        st.markdown("<h3 style='font-weight: bold; color: white;'>Full Candidate Report</h3>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        # --- Save to Excel
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_path = f"results/final_ranking_{now}.xlsx"
        df.to_excel(file_path, index=False)

        with open(file_path, "rb") as f:
            st.download_button("⬇️ Download Result as Excel", data=f, file_name="ranked_candidates.xlsx")
