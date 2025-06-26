import streamlit as st
st.set_page_config(layout="wide")

import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from faster_whisper import WhisperModel
import requests
import spacy

# Load spaCy model and cache it
@st.cache_resource

def load_models():
    nlp = spacy.load("en_core_web_sm")
    model = WhisperModel("base", compute_type="int8")
    return model, nlp

model, nlp = load_models()

# Define error weightings and readable names
issue_weights = {
    "typographical": 0.5,
    "misspelling": 0.5,
    "grammar": 1.0,
    "style": 0.8,
    "punctuation": 0.5,
    "redundancy": 0.7,
    "semantics": 1.2,
    "uncategorized": 1.0
}

pretty_error_names = {
    "typographical": "Spelling Errors",
    "misspelling": "Spelling Errors",
    "grammar": "Grammar Mistakes",
    "style": "Style Issues",
    "punctuation": "Punctuation Errors",
    "redundancy": "Repetitions",
    "semantics": "Meaning Errors",
    "uncategorized": "Other Errors"
}

# Grammar evaluation using LanguageTool public API
def evaluate_grammar(text):
    if not text.strip():
        return {
            "Grammar": 0, "Fluency": 0, "Vocabulary": 0, "Total": 0,
            "Grammar Score (10)": 0, "Errors Found": 0, "Weighted Penalty": 0.0,
            "Error Types": {}, "Total Words": 0
        }

    response = requests.post(
        "paste api",
        data={"text": text, "language": "en-US"}
    )
    matches = response.json().get("matches", [])
    total_words = len(text.split())
    total_errors = len(matches)

    errors_per_100 = total_errors / max(total_words, 1) * 100
    grammar_score = 4 if errors_per_100 <= 2 else 3 if errors_per_100 <= 5 else 2 if errors_per_100 <= 9 else 1

    fluency_penalty = sum(1 for m in matches if "word order" in m['message'].lower() or "fragment" in m['message'].lower())
    fluency_score = max(0, 3 - fluency_penalty)

    doc = nlp(text)
    unique_words = set(token.lemma_ for token in doc if token.is_alpha and not token.is_stop)
    vocab_ratio = len(unique_words) / max(total_words, 1)
    vocab_score = 2 if total_words >= 15 and vocab_ratio > 0.4 else 1 if len(unique_words) > 5 else 0

    error_types = {}
    total_weighted_penalty = 0
    for m in matches:
        issue_type = m.get('rule', {}).get('issueType', 'uncategorized').lower()
        readable = pretty_error_names.get(issue_type, 'Unknown Oopsie')
        error_types[readable] = error_types.get(readable, 0) + 1
        total_weighted_penalty += issue_weights.get(issue_type, 1.0)

    grammar_score_10 = max(0, round(10 - total_weighted_penalty, 2))
    total_score = round(grammar_score + fluency_score + vocab_score, 2)

    return {
        "Grammar": grammar_score,
        "Fluency": fluency_score,
        "Vocabulary": vocab_score,
        "Total": total_score,
        "Grammar Score (10)": grammar_score_10,
        "Errors Found": total_errors,
        "Weighted Penalty": round(total_weighted_penalty, 2),
        "Error Types": error_types,
        "Total Words": total_words
    }

# Transcription function
def transcribe_audio(audio_path):
    segments, _ = model.transcribe(audio_path)
    transcription = " ".join([seg.text for seg in segments])
    return transcription

# Streamlit App UI
st.set_page_config(layout="wide")
st.title("🎙️ Grammar Scoring System from Spoken Audios")

uploaded_file = st.file_uploader("Upload your Audio file (mp3 or wav)", type=["mp3", "wav"])
if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    st.audio(uploaded_file, format=f"audio/{suffix[1:]}")

    with st.spinner("Transcribing and analyzing..."):
        transcribed_text = transcribe_audio(tmp_file_path)
        result = evaluate_grammar(transcribed_text)
        os.remove(tmp_file_path)

    st.subheader("📜 Transcription:")
    st.write(transcribed_text)

    st.subheader("📊 Scores:")
    st.write(result)

    # Visualizations
    st.subheader("📈 Score Breakdown")
    score_df = pd.DataFrame({"Category": ["Grammar", "Fluency", "Vocabulary"], "Score": [result['Grammar'], result['Fluency'], result['Vocabulary']]})
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(data=score_df, x="Category", y="Score", palette="Set2", ax=ax)
    ax.set_ylim(0, 5)
    st.pyplot(fig)

    st.subheader("❌ Error Type Distribution")
    if result['Error Types']:
        error_df = pd.DataFrame(list(result['Error Types'].items()), columns=['Error Type', 'Count'])
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        sns.barplot(data=error_df, x='Error Type', y='Count', palette='Reds_r', ax=ax2)
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)
    else:
        st.write("No errors detected! 🎉")
