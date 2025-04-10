 
import streamlit as st
st.set_page_config(page_title="Grammar-Scoring-System", layout="centered")

from faster_whisper import WhisperModel
import spacy
import language_tool_python
from textstat import flesch_reading_ease
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

@st.cache_resource
def load_models():
    # Use tiny or base model with optimized CPU settings
    whisper_model = WhisperModel("tiny", compute_type="int8")
    nlp_model = spacy.load("en_core_web_sm")
    grammar_tool = language_tool_python.LanguageTool('en-US')
    return whisper_model, nlp_model, grammar_tool

model, nlp, tool = load_models()

pretty_error_names = {
    'grammar': 'Grammar Goof',
    'misspelling': 'Spelling Slip',
    'typographical': 'Typo Trouble',
    'punctuation': 'Punctuation Panic',
    'style': 'Style Slip-Up',
    'non-conformance': 'Formatting Fail',
    'uncategorized': 'Mystery Mistake'
}

issue_weights = {
    'grammar': 1.5,
    'misspelling': 0.5,
    'typographical': 0.5,
    'punctuation': 0.5,
    'style': 0.2,
    'non-conformance': 1.0,
    'uncategorized': 1.0
}

def evaluate_grammar(text):
    if not text.strip():
        return {
            "Grammar": 0, "Fluency": 0, "Vocabulary": 0, "Total": 0,
            "Grammar Score (10)": 0, "Errors Found": 0, "Weighted Penalty": 0.0,
            "Error Types": {}, "Total Words": 0
        }

    matches = tool.check(text)
    total_words = len(text.split())
    total_errors = len(matches)

    errors_per_100 = total_errors / max(total_words, 1) * 100
    grammar_score = 4 if errors_per_100 <= 2 else 3 if errors_per_100 <= 5 else 2 if errors_per_100 <= 9 else 1

    fluency_penalty = sum(1 for m in matches if "word order" in m.message.lower() or "fragment" in m.message.lower())
    fluency_score = max(0, 3 - fluency_penalty)

    doc = nlp(text)
    unique_words = set(token.lemma_ for token in doc if token.is_alpha and not token.is_stop)
    vocab_ratio = len(unique_words) / max(total_words, 1)
    vocab_score = 2 if total_words >= 15 and vocab_ratio > 0.4 else 1 if len(unique_words) > 5 else 0

    error_types = {}
    total_weighted_penalty = 0
    for m in matches:
        issue_type = (m.ruleIssueType or 'uncategorized').lower()
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

def semantic_score(text):
    readability = flesch_reading_ease(text)
    return 3 if readability > 80 else 2 if readability > 60 else 1

def interpret_score(score):
    if score >= 9:
        return "Advanced (A)"
    elif score >= 7:
        return "Upper-Intermediate (B)"
    elif score >= 5:
        return "Intermediate (C)"
    elif score >= 3:
        return "Beginner (D)"
    else:
        return "Basic (F)"

def plot_error_types(error_types):
    if not error_types:
        st.success("Clean slate! No grammar crimes found.")
        return

    labels = list(error_types.keys())
    counts = list(error_types.values())

    plt.figure(figsize=(8, 4))
    sns.barplot(x=counts, y=labels, palette="crest")
    plt.xlabel("How Many Times You Messed Up")
    plt.ylabel("Type of Slip-Up")
    plt.title("The Breakdown of Your Blunders")
    st.pyplot(plt)

def download_csv(scores_dict):
    df = pd.DataFrame([scores_dict])
    return df.to_csv(index=False).encode('utf-8')
st.write("Drop your audio below and let this savage tool go full English teacher mode on you. It'll catch your grammar crimes, vocab blunders, and flow fails.")

uploaded_file = st.file_uploader("Upload your audio file (WAV, MP3, M4A):", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    with st.spinner("🔍 Transcribing your audio..."):
        segments, _ = model.transcribe(temp_audio_path)
        transcribed_text = " ".join([segment.text for segment in segments])

    st.subheader("📜 Here's What You Said")
    st.write(transcribed_text)

    with st.spinner("📚 Evaluating grammar and vocabulary..."):
        score = evaluate_grammar(transcribed_text)
        score["Semantic"] = semantic_score(transcribed_text)
        score["Final Total"] = round(0.4 * score["Total"] + 0.6 * score["Semantic"], 2)
        score["CEFR Level"] = interpret_score(score["Final Total"])

    st.subheader("📊 Your Scorecard")
    for key, val in score.items():
        if key != "Error Types":
            st.write(f"{key}: {val}")

    st.subheader("🧩 Your Mistake Report")
    plot_error_types(score.get("Error Types", {}))

    st.download_button(
        label="📥 Download My Roast Report as CSV",
        data=download_csv(score),
        file_name="audio_language_analysis.csv",
        mime="text/csv"
    )

    os.remove(temp_audio_path)
